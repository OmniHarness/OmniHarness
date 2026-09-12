"""OmniHarness feedback-guided execution and symbolic policy learning.

The runtime implements the method-level loop described by the OmniHarness
story:

1. retrieve symbolic policies and corrective strategies;
2. deliberate and verify an ordered plan;
3. synthesize Code-as-Policy and compile it without executing Python;
4. execute the resulting ComfyUI graph and verify it at technical, step, and
   semantic levels;
5. localize failures and integrate a repaired subworkflow within a fixed retry
   budget;
6. evolve ``S_t = (L_t, F_t)``, periodically consolidate it, and optionally
   export a frozen plug-and-play snapshot.

Model-backed judgment has no write or execution authority.  Deterministic
Python owns compilation, graph validation, ComfyUI I/O, retry/time budgets,
workflow admission, reliability statistics, and atomic memory updates.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import keyword
import mimetypes
import os
import re
import shutil
import struct
import tempfile
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from self_directed_inquiry import (
    CODEX_API_KEY_ENV,
    CODEX_BASE_URL,
    CODEX_MODEL,
    CODEX_MODEL_PROVIDER,
    CODEX_REASONING_EFFORT,
    CodexCandidateGenerator,
    SelfDirectedTaskProposer,
    Modality,
    ProposalRound,
    ReliabilityTier,
    ResourceInventory,
    TaskContract,
    config_section,
    context_capability_key,
    extract_json_object,
    load_proposal_state,
    load_omniharness_config,
    modality_requires_source,
    normalize_text,
    resolve_config_path,
    repository_root,
    runtime_resources_root,
    to_primitive,
    validate_provider_base_url,
    write_proposal_audit,
)
from symbolic_policy import (
    contains_unbound_policy_input,
    distill_workflow_graph,
    redact_instance_values,
)


class MemoryMode(str, Enum):
    ONLINE = "online"
    FROZEN = "frozen"


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    DRY_RUN = "dry_run"


class VerificationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


class PlanStrategy(str, Enum):
    REUSE = "reuse"
    ADAPT = "adapt"
    COMPOSE = "compose"
    BUILD = "build"


class FailureClass(str, Enum):
    PLANNING = "planning_failure"
    COMPILATION = "compilation_failure"
    INFRASTRUCTURE = "infrastructure_failure"
    EXECUTION = "execution_failure"
    VERIFIER = "verifier_failure"
    SEMANTIC = "semantic_failure"


@dataclass(frozen=True, init=False)
class ExecutionBudget:
    max_attempts: int = 5
    max_plan_repairs: int = 2
    max_wall_time_seconds: int = 1200

    def __init__(
        self,
        max_attempts: int | None = None,
        max_plan_repairs: int = 2,
        max_wall_time_seconds: int = 1200,
        *,
        max_retries: int | None = None,
    ) -> None:
        if max_attempts is not None and max_retries is not None:
            raise ValueError("max_retries and max_attempts are mutually exclusive")
        if max_retries is not None and (not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0):
            raise ValueError("max_retries must be a non-negative integer")
        attempts = max_attempts if max_attempts is not None else (4 if max_retries is None else max_retries) + 1
        object.__setattr__(self, "max_attempts", attempts)
        object.__setattr__(self, "max_plan_repairs", max_plan_repairs)
        object.__setattr__(self, "max_wall_time_seconds", max_wall_time_seconds)
        self.__post_init__()

    def __post_init__(self) -> None:
        values = (self.max_attempts, self.max_plan_repairs, self.max_wall_time_seconds)
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError("execution budgets must be positive integers")

    @property
    def max_retries(self) -> int:
        return self.max_attempts - 1


def resolve_execution_budget(
    config: Mapping[str, Any],
    *,
    max_retries: int | None = None,
    max_attempts: int | None = None,
    max_wall_time_seconds: int | None = None,
) -> ExecutionBudget:
    """Resolve CLI overrides, accepting the legacy total-attempts setting."""
    if config.get("max_retries") is not None and config.get("max_attempts") is not None:
        raise ValueError("runtime.max_retries and runtime.max_attempts are mutually exclusive")
    if max_retries is None and max_attempts is None:
        max_retries = config.get("max_retries")
        max_attempts = config.get("max_attempts")

    def integer(value: Any, name: str) -> int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value.strip()):
            return int(value)
        raise ValueError(f"{name} must be an integer")

    return ExecutionBudget(
        max_attempts=None if max_attempts is None else integer(max_attempts, "max_attempts"),
        max_retries=None if max_retries is None else integer(max_retries, "max_retries"),
        max_plan_repairs=integer(config.get("max_plan_repairs", 2), "max_plan_repairs"),
        max_wall_time_seconds=integer(config.get("max_wall_time_seconds", 1200) if max_wall_time_seconds is None else max_wall_time_seconds, "max_wall_time_seconds"),
    )


@dataclass(frozen=True)
class ExecutionRequest:
    task_id: str
    instruction: str
    modality: Modality
    capability_categories: tuple[str, ...]
    source_images: tuple[str, ...] = ()
    source_image_id: str | None = None
    generation_constraints: tuple[Mapping[str, Any], ...] = ()
    preservation_constraints: tuple[Mapping[str, Any], ...] = ()
    success_criteria: tuple[Mapping[str, Any], ...] = ()
    preferred_workflow_ids: tuple[str, ...] = ()
    preferred_failure_ids: tuple[str, ...] = ()
    task_signature: str = ""
    inquiry_iteration: int | None = None
    memory_mode: MemoryMode = MemoryMode.ONLINE
    budget: ExecutionBudget = field(default_factory=ExecutionBudget)

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.instruction.strip():
            raise ValueError("task_id and instruction are required")
        if modality_requires_source(self.modality) and not self.source_images:
            raise ValueError(f"{self.modality.value} execution requires source media")
        if self.inquiry_iteration is not None:
            if self.inquiry_iteration < 1:
                raise ValueError("inquiry_iteration must be positive")
            if self.modality not in {Modality.T2I, Modality.I2I}:
                raise ValueError("self-directed inquiry is image-only; video tasks belong to downstream execution")

    @classmethod
    def from_contract(
        cls,
        contract: TaskContract,
        *,
        memory_mode: MemoryMode = MemoryMode.ONLINE,
        budget: ExecutionBudget | None = None,
    ) -> "ExecutionRequest":
        return cls(
            task_id=contract.task_id,
            instruction=contract.description,
            modality=contract.modality,
            capability_categories=contract.capability_categories,
            source_images=(
                contract.source_image_paths
                if contract.source_image_paths
                else (() if contract.source_image_path is None else (contract.source_image_path,))
            ),
            source_image_id=contract.source_image_id,
            generation_constraints=tuple(to_primitive(item) for item in contract.generation_constraints),
            preservation_constraints=tuple(to_primitive(item) for item in contract.preservation_constraints),
            success_criteria=tuple(to_primitive(item) for item in contract.success_criteria),
            preferred_workflow_ids=contract.retrieved_workflow_ids,
            preferred_failure_ids=contract.retrieved_failure_ids,
            task_signature=contract.task_signature,
            inquiry_iteration=contract.inquiry_iteration,
            memory_mode=memory_mode,
            budget=budget or ExecutionBudget(),
        )


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    objective: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    required_nodes: tuple[str, ...]
    verification_type: str
    verification_criteria: tuple[str, ...]
    expected_output_nodes: tuple[str, ...]
    fallback: str = "repair_failed_step"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlanStep":
        verification = data.get("verification", {})
        if not isinstance(verification, Mapping):
            verification = {}
        return cls(
            step_id=str(data["step_id"]),
            objective=str(data["objective"]),
            inputs=_strings(data.get("inputs", ())),
            outputs=_strings(data.get("outputs", ())),
            required_nodes=_strings(data.get("required_nodes", ())),
            verification_type=str(verification.get("type", data.get("verification_type", "technical"))),
            verification_criteria=_strings(verification.get("criteria", data.get("verification_criteria", ()))),
            expected_output_nodes=_strings(data.get("expected_output_nodes", ())),
            fallback=str(data.get("fallback", "repair_failed_step")),
        )


@dataclass(frozen=True)
class ExecutionPlan:
    plan_id: str
    task_analysis: Mapping[str, Any]
    strategy: PlanStrategy
    selected_workflow_ids: tuple[str, ...]
    steps: tuple[PlanStep, ...]
    rationale: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionPlan":
        return cls(
            plan_id=str(data["plan_id"]),
            task_analysis=dict(data.get("task_analysis", {})),
            strategy=PlanStrategy(str(data["strategy"]).lower()),
            selected_workflow_ids=_strings(data.get("selected_workflow_ids", ())),
            steps=tuple(PlanStep.from_dict(item) for item in data.get("steps", ())),
            rationale=str(data.get("rationale", "")),
        )


@dataclass(frozen=True)
class WorkflowDraft:
    ir_code: str
    workflow_name: str
    description: str
    preconditions: str
    expected_effects: str
    dependencies: Mapping[str, Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowDraft":
        return cls(
            ir_code=str(data["ir_code"]),
            workflow_name=str(data.get("workflow_name", "OmniHarness_Workflow")),
            description=str(data.get("description", "OmniHarness synthesized workflow")),
            preconditions=str(data.get("preconditions", "Required inputs and dependencies are available.")),
            expected_effects=str(data.get("expected_effects", data.get("function", "Produces the requested visual result."))),
            dependencies=dict(data.get("dependencies", {})),
        )


@dataclass(frozen=True)
class LocalizedRepair:
    failed_step_id: str
    root_cause: str
    subworkflow_ir: str
    revised_ir_code: str
    reusable_name: str
    rationale: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LocalizedRepair":
        return cls(
            failed_step_id=str(data.get("failed_step_id", "unknown")),
            root_cause=str(data.get("root_cause", "")),
            subworkflow_ir=str(data.get("subworkflow_ir", "")),
            revised_ir_code=str(data["revised_ir_code"]),
            reusable_name=str(data.get("reusable_name", "Localized_Recovery")),
            rationale=str(data.get("rationale", "")),
        )


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    node_id: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class NodeInput:
    name: str
    type: str
    required: bool


@dataclass(frozen=True)
class NodeOutput:
    name: str
    type: str


@dataclass(frozen=True)
class NodeDefinition:
    class_type: str
    identifier: str
    description: str
    inputs: tuple[NodeInput, ...]
    outputs: tuple[NodeOutput, ...]
    output_node: bool = False

    @property
    def required_inputs(self) -> tuple[NodeInput, ...]:
        return tuple(item for item in self.inputs if item.required)


class NodeCatalog:
    def __init__(self, nodes: Iterable[NodeDefinition]) -> None:
        self._nodes = {item.class_type: item for item in nodes}
        self._identifiers = {item.identifier: item.class_type for item in nodes}

    def get(self, class_type: str) -> NodeDefinition | None:
        return self._nodes.get(class_type)

    def resolve(self, identifier: str) -> str | None:
        return identifier if identifier in self._nodes else self._identifiers.get(identifier)

    @property
    def class_types(self) -> tuple[str, ...]:
        return tuple(self._nodes)

    @classmethod
    def from_object_info(cls, payload: Mapping[str, Any]) -> "NodeCatalog":
        nodes: list[NodeDefinition] = []
        for class_type, raw in payload.items():
            if not isinstance(raw, Mapping):
                continue
            inputs: list[NodeInput] = []
            input_info = raw.get("input", {})
            if isinstance(input_info, Mapping):
                for required, section in ((True, "required"), (False, "optional")):
                    values = input_info.get(section, {})
                    if not isinstance(values, Mapping):
                        continue
                    for name, specification in values.items():
                        inputs.append(NodeInput(str(name), _node_type(specification), required))
            raw_outputs = raw.get("output", ())
            output_names = raw.get("output_name", ())
            outputs = tuple(
                NodeOutput(
                    str(output_names[index]) if index < len(output_names) else f"output_{index}",
                    _node_type(value),
                )
                for index, value in enumerate(raw_outputs)
            )
            nodes.append(
                NodeDefinition(
                    class_type=str(class_type),
                    identifier=_safe_identifier(str(class_type)),
                    description=str(raw.get("description", raw.get("display_name", ""))),
                    inputs=tuple(inputs),
                    outputs=outputs,
                    output_node=bool(raw.get("output_node", False)),
                )
            )
        return cls(nodes)

    def compact(self, query: str, *, required: Iterable[str] = (), limit: int = 100) -> tuple[Mapping[str, Any], ...]:
        query_tokens = _tokens(query)
        selected: dict[str, NodeDefinition] = {}
        for class_type in required:
            item = self.get(class_type)
            if item is not None:
                selected[class_type] = item
        ranked: list[tuple[int, str, NodeDefinition]] = []
        for item in self._nodes.values():
            haystack = _tokens(f"{item.class_type} {item.identifier} {item.description}")
            overlap = len(query_tokens & haystack)
            if overlap:
                ranked.append((overlap, item.class_type, item))
        ranked.sort(key=lambda value: (-value[0], value[1]))
        for _, _, item in ranked:
            selected.setdefault(item.class_type, item)
            if len(selected) >= limit:
                break
        return tuple(
            {
                "class_type": item.class_type,
                "identifier": item.identifier,
                "description": item.description[:500],
                "inputs": to_primitive(item.inputs),
                "outputs": to_primitive(item.outputs),
            }
            for item in selected.values()
        )


@dataclass(frozen=True)
class CompilationResult:
    prompt: Mapping[str, Mapping[str, Any]]
    provenance: Mapping[str, int]
    class_types: tuple[str, ...]


@dataclass(frozen=True)
class OutputArtifact:
    node_id: str
    output_type: str
    filename: str
    path: str


@dataclass(frozen=True)
class ComfyExecutionResult:
    prompt_id: str
    status: Mapping[str, Any]
    outputs: tuple[OutputArtifact, ...]
    history: Mapping[str, Any]


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    verifier: str
    scores: Mapping[str, float] = field(default_factory=dict)
    evidence: tuple[str, ...] = ()
    failed_criteria: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VerificationResult":
        return cls(
            status=VerificationStatus(str(data["status"]).lower()),
            verifier=str(data.get("verifier", "goal_verifier")),
            scores={str(key): float(value) for key, value in dict(data.get("scores", {})).items()},
            evidence=_strings(data.get("evidence", ())),
            failed_criteria=_strings(data.get("failed_criteria", ())),
        )


@dataclass(frozen=True)
class FailureDiagnosis:
    failure_class: FailureClass
    failed_stage: str
    root_cause: str
    workflow_antipattern: str
    remedy: str
    applicable_scope: str
    retry_scope: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FailureDiagnosis":
        return cls(
            failure_class=FailureClass(str(data["failure_class"]).lower()),
            failed_stage=str(data.get("failed_stage", "unknown")),
            root_cause=str(data.get("root_cause", "Unknown failure")),
            workflow_antipattern=str(data.get("workflow_antipattern", data.get("antipattern", "Unclassified antipattern"))),
            remedy=str(data.get("remedy", "Repair the failed stage")),
            applicable_scope=str(data.get("applicable_scope", "visual generation workflow")),
            retry_scope=str(data.get("retry_scope", "failed_step")),
        )


@dataclass(frozen=True)
class AttemptRecord:
    attempt: int
    submitted: bool
    workflow_path: str | None
    execution: ComfyExecutionResult | None = None
    step_verification: VerificationResult | None = None
    goal_verification: VerificationResult | None = None
    diagnosis: FailureDiagnosis | None = None
    localized_repair: LocalizedRepair | None = None
    error: str | None = None


@dataclass(frozen=True)
class AdmissionDecision:
    valid: bool
    novel: bool
    admitted: bool
    fingerprint: str | None
    matched_workflow_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class MemoryUpdate:
    updated: bool
    registered_workflow_id: str | None
    consolidated: bool


@dataclass(frozen=True)
class ExecutionOutcome:
    task_id: str
    status: ExecutionStatus
    plan: ExecutionPlan | None
    attempts: tuple[AttemptRecord, ...]
    selected_output_paths: tuple[str, ...]
    workflow_ids: tuple[str, ...]
    registered_workflow_id: str | None
    memory_updated: bool
    message: str


@dataclass(frozen=True)
class PolicyContext:
    workflows: tuple[Mapping[str, Any], ...]
    workflow_prompts: Mapping[str, Mapping[str, Any]]
    failures: tuple[Mapping[str, Any], ...]
    node_knowledge: tuple[Mapping[str, Any], ...]

    def compact(self, *, include_prompts: bool) -> Mapping[str, Any]:
        result: dict[str, Any] = {
            "workflows": self.workflows,
            "failure_library": self.failures,
            "node_knowledge": self.node_knowledge,
        }
        if include_prompts:
            result["workflow_prompts"] = self.workflow_prompts
        return result


def compile_code_as_policy(code: str, catalog: NodeCatalog) -> CompilationResult:
    """Compile a restricted Python-like DSL; the supplied code is never run."""

    try:
        module = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"invalid Code-as-Policy syntax: {exc}") from exc
    prompt: dict[str, dict[str, Any]] = {}
    variables: dict[str, list[Any]] = {}
    provenance: dict[str, int] = {}
    class_types: list[str] = []
    for index, statement in enumerate(module.body, start=1):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            raise ValueError(f"line {statement.lineno}: exactly one assignment is required")
        call = statement.value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.args:
            raise ValueError(f"line {statement.lineno}: use one plain node call with keyword arguments")
        class_type = catalog.resolve(call.func.id)
        if class_type is None:
            raise ValueError(f"line {statement.lineno}: unknown ComfyUI node {call.func.id!r}")
        definition = catalog.get(class_type)
        assert definition is not None
        allowed = {item.name for item in definition.inputs}
        inputs: dict[str, Any] = {}
        title: str | None = None
        for argument in call.keywords:
            if argument.arg is None:
                raise ValueError(f"line {statement.lineno}: **kwargs are forbidden")
            if argument.arg == "_meta_title":
                title = str(_literal(argument.value, statement.lineno))
                continue
            if allowed and argument.arg not in allowed:
                raise ValueError(f"line {statement.lineno}: {argument.arg!r} is not an input of {class_type}")
            inputs[argument.arg] = _ir_input(argument.value, variables, statement.lineno)
        missing = [item.name for item in definition.required_inputs if item.name not in inputs]
        if missing:
            raise ValueError(f"line {statement.lineno}: {class_type} missing inputs: {', '.join(missing)}")
        node_id = str(index)
        prompt[node_id] = {"class_type": class_type, "inputs": inputs}
        if title:
            prompt[node_id]["_meta"] = {"title": title}
        if definition.output_node:
            prompt[node_id].setdefault("_meta", {})["omniharness_output_node"] = True
        targets = _assignment_targets(statement.targets[0], statement.lineno)
        if len(targets) > max(1, len(definition.outputs)):
            raise ValueError(f"line {statement.lineno}: too many assigned outputs")
        for output_index, name in enumerate(targets):
            if name in variables:
                raise ValueError(f"line {statement.lineno}: variable {name!r} is reused")
            variables[name] = [node_id, output_index]
        provenance[node_id] = statement.lineno
        class_types.append(class_type)
    if not prompt:
        raise ValueError("Code-as-Policy workflow is empty")
    return CompilationResult(prompt, provenance, tuple(class_types))


def validate_workflow(prompt: Mapping[str, Any], catalog: NodeCatalog) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if contains_unbound_policy_input(prompt):
        issues.append(ValidationIssue("unbound_policy_input", "Bind all symbolic policy input roles for the current task before execution."))
    dependencies: dict[str, set[str]] = {str(key): set() for key in prompt}
    has_output = False
    node_ids = {str(key) for key in prompt}
    for raw_id, raw_node in prompt.items():
        node_id = str(raw_id)
        if not isinstance(raw_node, Mapping):
            issues.append(ValidationIssue("invalid_node", "node must be an object", node_id))
            continue
        class_type = str(raw_node.get("class_type", ""))
        definition = catalog.get(class_type)
        if definition is None:
            issues.append(ValidationIssue("unknown_node", f"unknown class_type {class_type!r}", node_id))
            continue
        has_output = has_output or _output_like(class_type)
        inputs = raw_node.get("inputs", {})
        if not isinstance(inputs, Mapping):
            issues.append(ValidationIssue("invalid_inputs", "inputs must be an object", node_id))
            continue
        missing = [item.name for item in definition.required_inputs if item.name not in inputs]
        if missing:
            issues.append(ValidationIssue("missing_input", ", ".join(missing), node_id))
        expected_types = {item.name: item.type for item in definition.inputs}
        for name, value in inputs.items():
            if not _link_like(value):
                continue
            upstream_id, slot = str(value[0]), int(value[1])
            if upstream_id not in node_ids:
                issues.append(ValidationIssue("missing_link", f"{name} references {upstream_id}", node_id))
                continue
            dependencies[node_id].add(upstream_id)
            upstream = prompt[upstream_id]
            upstream_definition = catalog.get(str(upstream.get("class_type", "")))
            if upstream_definition is None:
                continue
            if slot < 0 or slot >= len(upstream_definition.outputs):
                issues.append(ValidationIssue("invalid_output_slot", f"{name} references slot {slot}", node_id))
                continue
            expected = expected_types.get(str(name), "ANY")
            actual = upstream_definition.outputs[slot].type
            if not _compatible_types(expected, actual):
                issues.append(ValidationIssue("type_mismatch", f"{name} expects {expected}, got {actual}", node_id))
    if not has_output:
        issues.append(ValidationIssue("missing_output_node", "workflow has no image/video output node"))
    cycle = _find_cycle(node_ids, dependencies)
    if cycle:
        issues.append(ValidationIssue("cycle", " -> ".join(cycle)))
    return ValidationReport(tuple(issues))


def validate_plan(request: ExecutionRequest, plan: ExecutionPlan, context: PolicyContext) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if not plan.steps:
        issues.append(ValidationIssue("empty_plan", "plan contains no steps"))
    if len(plan.steps) > 24:
        issues.append(ValidationIssue("plan_too_long", "plan exceeds 24 steps"))
    identifiers = [item.step_id for item in plan.steps]
    if len(identifiers) != len(set(identifiers)):
        issues.append(ValidationIssue("duplicate_step", "step IDs must be unique"))
    for step in plan.steps:
        if not step.objective.strip():
            issues.append(ValidationIssue("missing_objective", f"{step.step_id} has no objective"))
        if not step.verification_criteria:
            issues.append(ValidationIssue("missing_step_verification", f"{step.step_id} has no criteria"))
    available = {str(item.get("workflow_id")) for item in context.workflows}
    unknown = set(plan.selected_workflow_ids) - available
    if unknown:
        issues.append(ValidationIssue("unknown_workflow", ", ".join(sorted(unknown))))
    applicability = plan.task_analysis.get("policy_applicability", {})
    for workflow_id in plan.selected_workflow_ids:
        check = applicability.get(workflow_id, {}) if isinstance(applicability, Mapping) else {}
        if not isinstance(check, Mapping) or check.get("satisfied") is not True or not check.get("evidence"):
            issues.append(ValidationIssue("unverified_preconditions", f"{workflow_id}: assess input roles, preconditions, and dependencies before reuse"))
    if plan.strategy is not PlanStrategy.BUILD and not plan.selected_workflow_ids:
        issues.append(ValidationIssue("missing_reuse_source", f"{plan.strategy.value} needs a source workflow"))
    components = {str(item.get("workflow_id")) for item in context.workflows if item.get("Retrieval Use") == "component_only"}
    if plan.strategy is PlanStrategy.REUSE and components.intersection(plan.selected_workflow_ids):
        issues.append(ValidationIssue("component_requires_adaptation", "Image policies must be adapted or composed with video-specific nodes; they are not complete video workflows."))
    if modality_requires_source(request.modality) and not request.source_images:
        issues.append(ValidationIssue("missing_source", f"{request.modality.value} plan has no input media"))
    return ValidationReport(tuple(issues))


GENERATION_TEAM_INSTRUCTIONS = """
You are the OmniHarness Generation Team. Operate through explicit roles:
Planner, Plan Verifier, Workflow Writer, Executor observer, Step Verifier,
Goal Verifier, and Failure Diagnoser. Retrieved workflows are symbolic policy
templates for task families; retrieved failures contain evidence and corrective
strategies. Bind the current task's inputs and adapt or compose templates before
execution. Respect preconditions, dependencies and component-only retrieval
roles. Image components require video-specific processing for video outputs.
Return only the
requested JSON. Never edit files, execute ComfyUI, change retry budgets, update
memory, or declare technical success. Code-as-Policy is a safe DSL with one
assignment and one allow-listed ComfyUI node call per line, keyword arguments
only, literals or earlier variables as inputs, and no imports, attributes,
control flow, nested calls, or arbitrary Python.
""".strip()


class GenerationTeam(Protocol):
    def plan(self, request: ExecutionRequest, context: PolicyContext) -> ExecutionPlan: ...
    def repair_plan(self, request: ExecutionRequest, context: PolicyContext, plan: ExecutionPlan, issues: Sequence[ValidationIssue]) -> ExecutionPlan: ...
    def write_workflow(self, request: ExecutionRequest, context: PolicyContext, plan: ExecutionPlan) -> WorkflowDraft: ...
    def verify_step(self, request: ExecutionRequest, step: PlanStep, result: ComfyExecutionResult) -> VerificationResult: ...
    def verify_goal(self, request: ExecutionRequest, plan: ExecutionPlan, result: ComfyExecutionResult) -> VerificationResult: ...
    def diagnose(self, request: ExecutionRequest, plan: ExecutionPlan, draft: WorkflowDraft, error: str | None, verification: VerificationResult | None) -> FailureDiagnosis: ...
    def repair_localized(self, request: ExecutionRequest, context: PolicyContext, plan: ExecutionPlan, draft: WorkflowDraft, diagnosis: FailureDiagnosis) -> LocalizedRepair: ...


class CodexGenerationTeam:
    """One task-scoped deliberative thread plus fresh localized recovery agents."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        model: str = CODEX_MODEL,
        model_provider: str = CODEX_MODEL_PROVIDER,
        reasoning_effort: str | None = CODEX_REASONING_EFFORT,
        provider_base_url: str | None = CODEX_BASE_URL,
        api_key_env: str = CODEX_API_KEY_ENV,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.model = model
        self.model_provider = model_provider
        self.reasoning_effort = reasoning_effort
        self.provider_base_url = validate_provider_base_url(provider_base_url)
        self.api_key_env = api_key_env.strip()
        if not self.api_key_env:
            raise ValueError("Codex API-key environment-variable name cannot be empty")
        self._sdk: Any = None
        self._codex: Any = None
        self._thread: Any = None

    def __enter__(self) -> "CodexGenerationTeam":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            import openai_codex as sdk  # type: ignore
        except ImportError as exc:
            raise RuntimeError("online OmniHarness execution requires openai-codex") from exc
        overrides = [
            f"model_provider={json.dumps(self.model_provider)}",
            f"developer_instructions={json.dumps(GENERATION_TEAM_INSTRUCTIONS)}",
        ]
        if self.provider_base_url:
            prefix = f"model_providers.{self.model_provider}"
            overrides.extend(
                (
                    f"{prefix}.name={json.dumps('OmniHarness third-party GPT-4o')}",
                    f"{prefix}.base_url={json.dumps(self.provider_base_url)}",
                    f"{prefix}.env_key={json.dumps(self.api_key_env)}",
                    f"{prefix}.wire_api={json.dumps('responses')}",
                    f"{prefix}.requires_openai_auth=false",
                )
            )
        if self.model.lower().startswith("gpt-4o"):
            overrides.extend(("model_context_window=128000", "model_supports_reasoning_summaries=false"))
        if self.reasoning_effort:
            overrides.append(f"model_reasoning_effort={json.dumps(self.reasoning_effort)}")
        config = sdk.CodexConfig(cwd=str(self.project_root), config_overrides=tuple(overrides))
        codex = sdk.Codex(config=config)
        codex.__enter__()
        try:
            thread = codex.thread_start(model=self.model, cwd=str(self.project_root), sandbox=sdk.Sandbox.read_only, approval_mode=sdk.ApprovalMode.deny_all)
        except Exception:
            codex.__exit__(*__import__("sys").exc_info())
            raise
        self._sdk, self._codex, self._thread = sdk, codex, thread

    def close(self) -> None:
        if self._codex is not None:
            self._codex.__exit__(None, None, None)
        self._sdk = self._codex = self._thread = None

    def plan(self, request: ExecutionRequest, context: PolicyContext) -> ExecutionPlan:
        payload = self._run_json(
            "Create a dependency-consistent ordered plan with explicit verification for every step. "
            "For each selected workflow, check its input roles, preconditions, and dependencies against "
            "the request and planned intermediate inputs. In task_analysis.policy_applicability return "
            "{workflow_id:{satisfied:true,evidence:[specific condition checks]}}. Select a policy only "
            "when its conditions can be met by the plan. "
            "Choose reuse, adapt, compose, or build. Return {plan_id, task_analysis, strategy, "
            "selected_workflow_ids, steps:[{step_id, objective, inputs, outputs, required_nodes, "
            "verification:{type,criteria}, expected_output_nodes, fallback}], rationale}.\nRequest:\n"
            + _json(request) + "\nContext:\n" + _json(context.compact(include_prompts=False))
        )
        return ExecutionPlan.from_dict(payload)

    def repair_plan(self, request: ExecutionRequest, context: PolicyContext, plan: ExecutionPlan, issues: Sequence[ValidationIssue]) -> ExecutionPlan:
        payload = self._run_json("Repair this plan and return the same schema.\nRequest:\n" + _json(request) + "\nPlan:\n" + _json(plan) + "\nIssues:\n" + _json(issues) + "\nContext:\n" + _json(context.compact(include_prompts=False)))
        return ExecutionPlan.from_dict(payload)

    def write_workflow(self, request: ExecutionRequest, context: PolicyContext, plan: ExecutionPlan) -> WorkflowDraft:
        payload = self._run_json(
            "Convert the verified plan to Code-as-Policy. Use $SOURCE_MEDIA_0 placeholders for media inputs; "
            "$SOURCE_IMAGE_0 remains accepted for compatibility. "
            "Retrieved graphs are templates: replace every $POLICY_* placeholder with task-specific "
            "text, numeric parameters, output names, or $SOURCE_MEDIA_n bindings. Check input-role "
            "preconditions and compose component_only image policies with video-specific nodes. "
            "Set _meta_title on every Code-as-Policy node to its owning plan step_id. Preserve "
            "existing assignment variable names during localized repairs. Each step must emit an "
            "inspectable intermediate output before dependent steps execute; expected_output_nodes "
            "should identify actual numeric node IDs of those output nodes, matching their "
            "one-based Code-as-Policy statement positions. A step title refers only to that step's "
            "output nodes; it does not require observing every internal node. "
            "Every node must contribute to at least one declared step output. "
            "If no applicable policy is available, construct a workflow from node knowledge. "
            "Describe the reusable task-family procedure, not the particular objects, prompt, or source "
            "filename of this execution, in workflow_name, description, preconditions and expected_effects. "
            "Return {ir_code, workflow_name, description, preconditions, expected_effects, "
            "dependencies:{Nodes,Models,Parameters}}.\nRequest:\n" + _json(request) + "\nPlan:\n" + _json(plan) + "\nContext:\n" + _json(context.compact(include_prompts=True))
        )
        return WorkflowDraft.from_dict(payload)

    def verify_step(self, request: ExecutionRequest, step: PlanStep, result: ComfyExecutionResult) -> VerificationResult:
        payload = self._run_json(
            "Inspect the supplied step output artifacts using image or video tools. Verify each listed criterion "
            "against these artifacts and the request. Do not infer an intermediate effect from the final result. "
            "Return {criteria:[{criterion,status:pass|fail|uncertain,evidence:[string]}]}. Copy each criterion "
            "exactly and use uncertain if its evidence cannot be inspected.\nRequest:\n" + _json(request)
            + "\nStep:\n" + _json(step) + "\nStep outputs:\n" + _json([item.path for item in result.outputs])
        )
        records = payload.get("criteria", ())
        by_criterion = {str(item.get("criterion", "")): item for item in records if isinstance(item, Mapping)} if isinstance(records, (list, tuple)) else {}
        statuses: list[VerificationStatus] = []
        evidence: list[str] = []
        failed: list[str] = []
        for criterion in step.verification_criteria or (step.objective,):
            record = by_criterion.get(criterion, {})
            support = _strings(record.get("evidence", ()))
            raw_status = str(record.get("status", "uncertain")).lower()
            status = VerificationStatus(raw_status) if raw_status in {item.value for item in VerificationStatus} else VerificationStatus.UNCERTAIN
            if status is VerificationStatus.PASS and not support:
                status = VerificationStatus.UNCERTAIN
            statuses.append(status)
            evidence.extend(f"{criterion}: {item}" for item in support)
            if status is not VerificationStatus.PASS:
                failed.append(criterion)
        status = VerificationStatus.FAIL if VerificationStatus.FAIL in statuses else VerificationStatus.UNCERTAIN if VerificationStatus.UNCERTAIN in statuses else VerificationStatus.PASS
        return VerificationResult(status, "semantic_step_verifier", {}, tuple(evidence), tuple(failed))

    def verify_goal(self, request: ExecutionRequest, plan: ExecutionPlan, result: ComfyExecutionResult) -> VerificationResult:
        payload = self._run_json(
            "Inspect every output path with the appropriate image or video tools. Evaluate every success and preservation criterion. "
            "Use uncertain when evidence is insufficient. Return {status:pass|fail|uncertain, verifier, "
            "scores, evidence, failed_criteria}.\nRequest:\n" + _json(request) + "\nPlan:\n" + _json(plan) + "\nOutputs:\n" + _json([item.path for item in result.outputs])
        )
        return VerificationResult.from_dict(payload)

    def diagnose(self, request: ExecutionRequest, plan: ExecutionPlan, draft: WorkflowDraft, error: str | None, verification: VerificationResult | None) -> FailureDiagnosis:
        payload = self._run_json(
            "Identify the smallest failed stage. Return {failure_class:planning_failure|compilation_failure|"
            "infrastructure_failure|execution_failure|verifier_failure|semantic_failure, failed_stage, "
            "root_cause, workflow_antipattern, remedy, applicable_scope, retry_scope}.\nRequest:\n" + _json(request) + "\nPlan:\n" + _json(plan) + "\nDraft:\n" + _json(draft) + "\nError:\n" + _json(error) + "\nVerification:\n" + _json(verification)
        )
        return FailureDiagnosis.from_dict(payload)

    def repair_localized(self, request: ExecutionRequest, context: PolicyContext, plan: ExecutionPlan, draft: WorkflowDraft, diagnosis: FailureDiagnosis) -> LocalizedRepair:
        prompt = (
            "Act as an isolated Failure Subagent. Repair only the diagnosed step and preserve unrelated "
            "workflow logic. Preserve the assignment variable names and node inputs outside that step and its dependent downstream nodes. "
            "Return {failed_step_id, root_cause, subworkflow_ir, revised_ir_code, "
            "reusable_name, rationale}. The full revised_ir_code must integrate the localized component.\n"
            "Request:\n" + _json(request) + "\nPlan:\n" + _json(plan) + "\nDraft:\n" + _json(draft) + "\nDiagnosis:\n" + _json(diagnosis) + "\nRecovery context:\n" + _json(context.compact(include_prompts=True))
        )
        return LocalizedRepair.from_dict(self._run_fresh_json(prompt))

    def _run_json(self, prompt: str) -> Mapping[str, Any]:
        self.start()
        result = self._thread.run(prompt)
        return extract_json_object(result.final_response)

    def _run_fresh_json(self, prompt: str) -> Mapping[str, Any]:
        self.start()
        thread = self._codex.thread_start(model=self.model, cwd=str(self.project_root), sandbox=self._sdk.Sandbox.read_only, approval_mode=self._sdk.ApprovalMode.deny_all)
        result = thread.run(prompt)
        return extract_json_object(result.final_response)


class ComfyUIClient:
    def __init__(self, base_url: str, *, timeout: float = 1200.0, poll_interval: float = 0.5) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.client_id = str(uuid.uuid4())

    def object_info(self) -> Mapping[str, Any]:
        payload = self._json("GET", "/object_info")
        if not isinstance(payload, Mapping):
            raise RuntimeError("ComfyUI /object_info returned no object")
        return payload

    def all_resources(self) -> ResourceInventory:
        resources = set(self.object_info())
        resources.update(("text_to_image", "image_input"))
        try:
            folders = self._json("GET", "/models")
        except RuntimeError:
            folders = ()
        if isinstance(folders, list):
            for folder in folders:
                try:
                    values = self._json("GET", f"/models/{folder}")
                except RuntimeError:
                    continue
                if isinstance(values, list):
                    resources.update(str(item) for item in values)
        return ResourceInventory(frozenset(resources))

    def preflight(self, prompt: Mapping[str, Any]) -> None:
        available = set(self.object_info())
        missing = sorted({str(item.get("class_type", "")) for item in prompt.values()} - available)
        if missing:
            raise RuntimeError("ComfyUI missing node classes: " + ", ".join(missing))

    def bind_source_images(self, prompt: Mapping[str, Any], source_images: Sequence[str]) -> Mapping[str, Any]:
        uploaded = [self.upload_image(path) for path in source_images]

        def bind(value: Any) -> Any:
            if isinstance(value, str) and value.startswith(("$SOURCE_IMAGE_", "$SOURCE_MEDIA_")):
                try:
                    return uploaded[int(value.rsplit("_", 1)[1])]
                except (ValueError, IndexError) as exc:
                    raise RuntimeError(f"invalid source placeholder: {value}") from exc
            if isinstance(value, list):
                return [bind(item) for item in value]
            if isinstance(value, Mapping):
                return {str(key): bind(item) for key, item in value.items()}
            return value

        return bind(prompt)

    def upload_image(self, path: str | Path) -> str:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        boundary = "----OmniHarness" + uuid.uuid4().hex
        body = bytearray()
        for name, value in (("overwrite", "true"), ("type", "input")):
            body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
        mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{source.name}\"\r\nContent-Type: {mime}\r\n\r\n".encode())
        body.extend(source.read_bytes())
        body.extend(f"\r\n--{boundary}--\r\n".encode())
        response = self._json("POST", "/upload/image", data=bytes(body), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        name = str(response.get("name", source.name))
        subfolder = str(response.get("subfolder", "")).strip("/\\")
        return f"{subfolder}/{name}" if subfolder else name

    def execute(self, prompt: Mapping[str, Any], output_dir: str | Path) -> ComfyExecutionResult:
        payload = self._json("POST", "/prompt", json_body={"prompt": prompt, "client_id": self.client_id})
        prompt_id = str(payload.get("prompt_id", ""))
        if not prompt_id:
            raise RuntimeError(f"ComfyUI returned no prompt_id: {payload}")
        deadline = time.monotonic() + self.timeout
        history: Mapping[str, Any] | None = None
        while time.monotonic() < deadline:
            response = self._json("GET", f"/history/{prompt_id}")
            record = response.get(prompt_id) if isinstance(response, Mapping) else None
            if isinstance(record, Mapping) and (record.get("outputs") or record.get("status", {}).get("completed") is not None):
                history = record
                break
            time.sleep(self.poll_interval)
        if history is None:
            try:
                self._json("POST", "/interrupt", json_body={})
            finally:
                raise TimeoutError(f"ComfyUI timed out for {prompt_id}")
        status = history.get("status", {})
        if str(status.get("status_str", "")).lower() in {"error", "failed"} or status.get("completed") is False:
            raise RuntimeError(f"ComfyUI execution failed: {status}")
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        artifacts: list[OutputArtifact] = []
        for node_id, node_output in dict(history.get("outputs", {})).items():
            if not isinstance(node_output, Mapping):
                continue
            for output_type, specifications in node_output.items():
                if not isinstance(specifications, list):
                    continue
                for specification in specifications:
                    if not isinstance(specification, Mapping) or "filename" not in specification:
                        continue
                    query = urlencode({"filename": specification["filename"], "subfolder": specification.get("subfolder", ""), "type": specification.get("type", "output")})
                    data = self._bytes("GET", "/view?" + query)
                    name = os.path.basename(str(specification["filename"])).replace("..", "_")
                    path = _unique_path(root, f"node_{node_id}_{name}")
                    path.write_bytes(data)
                    artifacts.append(OutputArtifact(str(node_id), str(output_type), str(specification["filename"]), str(path.resolve())))
        return ComfyExecutionResult(prompt_id, dict(status), tuple(artifacts), dict(history))

    def _json(self, method: str, path: str, *, json_body: Mapping[str, Any] | None = None, data: bytes | None = None, headers: Mapping[str, str] | None = None) -> Any:
        request_headers = dict(headers or {})
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        raw = self._bytes(method, path, data=data, headers=request_headers)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid ComfyUI JSON from {path}") from exc

    def _bytes(self, method: str, path: str, *, data: bytes | None = None, headers: Mapping[str, str] | None = None) -> bytes:
        request = Request(self.base_url + path, method=method, data=data, headers=dict(headers or {}))
        try:
            with urlopen(request, timeout=min(60.0, self.timeout)) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ComfyUI HTTP {exc.code} for {path}: {detail}") from exc
        except (URLError, OSError) as exc:
            raise RuntimeError(f"cannot reach ComfyUI at {self.base_url}: {exc}") from exc


class SymbolicPolicyStore:
    """Symbolic policies, failure evidence, and corrective strategies, with frozen exports."""

    def __init__(self, root: str | Path, *, consolidation_interval: int = 5) -> None:
        self.root = Path(root).resolve()
        self.workflow_metadata_path = self.root / "workflow_metadata.json"
        self.failure_metadata_path = self.root / "failure_metadata.json"
        self.state_path = self.root / ".omniharness_state.json"
        self.lock_path = self.root / ".omniharness.lock"
        self.consolidation_interval = consolidation_interval
        if consolidation_interval <= 0:
            raise ValueError("consolidation_interval must be positive")

    @property
    def is_snapshot(self) -> bool:
        return (self.root / "snapshot_manifest.json").exists()

    def assert_writable(self) -> None:
        if self.is_snapshot:
            raise ValueError("Frozen snapshots cannot be used in online mode or consolidated; use memory_mode='frozen'.")

    def validate(self) -> None:
        """Validate existing state without creating files in a frozen library."""
        self._workflow_metadata()
        self._failure_metadata()
        _read_json(self.state_path, missing_ok=True)
        if self.is_snapshot:
            manifest = _read_json(self.root / "snapshot_manifest.json")
            required = {self.workflow_metadata_path.name, self.failure_metadata_path.name}
            if not required.issubset(manifest):
                raise ValueError("snapshot manifest does not cover its metadata")
            actual_files = {path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()}
            if actual_files != set(manifest) | {"snapshot_manifest.json"}:
                raise ValueError("snapshot integrity check failed: file set differs from manifest")
            for filename, record in manifest.items():
                path = self._contained_path(str(filename))
                if not isinstance(record, Mapping) or not path.is_file():
                    raise ValueError(f"invalid snapshot manifest entry: {filename}")
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != record.get("sha256") or len(data) != record.get("size"):
                    raise ValueError(f"snapshot integrity check failed: {filename}")
            for workflow_id, raw in self._workflow_metadata().items():
                if isinstance(raw, Mapping):
                    relative = self._workflow_path(workflow_id, raw).relative_to(self.root).as_posix()
                    if relative not in manifest:
                        raise ValueError(f"snapshot manifest omits workflow: {relative}")

    def initialize(self) -> None:
        """Create empty S_1 only in a new/empty directory; never reset a library."""
        self.assert_writable()
        self.root.mkdir(parents=True, exist_ok=True)
        with _exclusive_lock(self.lock_path):
            entries = [path for path in self.root.iterdir() if path != self.lock_path]
            if entries:
                if not self.workflow_metadata_path.is_file() or not self.failure_metadata_path.is_file():
                    raise ValueError("Non-empty policy library is missing metadata; refusing to initialize or overwrite it.")
                self.validate()
                return
            _atomic_write_json(self.workflow_metadata_path, {})
            _atomic_write_json(self.failure_metadata_path, {})
            _atomic_write_json(self.state_path, {"attempt_counts": {}, "task_history": [], "committed_events": []})

    def _contained_path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"workflow path escapes its policy library: {relative}")
        return path

    def _workflow_path(self, workflow_id: str, raw: Mapping[str, Any]) -> Path:
        return self._contained_path(str(raw.get("Workflow Template", raw.get("Workflow Source Code", raw.get("Executable Workflow", f"{workflow_id}.json")))))

    def retrieve(self, request: ExecutionRequest, catalog: NodeCatalog, *, workflow_limit: int = 5, failure_limit: int = 8) -> PolicyContext:
        self.validate()
        workflows = self._workflow_metadata()
        query = _tokens(" ".join((request.instruction, request.modality.value, *request.capability_categories)))
        preferred = set(request.preferred_workflow_ids)
        ranked: list[tuple[float, str, Mapping[str, Any]]] = []
        templates: dict[str, Any] = {}
        for workflow_id, raw in workflows.items():
            if not isinstance(raw, Mapping) or str(raw.get("Reliability Tier", "")).lower() == "suspended":
                continue
            if int(raw.get("Usage Count", 0)) < 1 or int(raw.get("Success Count", 0)) < 1:
                continue
            modality = str(raw.get("Modality", "")).upper()
            component_only = request.modality in {Modality.T2V, Modality.I2V, Modality.V2V} and modality in {"T2I", "I2I"}
            if modality != request.modality.value and not component_only:
                continue
            text = " ".join(str(raw.get(key, "")) for key in ("Capability Category", "Capability Categories", "Workflow Name", "Workflow Description", "Expected Effects"))
            overlap = len(query & _tokens(text)) / max(1, len(query))
            rate = int(raw.get("Success Count", 0)) / max(1, int(raw.get("Usage Count", 0)))
            tier = {"validated": 1.0, "provisional": 0.35}.get(str(raw.get("Reliability Tier", "")).lower(), 0.0)
            capability_match = bool(set(_split_categories(raw.get("Capability Categories", raw.get("Capability Category", "")))).intersection(request.capability_categories))
            path = self._workflow_path(workflow_id, raw)
            if not path.is_file():
                continue
            template = distill_workflow_graph(_read_json(path))
            if any(catalog.get(str(node.get("class_type", ""))) is None for node in template.graph.values()):
                continue
            if component_only:
                # Images may be generated intermediates or extracted frames. The
                # image graph itself is never a complete video-task solution.
                if not (capability_match or overlap or workflow_id in preferred):
                    continue
            templates[workflow_id] = template
            score = (2.0 if workflow_id in preferred else 0.0) + 0.5 * float(capability_match) + 0.25 * overlap + 0.15 * tier + 0.1 * rate
            record = dict(raw)
            record["Retrieval Use"] = "component_only" if component_only else "task_workflow"
            record["Input Roles"] = list(template.input_roles)
            record["Binding Required"] = True
            if component_only:
                record["Adaptation Requirements"] = "Reuse only as an image-processing component; supply compatible images or frames and add video-specific nodes for temporal processing and output."
            ranked.append((score - (0.1 if component_only else 0.0), workflow_id, record))
        ranked.sort(key=lambda item: (-item[0], _id_sort_key(item[1])))
        selected = ranked[:workflow_limit]
        compact_workflows: list[Mapping[str, Any]] = []
        prompts: dict[str, Mapping[str, Any]] = {}
        required_nodes: list[str] = []
        for _, workflow_id, raw in selected:
            compact = dict(raw)
            compact["workflow_id"] = workflow_id
            compact_workflows.append(compact)
            dependencies = raw.get("Dependencies", {})
            if isinstance(dependencies, Mapping):
                required_nodes.extend(_split_items(dependencies.get("Nodes", ())))
        for _, workflow_id, _ in selected:
            prompts[workflow_id] = templates[workflow_id].graph
        failures = self._failure_metadata()
        preferred_failures = set(request.preferred_failure_ids)
        failure_ranked: list[tuple[int, str, Mapping[str, Any]]] = []
        for key, raw in failures.items():
            if key == "_metadata" or not isinstance(raw, Mapping):
                continue
            if str(raw.get("Modality", request.modality.value)).upper() != request.modality.value:
                continue
            text = " ".join(str(raw.get(field, "")) for field in ("Capability Category", "Root Cause", "Workflow Antipattern", "Remedy", "Applicable Scope"))
            score = len(query & _tokens(text)) + (5 if str(raw.get("Failure ID", key)) in preferred_failures else 0)
            if score:
                failure_ranked.append((score, key, raw))
        failure_ranked.sort(key=lambda item: (-item[0], item[1]))
        return PolicyContext(
            workflows=tuple(compact_workflows),
            workflow_prompts=prompts,
            failures=tuple(
                {key: item[2][key] for key in (
                    "Failure ID", "Modality", "Capability Category", "Root Cause",
                    "Workflow Antipattern", "Remedy", "Applicable Scope", "Occurrence Count",
                ) if key in item[2]}
                for item in failure_ranked[:failure_limit]
            ),
            node_knowledge=catalog.compact(" ".join((request.instruction, *request.capability_categories)), required=required_nodes),
        )

    def admission(
        self,
        request: ExecutionRequest,
        draft: WorkflowDraft,
        success: bool,
        prompt: Mapping[str, Any],
    ) -> AdmissionDecision:
        if not success or not prompt:
            return AdmissionDecision(False, False, False, None, (), "Only verified workflows are eligible.")
        fingerprint = workflow_fingerprint(prompt)
        matches: list[str] = []
        requested_capabilities = set(request.capability_categories)
        requested_intent = _intent_signature(
            " ".join(
                (
                    request.instruction,
                    draft.workflow_name,
                    draft.description,
                    draft.expected_effects,
                    _json(request.generation_constraints),
                    _json(request.preservation_constraints),
                )
            ),
            request.capability_categories,
        )
        for workflow_id, raw in self._workflow_metadata().items():
            if not isinstance(raw, Mapping):
                continue
            if str(raw.get("Modality", "")).upper() != request.modality.value:
                continue
            workflow_capabilities = set(
                _split_categories(
                    raw.get(
                        "Capability Categories",
                        raw.get("Capability Category", ""),
                    )
                )
            )
            if workflow_capabilities != requested_capabilities:
                continue
            workflow_intent = tuple(raw["Intent Signature"]) if raw.get("Intent Signature") else _intent_signature(
                " ".join(
                    str(raw.get(field, ""))
                    for field in (
                        "Workflow Name",
                        "Workflow Description",
                        "Preconditions",
                        "Expected Effects",
                    )
                ),
                workflow_capabilities,
            )
            if workflow_intent != requested_intent:
                continue
            path = self._workflow_path(workflow_id, raw)
            if not path.is_file():
                continue
            try:
                if workflow_fingerprint(_read_json(path)) == fingerprint:
                    matches.append(workflow_id)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        if matches:
            return AdmissionDecision(True, False, False, fingerprint, tuple(sorted(matches, key=_id_sort_key)), "Equivalent symbolic workflow already exists; update it instead of duplicating it.")
        return AdmissionDecision(True, True, True, fingerprint, (), "Verified operational graph is new.")

    def _record_task_state(self, state: dict[str, Any], request: ExecutionRequest, success: bool, *, error: str | None = None) -> None:
        history = list(state.get("task_history", ()))
        record = {
            "task_id": request.task_id,
            "task_signature": normalize_text(request.task_signature or request.instruction),
            "modality": request.modality.value,
            "capability_categories": list(request.capability_categories),
            "source_image_id": request.source_image_id,
            "success": success,
            "inquiry_iteration": request.inquiry_iteration,
        }
        if error is not None:
            record.update({"failure_stage": "pre_execution", "error": error})
        history.append(record)
        state["task_history"] = history[-500:]
        state["task_count"] = int(state.get("task_count", len(history) - 1)) + 1
        counts = {str(key): int(value) for key, value in dict(state.get("attempt_counts", {})).items()}
        if request.inquiry_iteration is not None:
            prefix = request.modality.value.lower()
            context = prefix if not modality_requires_source(request.modality) else f"{prefix}:{request.source_image_id or 'missing-source'}"
            for capability in request.capability_categories:
                key = context_capability_key(context, capability)
                counts[key] = counts.get(key, 0) + 1
            state["last_inquiry_iteration"] = max(int(state.get("last_inquiry_iteration", state.get("last_play_iteration", 0))), request.inquiry_iteration)
        state["attempt_counts"] = dict(sorted(counts.items()))
        committed = set(str(item) for item in state.get("committed_events", ()))
        committed.add(f"task:{request.task_id}")
        state["committed_events"] = sorted(committed)

    def record_pre_execution_failure(self, request: ExecutionRequest, reason: str) -> MemoryUpdate:
        """Record a selected task that failed before submission, without fake usage."""
        if request.memory_mode is MemoryMode.FROZEN:
            return MemoryUpdate(False, None, False)
        self.assert_writable()
        with _exclusive_lock(self.lock_path):
            self.validate()
            state = _read_json(self.state_path, missing_ok=True)
            if f"task:{request.task_id}" in state.get("committed_events", ()):
                return MemoryUpdate(False, None, False)
            self._record_task_state(state, request, False, error=reason)
            _atomic_write_json(self.state_path, state)
            consolidated = False
            if state["task_count"] % self.consolidation_interval == 0:
                consolidated = self._consolidate_locked()
            return MemoryUpdate(True, None, consolidated)

    def commit(
        self,
        *,
        request: ExecutionRequest,
        plan: ExecutionPlan,
        draft: WorkflowDraft,
        prompt: Mapping[str, Any],
        attempts: Sequence[AttemptRecord],
        success: bool,
        submitted: bool,
        admission: AdmissionDecision,
    ) -> MemoryUpdate:
        if request.memory_mode is MemoryMode.FROZEN:
            return MemoryUpdate(False, None, False)
        self.assert_writable()
        with _exclusive_lock(self.lock_path):
            workflows = self._workflow_metadata()
            failures = self._failure_metadata()
            state = _read_json(self.state_path, missing_ok=True)
            committed = set(str(item) for item in state.get("committed_events", ()))
            event = f"task:{request.task_id}"
            if event in committed:
                return MemoryUpdate(False, None, False)
            if success and admission.valid:
                admission = self.admission(request, draft, success, prompt)
            # Count submitted attempts, including failures before a later retry
            # succeeds. Plan-selected entries are the available reuse attribution;
            # this is not a claim of separately verified component-level success.
            invocations = [item for item in attempts if item.submitted]
            usage_delta = len(invocations) if attempts else int(submitted)
            success_delta = sum(
                item.goal_verification is not None and item.goal_verification.status is VerificationStatus.PASS
                for item in invocations
            ) if attempts else int(submitted and success)
            observed = {workflow_id: (usage_delta, success_delta) for workflow_id in set(plan.selected_workflow_ids)} if usage_delta else {}
            if submitted and success and admission.matched_workflow_ids:
                observed.setdefault(admission.matched_workflow_ids[0], (1, 1))
            for workflow_id, (usage_increment, success_increment) in observed.items():
                raw = workflows.get(workflow_id)
                if not isinstance(raw, dict):
                    continue
                usage = int(raw.get("Usage Count", 0)) + usage_increment
                successes = int(raw.get("Success Count", 0)) + success_increment
                raw["Usage Count"] = usage
                raw["Success Count"] = successes
                raw["Reliability Tier"] = reliability_tier(usage, successes)
            registered: str | None = None
            if success and admission.admitted:
                registered = _next_numeric_key(workflows)
                source_name = registered + ".json"
                template = distill_workflow_graph(prompt, instruction=request.instruction, source_images=request.source_images)
                _atomic_write_json(self.root / source_name, template.graph)
                description = redact_instance_values({
                    "Workflow Name": draft.workflow_name,
                    "Workflow Description": draft.description,
                    "Preconditions": draft.preconditions,
                    "Expected Effects": draft.expected_effects,
                    "Dependencies": dict(draft.dependencies),
                }, request.instruction, request.source_images)
                workflows[registered] = {
                    "Modality": request.modality.value,
                    "Capability Categories": list(request.capability_categories),
                    "Capability Category": ", ".join(request.capability_categories),
                    "Workflow Template": source_name,
                    "Workflow Source Code": source_name,
                    **description,
                    "Symbolic Policy Version": 1,
                    "Task Family": list(request.capability_categories),
                    "Input Roles": list(template.input_roles),
                    "Binding Required": True,
                    "Intent Signature": list(_intent_signature(" ".join((request.instruction, draft.workflow_name, draft.description, draft.expected_effects, _json(request.generation_constraints), _json(request.preservation_constraints))), request.capability_categories)),
                    "Usage Count": 1,
                    "Success Count": 1,
                    "Reliability Tier": ReliabilityTier.PROVISIONAL.value,
                    "Operational Fingerprint": admission.fingerprint,
                }
            diagnoses = [item for item in attempts if item.diagnosis is not None]
            if diagnoses:
                key = _next_numeric_key(failures)
                last = diagnoses[-1]
                diagnosis = last.diagnosis
                assert diagnosis is not None
                failures[key] = {
                    "Failure ID": f"F{int(key):03d}",
                    "Task": {
                        "Task Description": request.instruction,
                        "Modality": request.modality.value,
                        "Required Capability Set": list(request.capability_categories),
                        "Source Image": request.source_image_id,
                    },
                    "Task ID": request.task_id,
                    "Modality": request.modality.value,
                    "Capability Category": ", ".join(request.capability_categories),
                    "Failure Context": {
                        "Generation Constraints": list(request.generation_constraints),
                        "Preservation Constraints": list(request.preservation_constraints),
                        "Resolved By Retry": success,
                    },
                    "Failed Workflow": {
                        "Workflow Name": draft.workflow_name,
                        "Selected Workflow IDs": list(plan.selected_workflow_ids),
                        "Failed Stage": diagnosis.failed_stage,
                        "Workflow Path": last.workflow_path,
                    },
                    "Failure Evidence": {
                        "Attempt": last.attempt,
                        "Error": last.error,
                        "Step Verification": to_primitive(last.step_verification),
                        "Goal Verification": to_primitive(last.goal_verification),
                        "Intermediate and Final Outputs": [] if last.execution is None else [item.path for item in last.execution.outputs],
                        "Attempt History": [to_primitive(item) for item in diagnoses],
                    },
                    "Root Cause": diagnosis.root_cause,
                    "Workflow Antipattern": diagnosis.workflow_antipattern,
                    "Remedy": diagnosis.remedy,
                    "Applicable Scope": diagnosis.applicable_scope,
                    "Reusable Recovery Subworkflow": None if last.localized_repair is None else last.localized_repair.subworkflow_ir,
                    "Occurrence Count": 1,
                }
            self._record_task_state(state, request, success)
            _atomic_write_json(self.workflow_metadata_path, workflows)
            _atomic_write_json(self.failure_metadata_path, failures)
            _atomic_write_json(self.state_path, state)
            consolidated = False
            if state["task_count"] % self.consolidation_interval == 0:
                consolidated = self._consolidate_locked()
            return MemoryUpdate(True, registered, consolidated)

    def consolidate(self) -> bool:
        self.assert_writable()
        with _exclusive_lock(self.lock_path):
            return self._consolidate_locked()

    def _consolidate_locked(self) -> bool:
        workflows = self._workflow_metadata()
        failures = self._failure_metadata()
        changed = False
        groups: dict[str, list[str]] = {}
        for workflow_id, raw in workflows.items():
            if not isinstance(raw, Mapping):
                continue
            path = self._workflow_path(workflow_id, raw)
            if path.is_file():
                modality = str(raw.get("Modality", "")).upper()
                capabilities = tuple(
                    sorted(
                        _split_categories(
                            raw.get(
                                "Capability Categories",
                                raw.get("Capability Category", ""),
                            )
                        )
                    )
                )
                intent = tuple(raw["Intent Signature"]) if raw.get("Intent Signature") else _intent_signature(
                    " ".join(
                        str(raw.get(field, ""))
                        for field in (
                            "Workflow Name",
                            "Workflow Description",
                            "Preconditions",
                            "Expected Effects",
                        )
                    ),
                    capabilities,
                )
                group_key = "|".join(
                    (
                        modality,
                        *capabilities,
                        *intent,
                        workflow_fingerprint(_read_json(path)),
                    )
                )
                groups.setdefault(group_key, []).append(workflow_id)
        for identifiers in groups.values():
            if len(identifiers) < 2:
                continue
            identifiers.sort(key=_id_sort_key)
            canonical = identifiers[0]
            raw = workflows[canonical]
            usage = sum(int(workflows[item].get("Usage Count", 0)) for item in identifiers)
            success = sum(int(workflows[item].get("Success Count", 0)) for item in identifiers)
            raw["Usage Count"], raw["Success Count"] = usage, success
            raw["Reliability Tier"] = reliability_tier(usage, success)
            raw["Merged Workflow IDs"] = identifiers[1:]
            for duplicate in identifiers[1:]:
                del workflows[duplicate]
            changed = True
        failure_groups: dict[str, list[str]] = {}
        for key, raw in failures.items():
            if key == "_metadata" or not isinstance(raw, Mapping):
                continue
            fingerprint = normalize_text(" | ".join(str(raw.get(name, "")) for name in ("Root Cause", "Workflow Antipattern", "Remedy", "Applicable Scope")))
            failure_groups.setdefault(fingerprint, []).append(key)
        for identifiers in failure_groups.values():
            if len(identifiers) < 2:
                continue
            identifiers.sort(key=_id_sort_key)
            canonical = identifiers[0]
            failures[canonical]["Occurrence Count"] = sum(int(failures[item].get("Occurrence Count", 1)) for item in identifiers)
            failures[canonical]["Merged Failure IDs"] = [str(failures[item].get("Failure ID", item)) for item in identifiers[1:]]
            for duplicate in identifiers[1:]:
                del failures[duplicate]
            changed = True
        if changed:
            _atomic_write_json(self.workflow_metadata_path, workflows)
            _atomic_write_json(self.failure_metadata_path, failures)
        return changed

    def snapshot(self, destination: str | Path) -> Path:
        target = Path(destination).resolve()
        if target.is_relative_to(self.root):
            raise ValueError("snapshot destination must be outside the source library")
        if target.exists():
            raise FileExistsError(f"snapshot destination already exists: {target}")
        self.validate()
        target.parent.mkdir(parents=True, exist_ok=True)
        # A frozen source needs no write lock; a live source must be copied as
        # one committed state, not metadata from one round and graphs from another.
        with (nullcontext() if self.is_snapshot else _exclusive_lock(self.lock_path)):
            self.validate()
            staging = target.parent / (".policy-snapshot-" + uuid.uuid4().hex)
            staging.mkdir()
            try:
                workflows = {
                    key: raw for key, raw in self._workflow_metadata().items()
                    if isinstance(raw, dict) and int(raw.get("Usage Count", 0)) >= 1 and int(raw.get("Success Count", 0)) >= 1
                }
                for workflow_id, raw in workflows.items():
                    if not isinstance(raw, dict):
                        continue
                    source = self._workflow_path(workflow_id, raw)
                    relative = source.relative_to(self.root)
                    if relative.as_posix() in {"workflow_metadata.json", "failure_metadata.json", ".omniharness_state.json", "snapshot_manifest.json"}:
                        raise ValueError("workflow path conflicts with library metadata")
                    template = distill_workflow_graph(_read_json(source))
                    _atomic_write_json(staging / relative, template.graph)
                    raw["Workflow Source Code"] = relative.as_posix()
                    raw.pop("Executable Workflow", None)
                    raw["Workflow Template"] = relative.as_posix()
                    raw["Input Roles"] = list(template.input_roles)
                    raw["Binding Required"] = True
                    raw.pop("Symbolic Harness Version", None)
                    raw["Symbolic Policy Version"] = 1
                _atomic_write_json(staging / self.workflow_metadata_path.name, workflows)
                shutil.copy2(self.failure_metadata_path, staging / self.failure_metadata_path.name)
                # Export S=(L,F), not task-instance proposal history or commit
                # bookkeeping. Invocation/reliability statistics remain in L.
                _atomic_write_json(staging / self.state_path.name, {"frozen": True})
                manifest = {
                    path.relative_to(staging).as_posix(): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}
                    for path in staging.rglob("*") if path.is_file()
                }
                _atomic_write_json(staging / "snapshot_manifest.json", manifest)
                # Both paths are resolved children of the explicitly named
                # destination parent; publish only the completed snapshot.
                if not staging.is_relative_to(target.parent) or target.exists():
                    raise FileExistsError(f"snapshot destination already exists: {target}")
                staging.rename(target)
            finally:
                if staging.exists() and staging.resolve().parent == target.parent:
                    shutil.rmtree(staging)
        return target

    def _workflow_metadata(self) -> dict[str, Any]:
        return _read_json(self.workflow_metadata_path)

    def _failure_metadata(self) -> dict[str, Any]:
        return _read_json(self.failure_metadata_path)


class OmniHarnessRuntime:
    def __init__(
        self,
        *,
        team: GenerationTeam,
        policies: SymbolicPolicyStore,
        catalog: NodeCatalog,
        comfyui: ComfyUIClient,
        output_root: str | Path,
    ) -> None:
        self.team = team
        self.policies = policies
        self.catalog = catalog
        self.comfyui = comfyui
        self.output_root = Path(output_root)

    def run(self, request: ExecutionRequest, *, dry_run: bool = False) -> ExecutionOutcome:
        if request.memory_mode is MemoryMode.ONLINE:
            self.policies.assert_writable()
        self.policies.validate()
        started = time.monotonic()
        task_root = self.output_root / _safe_identifier(request.task_id)
        if task_root.resolve().is_relative_to(self.policies.root):
            raise ValueError("execution outputs must be outside the symbolic policy library")
        task_root.mkdir(parents=True, exist_ok=True)
        context = self.policies.retrieve(request, self.catalog)
        _append_jsonl(task_root / "runtime_audit.jsonl", {"event": "context_retrieved", "payload": context.compact(include_prompts=False)})
        plan: ExecutionPlan | None = None
        try:
            plan = self.team.plan(request, context)
            for repair_index in range(request.budget.max_plan_repairs + 1):
                report = validate_plan(request, plan, context)
                if report.valid:
                    break
                if repair_index >= request.budget.max_plan_repairs:
                    raise RuntimeError("plan verification failed: " + "; ".join(item.message for item in report.issues))
                plan = self.team.repair_plan(request, context, plan, report.issues)
            draft = self.team.write_workflow(request, context, plan)
        except Exception as exc:
            memory = self.policies.record_pre_execution_failure(request, str(exc)) if not dry_run else MemoryUpdate(False, None, False)
            outcome = ExecutionOutcome(request.task_id, ExecutionStatus.FAILED, plan, (), (), () if plan is None else plan.selected_workflow_ids, None, memory.updated, f"Planning or workflow synthesis failed before execution: {exc}")
            _append_jsonl(task_root / "runtime_audit.jsonl", {"event": "planning_failed", "error": str(exc)})
            _atomic_write_json(task_root / "outcome.json", to_primitive(outcome))
            return outcome
        attempts: list[AttemptRecord] = []
        final_prompt: Mapping[str, Any] = {}
        selected_outputs: tuple[str, ...] = ()
        submitted_any = False
        for attempt_index in range(1, request.budget.max_attempts + 1):
            if time.monotonic() - started >= request.budget.max_wall_time_seconds:
                break
            root = task_root / f"attempt_{attempt_index:02d}"
            root.mkdir(parents=True, exist_ok=True)
            execution: ComfyExecutionResult | None = None
            step_result: VerificationResult | None = None
            goal_result: VerificationResult | None = None
            diagnosis: FailureDiagnosis | None = None
            localized: LocalizedRepair | None = None
            error: Exception | None = None
            submitted = False
            workflow_path: str | None = None
            active_step_id: str | None = None
            try:
                compilation = compile_code_as_policy(draft.ir_code, self.catalog)
                report = validate_workflow(compilation.prompt, self.catalog)
                if not report.valid:
                    raise ValueError("workflow validation failed: " + "; ".join(f"{item.code}:{item.message}" for item in report.issues))
                stages = workflow_stages(plan, compilation.prompt)
                final_prompt = compilation.prompt
                (root / "workflow.omniharness.py").write_text(draft.ir_code, encoding="utf-8")
                _atomic_write_json(root / "workflow.json", compilation.prompt)
                _atomic_write_json(root / "provenance.json", compilation.provenance)
                workflow_path = str((root / "workflow.json").resolve())
                if dry_run:
                    attempts.append(AttemptRecord(attempt_index, False, workflow_path))
                    outcome = ExecutionOutcome(request.task_id, ExecutionStatus.DRY_RUN, plan, tuple(attempts), (), plan.selected_workflow_ids, None, False, "Plan and Code-as-Policy passed deterministic validation.")
                    _atomic_write_json(task_root / "outcome.json", to_primitive(outcome))
                    return outcome
                self.comfyui.preflight(compilation.prompt)
                bound = self.comfyui.bind_source_images(compilation.prompt, request.source_images)
                _atomic_write_json(root / "bound_workflow.json", bound)
                semantic_results: dict[str, VerificationResult] = {}
                for stage_index, (step, node_ids) in enumerate(zip(plan.steps, stages)):
                    active_step_id = step.step_id
                    if time.monotonic() - started >= request.budget.max_wall_time_seconds:
                        raise TimeoutError("execution budget exhausted before the next plan step")
                    stage_prompt = {node_id: bound[node_id] for node_id in node_ids}
                    submitted = submitted_any = True
                    stage_execution = self.comfyui.execute(stage_prompt, root / "outputs" / _safe_identifier(step.step_id))
                    execution = merge_execution_evidence(execution, stage_execution)
                    prefix_plan = replace(plan, steps=plan.steps[:stage_index + 1])
                    step_result = verify_steps(prefix_plan, execution, semantic_results)
                    artifacts = _step_artifacts(step, execution)
                    if _requires_semantic_verification(step) and artifacts:
                        if time.monotonic() - started >= request.budget.max_wall_time_seconds:
                            raise TimeoutError("execution budget exhausted before intermediate verification")
                        try:
                            semantic_results[step.step_id] = self.team.verify_step(request, step, replace(execution, outputs=artifacts))
                        except Exception as exc:
                            semantic_results[step.step_id] = VerificationResult(VerificationStatus.UNCERTAIN, "semantic_step_verifier", {}, (), (str(exc),))
                    step_result = verify_steps(prefix_plan, execution, semantic_results)
                    if step_result.status is not VerificationStatus.PASS:
                        diagnosis = _missing_step_evidence_diagnosis(prefix_plan, execution, step_result) or self._diagnose(request, plan, draft, None, step_result)
                        diagnosis = replace(diagnosis, failed_stage=step.step_id)
                        break
                assert execution is not None and step_result is not None
                if diagnosis is None:
                    # Only a completely verified plan reaches final task verification.
                    active_step_id = None
                    final_artifacts = _step_artifacts(plan.steps[-1], execution)
                    if time.monotonic() - started >= request.budget.max_wall_time_seconds:
                        raise TimeoutError("execution budget exhausted before goal verification")
                    goal_result = self.team.verify_goal(request, plan, replace(execution, outputs=final_artifacts))
                else:
                    goal_result = step_result
                if goal_result.status is VerificationStatus.PASS and not goal_result.evidence:
                    goal_result = VerificationResult(VerificationStatus.UNCERTAIN, goal_result.verifier, goal_result.scores, (), ("Goal verification returned no supporting evidence.",))
                if goal_result.status is VerificationStatus.PASS:
                    selected_outputs = tuple(item.path for item in final_artifacts)
                    attempts.append(AttemptRecord(attempt_index, True, workflow_path, execution, step_result, goal_result))
                    break
                diagnosis = diagnosis or self._diagnose(request, plan, draft, None, goal_result)
            except Exception as exc:
                error = exc
                diagnosis = self._diagnose(request, plan, draft, exc, goal_result)
                if active_step_id is not None:
                    diagnosis = replace(diagnosis, failed_stage=active_step_id)
            if diagnosis is not None and attempt_index < request.budget.max_attempts and diagnosis.failure_class is not FailureClass.VERIFIER and time.monotonic() - started < request.budget.max_wall_time_seconds:
                try:
                    proposal = self.team.repair_localized(request, context, plan, draft, diagnosis)
                    if diagnosis.failed_stage in {step.step_id for step in plan.steps} and proposal.failed_step_id != diagnosis.failed_stage:
                        raise ValueError("repair scope does not match the diagnosed plan step")
                    if not proposal.subworkflow_ir.strip():
                        raise ValueError("Failure Subagent returned no reusable subworkflow")
                    compile_code_as_policy(proposal.subworkflow_ir, self.catalog)
                    verified = {step.step_id for step in plan.steps if step_result is not None and step_result.scores.get(f"verified:{step.step_id}") == 1.0}
                    validate_localized_repair(plan, draft, proposal, self.catalog, verified)
                    plan = _rebind_plan_node_references(plan, draft.ir_code, proposal.revised_ir_code)
                    localized = proposal
                    draft = replace(draft, ir_code=localized.revised_ir_code)
                except Exception as exc:
                    error = ValueError(f"localized repair rejected: {exc}")
                    localized = None
            attempts.append(AttemptRecord(attempt_index, submitted, workflow_path, execution, step_result, goal_result, diagnosis, localized, None if error is None else str(error)))
            _append_jsonl(task_root / "runtime_audit.jsonl", {"event": "attempt_failed", "payload": attempts[-1]})
            if localized is None:
                break
        success = bool(attempts and attempts[-1].goal_verification and attempts[-1].goal_verification.status is VerificationStatus.PASS)
        admission = self.policies.admission(request, draft, success, final_prompt)
        _atomic_write_json(task_root / "workflow_admission.json", to_primitive(admission))
        memory = self.policies.commit(
            request=request,
            plan=plan,
            draft=draft,
            prompt=final_prompt,
            attempts=attempts,
            success=success,
            submitted=submitted_any,
            admission=admission,
        )
        outcome = ExecutionOutcome(
            task_id=request.task_id,
            status=ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED,
            plan=plan,
            attempts=tuple(attempts),
            selected_output_paths=selected_outputs if success else (),
            workflow_ids=plan.selected_workflow_ids,
            registered_workflow_id=memory.registered_workflow_id,
            memory_updated=memory.updated,
            message="Task passed plan, workflow, step, and goal verification." if success else "Retry budget exhausted, execution failed, or verification remained uncertain.",
        )
        _atomic_write_json(task_root / "outcome.json", to_primitive(outcome))
        return outcome

    def _diagnose(self, request: ExecutionRequest, plan: ExecutionPlan, draft: WorkflowDraft, error: Exception | None, verification: VerificationResult | None) -> FailureDiagnosis:
        try:
            return self.team.diagnose(request, plan, draft, None if error is None else str(error), verification)
        except Exception:
            if verification is not None and verification.status is VerificationStatus.UNCERTAIN:
                failure_class = FailureClass.VERIFIER
            elif isinstance(error, (TimeoutError, URLError, HTTPError)):
                failure_class = FailureClass.INFRASTRUCTURE
            elif isinstance(error, ValueError):
                failure_class = FailureClass.COMPILATION
            elif error is not None:
                failure_class = FailureClass.EXECUTION
            else:
                failure_class = FailureClass.SEMANTIC
            detail = str(error) if error is not None else "; ".join(verification.failed_criteria if verification else ())
            return FailureDiagnosis(failure_class, "execution", detail or "Acceptance criteria were not met.", "Proceeding without stage-specific verification.", "Repair only the responsible stage and re-verify it.", "staged visual generation workflows", "verifier" if failure_class is FailureClass.VERIFIER else "failed_step")


def workflow_stages(plan: ExecutionPlan, graph: Mapping[str, Mapping[str, Any]]) -> tuple[tuple[str, ...], ...]:
    """Submit each step's output and its ancestors before any dependent step."""
    stages: list[tuple[str, ...]] = []
    covered: set[str] = set()
    order = {step.step_id: index for index, step in enumerate(plan.steps)}
    for index, step in enumerate(plan.steps):
        nodes = _step_node_ids(step, graph, expected_only=True)
        if not nodes or any(not _resolve_step_reference(reference, graph, outputs_only=True) for reference in step.expected_output_nodes):
            raise ValueError(f"{step.step_id}: missing or ambiguous intermediate output nodes")
        while True:
            expanded = nodes | {str(value[0]) for node_id in nodes for value in graph[node_id].get("inputs", {}).values() if _link_to_existing(value, graph)}
            if expanded == nodes:
                break
            nodes = expanded
        if any(order.get(str(graph[node_id].get("_meta", {}).get("title", "")), index) > index for node_id in nodes):
            raise ValueError(f"{step.step_id}: output depends on a later plan step")
        stages.append(tuple(node_id for node_id in graph if node_id in nodes))
        covered.update(nodes)
    if covered != set(graph):
        raise ValueError("workflow contains nodes outside the declared step outputs and their dependencies")
    return tuple(stages)


def merge_execution_evidence(previous: ComfyExecutionResult | None, current: ComfyExecutionResult) -> ComfyExecutionResult:
    """Combine evidence only from subgraphs that have actually executed."""
    if previous is None:
        return current
    graph = {**_history_graph(previous.history), **_history_graph(current.history)}
    artifacts = {(item.node_id, item.filename): item for item in (*previous.outputs, *current.outputs)}
    history = dict(current.history)
    prompt = list(history.get("prompt", (None, None, {})))
    prompt[2] = graph
    history["prompt"] = prompt
    history["outputs"] = {**previous.history.get("outputs", {}), **current.history.get("outputs", {})}
    status = dict(current.status)
    status["messages"] = [*previous.status.get("messages", ()), *current.status.get("messages", ())]
    return replace(current, outputs=tuple(artifacts.values()), history=history, status=status)


def _execution_evidence_nodes(result: ComfyExecutionResult) -> set[str]:
    observed = {item.node_id for item in result.outputs}
    observed.update(str(node) for node, value in dict(result.history.get("outputs", {})).items() if value)
    for message in result.status.get("messages", ()):
        if not isinstance(message, (list, tuple)) or len(message) != 2 or not isinstance(message[1], Mapping):
            continue
        event, payload = message
        if event == "execution_cached":
            observed.update(str(node) for node in payload.get("nodes", ()))
        elif event in {"executed", "execution_node_completed"} and payload.get("node") is not None:
            observed.add(str(payload["node"]))
    return observed


def _resolve_step_reference(reference: str, graph: Mapping[str, Mapping[str, Any]], *, outputs_only: bool = False) -> set[str]:
    if reference in graph:
        return {reference}
    titled = {node_id for node_id, node in graph.items() if str(node.get("_meta", {}).get("title", "")) == reference}
    if titled:
        return {node_id for node_id in titled if _is_output_node(graph[node_id])} if outputs_only else titled
    matches = {node_id for node_id, node in graph.items() if _normalize_node(str(node.get("class_type", ""))) == _normalize_node(reference)}
    return matches if len(matches) == 1 else set()


def _step_node_ids(step: PlanStep, graph: Mapping[str, Mapping[str, Any]], *, expected_only: bool = False) -> set[str]:
    references = step.expected_output_nodes
    nodes = set().union(*(_resolve_step_reference(reference, graph, outputs_only=expected_only) for reference in references)) if references else set()
    titled = {node_id for node_id, node in graph.items() if str(node.get("_meta", {}).get("title", "")) == step.step_id}
    if expected_only:
        return nodes or {node_id for node_id in titled if _is_output_node(graph[node_id])}
    if titled:
        return titled | nodes
    for reference in step.required_nodes:
        nodes.update(_resolve_step_reference(reference, graph))
    return nodes


def _is_output_node(node: Mapping[str, Any]) -> bool:
    return bool(node.get("_meta", {}).get("omniharness_output_node")) or _output_like(str(node.get("class_type", "")))


def _requires_semantic_verification(step: PlanStep) -> bool:
    return step.verification_type.strip().lower() not in {"technical", "structural", "execution", "artifact"}


def _step_artifacts(step: PlanStep, result: ComfyExecutionResult) -> tuple[OutputArtifact, ...]:
    nodes = _step_node_ids(step, _history_graph(result.history), expected_only=True)
    return tuple(item for item in result.outputs if item.node_id in nodes)


def _missing_step_evidence_diagnosis(plan: ExecutionPlan, result: ComfyExecutionResult, verdict: VerificationResult) -> FailureDiagnosis | None:
    for step in plan.steps:
        if verdict.scores.get(f"observed:{step.step_id}") == 1.0 and (not _requires_semantic_verification(step) or _step_artifacts(step, result)):
            continue
        return FailureDiagnosis(
            FailureClass.COMPILATION, step.step_id,
            "The step does not expose a uniquely identified output with execution evidence.",
            "Treating a planned node or an unobserved intermediate result as verified execution.",
            "Expose an inspectable output for this step and preserve existing verified logic. Give the output node a unique title matching the step's expected output reference; retain the current task procedure and constraints.",
            "Workflows with missing or ambiguous intermediate verification outputs.", "failed_step",
        )
    return None


def verify_steps(plan: ExecutionPlan, result: ComfyExecutionResult, semantic_results: Mapping[str, VerificationResult] | None = None) -> VerificationResult:
    technical = verify_artifacts(result)
    if technical.status is not VerificationStatus.PASS:
        return technical
    graph = _history_graph(result.history)
    observed = _execution_evidence_nodes(result)
    missing: list[str] = []
    uncertain: list[str] = []
    evidence = list(technical.evidence)
    scores: dict[str, float] = {}
    for step in plan.steps:
        scores[f"observed:{step.step_id}"] = 0.0
        scores[f"verified:{step.step_id}"] = 0.0
        unresolved = False
        for reference in step.expected_output_nodes:
            if not _resolve_step_reference(reference, graph, outputs_only=True):
                uncertain.append(f"{step.step_id}: missing or ambiguous expected node {reference}")
                unresolved = True
        nodes = _step_node_ids(step, graph, expected_only=True)
        if unresolved or not nodes or not nodes.issubset(observed):
            if not unresolved:
                uncertain.append(f"{step.step_id}: no execution, cache, or output evidence for {sorted(nodes - observed) if nodes else 'declared outputs'}")
            continue
        scores[f"observed:{step.step_id}"] = 1.0
        evidence.append(f"{step.step_id}: execution evidence for nodes {sorted(nodes)}")
        if _requires_semantic_verification(step):
            verdict = (semantic_results or {}).get(step.step_id)
            if not _step_artifacts(step, result):
                uncertain.append(f"{step.step_id}: no inspectable intermediate output for semantic verification")
                continue
            if verdict is None or verdict.status is VerificationStatus.UNCERTAIN or not verdict.evidence:
                uncertain.append(f"{step.step_id}: semantic criteria lack verified evidence")
                continue
            evidence.extend(f"{step.step_id}: {item}" for item in verdict.evidence)
            if verdict.status is VerificationStatus.FAIL:
                missing.extend(f"{step.step_id}: {item}" for item in verdict.failed_criteria or ("semantic verification failed",))
                continue
        scores[f"verified:{step.step_id}"] = 1.0
    status = VerificationStatus.FAIL if missing else VerificationStatus.UNCERTAIN if uncertain else VerificationStatus.PASS
    scores["step_completion"] = sum(scores[f"verified:{step.step_id}"] for step in plan.steps) / len(plan.steps) if plan.steps else 0.0
    return VerificationResult(status, "step_verifier", scores, tuple(evidence), tuple(missing + uncertain))


def validate_localized_repair(plan: ExecutionPlan, draft: WorkflowDraft, repair: LocalizedRepair, catalog: NodeCatalog, verified_steps: set[str]) -> None:
    step = next((item for item in plan.steps if item.step_id == repair.failed_step_id), None)
    if step is None:
        raise ValueError("localized repair must identify an existing plan step")
    before = compile_code_as_policy(draft.ir_code, catalog)
    after = compile_code_as_policy(repair.revised_ir_code, catalog)
    allowed = _step_node_ids(step, before.prompt)
    if not allowed:
        raise ValueError("cannot locate the failed step in the compiled workflow")
    while True:
        dependents = {node_id for node_id, node in before.prompt.items() if any(_link_like(value) and str(value[0]) in allowed for value in node.get("inputs", {}).values())}
        expanded = allowed | dependents
        if expanded == allowed:
            break
        allowed = expanded
    protected = set(before.prompt) - allowed
    for candidate in plan.steps:
        if candidate.step_id in verified_steps and candidate.step_id != step.step_id:
            protected.update(_step_node_ids(candidate, before.prompt))

    def node_names(ir_code: str) -> dict[str, tuple[str, ...]]:
        return {str(index): tuple(_assignment_targets(statement.targets[0], statement.lineno)) for index, statement in enumerate(ast.parse(ir_code).body, 1)}

    old_names, new_names = node_names(draft.ir_code), node_names(repair.revised_ir_code)
    reverse = {names: node_id for node_id, names in new_names.items()}

    def normalized(node: Mapping[str, Any], names: Mapping[str, tuple[str, ...]]) -> Mapping[str, Any]:
        inputs = {key: {"variable_outputs": names[str(value[0])], "slot": value[1]} if _link_like(value) else value for key, value in node.get("inputs", {}).items()}
        return {"class_type": node.get("class_type"), "inputs": inputs, "_meta": node.get("_meta", {})}

    for node_id in protected:
        replacement = reverse.get(old_names[node_id])
        if replacement is None or normalized(before.prompt[node_id], old_names) != normalized(after.prompt[replacement], new_names):
            raise ValueError(f"localized repair changed protected node {node_id} ({', '.join(old_names[node_id])})")
    anchors = {reverse[old_names[node_id]] for node_id in allowed if old_names[node_id] in reverse}
    new_nodes = {node_id for node_id, names in new_names.items() if names not in set(old_names.values())}
    ancestors, descendants = set(anchors), set(anchors)
    while True:
        expanded = ancestors | {str(value[0]) for node_id in ancestors for value in after.prompt[node_id].get("inputs", {}).values() if _link_like(value)}
        if expanded == ancestors:
            break
        ancestors = expanded
    while True:
        expanded = descendants | {node_id for node_id, node in after.prompt.items() if any(_link_like(value) and str(value[0]) in descendants for value in node.get("inputs", {}).values())}
        if expanded == descendants:
            break
        descendants = expanded
    if new_nodes - ancestors - descendants:
        raise ValueError("localized repair introduced nodes unrelated to the affected component")


def _rebind_plan_node_references(plan: ExecutionPlan, previous_ir: str, revised_ir: str) -> ExecutionPlan:
    def names(ir_code: str) -> dict[str, tuple[str, ...]]:
        return {str(index): tuple(_assignment_targets(statement.targets[0], statement.lineno)) for index, statement in enumerate(ast.parse(ir_code).body, 1)}
    previous, revised = names(previous_ir), names(revised_ir)
    by_name = {variables: node_id for node_id, variables in revised.items()}
    remapping = {node_id: by_name[variables] for node_id, variables in previous.items() if variables in by_name}
    steps = tuple(replace(step, expected_output_nodes=tuple(remapping.get(reference, reference) for reference in step.expected_output_nodes)) for step in plan.steps)
    return replace(plan, steps=steps)


def verify_artifacts(result: ComfyExecutionResult, *, minimum_bytes: int = 128) -> VerificationResult:
    failed: list[str] = []
    evidence: list[str] = []
    if not result.outputs:
        failed.append("no output artifact was produced")
    for artifact in result.outputs:
        path = Path(artifact.path)
        if not path.is_file():
            failed.append(f"missing artifact: {path}")
            continue
        size = path.stat().st_size
        if size < minimum_bytes:
            failed.append(f"artifact too small: {path.name}")
            continue
        dimensions = _image_dimensions(path)
        evidence.append(f"{path.name}: {size} bytes" + ("" if dimensions is None else f", {dimensions[0]}x{dimensions[1]}"))
    return VerificationResult(VerificationStatus.FAIL if failed else VerificationStatus.PASS, "technical_verifier", {"artifact_validity": 0.0 if failed else 1.0}, tuple(evidence), tuple(failed))


def workflow_fingerprint(prompt: Mapping[str, Any]) -> str:
    nodes = dict(distill_workflow_graph(prompt).graph)
    memo: dict[str, Any] = {}

    def signature(node_id: str, visiting: frozenset[str] = frozenset()) -> Any:
        if node_id in memo:
            return memo[node_id]
        if node_id in visiting:
            return ["cycle"]
        node = nodes[node_id]
        class_type = str(node.get("class_type", ""))
        inputs = []
        for name, value in sorted(dict(node.get("inputs", {})).items()):
            normalized = ["edge", signature(str(value[0]), visiting | {node_id}), int(value[1])] if _link_to_existing(value, nodes) else ["literal", _normalize_literal(class_type, str(name), value)]
            inputs.append([str(name), normalized])
        memo[node_id] = [class_type, inputs]
        return memo[node_id]

    canonical = sorted((signature(node_id) for node_id in nodes), key=lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False))
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def reliability_tier(usage: int, success: int) -> str:
    rate = success / usage if usage else 0.0
    if usage >= 10 and rate <= 0.2:
        return ReliabilityTier.SUSPENDED.value
    if usage >= 3 and rate >= 0.5:
        return ReliabilityTier.VALIDATED.value
    return ReliabilityTier.PROVISIONAL.value


def _literal(node: ast.AST, line: int) -> Any:
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line}: only literals and earlier variables are allowed") from exc
    if not _safe_literal(value):
        raise ValueError(f"line {line}: unsupported literal")
    return value


def _ir_input(node: ast.AST, variables: Mapping[str, list[Any]], line: int) -> Any:
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"line {line}: {node.id!r} referenced before assignment")
        return list(variables[node.id])
    return _literal(node, line)


def _safe_literal(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_safe_literal(item) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _safe_literal(item) for key, item in value.items())
    return False


def _assignment_targets(target: ast.AST, line: int) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)) and target.elts and all(isinstance(item, ast.Name) for item in target.elts):
        return [item.id for item in target.elts]
    raise ValueError(f"line {line}: invalid assignment target")


def _find_cycle(nodes: set[str], dependencies: Mapping[str, set[str]]) -> tuple[str, ...]:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, path: tuple[str, ...]) -> tuple[str, ...]:
        if node in visiting:
            return (*path[path.index(node):], node) if node in path else (*path, node)
        if node in visited:
            return ()
        visiting.add(node)
        for dependency in dependencies.get(node, set()):
            cycle = visit(dependency, (*path, node))
            if cycle:
                return cycle
        visiting.remove(node)
        visited.add(node)
        return ()

    for node in nodes:
        cycle = visit(node, ())
        if cycle:
            return cycle
    return ()


def _normalize_literal(class_type: str, name: str, value: Any) -> Any:
    if isinstance(value, str) and value.startswith(("$POLICY_", "$HARNESS_")):
        return "$POLICY_" + value.split("_", 1)[1].rsplit("_", 1)[0]
    class_key, name_key = class_type.lower(), name.lower()
    if name_key in {"seed", "noise_seed", "random_seed"}:
        return "$SEED"
    if name_key in {"filename", "filename_prefix", "output_name"} and _output_like(class_type):
        return "$OUTPUT_NAME"
    if name_key in {"image", "image_path", "file", "path"} and "load" in class_key:
        return "$SOURCE_IMAGE"
    if name_key in {"text", "prompt", "positive", "negative", "text_positive", "text_negative"} and any(token in class_key for token in ("textencode", "text_encode", "prompt")):
        return "$TASK_PROMPT"
    if isinstance(value, Mapping):
        return {str(key): _normalize_literal(class_type, f"{name}.{key}", item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_literal(class_type, name, item) for item in value]
    return int(value) if isinstance(value, float) and value.is_integer() else value


def _history_graph(history: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    prompt = history.get("prompt")
    if not isinstance(prompt, (list, tuple)) or len(prompt) < 3 or not isinstance(prompt[2], Mapping):
        return {}
    return {str(key): value for key, value in prompt[2].items() if isinstance(value, Mapping)}


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    data = path.read_bytes()[:65536]
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data.startswith((b"GIF87a", b"GIF89a")) and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    if data.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                height, width = struct.unpack(">HH", data[index + 5:index + 9])
                return width, height
            length = struct.unpack(">H", data[index + 2:index + 4])[0]
            index += max(2, length + 2)
    return None


def _node_type(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return "ENUM" if isinstance(value, (list, tuple)) else str(value)


def _compatible_types(expected: str, actual: str) -> bool:
    return expected in {"ANY", "*", "ENUM"} or actual in {"ANY", "*"} or expected == actual


def _link_like(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)) and isinstance(value[1], int)


def _link_to_existing(value: Any, nodes: Mapping[str, Any]) -> bool:
    return _link_like(value) and str(value[0]) in nodes


def _output_like(class_type: str) -> bool:
    value = class_type.lower()
    return any(token in value for token in ("saveimage", "previewimage", "savevideo", "combinevideo", "vhs_videocombine"))


def _normalize_node(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _safe_identifier(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "item"
    return "v_" + rendered if rendered[0].isdigit() or keyword.iskeyword(rendered) else rendered


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_\u4e00-\u9fff]+", value.replace("_", " ").lower()))


def _intent_signature(
    text: str,
    capabilities: Iterable[str],
) -> tuple[str, ...]:
    """Conservative intended-function key used together with graph identity.

    A shared graph topology is not enough: adding, removing, recoloring, and
    repainting can all be prompt-parameterizations of the same ComfyUI graph.
    The key therefore preserves capability semantics and explicit operation or
    style markers.  When no marker is present, the capability set is the safe
    fallback.
    """

    normalized = normalize_text(text.replace("_", " "))
    labels = {"cap:" + normalize_text(item) for item in capabilities}
    patterns = (
        ("scene-replace", r"scene replacement|replace[^.]{0,40}scene|场景替换|替换场景"),
        ("replace", r"\breplac(?:e|es|ed|ement|ing)\b|\bswap\b|替换|置换"),
        ("remove", r"\bremov(?:e|es|ed|al|ing)\b|\bdelete\b|\berase\b|删除|移除|消除"),
        ("add", r"\badd(?:s|ed|ition|ing)?\b|\binsert\b|添加|增加|插入"),
        ("colorize", r"\bcolou?ri[sz](?:e|es|ed|ation|ing)\b|black[- ]and[- ]white|黑白|上色"),
        ("recolor", r"\brecolou?r\b|change[^.]{0,30}colou?r|颜色修改|改色"),
        ("repair", r"\brepair\b|\brestore\b|\brestoration\b|\brefine\b|修复|恢复|细化"),
        ("upscale", r"super[- ]resolution|\bupscal(?:e|ing)\b|超分辨率|放大"),
        ("outpaint", r"\boutpaint(?:ing)?\b|extend[^.]{0,30}boundar|扩图|外扩"),
        ("pose", r"\bpose\b|姿态|姿势"),
        ("repaint", r"\brepaint(?:s|ed|ing)?\b|重绘"),
        ("scribble-style", r"\bscribble\b|涂鸦"),
        ("watercolor-style", r"\bwatercolou?r\b|水彩"),
        ("oil-style", r"\boil painting\b|油画"),
        ("poster-style", r"\bposter\b|海报"),
        ("comic-style", r"\bcomic\b|漫画"),
        ("realistic-style", r"\brealistic style\b|写实风格"),
        ("text-render", r"in[- ]image text|text rendering|文字生成|图中文字"),
        ("position", r"position[- ]constrained|spatial placement|位置约束|空间位置"),
    )
    for label, pattern in patterns:
        if re.search(pattern, normalized):
            labels.add(label)
    return tuple(sorted(labels))


def _split_items(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _split_categories(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in re.split(r"[,;]", value) if item.strip())
    return _strings(value)


def _strings(value: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(item) for item in value)


def _json(value: Any) -> str:
    return json.dumps(_runtime_primitive(value), ensure_ascii=False, separators=(",", ":"))


def _runtime_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _runtime_primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _runtime_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_runtime_primitive(item) for item in value]
    return value


def _read_json(path: Path, *, missing_ok: bool = False) -> dict[str, Any]:
    if missing_ok and not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(_runtime_primitive(payload), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_runtime_primitive(payload), ensure_ascii=False, sort_keys=True) + "\n")


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _next_numeric_key(payload: Mapping[str, Any]) -> str:
    return str(max((int(key) for key in payload if str(key).isdigit()), default=0) + 1)


def _id_sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):020d}") if value.isdigit() else (1, value)


def _unique_path(root: Path, name: str) -> Path:
    candidate = root / name
    index = 1
    while candidate.exists():
        candidate = root / f"{Path(name).stem}_{index}{Path(name).suffix}"
        index += 1
    return candidate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OmniHarness feedback-guided execution and symbolic policy learning")
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("OMNIHARNESS_CONFIG", "config.yaml")))
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inquiry", "execute"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--comfyui-url", default=None)
        sub.add_argument("--symbolic-policy", type=Path, default=None)
        sub.add_argument("--source-image-pool", type=Path, default=None)
        sub.add_argument("--output-root", type=Path, default=None)
        sub.add_argument("--project-root", type=Path, default=None)
        sub.add_argument("--memory-mode", choices=("online", "frozen"), default=None)
        attempts = sub.add_mutually_exclusive_group()
        attempts.add_argument("--max-retries", type=int, default=None, help="Retries after the initial attempt (default: 4).")
        attempts.add_argument("--max-attempts", type=int, default=None, help="Legacy total-attempt budget, including the initial attempt.")
        sub.add_argument("--max-wall-time", type=int, default=None)
        sub.add_argument("--dry-run", action="store_true", default=None)
        sub.add_argument("--codex-model", default=None)
        sub.add_argument("--codex-provider", default=None)
        sub.add_argument("--codex-reasoning-effort", default=None)
        sub.add_argument("--codex-base-url", default=None)
        sub.add_argument("--codex-api-key-env", default=None)
    inquiry = subparsers.choices["inquiry"]
    inquiry.add_argument("--iterations", type=int, default=None)
    inquiry.add_argument("--candidate-count", type=int, default=None)
    inquiry.add_argument("--start-iteration", type=int, default=None)
    inquiry.add_argument("--modality", choices=("T2I", "I2I"), default=None)
    execute = subparsers.choices["execute"]
    execute.add_argument("--task-contract", type=Path, required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--symbolic-policy", type=Path, default=None)
    snapshot.add_argument("--destination", type=Path, required=True)
    consolidate = subparsers.add_parser("consolidate")
    consolidate.add_argument("--symbolic-policy", type=Path, default=None)
    return parser


def _runtime_for(args: argparse.Namespace, team: GenerationTeam, client: ComfyUIClient, run_root: Path) -> OmniHarnessRuntime:
    catalog = NodeCatalog.from_object_info(client.object_info())
    return OmniHarnessRuntime(team=team, policies=SymbolicPolicyStore(args.symbolic_policy), catalog=catalog, comfyui=client, output_root=run_root)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_omniharness_config(args.config)
    paths = config_section(config, "paths")
    proposal_config = config_section(config, "proposal")
    runtime_config = config_section(config, "runtime")
    codex_config = config_section(config, "codex")
    resources_root = runtime_resources_root()
    repo_root = repository_root()

    def configured_path(cli_value: Path | None, key: str, default: Path) -> Path:
        if cli_value is not None:
            return cli_value.expanduser().resolve()
        return resolve_config_path(paths.get(key), args.config, default)

    args.symbolic_policy = configured_path(
        args.symbolic_policy,
        "symbolic_policy",
        repo_root / "runs" / "policy_library",
    )
    if args.command == "snapshot":
        path = SymbolicPolicyStore(args.symbolic_policy).snapshot(args.destination)
        print(path)
        return 0
    if args.command == "consolidate":
        changed = SymbolicPolicyStore(args.symbolic_policy).consolidate()
        print(json.dumps({"consolidated": changed}))
        return 0
    args.source_image_pool = configured_path(
        args.source_image_pool,
        "source_image_pool",
        resources_root / "source_image_pool_comfybench",
    )
    args.output_root = configured_path(args.output_root, "output_root", repo_root / "runs")
    args.project_root = configured_path(args.project_root, "project_root", repo_root)
    args.comfyui_url = args.comfyui_url or runtime_config.get("comfyui_url")
    if not args.comfyui_url:
        parser.error("ComfyUI URL is required via runtime.comfyui_url or --comfyui-url")
    args.memory_mode = args.memory_mode or str(runtime_config.get("memory_mode", "online"))
    try:
        budget = resolve_execution_budget(runtime_config, max_retries=args.max_retries, max_attempts=args.max_attempts, max_wall_time_seconds=args.max_wall_time)
        mode = MemoryMode(args.memory_mode)
        store = SymbolicPolicyStore(args.symbolic_policy)
        if mode is MemoryMode.ONLINE:
            store.initialize()
        else:
            store.validate()
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.max_attempts = budget.max_attempts
    args.max_wall_time = budget.max_wall_time_seconds
    args.dry_run = bool(args.dry_run if args.dry_run is not None else runtime_config.get("dry_run", False))
    args.codex_model = str(args.codex_model or codex_config.get("model") or CODEX_MODEL)
    args.codex_provider = str(args.codex_provider or codex_config.get("provider") or CODEX_MODEL_PROVIDER)
    args.codex_reasoning_effort = args.codex_reasoning_effort or codex_config.get("reasoning_effort") or CODEX_REASONING_EFFORT
    args.codex_base_url = args.codex_base_url or codex_config.get("base_url") or CODEX_BASE_URL
    args.codex_api_key_env = str(args.codex_api_key_env or codex_config.get("api_key_env") or CODEX_API_KEY_ENV)
    if args.command == "inquiry":
        args.iterations = int(args.iterations if args.iterations is not None else runtime_config.get("iterations", 50))
        args.candidate_count = int(args.candidate_count if args.candidate_count is not None else proposal_config.get("candidate_count", 10))
        args.start_iteration = int(args.start_iteration if args.start_iteration is not None else proposal_config.get("iteration", 1))
        args.modality = args.modality or proposal_config.get("modality")
        if args.modality not in (None, "", "T2I", "I2I"):
            parser.error("proposal.modality must be T2I, I2I, or null during image-only inquiry")
        if min(args.iterations, args.start_iteration, args.candidate_count) <= 0:
            parser.error("iterations, start_iteration, and candidate_count must be positive")
        history = _read_json(store.state_path, missing_ok=True)
        last_iteration = max(
            [int(history.get("last_inquiry_iteration", history.get("last_play_iteration", 0)))]
            + [int(item.get("inquiry_iteration", item.get("play_iteration")) or 0) for item in history.get("task_history", ()) if isinstance(item, Mapping)]
        )
        if last_iteration and args.start_iteration != last_iteration + 1:
            parser.error(f"This library already completed inquiry iteration {last_iteration}; use --start-iteration {last_iteration + 1} or a new library.")
        if not last_iteration and history.get("attempt_counts") and args.start_iteration == 1:
            parser.error("This library contains legacy practice history; supply the continuation --start-iteration from its audit or use a new library.")
    client = ComfyUIClient(args.comfyui_url, timeout=float(args.max_wall_time))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_root.resolve() / f"omniharness_{args.command}_{stamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    if args.command == "execute":
        contract = TaskContract.from_dict(_read_json(args.task_contract))
        with CodexGenerationTeam(
            args.project_root,
            model=args.codex_model,
            model_provider=args.codex_provider,
            reasoning_effort=args.codex_reasoning_effort or None,
            provider_base_url=args.codex_base_url,
            api_key_env=args.codex_api_key_env,
        ) as team:
            outcome = _runtime_for(args, team, client, run_root).run(ExecutionRequest.from_contract(contract, memory_mode=mode, budget=budget), dry_run=args.dry_run)
        print(json.dumps(_runtime_primitive(outcome), ensure_ascii=False, indent=2))
        return 0 if outcome.status in {ExecutionStatus.COMPLETED, ExecutionStatus.DRY_RUN} else 2

    resources = client.all_resources()
    results: list[ExecutionOutcome] = []
    with CodexCandidateGenerator(
        args.project_root,
        model=args.codex_model,
        model_provider=args.codex_provider,
        reasoning_effort=args.codex_reasoning_effort or None,
        provider_base_url=args.codex_base_url,
        api_key_env=args.codex_api_key_env,
    ) as generator:
        proposer = SelfDirectedTaskProposer(generator)
        for offset in range(args.iterations):
            iteration = args.start_iteration + offset
            state = load_proposal_state(
                args.symbolic_policy,
                args.source_image_pool,
                iteration=iteration,
                requested_modality=None if args.modality is None else Modality(args.modality),
                candidate_count=args.candidate_count,
                resource_inventory=resources,
            )
            proposal: ProposalRound = proposer.propose(state)
            write_proposal_audit(run_root / "proposal_audit.jsonl", proposal)
            _atomic_write_json(run_root / f"task_contract_{iteration:04d}.json", proposal.selected)
            with CodexGenerationTeam(
                args.project_root,
                model=args.codex_model,
                model_provider=args.codex_provider,
                reasoning_effort=args.codex_reasoning_effort or None,
                provider_base_url=args.codex_base_url,
                api_key_env=args.codex_api_key_env,
            ) as team:
                runtime = _runtime_for(args, team, client, run_root / "execution")
                outcome = runtime.run(ExecutionRequest.from_contract(proposal.selected, memory_mode=mode, budget=budget), dry_run=args.dry_run)
            results.append(outcome)
            if args.dry_run:
                break
    summary = {"iterations": len(results), "completed": sum(item.status is ExecutionStatus.COMPLETED for item in results), "failed": sum(item.status is ExecutionStatus.FAILED for item in results), "run_root": str(run_root)}
    _atomic_write_json(run_root / "inquiry_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
