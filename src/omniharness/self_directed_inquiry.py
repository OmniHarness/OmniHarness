"""Self-directed inquiry task proposal for OmniHarness.

Task proposal and selection follow the Self-Directed Inquiry formulation:

    T_t = Propose(G, c_t, S_t)
    tau_t = argmax_tau N(tau) C(tau)

where ``G`` is the fixed visual capability space, ``c_t`` is the current
scene context, and ``S_t = (L_t, F_t)`` contains task-family symbolic policies
in the workflow library and corrective strategies in the failure library.
Model-backed generation is separated from deterministic validation, scoring,
selection, and audit logging.

Candidate generation uses ``openai-codex``. Configuration is loaded from
one optional YAML file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol
from urllib.parse import urlsplit


CODEX_MODEL = os.environ.get("OMNIHARNESS_CODEX_MODEL", "gpt-4o").strip() or "gpt-4o"
CODEX_MODEL_PROVIDER = (
    os.environ.get("OMNIHARNESS_CODEX_PROVIDER", "omniharness_gpt4o").strip()
    or "omniharness_gpt4o"
)
# GPT-4o is not a reasoning-family model.  Leave this unset by default; the
# override remains available for compatible third-party model aliases.
CODEX_REASONING_EFFORT = os.environ.get("OMNIHARNESS_CODEX_REASONING_EFFORT", "").strip() or None
CODEX_BASE_URL = os.environ.get("OMNIHARNESS_CODEX_BASE_URL", "").strip() or None
CODEX_API_KEY_ENV = os.environ.get("OMNIHARNESS_CODEX_API_KEY_ENV", "OMNIHARNESS_API_KEY").strip() or "OMNIHARNESS_API_KEY"
UNSUPPORTED_COMPETENCE_PRIOR = 0.05


def repository_root() -> Path:
    """Return the repository root for the ``src/omniharness`` layout."""

    return Path(__file__).resolve().parents[2]


def runtime_resources_root() -> Path:
    """Return the root of versioned resources consumed by OmniHarness."""

    return repository_root() / "resources"


def load_omniharness_config(path: str | Path | None) -> dict[str, Any]:
    """Load the single user-owned YAML configuration file when present."""

    if path is None:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("YAML configuration requires PyYAML; install requirements.txt") from exc
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"configuration root must be a mapping: {config_path}")
    return dict(payload)


def config_section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"configuration section '{name}' must be a mapping")
    return dict(value)


def resolve_config_path(value: Any, config_path: str | Path | None, default: Path) -> Path:
    if value in (None, ""):
        return default.resolve()
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    base = Path.cwd() if config_path is None else Path(config_path).expanduser().resolve().parent
    return (base / candidate).resolve()


def validate_provider_base_url(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("codex.base_url must be an absolute HTTP(S) API root")
    if parsed.path.endswith("/responses"):
        raise ValueError("codex.base_url must be the API root, not the /responses endpoint")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("remote Codex providers must use HTTPS")
    return normalized


class Modality(str, Enum):
    T2I = "T2I"
    I2I = "I2I"
    T2V = "T2V"
    I2V = "I2V"
    V2V = "V2V"


SOURCE_CONDITIONED_MODALITIES = frozenset(
    {Modality.I2I, Modality.I2V, Modality.V2V}
)


def modality_requires_source(modality: Modality) -> bool:
    return modality in SOURCE_CONDITIONED_MODALITIES


class ReliabilityTier(str, Enum):
    PROVISIONAL = "Provisional"
    VALIDATED = "Validated"
    SUSPENDED = "Suspended"


class RejectionCode(str, Enum):
    INVALID_SCHEMA = "INVALID_SCHEMA"
    INVALID_CAPABILITY = "INVALID_CAPABILITY"
    MODALITY_MISMATCH = "MODALITY_MISMATCH"
    MISSING_SOURCE_IMAGE = "MISSING_SOURCE_IMAGE"
    UNEXPECTED_SOURCE_IMAGE = "UNEXPECTED_SOURCE_IMAGE"
    UNAVAILABLE_RESOURCE = "UNAVAILABLE_RESOURCE"
    UNVERIFIABLE_OBJECTIVE = "UNVERIFIABLE_OBJECTIVE"
    SEMANTIC_DUPLICATE = "SEMANTIC_DUPLICATE"
    KNOWN_FAILURE_ANTIPATTERN = "KNOWN_FAILURE_ANTIPATTERN"
    EXCESSIVE_COMPLEXITY = "EXCESSIVE_COMPLEXITY"
    UNSAFE_CONTENT = "UNSAFE_CONTENT"


@dataclass(frozen=True)
class CapabilityDefinition:
    name: str
    modality: Modality
    description: str


GENERATIVE_CAPABILITY_SPACE: tuple[CapabilityDefinition, ...] = (
    CapabilityDefinition(
        "Photorealistic Generation",
        Modality.T2I,
        "Create realistic people, animals, objects, indoor and outdoor scenes with high perceptual fidelity.",
    ),
    CapabilityDefinition(
        "Futuristic & Fantasy Generation",
        Modality.T2I,
        "Create imaginative science-fiction, futuristic, and fantasy scenes, objects, and concepts.",
    ),
    CapabilityDefinition(
        "Illustration & Comic Generation",
        Modality.T2I,
        "Create illustrated, cartoon, comic, multi-panel, and narrative visual content.",
    ),
    CapabilityDefinition(
        "Poster & Graphic Design",
        Modality.T2I,
        "Create posters, covers, advertisements, and other layout-aware graphic compositions.",
    ),
    CapabilityDefinition(
        "In-Image Text Generation",
        Modality.T2I,
        "Render specified text with correct content, legibility, appearance, and spatial placement.",
    ),
    CapabilityDefinition(
        "Position-Constrained Generation",
        Modality.T2I,
        "Place objects or visual elements in explicitly specified image regions or locations.",
    ),
    CapabilityDefinition(
        "Localized Image Editing",
        Modality.I2I,
        "Modify selected regions while preserving unrelated content, including insertion, removal, replacement, and background editing.",
    ),
    CapabilityDefinition(
        "Style Transfer & Repainting",
        Modality.I2I,
        "Change visual style or rendering characteristics while preserving the main semantic content and structure.",
    ),
    CapabilityDefinition(
        "Reference-Guided Generation",
        Modality.I2I,
        "Generate content from transferable reference properties such as pose, structure, style, or semantics.",
    ),
    CapabilityDefinition(
        "Image Super-Resolution",
        Modality.I2I,
        "Improve spatial resolution and visual detail while preserving semantic content and global structure.",
    ),
    CapabilityDefinition(
        "Image Outpainting",
        Modality.I2I,
        "Extend content beyond the original boundaries while maintaining semantic, structural, and visual consistency.",
    ),
    CapabilityDefinition(
        "Image Restoration & Refinement",
        Modality.I2I,
        "Repair degraded or defective content through artifact correction, detail refinement, restoration, and quality enhancement.",
    ),
)

# Video modalities remain available to downstream execution, not image-only inquiry.
INQUIRY_MODALITIES = frozenset({Modality.T2I, Modality.I2I})


@dataclass(frozen=True)
class SourceImage:
    image_id: str
    path: str
    width: int
    height: int
    format: str
    caption: str = ""
    semantic_tags: tuple[str, ...] = ()
    quality_flags: tuple[str, ...] = ()
    applicable_capabilities: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, key: str, data: Mapping[str, Any], root: Path) -> "SourceImage":
        image_id = str(data.get("image_id", key))
        raw_path = Path(str(data["path"]))
        path = raw_path if raw_path.is_absolute() else root / raw_path
        path = path.resolve()
        if root.resolve() not in path.parents and path != root.resolve():
            raise ValueError(f"source image escapes its pool: {raw_path}")
        return cls(
            image_id=image_id,
            path=str(path),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            format=str(data.get("format", path.suffix.lstrip("."))),
            caption=str(data.get("caption", "")),
            semantic_tags=_strings(data.get("semantic_tags", ())),
            quality_flags=_strings(data.get("quality_flags", ())),
            applicable_capabilities=_strings(data.get("applicable_capabilities", ())),
        )


@dataclass(frozen=True)
class SymbolicPolicy:
    """Symbolic policy represented by a workflow template and applicability evidence.

    Legacy workflow field names are retained for resource and API compatibility.
    """

    workflow_id: str
    modality: Modality
    capability_categories: tuple[str, ...]
    name: str
    description: str
    preconditions: str
    expected_effects: str
    dependencies: tuple[str, ...]
    usage_count: int
    success_count: int
    reliability_tier: ReliabilityTier
    source_code: str = ""

    @property
    def empirical_success_rate(self) -> float:
        return self.success_count / self.usage_count if self.usage_count else 0.0

    @classmethod
    def from_metadata(cls, workflow_id: str, data: Mapping[str, Any]) -> "SymbolicPolicy":
        dependencies = data.get("Dependencies", {})
        dependency_items: list[str] = []
        if isinstance(dependencies, Mapping):
            for value in dependencies.values():
                dependency_items.extend(_split_items(value))
        return cls(
            workflow_id=str(workflow_id),
            modality=Modality(str(data.get("Modality", "T2I")).upper()),
            capability_categories=_split_categories(data.get("Capability Categories", data.get("Capability Category", ""))),
            name=str(data.get("Workflow Name", f"Workflow_{workflow_id}")),
            description=str(data.get("Workflow Description", "")),
            preconditions=str(data.get("Preconditions", "")),
            expected_effects=str(data.get("Expected Effects", "")),
            dependencies=tuple(dict.fromkeys(dependency_items)),
            usage_count=int(data.get("Usage Count", 0)),
            success_count=int(data.get("Success Count", 0)),
            reliability_tier=ReliabilityTier(str(data.get("Reliability Tier", "Provisional"))),
            source_code=str(data.get("Workflow Source Code", f"{workflow_id}.json")),
        )


@dataclass(frozen=True)
class FailureEntry:
    failure_id: str
    modality: Modality
    capability_categories: tuple[str, ...]
    root_cause: str
    antipattern: str
    remedy: str
    applicable_scope: str
    task_signature: str | None = None
    blocking: bool = False

    @classmethod
    def from_metadata(cls, key: str, data: Mapping[str, Any]) -> "FailureEntry":
        return cls(
            failure_id=str(data.get("Failure ID", key)),
            modality=Modality(str(data.get("Modality", "T2I")).upper()),
            capability_categories=_split_categories(data.get("Capability Categories", data.get("Capability Category", ""))),
            root_cause=str(data.get("Root Cause", "")),
            antipattern=str(data.get("Workflow Antipattern", "")),
            remedy=str(data.get("Remedy", "")),
            applicable_scope=str(data.get("Applicable Scope", "")),
            task_signature=_optional_text(data.get("Task Signature")),
            blocking=bool(data.get("Blocking", False)),
        )


@dataclass(frozen=True)
class ResourceInventory:
    resources: frozenset[str] = frozenset()

    def missing(self, required: Iterable[str]) -> tuple[str, ...]:
        available = {_resource_key(item) for item in self.resources}
        return tuple(sorted(item for item in required if _resource_key(item) not in available))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResourceInventory":
        values: list[str] = []
        for key in ("models", "custom_nodes", "features", "resources"):
            values.extend(_split_items(data.get(key, ())))
        return cls(frozenset(values))


@dataclass(frozen=True)
class TaskHistoryRecord:
    task_id: str
    task_signature: str
    modality: Modality
    capability_categories: tuple[str, ...]
    source_image_id: str | None = None
    success: bool | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskHistoryRecord":
        success = data.get("success")
        return cls(
            task_id=str(data["task_id"]),
            task_signature=normalize_text(str(data.get("task_signature", ""))),
            modality=Modality(str(data.get("modality", "T2I")).upper()),
            capability_categories=_strings(data.get("capability_categories", ())),
            source_image_id=_optional_text(data.get("source_image_id")),
            success=None if success is None else bool(success),
        )


@dataclass(frozen=True)
class ProposalBudget:
    candidate_count: int = 10
    max_capabilities_per_task: int = 3

    def __post_init__(self) -> None:
        if self.candidate_count <= 0 or self.max_capabilities_per_task <= 0:
            raise ValueError("proposal budgets must be positive")


@dataclass(frozen=True)
class ProposalState:
    iteration: int
    capabilities: tuple[CapabilityDefinition, ...]
    workflows: tuple[SymbolicPolicy, ...]
    failures: tuple[FailureEntry, ...]
    source_images: tuple[SourceImage, ...]
    resources: ResourceInventory
    recent_tasks: tuple[TaskHistoryRecord, ...] = ()
    attempt_counts: Mapping[str, int] = field(default_factory=dict)
    requested_modality: Modality | None = None
    budget: ProposalBudget = field(default_factory=ProposalBudget)

    def __post_init__(self) -> None:
        if self.iteration < 1:
            raise ValueError("inquiry iteration must start at 1")
        if not self.capabilities:
            raise ValueError("generative capability space G cannot be empty")
        if self.requested_modality is not None and self.requested_modality not in INQUIRY_MODALITIES:
            raise ValueError("self-directed inquiry supports only T2I and I2I")
        if any(item.modality not in INQUIRY_MODALITIES for item in self.capabilities):
            raise ValueError("the inquiry capability space must contain only image capabilities")
        if any(int(value) < 0 for value in self.attempt_counts.values()):
            raise ValueError("attempt counts cannot be negative")


@dataclass(frozen=True)
class CurrentSceneContext:
    iteration: int
    modality_coverage: Mapping[str, int]
    capability_coverage: Mapping[str, int]
    reliability_distribution: Mapping[str, int]
    source_image_pool: tuple[str, ...]
    source_image_metadata: tuple[Mapping[str, Any], ...]
    recent_task_signatures: tuple[str, ...]


@dataclass(frozen=True)
class TaskConstraint:
    kind: str
    target: str
    value: Any = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskConstraint":
        return cls(
            kind=str(data.get("kind", data.get("type", ""))),
            target=str(data.get("target", "")),
            value=data.get("value"),
        )


@dataclass(frozen=True)
class SuccessCriterion:
    criterion: str
    verifier: str = "multimodal_goal_verifier"
    threshold: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | str) -> "SuccessCriterion":
        if isinstance(data, str):
            return cls(data)
        threshold = data.get("threshold")
        return cls(
            criterion=str(data.get("criterion", data.get("description", ""))),
            verifier=str(data.get("verifier", "multimodal_goal_verifier")),
            threshold=None if threshold is None else float(threshold),
        )


@dataclass(frozen=True)
class CandidateTask:
    task_id: str
    description: str
    modality: Modality
    capability_categories: tuple[str, ...]
    source_image_id: str | None = None
    generation_constraints: tuple[TaskConstraint, ...] = ()
    preservation_constraints: tuple[TaskConstraint, ...] = ()
    success_criteria: tuple[SuccessCriterion, ...] = ()
    required_resources: tuple[str, ...] = ()
    task_signature: str = ""
    proposal_rationale: str = ""
    safety_flags: tuple[str, ...] = ()
    applicable_workflow_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.capability_categories:
            raise ValueError("candidate requires at least one capability")

    def canonical_signature(self, source: SourceImage | None = None) -> str:
        if self.task_signature.strip():
            return normalize_text(self.task_signature)
        source_key = "none" if source is None else source.image_id
        constraints = " ".join(
            f"{item.kind} {item.target} {item.value}"
            for item in (*self.generation_constraints, *self.preservation_constraints)
        )
        return normalize_text(
            " | ".join(
                (
                    self.modality.value,
                    ",".join(sorted(self.capability_categories)),
                    source_key,
                    constraints or self.description,
                )
            )
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateTask":
        capabilities = data.get("capability_categories", data.get("required_capabilities"))
        if capabilities is None:
            primary = str(data.get("primary_capability", "")).strip()
            auxiliary = _strings(data.get("auxiliary_capabilities", ()))
            capabilities = (primary, *auxiliary)
        return cls(
            task_id=str(data["task_id"]),
            description=str(data.get("description", data.get("task_description", ""))),
            modality=Modality(str(data["modality"]).upper()),
            capability_categories=tuple(item for item in _strings(capabilities) if item),
            source_image_id=_optional_text(data.get("source_image_id")),
            generation_constraints=tuple(TaskConstraint.from_dict(item) for item in data.get("generation_constraints", ())),
            preservation_constraints=tuple(TaskConstraint.from_dict(item) for item in data.get("preservation_constraints", ())),
            success_criteria=tuple(SuccessCriterion.from_dict(item) for item in data.get("success_criteria", ())),
            required_resources=_strings(data.get("required_resources", ())),
            task_signature=str(data.get("task_signature", "")),
            proposal_rationale=str(data.get("proposal_rationale", "")),
            safety_flags=_strings(data.get("safety_flags", ())),
            applicable_workflow_ids=(
                _strings(data["applicable_workflow_ids"])
                if isinstance(data.get("applicable_workflow_ids"), list) else None
            ),
        )


@dataclass(frozen=True)
class ValidationIssue:
    code: RejectionCode
    message: str


@dataclass(frozen=True)
class TaskScore:
    novelty: float
    competence: float
    competence_frontier: float
    objective: float
    context_key: str
    capability_reliability: Mapping[str, float]
    applicable_workflow_ids: tuple[str, ...]


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: CandidateTask
    issues: tuple[ValidationIssue, ...] = ()
    score: TaskScore | None = None

    @property
    def accepted(self) -> bool:
        return not self.issues and self.score is not None


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    inquiry_iteration: int | None
    description: str
    modality: Modality
    capability_categories: tuple[str, ...]
    source_image_id: str | None
    source_image_path: str | None
    generation_constraints: tuple[TaskConstraint, ...]
    preservation_constraints: tuple[TaskConstraint, ...]
    success_criteria: tuple[SuccessCriterion, ...]
    retrieved_workflow_ids: tuple[str, ...]
    retrieved_failure_ids: tuple[str, ...]
    selection_evidence: TaskScore
    task_signature: str
    source_image_paths: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskContract":
        source_image_path = _optional_text(data.get("source_image_path"))
        raw_source_paths = data.get("source_image_paths", ())
        if isinstance(raw_source_paths, str):
            source_image_paths = (raw_source_paths,)
        else:
            source_image_paths = _strings(raw_source_paths)
        if not source_image_paths and source_image_path is not None:
            source_image_paths = (source_image_path,)
        if source_image_path is None and source_image_paths:
            source_image_path = source_image_paths[0]
        # Read existing contracts while emitting the current inquiry terminology.
        iteration = data.get("inquiry_iteration", data.get("play_iteration", data.get("iteration")))
        return cls(
            task_id=str(data["task_id"]),
            inquiry_iteration=None if iteration is None else int(iteration),
            description=str(data["description"]),
            modality=Modality(str(data["modality"]).upper()),
            capability_categories=_strings(data["capability_categories"]),
            source_image_id=_optional_text(data.get("source_image_id")),
            source_image_path=source_image_path,
            generation_constraints=tuple(TaskConstraint.from_dict(item) for item in data.get("generation_constraints", ())),
            preservation_constraints=tuple(TaskConstraint.from_dict(item) for item in data.get("preservation_constraints", ())),
            success_criteria=tuple(SuccessCriterion.from_dict(item) for item in data.get("success_criteria", ())),
            retrieved_workflow_ids=_strings(data.get("retrieved_workflow_ids", ())),
            retrieved_failure_ids=_strings(data.get("retrieved_failure_ids", ())),
            selection_evidence=TaskScore(**dict(data["selection_evidence"])),
            task_signature=str(data.get("task_signature", "")),
            source_image_paths=source_image_paths,
        )


@dataclass(frozen=True)
class ProposalRound:
    context: CurrentSceneContext
    evaluations: tuple[CandidateEvaluation, ...]
    selected: TaskContract


class CandidateGenerator(Protocol):
    def generate(
        self,
        *,
        state: ProposalState,
        context: CurrentSceneContext,
        candidate_count: int,
    ) -> list[CandidateTask]: ...


PROPOSER_INSTRUCTIONS = """
You are the Self-Directed Inquiry Task Proposer inside OmniHarness. Generate
practice tasks before downstream objectives are specified. Each task must use
the supplied capability space, be feasible with current resources, differ from
recent tasks, and be independently verifiable. Use failure evidence and
corrective strategies to avoid known antipatterns. For I2I, use exactly one
supplied image ID and state preservation constraints. List applicable_workflow_ids only when
the workflow's preconditions hold for the proposed task and source image.
Exclude suspended workflows; return an empty list when none apply.
Use only the supplied context and library. Do not inspect downstream task
instructions, reference workflows, target outputs, or benchmark annotations.
Return only the requested JSON object.
""".strip()


class CodexCandidateGenerator:
    """Persistent read-only Codex proposer thread; no direct model API calls."""

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
        self._codex: Any = None
        self._thread: Any = None

    def __enter__(self) -> "CodexCandidateGenerator":
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
            raise RuntimeError("online proposal requires openai-codex") from exc
        overrides = [
            f"model_provider={json.dumps(self.model_provider)}",
            f"developer_instructions={json.dumps(PROPOSER_INSTRUCTIONS)}",
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
        try:
            codex.__enter__()
            thread = codex.thread_start(
                model=self.model,
                cwd=str(self.project_root),
                sandbox=sdk.Sandbox.read_only,
                approval_mode=sdk.ApprovalMode.deny_all,
            )
        except Exception:
            codex.__exit__(*__import__("sys").exc_info())
            raise
        self._codex, self._thread = codex, thread

    def close(self) -> None:
        if self._codex is not None:
            self._codex.__exit__(None, None, None)
        self._codex = self._thread = None

    def generate(
        self,
        *,
        state: ProposalState,
        context: CurrentSceneContext,
        candidate_count: int,
    ) -> list[CandidateTask]:
        self.start()
        compact = {
            "iteration": state.iteration,
            "requested_modality": None if state.requested_modality is None else state.requested_modality.value,
            "generative_capability_space": to_primitive(state.capabilities),
            "current_scene_context": to_primitive(context),
            "workflow_library": [
                {
                    "workflow_id": item.workflow_id,
                    "modality": item.modality.value,
                    "capability_categories": item.capability_categories,
                    "preconditions": item.preconditions,
                    "expected_effects": item.expected_effects,
                    "usage_count": item.usage_count,
                    "success_count": item.success_count,
                    "reliability_tier": item.reliability_tier.value,
                }
                for item in state.workflows
            ],
            "failure_library": to_primitive(state.failures),
            "available_resources": sorted(state.resources.resources),
        }
        prompt = f"""
Generate exactly {candidate_count} candidate tasks. Return
{{"candidates": [CandidateTask, ...]}}. Each CandidateTask contains task_id,
description, modality (T2I or I2I), capability_categories (one to three exact
names), source_image_id (null for T2I), generation_constraints,
preservation_constraints, success_criteria, required_resources,
task_signature, proposal_rationale, safety_flags (empty for acceptable tasks),
and applicable_workflow_ids. State:\n{json.dumps(compact, ensure_ascii=False, indent=2)}
""".strip()
        result = self._thread.run(prompt)
        payload = extract_json_object(result.final_response)
        raw = payload.get("candidates")
        if not isinstance(raw, list):
            raise ValueError("candidate response must contain a candidates array")
        candidates: list[CandidateTask] = []
        for item in raw:
            try:
                candidates.append(CandidateTask.from_dict(item))
            except (KeyError, TypeError, ValueError):
                continue
        if not candidates:
            raise ValueError("candidate generator returned no valid items")
        return candidates[:candidate_count]


class CandidateValidator:
    def validate(
        self,
        candidate: CandidateTask,
        state: ProposalState,
        *,
        batch_signatures: Iterable[str] = (),
    ) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        capability_map = {item.name: item for item in state.capabilities}
        source_map = {item.image_id: item for item in state.source_images}
        source = source_map.get(candidate.source_image_id or "")
        capabilities = candidate.capability_categories

        if not candidate.task_id.strip() or not candidate.description.strip():
            issues.append(ValidationIssue(RejectionCode.INVALID_SCHEMA, "task_id and description are required"))
        if not capabilities:
            issues.append(ValidationIssue(RejectionCode.INVALID_CAPABILITY, "at least one capability is required"))
        if candidate.modality not in INQUIRY_MODALITIES:
            issues.append(ValidationIssue(RejectionCode.MODALITY_MISMATCH, "inquiry is restricted to T2I and I2I"))
        if len(set(capabilities)) != len(capabilities):
            issues.append(ValidationIssue(RejectionCode.INVALID_SCHEMA, "capability categories must be unique"))
        if len(capabilities) > state.budget.max_capabilities_per_task:
            issues.append(ValidationIssue(RejectionCode.EXCESSIVE_COMPLEXITY, "too many capabilities for one practice task"))
        unknown = sorted(set(capabilities) - set(capability_map))
        if unknown:
            issues.append(ValidationIssue(RejectionCode.INVALID_CAPABILITY, "unknown capabilities: " + ", ".join(unknown)))
        mismatch = [name for name in capabilities if name in capability_map and capability_map[name].modality is not candidate.modality]
        if mismatch:
            issues.append(ValidationIssue(RejectionCode.MODALITY_MISMATCH, "capability/modality mismatch: " + ", ".join(mismatch)))
        if state.requested_modality is not None and candidate.modality is not state.requested_modality:
            issues.append(ValidationIssue(RejectionCode.MODALITY_MISMATCH, "candidate violates the requested modality"))
        if candidate.applicable_workflow_ids is None:
            issues.append(ValidationIssue(RejectionCode.INVALID_SCHEMA, "applicable_workflow_ids must list workflows whose preconditions hold"))
        elif set(candidate.applicable_workflow_ids) - {item.workflow_id for item in state.workflows}:
            issues.append(ValidationIssue(RejectionCode.INVALID_SCHEMA, "applicable_workflow_ids contains unknown workflows"))

        if modality_requires_source(candidate.modality):
            if source is None:
                issues.append(ValidationIssue(RejectionCode.MISSING_SOURCE_IMAGE, f"{candidate.modality.value} tasks require existing source media"))
            elif candidate.modality is Modality.I2I and source.applicable_capabilities and not set(capabilities).intersection(source.applicable_capabilities):
                issues.append(ValidationIssue(RejectionCode.MODALITY_MISMATCH, "source image is not applicable to the selected capabilities"))
            if not candidate.preservation_constraints:
                issues.append(ValidationIssue(RejectionCode.UNVERIFIABLE_OBJECTIVE, f"{candidate.modality.value} tasks require preservation constraints"))
        elif candidate.source_image_id is not None:
            issues.append(ValidationIssue(RejectionCode.UNEXPECTED_SOURCE_IMAGE, f"{candidate.modality.value} tasks use no source media"))

        missing = state.resources.missing(candidate.required_resources)
        if missing:
            issues.append(ValidationIssue(RejectionCode.UNAVAILABLE_RESOURCE, "missing resources: " + ", ".join(missing)))
        if not candidate.success_criteria or any(not item.criterion.strip() or not item.verifier.strip() for item in candidate.success_criteria):
            issues.append(ValidationIssue(RejectionCode.UNVERIFIABLE_OBJECTIVE, "at least one explicit success criterion is required"))
        if candidate.safety_flags:
            issues.append(ValidationIssue(RejectionCode.UNSAFE_CONTENT, "safety flags: " + ", ".join(candidate.safety_flags)))

        signature = candidate.canonical_signature(source)
        comparison = [record.task_signature for record in state.recent_tasks]
        comparison.extend(batch_signatures)
        if any(_semantic_similarity(signature, other) >= 0.88 for other in comparison if other):
            issues.append(ValidationIssue(RejectionCode.SEMANTIC_DUPLICATE, "task is a near duplicate of recent exploration"))
        for failure in state.failures:
            if (
                failure.blocking
                and failure.modality is candidate.modality
                and failure.task_signature
                and _semantic_similarity(signature, failure.task_signature) >= 0.92
            ):
                issues.append(ValidationIssue(RejectionCode.KNOWN_FAILURE_ANTIPATTERN, f"blocked by {failure.failure_id}"))
        return tuple(dict.fromkeys(issues))


class InquiryScorer:
    """Capability novelty and competence frontier scoring for Exploration within Reach."""

    def __init__(self, unsupported_prior: float = UNSUPPORTED_COMPETENCE_PRIOR) -> None:
        if not 0.0 <= unsupported_prior <= 1.0:
            raise ValueError("unsupported prior must be in [0, 1]")
        self.unsupported_prior = unsupported_prior

    def score(self, candidate: CandidateTask, state: ProposalState) -> TaskScore:
        source = next((item for item in state.source_images if item.image_id == candidate.source_image_id), None)
        context = task_context_key(candidate)
        novelty_terms = [
            1.0 / math.sqrt(state.attempt_counts.get(context_capability_key(context, capability), 0) + 1)
            for capability in candidate.capability_categories
        ]
        novelty = sum(novelty_terms) / len(novelty_terms)

        capability_reliability: dict[str, float] = {}
        workflow_ids: set[str] = set()
        for capability in candidate.capability_categories:
            ranked = [
                (wilson_lower_bound(item.success_count, item.usage_count), item)
                for item in state.workflows
                if _workflow_applies(item, candidate, source, state)
                and capability in item.capability_categories
            ]
            if not ranked:
                capability_reliability[capability] = self.unsupported_prior
                continue
            reliability, workflow = max(ranked, key=lambda pair: (pair[0], _stable_id(pair[1].workflow_id)))
            capability_reliability[capability] = reliability
            workflow_ids.add(workflow.workflow_id)
        competence = min(capability_reliability.values())
        frontier = 4.0 * competence * (1.0 - competence)
        return TaskScore(
            novelty=novelty,
            competence=competence,
            competence_frontier=frontier,
            objective=novelty * frontier,
            context_key=context,
            capability_reliability=dict(sorted(capability_reliability.items())),
            applicable_workflow_ids=tuple(sorted(workflow_ids, key=_id_sort_key)),
        )


class SelfDirectedTaskProposer:
    """Generate, gate, score, select, and audit one self-directed task."""

    def __init__(
        self,
        generator: CandidateGenerator,
        *,
        validator: CandidateValidator | None = None,
        scorer: InquiryScorer | None = None,
    ) -> None:
        self.generator = generator
        self.validator = validator or CandidateValidator()
        self.scorer = scorer or InquiryScorer()

    def propose(self, state: ProposalState, *, candidate_count: int | None = None) -> ProposalRound:
        context = build_current_scene_context(state)
        count = min(candidate_count or state.budget.candidate_count, state.budget.candidate_count)
        candidates = self.generator.generate(state=state, context=context, candidate_count=count)
        evaluations: list[CandidateEvaluation] = []
        seen: list[str] = []
        sources = {item.image_id: item for item in state.source_images}
        for candidate in candidates:
            issues = self.validator.validate(candidate, state, batch_signatures=seen)
            signature = candidate.canonical_signature(sources.get(candidate.source_image_id or ""))
            seen.append(signature)
            evaluations.append(
                CandidateEvaluation(
                    candidate=candidate,
                    issues=issues,
                    score=None if issues else self.scorer.score(candidate, state),
                )
            )
        accepted = [item for item in evaluations if item.accepted]
        if not accepted:
            detail = "; ".join(f"{item.candidate.task_id}: {','.join(issue.code.value for issue in item.issues)}" for item in evaluations)
            raise RuntimeError("all proposed tasks were rejected: " + detail)
        selected = max(
            accepted,
            key=lambda item: (
                item.score.objective if item.score else -1.0,
                item.score.novelty if item.score else -1.0,
                -len(item.candidate.capability_categories),
                item.candidate.task_id,
            ),
        )
        return ProposalRound(context, tuple(evaluations), _build_contract(selected, state))


def build_current_scene_context(state: ProposalState) -> CurrentSceneContext:
    modality = {item.value: 0 for item in Modality}
    capability = {item.name: 0 for item in state.capabilities}
    reliability = {item.value: 0 for item in ReliabilityTier}
    for workflow in state.workflows:
        modality[workflow.modality.value] = modality.get(workflow.modality.value, 0) + 1
        reliability[workflow.reliability_tier.value] = reliability.get(workflow.reliability_tier.value, 0) + 1
        for name in workflow.capability_categories:
            capability[name] = capability.get(name, 0) + 1
    source_metadata = tuple(
        {
            "image_id": item.image_id,
            "filename": Path(item.path).name,
            "width": item.width,
            "height": item.height,
            "format": item.format,
            "caption": item.caption,
            "semantic_tags": item.semantic_tags,
            "quality_flags": item.quality_flags,
            "applicable_capabilities": item.applicable_capabilities,
        }
        for item in state.source_images
    )
    return CurrentSceneContext(
        iteration=state.iteration,
        modality_coverage=dict(sorted(modality.items())),
        capability_coverage=dict(sorted(capability.items())),
        reliability_distribution=dict(sorted(reliability.items())),
        source_image_pool=tuple(item.image_id for item in state.source_images),
        source_image_metadata=source_metadata,
        recent_task_signatures=tuple(item.task_signature for item in state.recent_tasks),
    )


def load_proposal_state(
    symbolic_policy_root: str | Path,
    source_image_pool_root: str | Path,
    *,
    iteration: int,
    requested_modality: Modality | None = None,
    candidate_count: int = 10,
    resource_inventory: ResourceInventory | None = None,
) -> ProposalState:
    policy_root = Path(symbolic_policy_root).resolve()
    source_root = Path(source_image_pool_root).resolve()
    # Read a new/empty library as S_1 = (empty, empty), without writing files.
    # A partly populated library must never be silently reset.
    fresh = not policy_root.exists() or (policy_root.is_dir() and not any(policy_root.iterdir()))
    workflow_raw = _read_json_object(policy_root / "workflow_metadata.json", missing_ok=fresh)
    failure_raw = _read_json_object(policy_root / "failure_metadata.json", missing_ok=fresh)
    # T2I-only inquiry uses no source-image context or I2I capability space.
    source_raw = {} if requested_modality is Modality.T2I else _read_json_object(source_root / "source_image_metadata.json")
    workflows = tuple(SymbolicPolicy.from_metadata(key, value) for key, value in workflow_raw.items() if isinstance(value, Mapping))
    failures = tuple(FailureEntry.from_metadata(key, value) for key, value in failure_raw.items() if key != "_metadata" and isinstance(value, Mapping))
    sources = tuple(SourceImage.from_dict(key, value, source_root) for key, value in source_raw.items() if isinstance(value, Mapping))

    memory_path = policy_root / ".omniharness_state.json"
    memory = _read_json_object(memory_path, missing_ok=True)
    history = tuple(TaskHistoryRecord.from_dict(item) for item in memory.get("task_history", ()) if isinstance(item, Mapping))
    attempts = {str(key): int(value) for key, value in dict(memory.get("attempt_counts", {})).items()}
    if resource_inventory is None:
        inferred = {"text_to_image"}
        if sources:
            inferred.add("image_input")
        for workflow in workflows:
            inferred.update(workflow.dependencies)
        resource_inventory = ResourceInventory(frozenset(inferred))
    return ProposalState(
        iteration=iteration,
        capabilities=tuple(
            item for item in GENERATIVE_CAPABILITY_SPACE
            if requested_modality is None or item.modality is requested_modality
        ),
        workflows=workflows,
        failures=failures,
        source_images=sources,
        resources=resource_inventory,
        recent_tasks=history,
        attempt_counts=attempts,
        requested_modality=requested_modality,
        budget=ProposalBudget(candidate_count=candidate_count),
    )


def task_context_key(candidate: CandidateTask) -> str:
    """Use fixed contexts for source-free tasks and media contexts otherwise."""

    prefix = candidate.modality.value.lower()
    if not modality_requires_source(candidate.modality):
        return prefix
    return f"{prefix}:{candidate.source_image_id or 'missing-source'}"


def context_capability_key(context: str, capability: str) -> str:
    return f"{normalize_text(context)}::{normalize_text(capability)}"


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    if not 0 <= successes <= trials:
        raise ValueError("Wilson counts require 0 <= successes <= trials")
    if trials == 0:
        return 0.0
    p = successes / trials
    z2 = z * z
    centre = p + z2 / (2.0 * trials)
    spread = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
    return max(0.0, (centre - spread) / (1.0 + z2 / trials))


def write_proposal_audit(path: str | Path, round_: ProposalRound) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(to_primitive(round_), ensure_ascii=False, sort_keys=True) + "\n")


def extract_json_object(text: str) -> Mapping[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return payload
    raise ValueError("no JSON object found")


def to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: to_primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [to_primitive(item) for item in value]
    return value


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _build_contract(evaluation: CandidateEvaluation, state: ProposalState) -> TaskContract:
    candidate = evaluation.candidate
    assert evaluation.score is not None
    source = next((item for item in state.source_images if item.image_id == candidate.source_image_id), None)
    relevant_failures = [
        item.failure_id
        for item in state.failures
        if item.modality is candidate.modality and set(item.capability_categories).intersection(candidate.capability_categories)
    ]
    return TaskContract(
        task_id=f"inquiry_{state.iteration:04d}_{candidate.task_id}",
        inquiry_iteration=state.iteration,
        description=candidate.description,
        modality=candidate.modality,
        capability_categories=candidate.capability_categories,
        source_image_id=candidate.source_image_id,
        source_image_path=None if source is None else source.path,
        generation_constraints=candidate.generation_constraints,
        preservation_constraints=candidate.preservation_constraints,
        success_criteria=candidate.success_criteria,
        retrieved_workflow_ids=evaluation.score.applicable_workflow_ids,
        retrieved_failure_ids=tuple(sorted(relevant_failures)),
        selection_evidence=evaluation.score,
        task_signature=candidate.canonical_signature(source),
        source_image_paths=() if source is None else (source.path,),
    )


def _workflow_applies(
    workflow: SymbolicPolicy,
    candidate: CandidateTask,
    source: SourceImage | None,
    state: ProposalState,
) -> bool:
    if candidate.applicable_workflow_ids is None or workflow.workflow_id not in candidate.applicable_workflow_ids:
        return False
    if workflow.reliability_tier is ReliabilityTier.SUSPENDED:
        return False
    if workflow.usage_count < 1 or workflow.success_count < 1:
        return False
    if workflow.modality is not candidate.modality:
        return False
    if state.resources.missing(workflow.dependencies):
        return False
    if modality_requires_source(candidate.modality) and source is None:
        return False
    return bool(set(workflow.capability_categories).intersection(candidate.capability_categories))


def _semantic_similarity(left: str, right: str) -> float:
    a = set(re.findall(r"[a-z0-9_\u4e00-\u9fff]+", normalize_text(left)))
    b = set(re.findall(r"[a-z0-9_\u4e00-\u9fff]+", normalize_text(right)))
    return len(a & b) / len(a | b) if a or b else 1.0


def _resource_key(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def _split_categories(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in re.split(r"[,;]", value) if item.strip())
    return _strings(value)


def _split_items(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _strings(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stable_id(value: str) -> str:
    return f"{int(value):020d}" if value.isdigit() else value


def _id_sort_key(value: str) -> tuple[int, str]:
    return (0, _stable_id(value)) if value.isdigit() else (1, value)


def _read_json_object(path: Path, *, missing_ok: bool = False) -> dict[str, Any]:
    if missing_ok and not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OmniHarness self-directed inquiry task proposal")
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("OMNIHARNESS_CONFIG", "config.yaml")))
    parser.add_argument("--symbolic-policy", type=Path, default=None)
    parser.add_argument("--source-image-pool", type=Path, default=None)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--candidate-count", type=int, default=None)
    parser.add_argument("--modality", choices=("T2I", "I2I"), default=None)
    parser.add_argument("--resource-inventory", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--audit", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-provider", default=None)
    parser.add_argument("--codex-reasoning-effort", default=None)
    parser.add_argument("--codex-base-url", default=None)
    parser.add_argument("--codex-api-key-env", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_omniharness_config(args.config)
    paths = config_section(config, "paths")
    proposal_config = config_section(config, "proposal")
    codex_config = config_section(config, "codex")
    resources_root = runtime_resources_root()

    def configured_path(cli_value: Path | None, key: str, default: Path) -> Path:
        if cli_value is not None:
            return cli_value.expanduser().resolve()
        return resolve_config_path(paths.get(key), args.config, default)

    symbolic_policy = configured_path(
        args.symbolic_policy,
        "symbolic_policy",
        repository_root() / "runs" / "policy_library",
    )
    source_image_pool = configured_path(
        args.source_image_pool,
        "source_image_pool",
        resources_root / "source_image_pool_comfybench",
    )
    project_root = configured_path(args.project_root, "project_root", repository_root())
    output = configured_path(args.output, "proposal_output", Path.cwd() / "task_contract.json")
    audit = configured_path(args.audit, "proposal_audit", Path.cwd() / "proposal_audit.jsonl")
    iteration = int(args.iteration if args.iteration is not None else proposal_config.get("iteration", 1))
    candidate_count = int(args.candidate_count if args.candidate_count is not None else proposal_config.get("candidate_count", 10))
    modality_value = args.modality if args.modality is not None else proposal_config.get("modality")
    model = str(args.codex_model or codex_config.get("model") or CODEX_MODEL)
    provider = str(args.codex_provider or codex_config.get("provider") or CODEX_MODEL_PROVIDER)
    reasoning_effort = args.codex_reasoning_effort or codex_config.get("reasoning_effort") or CODEX_REASONING_EFFORT
    provider_base_url = args.codex_base_url or codex_config.get("base_url") or CODEX_BASE_URL
    api_key_env = str(args.codex_api_key_env or codex_config.get("api_key_env") or CODEX_API_KEY_ENV)
    resources = None
    if args.resource_inventory is not None:
        resources = ResourceInventory.from_dict(_read_json_object(args.resource_inventory))
    state = load_proposal_state(
        symbolic_policy,
        source_image_pool,
        iteration=iteration,
        requested_modality=None if modality_value in (None, "") else Modality(str(modality_value).upper()),
        candidate_count=candidate_count,
        resource_inventory=resources,
    )
    generator = CodexCandidateGenerator(
        project_root,
        model=model,
        model_provider=provider,
        reasoning_effort=None if reasoning_effort in (None, "") else str(reasoning_effort),
        provider_base_url=None if provider_base_url in (None, "") else str(provider_base_url),
        api_key_env=api_key_env,
    )
    try:
        round_ = SelfDirectedTaskProposer(generator).propose(state)
    finally:
        generator.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(to_primitive(round_.selected), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_proposal_audit(audit, round_)
    print(json.dumps({"task_contract": str(output), "selected_task": round_.selected.task_id, "objective": round_.selected.selection_evidence.objective}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
