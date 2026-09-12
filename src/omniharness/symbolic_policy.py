"""Input abstraction for symbolic policies represented by workflow templates.

These templates are planning inputs, not directly executable graphs. The
workflow writer must bind their input roles for each task before execution.
This does not infer arbitrary semantic task families from an execution trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


TEMPLATE_PREFIX = "$POLICY_"


@dataclass(frozen=True)
class PolicyTemplate:
    graph: Mapping[str, Any]
    input_roles: tuple[Mapping[str, Any], ...]


def distill_workflow_graph(
    graph: Mapping[str, Any],
    *,
    instruction: str = "",
    source_images: Sequence[str] = (),
) -> PolicyTemplate:
    """Abstract prompt/media/seed/output literals while retaining graph edges.

    Existing resource graphs can be distilled in memory without modifying them.
    Each binding location has an explicit role; independent prompt branches
    remain independently bindable. Models, samplers and other procedural
    parameters are retained.
    """

    nodes = {str(key): value for key, value in graph.items() if isinstance(value, Mapping)}
    roles: list[Mapping[str, Any]] = []
    counts: dict[str, int] = {}
    sources = {str(value) for value in source_images if value}
    sources.update(Path(value).name for value in source_images if value)

    def abstract(node_id: str, class_type: str, name: str, value: Any, path: tuple[Any, ...]) -> Any:
        if (
            isinstance(value, list) and len(value) == 2
            and isinstance(value[0], (str, int)) and isinstance(value[1], int)
            and str(value[0]) in nodes
        ):
            return list(value)
        if isinstance(value, Mapping):
            return {str(key): abstract(node_id, class_type, str(key), item, (*path, str(key))) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [abstract(node_id, class_type, name, item, (*path, index)) for index, item in enumerate(value)]
        role = _input_role(class_type, name, value, instruction, sources)
        if role is None:
            return value
        index = counts.get(role, 0)
        counts[role] = index + 1
        token = f"{TEMPLATE_PREFIX}{role.upper()}_{index}"
        roles.append({
            "role": role,
            "placeholder": token,
            "node_id": node_id,
            "input_path": list(path),
            "binding": "Bind for the current task before compilation and execution.",
        })
        return token

    distilled: dict[str, Any] = {}
    for node_id, node in nodes.items():
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {})
        if not isinstance(inputs, Mapping):
            raise ValueError(f"node {node_id} inputs must be a mapping")
        # Display titles and arbitrary annotations may contain instance prompts.
        distilled[node_id] = {
            "class_type": class_type,
            "inputs": {
                str(name): abstract(node_id, class_type, str(name), value, (str(name),))
                for name, value in inputs.items()
            },
        }
    return PolicyTemplate(distilled, tuple(roles))


def contains_unbound_policy_input(value: Any) -> bool:
    if isinstance(value, str):
        return TEMPLATE_PREFIX in value or "$HARNESS_" in value
    if isinstance(value, Mapping):
        return any(contains_unbound_policy_input(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_unbound_policy_input(item) for item in value)
    return False


def redact_instance_values(value: Any, instruction: str, source_images: Sequence[str]) -> Any:
    """Remove exact instance inputs from descriptive metadata, not failure logs."""
    if isinstance(value, str):
        if instruction:
            value = value.replace(instruction, "the current task instruction")
        for source in sorted({str(item) for item in source_images if item}, key=len, reverse=True):
            value = value.replace(source, "the current source media")
            name = Path(source).name
            if name:
                value = value.replace(name, "the current source media")
        return value
    if isinstance(value, Mapping):
        return {str(key): redact_instance_values(item, instruction, source_images) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [redact_instance_values(item, instruction, source_images) for item in value]
    return value


def _input_role(class_type: str, name: str, value: Any, instruction: str, sources: set[str]) -> str | None:
    key = name.lower()
    kind = class_type.lower()
    if isinstance(value, str) and value.startswith((TEMPLATE_PREFIX, "$HARNESS_")):
        # Re-distillation is idempotent, including role names containing '_'.
        return value.split("_", 1)[1].rsplit("_", 1)[0].lower()
    if key in {"seed", "noise_seed", "random_seed"}:
        return "seed"
    if key in {"filename", "filename_prefix", "output_name"} and any(term in kind for term in ("save", "preview", "combine")):
        return "output_name"
    if not isinstance(value, str):
        return None
    if value.startswith(("$SOURCE_IMAGE_", "$SOURCE_MEDIA_")) or value in sources:
        return "source_media"
    media_field = key in {"image", "images", "image_path", "video", "videos", "video_path", "source_image", "source_video", "mask"}
    media_loader = any(term in kind for term in ("image", "video", "media", "frame", "mask")) and any(term in kind for term in ("load", "input", "upload"))
    if media_field or (media_loader and key in {"file", "path", "filename", "url"}):
        return "source_media"
    prompt_field = key in {"text", "prompt", "positive", "negative", "text_positive", "text_negative", "positive_prompt", "negative_prompt", "prompt_text", "instruction"}
    if prompt_field or (key in {"string", "value"} and any(term in kind for term in ("text", "string", "prompt"))):
        return "negative_prompt" if "negative" in key else "task_prompt"
    if instruction and instruction in value:
        return "task_prompt"
    return None
