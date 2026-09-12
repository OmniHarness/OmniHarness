"""Shared, reproducible benchmark execution support for OmniHarness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "config.yaml"
DEFAULT_BENCHMARK_ROOT = REPOSITORY_ROOT / "bench data"
SAFE_IDENTIFIER = re.compile(r"[^A-Za-z0-9_.-]+")


class BenchmarkError(RuntimeError):
    """Raised when benchmark data or execution settings are invalid."""


@dataclass(frozen=True)
class BenchmarkTask:
    benchmark: str
    sample_id: str
    instruction: str
    modality: str
    capability_categories: tuple[str, ...]
    source_image_paths: tuple[Path, ...] = ()
    generation_constraints: tuple[Mapping[str, Any], ...] = ()
    preservation_constraints: tuple[Mapping[str, Any], ...] = ()
    success_criteria: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return _safe_identifier(f"{self.benchmark}_{self.sample_id}")


@dataclass(frozen=True)
class SkippedTask:
    sample_id: str
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkLoadResult:
    tasks: tuple[BenchmarkTask, ...]
    skipped: tuple[SkippedTask, ...] = ()


BenchmarkLoader = Callable[[Path, argparse.Namespace], BenchmarkLoadResult]
ParserConfigurator = Callable[[argparse.ArgumentParser], None]


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise BenchmarkError(f"benchmark file is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot parse JSON {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise BenchmarkError(f"benchmark file is missing: {path}")
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise BenchmarkError(
                        f"JSONL row must be an object: {path}:{line_number}"
                    )
                rows.append(dict(payload))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot parse JSONL {path}: {exc}") from exc
    return rows


def natural_key(value: str) -> tuple[tuple[int, Any], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", value)
        if part
    )


def infer_capabilities(instruction: str, modality: str) -> tuple[str, ...]:
    """Map public benchmark instructions to the fixed OmniHarness capability space."""

    text = instruction.lower()
    normalized_modality = modality.upper()
    if normalized_modality == "T2V":
        return ("Text-to-Video Generation",)
    if normalized_modality == "I2V":
        return ("Image-to-Video Animation",)
    if normalized_modality == "V2V":
        return ("Video Editing & Enhancement",)
    if normalized_modality == "T2I":
        capabilities: list[str] = []
        if any(token in text for token in ("left of", "right of", "above", "below", "behind", "in front of", "under ", "on top of")):
            capabilities.append("Position-Constrained Generation")
        if any(token in text for token in ("text", "letter", "word", "caption", "typography")):
            capabilities.append("In-Image Text Generation")
        if any(token in text for token in ("poster", "logo", "cover", "advertisement")):
            capabilities.append("Poster & Graphic Design")
        if any(token in text for token in ("comic", "illustration", "cartoon", "sketch")):
            capabilities.append("Illustration & Comic Generation")
        if any(token in text for token in ("fantasy", "futuristic", "science-fiction", "sci-fi")):
            capabilities.append("Futuristic & Fantasy Generation")
        if not capabilities:
            capabilities.append("Photorealistic Generation")
        return tuple(dict.fromkeys(capabilities))

    capabilities = []
    if any(token in text for token in ("upscale", "super-resolution", "high-resolution", "resolution by")):
        capabilities.append("Image Super-Resolution")
    if any(token in text for token in ("extend the image", "outpaint", "outside the image", "pixels on")):
        capabilities.append("Image Outpainting")
    if any(token in text for token in ("restore", "refine", "repair", "correct unreasonable", "artifact", "deblur", "denoise")):
        capabilities.append("Image Restoration & Refinement")
    if any(token in text for token in ("style", "repaint", "painting", "watercolor", "cubist", "convert it into a portrait")):
        capabilities.append("Style Transfer & Repainting")
    if any(token in text for token in ("same pose", "reference", "match", "view", "predict", "frame", "according to the images")):
        capabilities.append("Reference-Guided Generation")
    if not capabilities or any(token in text for token in ("add ", "remove ", "replace ", "change ", "move ", "place ", "complete ", "draw ", "fill ")):
        capabilities.append("Localized Image Editing")
    return tuple(dict.fromkeys(capabilities))


def public_success_criterion(instruction: str) -> tuple[Mapping[str, Any], ...]:
    return (
        {
            "criterion": f"The output faithfully satisfies the benchmark instruction: {instruction}",
            "verifier": "multimodal_goal_verifier",
        },
    )


def _safe_identifier(value: str) -> str:
    normalized = SAFE_IDENTIFIER.sub("-", value.strip()).strip("-._")
    if not normalized:
        raise BenchmarkError(f"cannot create a task identifier from {value!r}")
    return normalized.lower()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _task_contract(task: BenchmarkTask, ordinal: int) -> dict[str, Any]:
    source_paths = [str(path.resolve()) for path in task.source_image_paths]
    signature_payload = {
        "benchmark": task.benchmark,
        "sample_id": task.sample_id,
        "instruction": task.instruction,
        "source_image_paths": source_paths,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    capability_reliability = {
        capability: 0.5 for capability in task.capability_categories
    }
    return {
        "task_id": task.task_id,
        "inquiry_iteration": None,
        "downstream_index": ordinal,
        "description": task.instruction,
        "modality": task.modality.upper(),
        "capability_categories": list(task.capability_categories),
        "source_image_id": task.sample_id if source_paths else None,
        "source_image_path": source_paths[0] if source_paths else None,
        "source_image_paths": source_paths,
        "generation_constraints": [dict(item) for item in task.generation_constraints],
        "preservation_constraints": [
            dict(item) for item in task.preservation_constraints
        ],
        "success_criteria": [dict(item) for item in task.success_criteria],
        "retrieved_workflow_ids": [],
        "retrieved_failure_ids": [],
        "selection_evidence": {
            "novelty": 0.0,
            "competence": 0.5,
            "competence_frontier": 1.0,
            "objective": 0.0,
            "context_key": f"benchmark:{task.benchmark}",
            "capability_reliability": capability_reliability,
            "applicable_workflow_ids": [],
        },
        "task_signature": signature,
    }


def _evaluation_metadata(task: BenchmarkTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "benchmark": task.benchmark,
        "sample_id": task.sample_id,
        "instruction": task.instruction,
        "source_media_paths": [
            str(path.resolve()) for path in task.source_image_paths
        ],
        "evaluation": dict(task.metadata),
    }


def _validate_tasks(tasks: Sequence[BenchmarkTask]) -> None:
    seen: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            raise BenchmarkError(f"duplicate benchmark task ID: {task.task_id}")
        seen.add(task.task_id)
        if not task.instruction.strip():
            raise BenchmarkError(f"empty instruction: {task.task_id}")
        modality = task.modality.upper()
        if modality not in {"T2I", "I2I", "T2V", "I2V", "V2V"}:
            raise BenchmarkError(
                f"unsupported OmniHarness modality {task.modality!r}: {task.task_id}"
            )
        if modality in {"T2I", "T2V"} and task.source_image_paths:
            raise BenchmarkError(f"{modality} task has source media: {task.task_id}")
        if modality in {"I2I", "I2V", "V2V"} and not task.source_image_paths:
            raise BenchmarkError(f"{modality} task has no source media: {task.task_id}")
        for path in task.source_image_paths:
            if not path.is_file():
                raise BenchmarkError(f"source image is missing: {path}")
        if not task.capability_categories:
            raise BenchmarkError(f"task has no capabilities: {task.task_id}")
        if not task.success_criteria:
            raise BenchmarkError(f"task has no success criteria: {task.task_id}")


def _select_tasks(
    tasks: Sequence[BenchmarkTask], args: argparse.Namespace
) -> list[BenchmarkTask]:
    selected = list(tasks)
    if args.sample_id:
        requested = set(args.sample_id)
        selected = [
            task
            for task in selected
            if task.sample_id in requested or task.task_id in requested
        ]
        matched = {task.sample_id for task in selected} | {
            task.task_id for task in selected
        }
        missing = sorted(requested - matched)
        if missing:
            raise BenchmarkError(
                "requested sample IDs were not found: " + ", ".join(missing)
            )
    if args.num_shards < 1:
        raise BenchmarkError("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise BenchmarkError("--shard-index must be in [0, num-shards)")
    if args.num_shards > 1:
        selected = [
            task
            for task in selected
            if int(hashlib.sha256(task.task_id.encode()).hexdigest(), 16)
            % args.num_shards
            == args.shard_index
        ]
    if args.start < 0:
        raise BenchmarkError("--start cannot be negative")
    selected = selected[args.start :]
    if args.limit is not None:
        if args.limit < 1:
            raise BenchmarkError("--limit must be at least 1")
        selected = selected[: args.limit]
    return selected


def _launcher_arguments(
    args: argparse.Namespace,
    contract_path: Path,
    execution_root: Path,
    agent_project_root: Path,
    *,
    skip_comfyui_check: bool,
) -> list[str]:
    arguments = [
        "execute",
        "--config",
        str(args.config.resolve()),
        "--task-contract",
        str(contract_path.resolve()),
        "--output-root",
        str(execution_root.resolve()),
        "--project-root",
        str(agent_project_root.resolve()),
        "--preflight-timeout",
        str(args.preflight_timeout),
    ]
    option_values = (
        ("--memory-mode", args.memory_mode),
        ("--symbolic-policy", args.symbolic_policy),
        ("--source-image-pool", args.source_image_pool),
        ("--comfyui-url", args.comfyui_url),
        ("--max-retries", args.max_retries),
        ("--max-attempts", args.max_attempts),
        ("--max-wall-time", args.max_wall_time),
        ("--codex-model", args.codex_model),
        ("--codex-provider", args.codex_provider),
        ("--codex-reasoning-effort", args.codex_reasoning_effort),
        ("--codex-base-url", args.codex_base_url),
        ("--codex-api-key-env", args.codex_api_key_env),
    )
    for flag, value in option_values:
        if value not in (None, ""):
            arguments.extend((flag, str(value)))
    if args.dry_run:
        arguments.append("--dry-run")
    if skip_comfyui_check:
        arguments.append("--skip-comfyui-check")
    if args.verify_resource_hashes:
        arguments.append("--verify-resource-hashes")
    if args.debug:
        arguments.append("--debug")
    return arguments


def _build_parser(
    benchmark: str,
    default_data_root: Path,
    configure: ParserConfigurator | None,
) -> argparse.ArgumentParser:
    slug = _safe_identifier(benchmark)
    parser = argparse.ArgumentParser(
        description=f"Run the {benchmark} benchmark with OmniHarness"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "runs" / "benchmarks" / slug,
    )
    parser.add_argument("--sample-id", action="append")
    parser.add_argument(
        "--agent-project-root",
        type=Path,
        help="optional sanitized Codex working directory; defaults to an empty session directory",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    error_group = parser.add_mutually_exclusive_group()
    error_group.add_argument(
        "--continue-on-error", dest="continue_on_error", action="store_true"
    )
    error_group.add_argument(
        "--no-continue-on-error", dest="continue_on_error", action="store_false"
    )
    parser.set_defaults(continue_on_error=True)
    parser.add_argument("--memory-mode", choices=("online", "frozen"), help="override policy updates; use frozen for the online policy update ablation and frozen snapshot studies")
    parser.add_argument("--symbolic-policy", type=Path, help="policy library for this evaluation run")
    parser.add_argument("--source-image-pool", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--comfyui-url")
    retry_group = parser.add_mutually_exclusive_group()
    retry_group.add_argument("--max-retries", type=int)
    retry_group.add_argument("--max-attempts", type=int)
    parser.add_argument("--max-wall-time", type=int)
    parser.add_argument("--codex-model")
    parser.add_argument("--codex-provider")
    parser.add_argument("--codex-reasoning-effort")
    parser.add_argument("--codex-base-url")
    parser.add_argument("--codex-api-key-env")
    parser.add_argument("--skip-comfyui-check", action="store_true")
    parser.add_argument("--verify-resource-hashes", action="store_true")
    parser.add_argument("--preflight-timeout", type=float, default=10.0)
    parser.add_argument("--debug", action="store_true")
    if configure is not None:
        configure(parser)
    return parser


def run_benchmark(
    benchmark: str,
    default_data_root: Path,
    loader: BenchmarkLoader,
    *,
    configure_parser: ParserConfigurator | None = None,
    argv: Sequence[str] | None = None,
) -> int:
    parser = _build_parser(benchmark, default_data_root, configure_parser)
    args = parser.parse_args(argv)
    if args.preflight_timeout <= 0:
        parser.error("--preflight-timeout must be greater than zero")
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    args.config = args.config.expanduser().resolve()

    try:
        if not data_root.is_dir():
            raise BenchmarkError(f"benchmark data directory is missing: {data_root}")
        loaded = loader(data_root, args)
        _validate_tasks(loaded.tasks)
        selected = _select_tasks(loaded.tasks, args)
        print(
            json.dumps(
                {
                    "benchmark": benchmark,
                    "data_root": str(data_root),
                    "loaded": len(loaded.tasks),
                    "skipped_by_adapter": len(loaded.skipped),
                    "selected": len(selected),
                    "shard": f"{args.shard_index}/{args.num_shards}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if args.list_only:
            return 0
        if not selected:
            raise BenchmarkError("no benchmark tasks remain after filtering")

        contracts_root = output_root / "contracts"
        evaluation_root = output_root / "evaluation_metadata"
        contracts: dict[str, Path] = {}
        for ordinal, task in enumerate(selected, start=1):
            contract_path = contracts_root / f"{task.task_id}.json"
            _atomic_write_json(contract_path, _task_contract(task, ordinal))
            _atomic_write_json(
                evaluation_root / f"{task.task_id}.json",
                _evaluation_metadata(task),
            )
            contracts[task.task_id] = contract_path

        manifest = {
            "benchmark": benchmark,
            "created_at": _utc_now(),
            "data_root": str(data_root),
            "memory_mode": args.memory_mode or "from_config",
            "symbolic_policy_override": None if args.symbolic_policy is None else str(args.symbolic_policy.resolve()),
            "dry_run": args.dry_run,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "selected_task_ids": [task.task_id for task in selected],
            "skipped_by_adapter": [asdict(item) for item in loaded.skipped],
        }
        _atomic_write_json(output_root / "benchmark_manifest.json", manifest)
        if args.prepare_only:
            print(
                f"Prepared {len(selected)} public contracts in {contracts_root}; "
                f"private evaluation metadata is in {evaluation_root}"
            )
            return 0
        if not args.config.is_file():
            raise BenchmarkError(
                f"configuration file is missing: {args.config}; create it from config.example.yaml"
            )

        from run_omniharness import main as launch_omniharness

        state_path = output_root / "benchmark_state.jsonl"
        # Task IDs alone do not identify the input media, model, or policy library state.
        # Evaluate every selected task in this session instead of reusing old status.
        pending = selected
        session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_{os.getpid()}"
        session_root = output_root / "sessions" / session_id
        if args.agent_project_root is None:
            agent_project_root = session_root / "agent_workspace"
            agent_project_root.mkdir(parents=True, exist_ok=False)
        else:
            agent_project_root = args.agent_project_root.expanduser().resolve()
            if not agent_project_root.is_dir():
                raise BenchmarkError(
                    f"agent project root is missing: {agent_project_root}"
                )
        completed = 0
        failed = 0
        dry_runs = 0
        for position, task in enumerate(pending):
            print(
                f"[{benchmark}] {position + 1}/{len(pending)} {task.task_id}",
                flush=True,
            )
            started_at = _utc_now()
            execution_root = session_root / task.task_id
            launcher_argv = _launcher_arguments(
                args,
                contracts[task.task_id],
                execution_root,
                agent_project_root,
                skip_comfyui_check=args.skip_comfyui_check or position > 0,
            )
            try:
                return_code = int(launch_omniharness(launcher_argv))
            except KeyboardInterrupt:
                _append_jsonl(
                    state_path,
                    {
                        "task_id": task.task_id,
                        "status": "interrupted",
                        "started_at": started_at,
                        "finished_at": _utc_now(),
                        "session_id": session_id,
                    },
                )
                raise
            except Exception:
                if args.debug:
                    raise
                return_code = 1
            if return_code == 0:
                status = "dry_run" if args.dry_run else "completed"
                completed += int(status == "completed")
                dry_runs += int(status == "dry_run")
            else:
                status = "failed"
                failed += 1
            _append_jsonl(
                state_path,
                {
                    "task_id": task.task_id,
                    "sample_id": task.sample_id,
                    "status": status,
                    "return_code": return_code,
                    "started_at": started_at,
                    "finished_at": _utc_now(),
                    "session_id": session_id,
                    "contract": str(contracts[task.task_id]),
                    "execution_root": str(execution_root),
                },
            )
            if return_code != 0 and not args.continue_on_error:
                break

        summary = {
            "benchmark": benchmark,
            "session_id": session_id,
            "selected": len(selected),
            "attempted": completed + dry_runs + failed,
            "completed": completed,
            "dry_run": dry_runs,
            "failed": failed,
            "session_root": str(session_root),
            "agent_project_root": str(agent_project_root),
        }
        _atomic_write_json(session_root / "benchmark_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if failed == 0 else 2
    except KeyboardInterrupt:
        print(f"\n[{benchmark}] interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.debug:
            raise
        print(f"[{benchmark}] {exc}", file=sys.stderr)
        return 1
