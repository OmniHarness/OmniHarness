"""Validated command-line launcher for OmniHarness experiments.

The launcher is intentionally thin at execution time: it resolves and checks
the experiment configuration, then delegates all proposal, planning, workflow
generation, execution, verification, recovery, and symbolic policy learning to
``src/omniharness/omniharness_runtime.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "omniharness"
RUNTIME_PATH = SOURCE_ROOT / "omniharness_runtime.py"
PROPOSAL_PATH = SOURCE_ROOT / "self_directed_inquiry.py"
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "config.yaml"
COMMANDS = ("inquiry", "execute", "snapshot", "consolidate")
ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class LaunchError(RuntimeError):
    """Raised when OmniHarness cannot be launched safely."""


@dataclass(frozen=True)
class LaunchSettings:
    command: str
    config_path: Path
    project_root: Path
    symbolic_policy: Path
    source_image_pool: Path
    output_root: Path
    comfyui_url: str | None
    memory_mode: str
    max_attempts: int
    max_wall_time: int
    dry_run: bool
    codex_model: str
    codex_provider: str
    codex_reasoning_effort: str | None
    codex_base_url: str | None
    codex_api_key_env: str
    iterations: int
    candidate_count: int
    start_iteration: int
    modality: str | None
    task_contract: Path | None
    snapshot_destination: Path | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run OmniHarness with configuration, dependency, resource, "
            "credential, and ComfyUI preflight checks."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=COMMANDS,
        default="inquiry",
        help="runtime operation; defaults to inquiry",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="experiment YAML file (default: repository/config.yaml)",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate the experiment without starting the runtime",
    )
    parser.add_argument(
        "--skip-comfyui-check",
        action="store_true",
        help="skip the network probe; the runtime will still require ComfyUI",
    )
    parser.add_argument(
        "--preflight-timeout",
        type=float,
        default=10.0,
        help="seconds allowed for the ComfyUI preflight request",
    )
    parser.add_argument(
        "--verify-resource-hashes",
        action="store_true",
        help="verify source images against SHA-256 values in their metadata",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show a traceback instead of a concise launcher error",
    )

    parser.add_argument("--comfyui-url")
    parser.add_argument("--symbolic-policy", type=Path)
    parser.add_argument("--source-image-pool", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--memory-mode", choices=("online", "frozen"))
    retry_group = parser.add_mutually_exclusive_group()
    retry_group.add_argument("--max-retries", type=int, help="retries after the initial attempt (default: 4)")
    retry_group.add_argument("--max-attempts", type=int, help="legacy total attempt budget, including the initial attempt")
    parser.add_argument("--max-wall-time", type=int)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--codex-model")
    parser.add_argument("--codex-provider")
    parser.add_argument("--codex-reasoning-effort")
    parser.add_argument("--codex-base-url")
    parser.add_argument("--codex-api-key-env")

    parser.add_argument("--iterations", type=int)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--start-iteration", type=int)
    parser.add_argument("--modality", choices=("T2I", "I2I"), help="image-only inquiry modality; downstream contracts may request video")
    parser.add_argument("--task-contract", type=Path)
    parser.add_argument("--destination", type=Path)
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        example = REPOSITORY_ROOT / "config.example.yaml"
        raise LaunchError(
            f"configuration file not found: {path}\n"
            f"Create it from {example} and insert the third-party provider URL."
        )
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise LaunchError(
            "PyYAML is unavailable; install the repository requirements first"
        ) from exc
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise LaunchError(f"cannot load configuration {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LaunchError("the YAML configuration root must be a mapping")
    return dict(payload)


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise LaunchError(f"configuration section '{name}' must be a mapping")
    return dict(value)


def _resolve_path(
    cli_value: Path | None,
    configured_value: Any,
    default: Path,
    config_path: Path,
) -> Path:
    if cli_value is not None:
        return cli_value.expanduser().resolve()
    if configured_value not in (None, ""):
        candidate = Path(str(configured_value)).expanduser()
        if not candidate.is_absolute():
            candidate = config_path.parent / candidate
        return candidate.resolve()
    return default.resolve()


def _boolean(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    raise LaunchError(f"{name} must be a boolean")


def _positive_integer(value: Any, *, name: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise LaunchError(f"{name} must be an integer") from exc
    if integer < 1:
        raise LaunchError(f"{name} must be at least 1")
    return integer


def _effective_settings(
    args: argparse.Namespace,
    config_path: Path,
    config: Mapping[str, Any],
) -> LaunchSettings:
    paths = _section(config, "paths")
    proposal = _section(config, "proposal")
    runtime = _section(config, "runtime")
    codex = _section(config, "codex")
    resources = REPOSITORY_ROOT / "resources"

    cli_dry_run = args.dry_run
    dry_run = (
        _boolean(runtime.get("dry_run", False), name="runtime.dry_run")
        if cli_dry_run is None
        else bool(cli_dry_run)
    )
    reasoning_effort = (
        args.codex_reasoning_effort
        or codex.get("reasoning_effort")
        or os.environ.get("OMNIHARNESS_CODEX_REASONING_EFFORT")
        or None
    )
    modality = args.modality or proposal.get("modality") or None
    if modality is not None:
        modality = str(modality).upper()
        if modality not in {"T2I", "I2I"}:
            raise LaunchError("proposal.modality must be T2I, I2I, or null; video is evaluated downstream")

    # Share the runtime's retry semantics so CLI/config overrides cannot diverge.
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from omniharness_runtime import resolve_execution_budget

    try:
        budget = resolve_execution_budget(
            runtime,
            max_retries=args.max_retries,
            max_attempts=args.max_attempts,
            max_wall_time_seconds=args.max_wall_time,
        )
    except (TypeError, ValueError) as exc:
        raise LaunchError(str(exc)) from exc

    return LaunchSettings(
        command=args.command,
        config_path=config_path,
        project_root=_resolve_path(
            args.project_root, paths.get("project_root"), REPOSITORY_ROOT, config_path
        ),
        symbolic_policy=_resolve_path(
            args.symbolic_policy,
            paths.get("symbolic_policy"),
            REPOSITORY_ROOT / "runs" / "policy_library",
            config_path,
        ),
        source_image_pool=_resolve_path(
            args.source_image_pool,
            paths.get("source_image_pool"),
            resources / "source_image_pool_comfybench",
            config_path,
        ),
        output_root=_resolve_path(
            args.output_root,
            paths.get("output_root"),
            REPOSITORY_ROOT / "runs",
            config_path,
        ),
        comfyui_url=(args.comfyui_url or runtime.get("comfyui_url") or None),
        memory_mode=str(args.memory_mode or runtime.get("memory_mode", "online")),
        max_attempts=budget.max_attempts,
        max_wall_time=budget.max_wall_time_seconds,
        dry_run=dry_run,
        codex_model=str(
            args.codex_model
            or codex.get("model")
            or os.environ.get("OMNIHARNESS_CODEX_MODEL")
            or "gpt-4o"
        ).strip(),
        codex_provider=str(
            args.codex_provider
            or codex.get("provider")
            or os.environ.get("OMNIHARNESS_CODEX_PROVIDER")
            or "omniharness_gpt4o"
        ).strip(),
        codex_reasoning_effort=(
            None if reasoning_effort in (None, "") else str(reasoning_effort).strip()
        ),
        codex_base_url=(
            args.codex_base_url
            or codex.get("base_url")
            or os.environ.get("OMNIHARNESS_CODEX_BASE_URL")
            or None
        ),
        codex_api_key_env=str(
            args.codex_api_key_env
            or codex.get("api_key_env")
            or os.environ.get("OMNIHARNESS_CODEX_API_KEY_ENV")
            or "OMNIHARNESS_API_KEY"
        ).strip(),
        iterations=_positive_integer(
            args.iterations
            if args.iterations is not None
            else runtime.get("iterations", 50),
            name="runtime.iterations",
        ),
        candidate_count=_positive_integer(
            args.candidate_count
            if args.candidate_count is not None
            else proposal.get("candidate_count", 10),
            name="proposal.candidate_count",
        ),
        start_iteration=_positive_integer(
            args.start_iteration
            if args.start_iteration is not None
            else proposal.get("iteration", 1),
            name="proposal.iteration",
        ),
        modality=modality,
        task_contract=(
            None if args.task_contract is None else args.task_contract.expanduser().resolve()
        ),
        snapshot_destination=(
            None if args.destination is None else args.destination.expanduser().resolve()
        ),
    )


def _read_json_mapping(path: Path, problems: list[str]) -> Mapping[str, Any] | None:
    if not path.is_file():
        problems.append(f"required JSON file is missing: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        problems.append(f"invalid JSON file {path}: {exc}")
        return None
    if not isinstance(payload, Mapping):
        problems.append(f"JSON root must be an object: {path}")
        return None
    return payload


def _existing_writable_anchor(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _check_writable_target(path: Path, label: str, problems: list[str]) -> None:
    anchor = _existing_writable_anchor(path)
    if not anchor.exists():
        problems.append(f"{label} has no existing parent: {path}")
    elif not os.access(anchor, os.W_OK):
        problems.append(f"{label} is not writable: {anchor}")


def _validate_symbolic_policy(
    root: Path, command: str, problems: list[str], notes: list[str], *, memory_mode: str = "online"
) -> None:
    can_initialize = command in {"inquiry", "execute"} and memory_mode == "online"
    if can_initialize and (not root.exists() or (root.is_dir() and not any(root.iterdir()))):
        _check_writable_target(root, "new policy library", problems)
        notes.append("new policy library: empty state will be initialized at launch")
        return
    if not root.is_dir():
        problems.append(f"policy library directory is missing: {root}")
        return
    if (root / "snapshot_manifest.json").exists() and (command == "consolidate" or can_initialize):
        problems.append("a frozen snapshot requires --memory-mode frozen and cannot be consolidated")
    metadata = None
    for filename in ("workflow_metadata.json", "failure_metadata.json"):
        loaded = _read_json_mapping(root / filename, problems)
        if filename == "workflow_metadata.json":
            metadata = loaded
    workflow_files = []
    for key, entry in (metadata or {}).items():
        if key == "_metadata":
            continue
        if not isinstance(entry, Mapping):
            problems.append(f"invalid policy metadata entry: {key}")
            continue
        path = (root / str(entry.get("Workflow Source Code", f"{key}.json"))).resolve()
        if root.resolve() not in path.parents:
            problems.append(f"workflow template escapes its library: {key}")
            continue
        workflow_files.append(path)
    for path in workflow_files:
        _read_json_mapping(path, problems)
    if command == "consolidate" or can_initialize:
        _check_writable_target(root, "policy library", problems)
    notes.append(f"symbolic policies: {len(workflow_files)}")


def _validate_source_images(
    root: Path,
    verify_hashes: bool,
    problems: list[str],
    notes: list[str],
) -> None:
    if not root.is_dir():
        problems.append(f"source image pool is missing: {root}")
        return
    metadata = _read_json_mapping(root / "source_image_metadata.json", problems)
    if metadata is None:
        return
    valid_images = 0
    root_resolved = root.resolve()
    for image_id, raw in metadata.items():
        if not isinstance(raw, Mapping) or not raw.get("path"):
            problems.append(f"source image metadata is incomplete: {image_id}")
            continue
        image_path = (root / str(raw["path"])).resolve()
        try:
            image_path.relative_to(root_resolved)
        except ValueError:
            problems.append(f"source image escapes its resource directory: {image_path}")
            continue
        if not image_path.is_file():
            problems.append(f"source image is missing: {image_path}")
            continue
        expected_size = raw.get("size_bytes")
        if expected_size is not None and image_path.stat().st_size != int(expected_size):
            problems.append(f"source image size mismatch: {image_path}")
            continue
        if verify_hashes and raw.get("sha256"):
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if digest.lower() != str(raw["sha256"]).lower():
                problems.append(f"source image SHA-256 mismatch: {image_path}")
                continue
        valid_images += 1
    notes.append(f"source images: {valid_images}/{len(metadata)}")


def _validate_api_url(value: str | None, problems: list[str]) -> None:
    if value is None or not str(value).strip():
        problems.append("codex.base_url is required for the third-party GPT-4o provider")
        return
    normalized = str(value).strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        problems.append("codex.base_url must be an absolute HTTP(S) API root")
    elif parsed.scheme == "http" and parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        problems.append("remote GPT-4o providers must use HTTPS")
    if parsed.path.endswith("/responses"):
        problems.append("codex.base_url must be the API root, not /responses")
    if "your_provider" in normalized.lower() or str(parsed.hostname).endswith(".example"):
        problems.append("replace the placeholder codex.base_url in config.yaml")


def _probe_comfyui(url: str, timeout: float) -> int:
    endpoint = url.rstrip("/") + "/object_info"
    request = Request(endpoint, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, socket.timeout, UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchError(f"ComfyUI preflight failed at {endpoint}: {exc}") from exc
    if not isinstance(payload, Mapping) or not payload:
        raise LaunchError(f"ComfyUI returned no node catalog at {endpoint}")
    return len(payload)


def _preflight(
    settings: LaunchSettings,
    *,
    skip_comfyui_check: bool,
    timeout: float,
    verify_hashes: bool,
) -> list[str]:
    problems: list[str] = []
    notes: list[str] = []

    if sys.version_info[:2] != (3, 10):
        problems.append(
            "the frozen OmniHarness environment requires CPython 3.10.x; "
            f"found {sys.version_info.major}.{sys.version_info.minor}"
        )
    for path in (PROPOSAL_PATH, RUNTIME_PATH):
        if not path.is_file():
            problems.append(f"OmniHarness source file is missing: {path}")
    _validate_symbolic_policy(
        settings.symbolic_policy, settings.command, problems, notes, memory_mode=settings.memory_mode
    )

    if settings.command in {"inquiry", "execute"}:
        if importlib.util.find_spec("openai_codex") is None:
            problems.append(
                "openai-codex is unavailable; install requirements.txt in this Python environment"
            )
        if not settings.project_root.is_dir():
            problems.append(f"Codex project root is missing: {settings.project_root}")
        if settings.command != "inquiry" or settings.modality != "T2I":
            _validate_source_images(
                settings.source_image_pool, verify_hashes, problems, notes
            )
        _check_writable_target(settings.output_root, "runtime output root", problems)
        if settings.memory_mode not in {"online", "frozen"}:
            problems.append("runtime.memory_mode must be online or frozen")
        if not settings.codex_model:
            problems.append("codex.model cannot be empty")
        elif "gpt-4o" not in settings.codex_model.lower():
            notes.append("backbone override: record model/provider settings with the evaluation")
        if not settings.codex_provider:
            problems.append("codex.provider cannot be empty")
        _validate_api_url(settings.codex_base_url, problems)
        if not ENVIRONMENT_NAME.fullmatch(settings.codex_api_key_env):
            problems.append(
                "codex.api_key_env must be a valid environment-variable name"
            )
        elif not os.environ.get(settings.codex_api_key_env, "").strip():
            problems.append(
                f"API key environment variable is unset: {settings.codex_api_key_env}"
            )
        if not settings.comfyui_url:
            problems.append("runtime.comfyui_url is required")
        else:
            parsed = urlsplit(str(settings.comfyui_url))
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                problems.append("runtime.comfyui_url must be an absolute HTTP(S) URL")
            elif not skip_comfyui_check and not problems:
                try:
                    nodes = _probe_comfyui(str(settings.comfyui_url), timeout)
                    notes.append(f"ComfyUI node classes: {nodes}")
                except LaunchError as exc:
                    problems.append(str(exc))

    if settings.command == "execute":
        if settings.task_contract is None:
            problems.append("execute requires --task-contract")
        else:
            contract = _read_json_mapping(settings.task_contract, problems)
            if contract is not None:
                required = {
                    "task_id",
                    "description",
                    "modality",
                    "capability_categories",
                    "selection_evidence",
                }
                missing = sorted(required - set(contract))
                if missing:
                    problems.append(
                        "task contract is missing fields: " + ", ".join(missing)
                    )

    if settings.command == "snapshot":
        if settings.snapshot_destination is None:
            problems.append("snapshot requires --destination")
        elif settings.snapshot_destination.exists():
            problems.append(
                f"snapshot destination already exists: {settings.snapshot_destination}"
            )
        else:
            _check_writable_target(
                settings.snapshot_destination, "snapshot destination", problems
            )

    if problems:
        formatted = "\n".join(f"  - {problem}" for problem in problems)
        raise LaunchError(f"OmniHarness preflight failed:\n{formatted}")
    return notes


def _append_option(arguments: list[str], flag: str, value: Any) -> None:
    if value not in (None, ""):
        arguments.extend((flag, str(value)))


def _runtime_arguments(settings: LaunchSettings) -> list[str]:
    arguments = ["--config", str(settings.config_path), settings.command]
    _append_option(
        arguments, "--symbolic-policy", settings.symbolic_policy
    )
    if settings.command == "snapshot":
        _append_option(arguments, "--destination", settings.snapshot_destination)
        return arguments
    if settings.command == "consolidate":
        return arguments

    _append_option(arguments, "--comfyui-url", settings.comfyui_url)
    _append_option(arguments, "--source-image-pool", settings.source_image_pool)
    _append_option(arguments, "--output-root", settings.output_root)
    _append_option(arguments, "--project-root", settings.project_root)
    _append_option(arguments, "--memory-mode", settings.memory_mode)
    _append_option(arguments, "--max-attempts", settings.max_attempts)
    _append_option(arguments, "--max-wall-time", settings.max_wall_time)
    _append_option(arguments, "--codex-model", settings.codex_model)
    _append_option(arguments, "--codex-provider", settings.codex_provider)
    _append_option(
        arguments,
        "--codex-reasoning-effort",
        settings.codex_reasoning_effort,
    )
    _append_option(arguments, "--codex-base-url", settings.codex_base_url)
    _append_option(
        arguments, "--codex-api-key-env", settings.codex_api_key_env
    )
    if settings.dry_run:
        arguments.append("--dry-run")
    if settings.command == "execute":
        _append_option(arguments, "--task-contract", settings.task_contract)
        return arguments

    _append_option(arguments, "--iterations", settings.iterations)
    _append_option(arguments, "--candidate-count", settings.candidate_count)
    _append_option(arguments, "--start-iteration", settings.start_iteration)
    _append_option(arguments, "--modality", settings.modality)
    return arguments


def _print_launch_summary(settings: LaunchSettings, notes: Sequence[str]) -> None:
    print("[OmniHarness] preflight passed", flush=True)
    print(f"  command: {settings.command}", flush=True)
    print(f"  config: {settings.config_path}", flush=True)
    print(f"  symbolic policy library: {settings.symbolic_policy}", flush=True)
    if settings.command in {"inquiry", "execute"}:
        print(f"  model: {settings.codex_model}", flush=True)
        print(f"  provider: {settings.codex_provider}", flush=True)
        print(f"  API key variable: {settings.codex_api_key_env}", flush=True)
        print(f"  ComfyUI: {settings.comfyui_url}", flush=True)
        print(f"  output root: {settings.output_root}", flush=True)
        print(f"  execution budget: {settings.max_attempts - 1} retries, {settings.max_attempts} total attempts", flush=True)
    for note in notes:
        print(f"  {note}", flush=True)


def _invoke_runtime(arguments: Sequence[str]) -> int:
    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        from omniharness_runtime import main as runtime_main
    except ImportError as exc:
        raise LaunchError(f"cannot import the OmniHarness runtime: {exc}") from exc
    result = runtime_main(list(arguments))
    return int(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.preflight_timeout <= 0:
        parser.error("--preflight-timeout must be greater than zero")
    config_path = args.config.expanduser().resolve()
    try:
        config = _load_yaml(config_path)
        settings = _effective_settings(args, config_path, config)
        notes = _preflight(
            settings,
            skip_comfyui_check=args.skip_comfyui_check,
            timeout=float(args.preflight_timeout),
            verify_hashes=args.verify_resource_hashes,
        )
        _print_launch_summary(settings, notes)
        if args.preflight_only:
            return 0
        return _invoke_runtime(_runtime_arguments(settings))
    except KeyboardInterrupt:
        print("\n[OmniHarness] interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.debug:
            raise
        print(f"[OmniHarness] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
