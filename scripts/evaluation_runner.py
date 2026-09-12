"""Shared infrastructure for official-protocol OmniHarness evaluations."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import mimetypes
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "config.yaml"
DEFAULT_BENCHMARK_ROOT = REPOSITORY_ROOT / "bench data"
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "results"
MEDIA_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".mp4",
    ".mov",
    ".webm",
    ".avi",
}
SAFE_NAME = re.compile(r"[^a-z0-9]+")


class EvaluationError(RuntimeError):
    """Raised for invalid evaluation inputs or protocol failures."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read JSON {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise EvaluationError(
                        f"JSONL row must be an object: {path}:{line_number}"
                    )
                rows.append(dict(row))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def atomic_write_json(path: Path, payload: Any) -> None:
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


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_token(value: str) -> str:
    return SAFE_NAME.sub("", value.lower())


def natural_key(value: str) -> tuple[tuple[int, Any], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", value)
        if part
    )


class ResultResolver:
    """Resolve flat or category-nested benchmark media without guessing silently."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        if self.root.exists() and not self.root.is_dir():
            raise EvaluationError(f"results path is not a directory: {self.root}")
        if not self.root.exists():
            self.files = ()
            return
        self.files = tuple(
            sorted(
                (
                    path.resolve()
                    for path in self.root.rglob("*")
                    if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
                ),
                key=lambda path: natural_key(str(path.relative_to(self.root))),
            )
        )

    def find(
        self,
        keys: Sequence[str],
        *,
        extensions: Sequence[str],
        category_hints: Sequence[str] = (),
    ) -> Path | None:
        allowed = {extension.lower() for extension in extensions}
        key_tokens = {canonical_token(key) for key in keys if key}
        candidates = [
            path
            for path in self.files
            if path.suffix.lower() in allowed
            and canonical_token(path.stem) in key_tokens
        ]
        if not candidates:
            return None
        hint_tokens = {canonical_token(hint) for hint in category_hints if hint}
        if hint_tokens:
            hinted = [
                path
                for path in candidates
                if any(
                    token in {canonical_token(part) for part in path.parts[:-1]}
                    for token in hint_tokens
                )
            ]
            if hinted:
                candidates = hinted
        direct = [path for path in candidates if path.parent == self.root]
        if len(direct) == 1:
            return direct[0]
        if len(candidates) == 1:
            return candidates[0]
        relative = ", ".join(str(path.relative_to(self.root)) for path in candidates[:8])
        raise EvaluationError(
            f"ambiguous result for keys {list(keys)!r}: {relative}"
        )


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    data_root: Path,
    results_dir: Path,
    output_dir: Path,
) -> None:
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--results-dir", type=Path, default=results_dir)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--strict-completeness", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="recompute scores from scratch; existing unbound evaluation records cannot be resumed")


def add_judge_arguments(parser: argparse.ArgumentParser, *, default_workers: int = 4) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument(
        "--wire-api", choices=("responses", "chat-completions"), default=None
    )
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-retries", type=int, default=3)


def select_items(
    items: Sequence[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    selected = list(items)
    if args.sample_id:
        requested = set(args.sample_id)
        selected = [item for item in selected if str(item["sample_id"]) in requested]
        found = {str(item["sample_id"]) for item in selected}
        missing = sorted(requested - found, key=natural_key)
        if missing:
            raise EvaluationError("unknown sample IDs: " + ", ".join(missing))
    if args.start < 0:
        raise EvaluationError("--start cannot be negative")
    selected = selected[args.start :]
    if args.limit is not None:
        if args.limit < 1:
            raise EvaluationError("--limit must be positive")
        selected = selected[: args.limit]
    return selected


def coverage_report(
    benchmark: str,
    items: Sequence[Mapping[str, Any]],
    *,
    results_dir: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    available = sum(item.get("result_path") is not None for item in items)
    report: dict[str, Any] = {
        "benchmark": benchmark,
        "results_dir": str(results_dir.resolve()),
        "selected": len(items),
        "results_found": available,
        "results_missing": len(items) - available,
        "coverage": available / len(items) if items else 0.0,
    }
    if extra:
        report.update(extra)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def load_evaluation_records(path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Start a new evaluation without reusing scores with unverified provenance."""
    if path.exists() and not overwrite:
        raise EvaluationError(
            f"existing evaluation records are not bound to media and protocol fingerprints: {path}. "
            "Use a new --output-dir or --overwrite to recompute all selected scores."
        )
    return {}


class JsonEvaluationStore:
    def __init__(self, path: Path, *, overwrite: bool = False) -> None:
        self.path = path
        self.records = load_evaluation_records(path, overwrite=overwrite)

    def pending(
        self, items: Sequence[dict[str, Any]], *, overwrite: bool
    ) -> list[dict[str, Any]]:
        return list(items)

    def update(self, sample_id: str, record: Mapping[str, Any]) -> None:
        self.records[str(sample_id)] = dict(record)
        ordered = {
            key: self.records[key]
            for key in sorted(self.records, key=natural_key)
        }
        atomic_write_json(self.path, ordered)


def run_threaded(
    items: Sequence[dict[str, Any]],
    *,
    workers: int,
    function: Callable[[dict[str, Any]], Mapping[str, Any]],
    on_result: Callable[[str, Mapping[str, Any]], None],
) -> None:
    if workers < 1:
        raise EvaluationError("--workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, item): item for item in items}
        for position, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            sample_id = str(item["sample_id"])
            try:
                record = dict(future.result())
            except Exception as exc:  # retain completed records from the current run
                record = {
                    "status": "error",
                    "sample_id": sample_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            on_result(sample_id, record)
            print(f"[{position}/{len(items)}] {sample_id}: {record.get('status')}")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvaluationError(
            f"configuration file is missing: {path}; create it from config.example.yaml"
        )
    try:
        import yaml
    except ImportError as exc:
        raise EvaluationError("PyYAML is required to read the evaluation config") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise EvaluationError(f"configuration root must be a mapping: {path}")
    return dict(payload)


@dataclass(frozen=True)
class JudgeSettings:
    model: str
    base_url: str
    api_key: str
    wire_api: str
    timeout: float
    max_retries: int


def load_judge_settings(args: argparse.Namespace) -> JudgeSettings:
    config = _load_yaml(args.config.expanduser().resolve())
    codex = config.get("codex", {})
    evaluation = config.get("evaluation", {})
    if not isinstance(codex, Mapping) or not isinstance(evaluation, Mapping):
        raise EvaluationError("codex and evaluation config sections must be mappings")
    model = str(args.model or evaluation.get("model") or codex.get("model") or "gpt-4o")
    base_url = str(
        args.base_url or evaluation.get("base_url") or codex.get("base_url") or ""
    ).strip().rstrip("/")
    key_env = str(
        args.api_key_env
        or evaluation.get("api_key_env")
        or codex.get("api_key_env")
        or "OMNIHARNESS_API_KEY"
    ).strip()
    wire_api = str(
        args.wire_api or evaluation.get("wire_api") or "responses"
    ).strip()
    timeout = float(args.timeout or evaluation.get("timeout_seconds") or 300)
    if not base_url:
        raise EvaluationError("evaluation API base URL is empty")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise EvaluationError("evaluation API base URL must be absolute HTTP(S)")
    if parsed.scheme == "http" and parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise EvaluationError("remote evaluation APIs must use HTTPS")
    if wire_api not in {"responses", "chat-completions"}:
        raise EvaluationError(f"unsupported wire API: {wire_api}")
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        raise EvaluationError(f"API key environment variable is unset: {key_env}")
    if timeout <= 0 or args.max_retries < 1:
        raise EvaluationError("timeout and max retries must be positive")
    return JudgeSettings(model, base_url, api_key, wire_api, timeout, args.max_retries)


@dataclass(frozen=True)
class JudgeResponse:
    text: str
    usage: Mapping[str, Any]


class VisionJudge:
    def __init__(self, settings: JudgeSettings) -> None:
        self.settings = settings

    def evaluate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_urls: Sequence[str],
        max_tokens: int = 1200,
    ) -> JudgeResponse:
        payload = self._payload(system_prompt, user_prompt, image_urls, max_tokens)
        endpoint = self.settings.base_url + (
            "/responses"
            if self.settings.wire_api == "responses"
            else "/chat/completions"
        )
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                with urlopen(request, timeout=self.settings.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return JudgeResponse(self._extract_text(body), body.get("usage", {}))
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = exc
                if attempt < self.settings.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise EvaluationError(f"judge request failed: {last_error}")

    def _payload(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: Sequence[str],
        max_tokens: int,
    ) -> dict[str, Any]:
        if self.settings.wire_api == "responses":
            user_content: list[dict[str, Any]] = [
                {"type": "input_text", "text": user_prompt}
            ]
            user_content.extend(
                {"type": "input_image", "image_url": url} for url in image_urls
            )
            input_messages: list[dict[str, Any]] = []
            if system_prompt:
                input_messages.append(
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    }
                )
            input_messages.append({"role": "user", "content": user_content})
            return {
                "model": self.settings.model,
                "input": input_messages,
                "temperature": 0,
                "max_output_tokens": max_tokens,
            }
        user_content = [{"type": "text", "text": user_prompt}]
        user_content.extend(
            {"type": "image_url", "image_url": {"url": url}}
            for url in image_urls
        )
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        return {
            "model": self.settings.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }

    @staticmethod
    def _extract_text(body: Mapping[str, Any]) -> str:
        direct = body.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
        output = body.get("output")
        if isinstance(output, list):
            texts: list[str] = []
            for item in output:
                for content in item.get("content", []):
                    text = content.get("text")
                    if isinstance(text, str):
                        texts.append(text)
            if texts:
                return "\n".join(texts).strip()
        raise EvaluationError("judge response contains no text output")


def pil_image_to_data_url(image: Any, *, max_size: tuple[int, int] = (512, 512)) -> str:
    image = image.convert("RGB")
    image.thumbnail(max_size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def image_to_data_url(path: Path, *, max_size: tuple[int, int] = (512, 512)) -> str:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise EvaluationError("Pillow is required for visual evaluation") from exc
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).copy()
    return pil_image_to_data_url(image, max_size=max_size)


def file_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def sample_video(
    path: Path,
    *,
    frame_limit: int = 5,
    max_size: tuple[int, int] = (512, 512),
) -> tuple[list[str], dict[str, Any]]:
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:
        raise EvaluationError("opencv-python-headless and Pillow are required for video evaluation") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise EvaluationError(f"cannot open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_rate <= 0:
        capture.release()
        raise EvaluationError(f"invalid frame rate for video: {path}")
    interval = max(6, frame_count // frame_limit)
    frames: list[str] = []
    index = 0
    while capture.isOpened():
        status, frame = capture.read()
        if not status:
            break
        if index % interval == 0 and len(frames) < frame_limit:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(pil_image_to_data_url(Image.fromarray(rgb), max_size=max_size))
        index += 1
    capture.release()
    if not frames:
        raise EvaluationError(f"video contains no readable frames: {path}")
    return frames, {
        "width": width,
        "height": height,
        "num_frames": frame_count,
        "frame_rate": frame_rate,
        "duration": frame_count / frame_rate,
        "sampled_frames": len(frames),
    }


def mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def exit_code_for_coverage(report: Mapping[str, Any], *, strict: bool) -> int:
    return 2 if strict and int(report.get("results_missing", 0)) else 0
