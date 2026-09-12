"""Run every ComfyBench image and video task through OmniHarness."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from benchmark_runner import (
    DEFAULT_BENCHMARK_ROOT,
    BenchmarkError,
    BenchmarkLoadResult,
    BenchmarkTask,
    infer_capabilities,
    public_success_criterion,
    read_json,
    run_benchmark,
)


MODALITIES = {name: name.upper() for name in ("t2i", "i2i", "t2v", "i2v", "v2v")}
CATEGORIES = ("vanilla", "complex", "creative")


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--modalities", nargs="+", choices=tuple(MODALITIES), default=None)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=None)


def _load(data_root: Path, args: argparse.Namespace) -> BenchmarkLoadResult:
    payload = read_json(data_root / "complete.json")
    if not isinstance(payload, Mapping):
        raise BenchmarkError("ComfyBench complete.json must contain an object")
    requested_modalities = set(args.modalities or MODALITIES)
    requested_categories = set(args.categories or CATEGORIES)
    tasks: list[BenchmarkTask] = []
    for sample_id, raw in sorted(payload.items(), key=lambda item: int(item[0])):
        if not isinstance(raw, Mapping):
            raise BenchmarkError(f"invalid ComfyBench sample: {sample_id}")
        raw_modality = str(raw.get("modality", "")).lower()
        category = str(raw.get("category", ""))
        if raw_modality not in MODALITIES:
            raise BenchmarkError(
                f"unknown ComfyBench modality {raw_modality!r}: {sample_id}"
            )
        if raw_modality not in requested_modalities or category not in requested_categories:
            continue
        modality = MODALITIES[raw_modality]
        instruction = str(raw.get("instruction", "")).strip()
        resource = raw.get("resource")
        source_paths: tuple[Path, ...] = ()
        if resource not in (None, "", "None"):
            source_paths = ((data_root / "resource" / str(resource)).resolve(),)
        constraints: tuple[Mapping[str, Any], ...] = ()
        preservation: tuple[Mapping[str, Any], ...] = ()
        if source_paths:
            preservation = (
                {
                    "kind": "preserve_unrequested_content",
                    "target": "source_media_0",
                    "value": True,
                },
            )
        tasks.append(
            BenchmarkTask(
                benchmark="comfybench",
                sample_id=str(sample_id),
                instruction=instruction,
                modality=modality,
                capability_categories=infer_capabilities(instruction, modality),
                source_image_paths=source_paths,
                generation_constraints=constraints,
                preservation_constraints=preservation,
                success_criteria=public_success_criterion(instruction),
                metadata={"name": raw.get("name"), "category": category, "raw": dict(raw)},
            )
        )
    return BenchmarkLoadResult(tuple(tasks))


def main() -> int:
    return run_benchmark(
        "ComfyBench",
        DEFAULT_BENCHMARK_ROOT / "ComfyBench",
        _load,
        configure_parser=_configure,
    )


if __name__ == "__main__":
    raise SystemExit(main())
