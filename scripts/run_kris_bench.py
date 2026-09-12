"""Run KRIS_Bench knowledge-intensive image tasks through OmniHarness."""

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
    natural_key,
    public_success_criterion,
    read_json,
    run_benchmark,
)


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--categories",
        nargs="+",
        help="KRIS_Bench category directory names; defaults to all categories",
    )


def _source_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise BenchmarkError(f"invalid KRIS_Bench ori_img value: {value!r}")


def _load(data_root: Path, args: argparse.Namespace) -> BenchmarkLoadResult:
    available = {
        path.name: path
        for path in data_root.iterdir()
        if path.is_dir() and (path / "annotation.json").is_file()
    }
    requested = set(args.categories or available)
    missing = sorted(requested - set(available))
    if missing:
        raise BenchmarkError("unknown KRIS_Bench categories: " + ", ".join(missing))
    tasks: list[BenchmarkTask] = []
    for category in sorted(requested):
        category_root = available[category]
        annotations = read_json(category_root / "annotation.json")
        if not isinstance(annotations, Mapping):
            raise BenchmarkError(f"annotation root must be an object: {category}")
        for sample_id, raw in sorted(
            annotations.items(), key=lambda item: natural_key(str(item[0]))
        ):
            if not isinstance(raw, Mapping):
                raise BenchmarkError(f"invalid KRIS_Bench sample: {category}/{sample_id}")
            instruction = str(raw.get("ins_en", "")).strip()
            source_names = _source_names(raw.get("ori_img"))
            source_paths = tuple((category_root / name).resolve() for name in source_names)
            constraints: tuple[Mapping[str, Any], ...] = ()
            if len(source_paths) > 1:
                constraints = (
                    {
                        "kind": "ordered_reference_media",
                        "target": "source_media",
                        "value": list(range(len(source_paths))),
                    },
                )
            tasks.append(
                BenchmarkTask(
                    benchmark="kris-bench",
                    sample_id=f"{category}-{sample_id}",
                    instruction=instruction,
                    modality="I2I",
                    capability_categories=infer_capabilities(instruction, "I2I"),
                    source_image_paths=source_paths,
                    generation_constraints=constraints,
                    preservation_constraints=(
                        {
                            "kind": "preserve_unrequested_content",
                            "target": "source_media_0",
                            "value": True,
                        },
                    ),
                    success_criteria=public_success_criterion(instruction),
                    metadata={
                        "category": category,
                        "annotation_id": str(sample_id),
                        "source_files": list(source_names),
                        "ground_truth_file": raw.get("gt_img"),
                        "explanation": raw.get("explain_en"),
                    },
                )
            )
    return BenchmarkLoadResult(tuple(tasks))


def main() -> int:
    return run_benchmark(
        "KRIS_Bench",
        DEFAULT_BENCHMARK_ROOT / "KRIS_Bench",
        _load,
        configure_parser=_configure,
    )


if __name__ == "__main__":
    raise SystemExit(main())
