"""Run ReasonEdit relation-aware image editing tasks through OmniHarness."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from benchmark_runner import (
    DEFAULT_BENCHMARK_ROOT,
    BenchmarkError,
    BenchmarkLoadResult,
    BenchmarkTask,
    infer_capabilities,
    public_success_criterion,
    run_benchmark,
)


CLIP_SUFFIX = re.compile(r"\s+CLIP:\s*(.*)$", re.IGNORECASE)


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--categories",
        nargs="+",
        help="ReasonEdit category directory names; defaults to all categories",
    )


def _load(data_root: Path, args: argparse.Namespace) -> BenchmarkLoadResult:
    available = {path.name: path for path in data_root.iterdir() if path.is_dir()}
    requested = set(args.categories or available)
    missing = sorted(requested - set(available))
    if missing:
        raise BenchmarkError("unknown ReasonEdit categories: " + ", ".join(missing))
    tasks: list[BenchmarkTask] = []
    for category in sorted(requested):
        category_root = available[category]
        text_files = sorted(category_root.glob("*.txt"))
        if len(text_files) != 1:
            raise BenchmarkError(
                f"ReasonEdit category must contain one instruction file: {category}"
            )
        lines = [
            line.strip()
            for line in text_files[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for index, line in enumerate(lines, start=1):
            match = CLIP_SUFFIX.search(line)
            clip_target = None if match is None else match.group(1).strip()
            instruction = line if match is None else line[: match.start()].strip()
            original = (category_root / f"{index:03d}.png").resolve()
            mask = (category_root / f"{index:03d}_mask.jpg").resolve()
            source_paths = (original,)
            preservation = (
                {
                    "kind": "preserve_unrequested_content",
                    "target": "source_media_0",
                    "value": True,
                },
            )
            tasks.append(
                BenchmarkTask(
                    benchmark="reasonedit",
                    sample_id=f"{category}-{index:03d}",
                    instruction=instruction,
                    modality="I2I",
                    capability_categories=infer_capabilities(instruction, "I2I"),
                    source_image_paths=source_paths,
                    preservation_constraints=preservation,
                    success_criteria=public_success_criterion(instruction),
                    metadata={
                        "category": category,
                        "index": index,
                        "clip_target": clip_target,
                        "evaluation_mask_file": str(mask) if mask.is_file() else None,
                    },
                )
            )
    return BenchmarkLoadResult(tuple(tasks))


def main() -> int:
    return run_benchmark(
        "ReasonEdit",
        DEFAULT_BENCHMARK_ROOT / "ReasonEdit",
        _load,
        configure_parser=_configure,
    )


if __name__ == "__main__":
    raise SystemExit(main())
