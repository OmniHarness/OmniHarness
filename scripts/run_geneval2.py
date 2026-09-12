"""Run GenEval2 compositional text-to-image prompts through OmniHarness."""

from __future__ import annotations

from pathlib import Path

from benchmark_runner import (
    DEFAULT_BENCHMARK_ROOT,
    BenchmarkLoadResult,
    BenchmarkTask,
    infer_capabilities,
    public_success_criterion,
    read_jsonl,
    run_benchmark,
)


def _load(data_root: Path, _args: object) -> BenchmarkLoadResult:
    rows = read_jsonl(data_root / "geneval2_data.jsonl")
    tasks = []
    for index, row in enumerate(rows):
        instruction = str(row.get("prompt", "")).strip()
        tasks.append(
            BenchmarkTask(
                benchmark="geneval2",
                sample_id=f"{index:04d}",
                instruction=instruction,
                modality="T2I",
                capability_categories=infer_capabilities(instruction, "T2I"),
                success_criteria=public_success_criterion(instruction),
                metadata={"row_index": index, "evaluation": dict(row)},
            )
        )
    return BenchmarkLoadResult(tuple(tasks))


def main() -> int:
    return run_benchmark(
        "GenEval2",
        DEFAULT_BENCHMARK_ROOT / "GenEval2",
        _load,
    )


if __name__ == "__main__":
    raise SystemExit(main())
