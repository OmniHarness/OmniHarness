"""Run WISE world-knowledge text-to-image prompts through OmniHarness."""

from __future__ import annotations

import argparse
from pathlib import Path

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


DOMAINS = (
    "cultural_common_sense",
    "natural_science",
    "spatio-temporal_reasoning",
)
VARIANTS = ("original", "rewrite", "both")


def _configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=None)
    parser.add_argument("--variant", choices=VARIANTS, default="original")


def _load(data_root: Path, args: argparse.Namespace) -> BenchmarkLoadResult:
    domains = args.domains or DOMAINS
    variants = ("original", "rewrite") if args.variant == "both" else (args.variant,)
    tasks: list[BenchmarkTask] = []
    for domain in domains:
        for variant in variants:
            suffix = "" if variant == "original" else "_rewrite"
            path = data_root / f"{domain}{suffix}.json"
            payload = read_json(path)
            if not isinstance(payload, list):
                raise BenchmarkError(f"WISE file must contain a list: {path}")
            for index, raw in enumerate(payload):
                if not isinstance(raw, dict):
                    raise BenchmarkError(f"invalid WISE sample: {path}:{index}")
                instruction = str(raw.get("Prompt", "")).strip()
                prompt_id = str(raw.get("prompt_id", index))
                tasks.append(
                    BenchmarkTask(
                        benchmark="wise",
                        sample_id=f"{domain}-{variant}-{prompt_id}",
                        instruction=instruction,
                        modality="T2I",
                        capability_categories=infer_capabilities(instruction, "T2I"),
                        success_criteria=public_success_criterion(instruction),
                        metadata={
                            "domain": domain,
                            "variant": variant,
                            "row_index": index,
                            "evaluation": dict(raw),
                        },
                    )
                )
    return BenchmarkLoadResult(tuple(tasks))


def main() -> int:
    return run_benchmark(
        "WISE",
        DEFAULT_BENCHMARK_ROOT / "WISE",
        _load,
        configure_parser=_configure,
    )


if __name__ == "__main__":
    raise SystemExit(main())
