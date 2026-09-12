"""Evaluate KRIS-Bench with its official GPT-4o category-specific rubrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation_runner import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_RESULTS_ROOT,
    EvaluationError,
    JsonEvaluationStore,
    ResultResolver,
    VisionJudge,
    add_common_arguments,
    add_judge_arguments,
    atomic_write_json,
    coverage_report,
    exit_code_for_coverage,
    image_to_data_url,
    load_judge_settings,
    mean,
    natural_key,
    read_json,
    run_threaded,
    select_items,
)
from kris_official_prompts import (
    prompt_consist,
    prompt_consist_multi,
    prompt_consist_temporal,
    prompt_dual_evaluation,
    prompt_instruction_following,
    prompt_instruction_multi,
    prompt_instruction_temporal,
    prompt_quality,
    prompt_view_instruction_following,
)


OFFICIAL_IMPLEMENTATION = "https://github.com/mercurystraw/Kris_Bench"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
COMMON_CATEGORIES = (
    "count_change",
    "color_change",
    "anomaly_correction",
    "position_movement",
    "size_adjustment",
    "part_completion",
    "multi-instruction_execution",
)
KNOWLEDGE_CATEGORIES = (
    "abstract_reasoning",
    "mathematics",
    "practical_knowledge",
    "medicine",
    "rule-based_reasoning",
    "biology",
    "geography",
    "chemistry",
    "humanities",
    "physics",
)
MULTI_CATEGORY = "multi-element_composition"
TEMPORAL_CATEGORY = "temporal_prediction"
VIEW_CATEGORY = "viewpoint_change"
ALL_CATEGORIES = (
    COMMON_CATEGORIES
    + KNOWLEDGE_CATEGORIES
    + (MULTI_CATEGORY, TEMPORAL_CATEGORY, VIEW_CATEGORY)
)
ALL_METRICS = (
    "consistency",
    "instruction_following",
    "knowledge_plausibility",
    "image_quality",
)
DEFAULT_PATTERNS = (
    r"([1-5])\s*/\s*5",
    r"([1-5])\s+out\s+of\s+5",
    r"\b([1-5])\b",
)


def _source_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise EvaluationError(f"invalid KRIS_Bench ori_img annotation: {value!r}")


def _load_items(data_root: Path, resolver: ResultResolver) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    available = {
        path.name: path
        for path in data_root.iterdir()
        if path.is_dir() and (path / "annotation.json").is_file()
    }
    missing_categories = sorted(set(ALL_CATEGORIES) - set(available))
    if missing_categories:
        raise EvaluationError(
            "missing KRIS_Bench categories: " + ", ".join(missing_categories)
        )
    for category in ALL_CATEGORIES:
        category_root = available[category]
        annotations = read_json(category_root / "annotation.json")
        if not isinstance(annotations, Mapping):
            raise EvaluationError(f"invalid KRIS annotation root: {category}")
        for image_id, raw in sorted(
            annotations.items(), key=lambda item: natural_key(str(item[0]))
        ):
            if not isinstance(raw, Mapping):
                raise EvaluationError(f"invalid KRIS annotation: {category}/{image_id}")
            source_paths = [
                (category_root / name).resolve()
                for name in _source_names(raw.get("ori_img"))
            ]
            for source in source_paths:
                if not source.is_file():
                    raise EvaluationError(f"missing KRIS source image: {source}")
            ground_truth = None
            if raw.get("gt_img"):
                ground_truth = (category_root / str(raw["gt_img"])).resolve()
                if not ground_truth.is_file():
                    raise EvaluationError(f"missing KRIS ground truth: {ground_truth}")
            sample_id = f"{category}-{image_id}"
            result = resolver.find(
                (
                    sample_id,
                    f"kris-bench_{sample_id}",
                    f"kris_bench_{sample_id}",
                    str(image_id),
                ),
                extensions=IMAGE_EXTENSIONS,
                category_hints=(category,),
            )
            items.append(
                {
                    "sample_id": sample_id,
                    "category": category,
                    "image_id": str(image_id),
                    "instruction": str(raw.get("ins_en", "")).strip(),
                    "explanation": str(raw.get("explain_en", "")).strip(),
                    "source_paths": source_paths,
                    "ground_truth_path": ground_truth,
                    "result_path": result,
                }
            )
    return items


def _extract_json_object(text: str) -> Mapping[str, Any] | None:
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, unused_end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return None


def _extract_score(
    response: str,
    score_key: str,
    reason_keys: Sequence[str],
    prefix: str,
) -> tuple[int, str | None]:
    payload = _extract_json_object(response)
    if payload and payload.get(score_key) is not None:
        score = int(payload[score_key])
        reason = next(
            (str(payload[key]) for key in reason_keys if payload.get(key) is not None),
            None,
        )
        if 1 <= score <= 5:
            return score, reason
    for pattern in (prefix,) + DEFAULT_PATTERNS:
        match = re.search(pattern, response, re.I | re.S)
        if match:
            return int(match.group(1)), None
    raise EvaluationError(f"cannot extract {score_key} from KRIS judge response")


def _judge_metric(
    judge: VisionJudge,
    *,
    prompt: str,
    image_paths: Sequence[Path],
    score_key: str,
    reason_keys: Sequence[str],
    prefix: str,
) -> tuple[int, str | None, str, Mapping[str, Any]]:
    response = judge.evaluate(
        system_prompt="",
        user_prompt=prompt,
        image_urls=[image_to_data_url(path) for path in image_paths],
        max_tokens=1000,
    )
    score, reason = _extract_score(response.text, score_key, reason_keys, prefix)
    return score, reason, response.text, response.usage


def _metrics_for(category: str, requested: Sequence[str] | None) -> tuple[str, ...]:
    available = (
        ("consistency", "instruction_following", "knowledge_plausibility", "image_quality")
        if category in KNOWLEDGE_CATEGORIES
        else ("consistency", "instruction_following", "image_quality")
    )
    if requested is None:
        return available
    invalid = sorted(set(requested) - set(available))
    if invalid:
        raise EvaluationError(
            f"metrics {invalid} are not defined by the official protocol for {category}"
        )
    return tuple(metric for metric in available if metric in requested)


def _evaluate_item(
    item: dict[str, Any],
    *,
    judge: VisionJudge,
    requested_metrics: Sequence[str] | None,
) -> dict[str, Any]:
    category = str(item["category"])
    metrics = _metrics_for(category, requested_metrics)
    sources = list(item["source_paths"])
    edited = Path(item["result_path"])
    instruction = str(item["instruction"])
    explanation = str(item["explanation"])
    record: dict[str, Any] = {
        "status": "ok",
        "sample_id": item["sample_id"],
        "category": category,
        "image_id": item["image_id"],
        "instruction": instruction,
        "explain": explanation,
        "result_path": str(edited),
        "raw_responses": {},
        "usage": {},
    }
    if category == MULTI_CATEGORY:
        consistency_prompt = prompt_consist_multi.format(instruct=instruction)
        instruction_prompt = prompt_instruction_multi.format(instruct=instruction)
        comparison_images = sources + [edited]
    elif category == TEMPORAL_CATEGORY:
        consistency_prompt = prompt_consist_temporal.format(
            N=len(sources), instruct=instruction
        )
        instruction_prompt = prompt_instruction_temporal.format(
            N=len(sources), instruct=instruction
        )
        comparison_images = sources + [edited]
    else:
        consistency_prompt = prompt_consist.format(instruct=instruction)
        instruction_prompt = (
            prompt_view_instruction_following.format(instruct=instruction)
            if category == VIEW_CATEGORY
            else prompt_instruction_following.format(instruct=instruction)
        )
        comparison_images = sources + [edited]

    if "consistency" in metrics:
        score, reason, raw, usage = _judge_metric(
            judge,
            prompt=consistency_prompt,
            image_paths=comparison_images,
            score_key="consistency_score",
            reason_keys=("reason", "reasoning"),
            prefix=r"consistency[_\s]*score\s*[:：]?\s*([1-5])",
        )
        record.update(consistency_score=score, consistency_reasoning=reason)
        record["raw_responses"]["consistency"] = raw
        record["usage"]["consistency"] = dict(usage)

    if category in KNOWLEDGE_CATEGORIES and (
        "instruction_following" in metrics or "knowledge_plausibility" in metrics
    ):
        response = judge.evaluate(
            system_prompt="",
            user_prompt=prompt_dual_evaluation.format(
                instruct=instruction, explanation=explanation
            ),
            image_urls=[image_to_data_url(path) for path in comparison_images],
            max_tokens=1200,
        )
        if "instruction_following" in metrics:
            score, reason = _extract_score(
                response.text,
                "instruction_score",
                ("instruction_reasoning", "reasoning", "reason"),
                r"instruction[_\s]*score\s*[:：]?\s*([1-5])",
            )
            record.update(instruction_score=score, instruction_reasoning=reason)
        if "knowledge_plausibility" in metrics:
            score, reason = _extract_score(
                response.text,
                "knowledge_score",
                ("knowledge_reasoning", "reasoning", "reason"),
                r"knowledge[_\s]*score\s*[:：]?\s*([1-5])",
            )
            record.update(knowledge_score=score, knowledge_reasoning=reason)
        record["raw_responses"]["dual_score"] = response.text
        record["usage"]["dual_score"] = dict(response.usage)
    elif "instruction_following" in metrics:
        instruction_images = comparison_images
        if category == VIEW_CATEGORY:
            if item["ground_truth_path"] is None:
                raise EvaluationError(
                    f"viewpoint task lacks ground truth: {item['sample_id']}"
                )
            instruction_images = sources + [edited, item["ground_truth_path"]]
        score, reason, raw, usage = _judge_metric(
            judge,
            prompt=instruction_prompt,
            image_paths=instruction_images,
            score_key="instruction_score",
            reason_keys=("reasoning", "reason"),
            prefix=r"instruction[_\s]*score\s*[:：]?\s*([1-5])",
        )
        record.update(instruction_score=score, instruction_reasoning=reason)
        record["raw_responses"]["instruction_following"] = raw
        record["usage"]["instruction_following"] = dict(usage)

    if "image_quality" in metrics:
        score, reason, raw, usage = _judge_metric(
            judge,
            prompt=prompt_quality,
            image_paths=(edited,),
            score_key="quality_score",
            reason_keys=("reasoning", "reason"),
            prefix=r"quality[_\s]*score\s*[:：]?\s*([1-5])",
        )
        record.update(quality_score=score, quality_reasoning=reason)
        record["raw_responses"]["image_quality"] = raw
        record["usage"]["image_quality"] = dict(usage)
    return record


def _summary(items: list[dict[str, Any]], records: Mapping[str, Any]) -> dict[str, Any]:
    score_keys = (
        "consistency_score",
        "instruction_score",
        "knowledge_score",
        "quality_score",
    )
    category_summary: dict[str, Any] = {}
    all_scores: dict[str, list[float]] = defaultdict(list)
    for category in ALL_CATEGORIES:
        category_records = [
            records.get(str(item["sample_id"]), {})
            for item in items
            if item["category"] == category
            and records.get(str(item["sample_id"]), {}).get("status") == "ok"
        ]
        metrics: dict[str, Any] = {}
        for key in score_keys:
            values = [float(record[key]) for record in category_records if key in record]
            if values:
                metrics[key] = mean(values)
                all_scores[key].extend(values)
        category_summary[category] = {"count": len(category_records), **metrics}
    return {
        "protocol": "KRIS-Bench official GPT-4o rubrics",
        "official_implementation": OFFICIAL_IMPLEMENTATION,
        "selected": len(items),
        "completed": sum(
            records.get(str(item["sample_id"]), {}).get("status") == "ok"
            for item in items
        ),
        "macro_sample_means": {
            key: mean(values) for key, values in all_scores.items()
        },
        "categories": category_summary,
        "official_code_compatibility_note": (
            "anomaly_correction follows the generic instruction rubric, matching the "
            "published script's category-name branch behavior"
        ),
    }


def _write_category_metrics(
    output_dir: Path,
    items: Sequence[dict[str, Any]],
    records: Mapping[str, Any],
) -> None:
    for category in ALL_CATEGORIES:
        payload: dict[str, Any] = {}
        for item in items:
            if item["category"] != category:
                continue
            record = records.get(str(item["sample_id"]), {})
            if record.get("status") != "ok":
                continue
            payload[str(item["image_id"])] = {
                key: value
                for key, value in record.items()
                if key
                not in {
                    "status",
                    "sample_id",
                    "category",
                    "image_id",
                    "result_path",
                    "raw_responses",
                    "usage",
                }
            }
        atomic_write_json(output_dir / category / "metrics.json", payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        data_root=DEFAULT_BENCHMARK_ROOT / "KRIS_Bench",
        results_dir=DEFAULT_RESULTS_ROOT / "kris_bench_results",
        output_dir=DEFAULT_RESULTS_ROOT / "kris_bench_results" / "evaluation",
    )
    add_judge_arguments(parser, default_workers=8)
    parser.add_argument("--categories", nargs="+", choices=ALL_CATEGORIES)
    parser.add_argument("--metrics", nargs="+", choices=ALL_METRICS)
    args = parser.parse_args(argv)
    try:
        results_dir = args.results_dir.expanduser().resolve()
        items = _load_items(
            args.data_root.expanduser().resolve(), ResultResolver(results_dir)
        )
        if args.categories:
            items = [item for item in items if item["category"] in args.categories]
        items = select_items(items, args)
        for item in items:
            _metrics_for(str(item["category"]), args.metrics)
        report = coverage_report(
            "KRIS_Bench",
            items,
            results_dir=results_dir,
            extra={"official_implementation": OFFICIAL_IMPLEMENTATION},
        )
        if args.validate_only:
            return exit_code_for_coverage(report, strict=args.strict_completeness)
        settings = load_judge_settings(args)
        judge = VisionJudge(settings)
        output_dir = args.output_dir.expanduser().resolve()
        store = JsonEvaluationStore(output_dir / "per_sample.json", overwrite=args.overwrite)
        pending = store.pending(items, overwrite=args.overwrite)
        with_results = [item for item in pending if item["result_path"] is not None]
        for item in pending:
            if item["result_path"] is None:
                store.update(
                    str(item["sample_id"]),
                    {"status": "missing_result", "sample_id": item["sample_id"]},
                )

        def evaluate(item: dict[str, Any]) -> dict[str, Any]:
            return _evaluate_item(
                item, judge=judge, requested_metrics=args.metrics
            )

        run_threaded(
            with_results,
            workers=args.workers,
            function=evaluate,
            on_result=store.update,
        )
        _write_category_metrics(output_dir, items, store.records)
        summary = _summary(items, store.records)
        summary["judge"] = {
            "model": settings.model,
            "base_url": settings.base_url,
            "wire_api": settings.wire_api,
        }
        atomic_write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        failures = sum(
            store.records.get(str(item["sample_id"]), {}).get("status") != "ok"
            for item in items
        )
        return 2 if failures else 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[KRIS_Bench evaluation] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
