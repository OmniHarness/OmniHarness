"""Evaluate WISE original/rewrite outputs with the official legacy WiScore."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

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
    atomic_write_jsonl,
    coverage_report,
    exit_code_for_coverage,
    image_to_data_url,
    load_judge_settings,
    mean,
    read_json,
    run_threaded,
    select_items,
)


OFFICIAL_IMPLEMENTATION = "https://github.com/PKU-YuanGroup/WISE/tree/main/WISE_legacy"
DOMAINS = (
    "cultural_common_sense",
    "spatio-temporal_reasoning",
    "natural_science",
)
VARIANTS = ("original", "rewrite", "both")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
REQUIRED_SCORES = ("consistency", "realism", "aesthetic_quality")
SCORE_PATTERN = re.compile(
    r"\*{0,2}(Consistency|Realism|Aesthetic Quality)\*{0,2}\s*[:：]?\s*([0-2])",
    re.I,
)
SYSTEM_PROMPT = (
    "You are a professional Vincennes image quality audit expert, "
    "please evaluate the image quality strictly according to the protocol."
)
EVALUATION_PROMPT = """Please evaluate strictly and return ONLY the three scores as requested.

# Text-to-Image Quality Evaluation Protocol

## System Instruction
You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE RUTHLESSNESS. Only images meeting the HIGHEST standards should receive top scores.

**Input Parameters**
- PROMPT: [User's original prompt to]
- EXPLANATION: [Further explanation of the original prompt]

---

## Scoring Criteria

**Consistency (0-2):** How accurately and completely the image reflects the PROMPT.
* **0 (Rejected):** Fails to capture key elements of the prompt, or contradicts the prompt.
* **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, or not accurately. Noticeable deviations from the prompt's intent.
* **2 (Exemplary):** Perfectly and completely aligns with the PROMPT. Every single element and nuance of the prompt is flawlessly represented in the image. The image is an ideal, unambiguous visual realization of the given prompt.

**Realism (0-2):** How realistically the image is rendered.
* **0 (Rejected):** Physically implausible and clearly artificial. Breaks fundamental laws of physics or visual realism.
* **1 (Conditional):** Contains minor inconsistencies or unrealistic elements. While somewhat believable, noticeable flaws detract from realism.
* **2 (Exemplary):** Achieves photorealistic quality, indistinguishable from a real photograph. Flawless adherence to physical laws, accurate material representation, and coherent spatial relationships. No visual cues betraying AI generation.

**Aesthetic Quality (0-2):** The overall artistic appeal and visual quality of the image.
* **0 (Rejected):** Poor aesthetic composition, visually unappealing, and lacks artistic merit.
* **1 (Conditional):** Demonstrates basic visual appeal, acceptable composition, and color harmony, but lacks distinction or artistic flair.
* **2 (Exemplary):** Possesses exceptional aesthetic quality, comparable to a masterpiece. Strikingly beautiful, with perfect composition, a harmonious color palette, and a captivating artistic style. Demonstrates a high degree of artistic vision and execution.

---

## Output Format

Do not include any other text, explanations, or labels. You must return only three lines of text, each containing a metric and the corresponding score, for example:

Consistency: 2
Realism: 1
Aesthetic Quality: 0

---

## IMPORTANT Enforcement

Be EXTREMELY strict in your evaluation. A score of '2' should be exceedingly rare and reserved only for images that truly excel and meet the highest possible standards in each metric. If there is any doubt, downgrade the score.

For Consistency, a score of '2' requires complete and flawless adherence to every aspect of the prompt, leaving no room for misinterpretation or omission.

For Realism, a score of '2' means the image is virtually indistinguishable from a real photograph in terms of detail, lighting, physics, and material properties.

For Aesthetic Quality, a score of '2' demands exceptional artistic merit, not just pleasant visuals.

---

Here are the Prompt and EXPLANATION for this evaluation:
PROMPT: "{prompt}"
EXPLANATION: "{explanation}"

Please strictly adhere to the scoring criteria and follow the template format when providing your results."""


def _parse_scores(text: str) -> dict[str, float]:
    parsed = {
        label.lower().replace(" ", "_"): float(value)
        for label, value in SCORE_PATTERN.findall(text)
    }
    if not set(REQUIRED_SCORES).issubset(parsed):
        numbers = re.findall(r"(?m)^\s*([0-2])\s*$", text)
        if len(numbers) >= 3:
            parsed = {
                "consistency": float(numbers[0]),
                "realism": float(numbers[1]),
                "aesthetic_quality": float(numbers[2]),
            }
    if not set(REQUIRED_SCORES).issubset(parsed):
        missing = sorted(set(REQUIRED_SCORES) - set(parsed))
        raise EvaluationError("WISE judge response is missing: " + ", ".join(missing))
    return {key: parsed[key] for key in REQUIRED_SCORES}


def _category(prompt_id: int) -> str:
    if 1 <= prompt_id <= 400:
        return "CULTURE"
    if 401 <= prompt_id <= 567:
        return "TIME"
    if 568 <= prompt_id <= 700:
        return "SPACE"
    if 701 <= prompt_id <= 800:
        return "BIOLOGY"
    if 801 <= prompt_id <= 900:
        return "PHYSICS"
    if 901 <= prompt_id <= 1000:
        return "CHEMISTRY"
    raise EvaluationError(f"WISE prompt_id is outside 1..1000: {prompt_id}")


def _load_items(
    data_root: Path,
    resolver: ResultResolver,
    *,
    domains: list[str],
    variant: str,
) -> list[dict[str, Any]]:
    variants = ("original", "rewrite") if variant == "both" else (variant,)
    items: list[dict[str, Any]] = []
    for domain in domains:
        for current_variant in variants:
            suffix = "" if current_variant == "original" else "_rewrite"
            payload = read_json(data_root / f"{domain}{suffix}.json")
            if not isinstance(payload, list):
                raise EvaluationError(f"WISE data file must contain a list: {domain}{suffix}")
            for row in payload:
                if not isinstance(row, Mapping):
                    raise EvaluationError(f"invalid WISE record in {domain}{suffix}")
                prompt_id = int(row["prompt_id"])
                sample_id = f"{domain}-{current_variant}-{prompt_id}"
                keys: tuple[str, ...] = (
                    sample_id,
                    f"wise_{sample_id}",
                    f"{current_variant}_{prompt_id}",
                )
                if variant != "both":
                    keys += (str(prompt_id), f"{prompt_id:04d}")
                result = resolver.find(
                    keys,
                    extensions=IMAGE_EXTENSIONS,
                    category_hints=(domain, current_variant),
                )
                items.append(
                    {
                        "sample_id": sample_id,
                        "prompt_id": prompt_id,
                        "domain": domain,
                        "variant": current_variant,
                        "category": _category(prompt_id),
                        "prompt": str(row.get("Prompt", "")).strip(),
                        "explanation": str(row.get("Explanation", "")).strip(),
                        "subcategory": row.get("Subcategory"),
                        "result_path": result,
                    }
                )
    return items


def _wiscore(record: Mapping[str, Any]) -> float:
    return (
        0.7 * float(record["consistency"])
        + 0.2 * float(record["realism"])
        + 0.1 * float(record["aesthetic_quality"])
    ) / 2.0


def _summary(items: list[dict[str, Any]], records: Mapping[str, Any]) -> dict[str, Any]:
    variant_categories: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    completed = 0
    for item in items:
        record = records.get(str(item["sample_id"]), {})
        if record.get("status") != "ok":
            continue
        completed += 1
        variant_categories[str(item["variant"])][str(item["category"])].append(
            _wiscore(record)
        )
    weights = {
        "CULTURE": 0.4,
        "TIME": 0.167,
        "SPACE": 0.133,
        "BIOLOGY": 0.1,
        "PHYSICS": 0.1,
        "CHEMISTRY": 0.1,
    }
    variant_summaries: dict[str, Any] = {}
    for variant, categories in variant_categories.items():
        category_scores = {
            category: mean(values) for category, values in categories.items()
        }
        complete = all(category_scores.get(category) is not None for category in weights)
        overall = (
            sum(weights[key] * float(category_scores[key]) for key in weights)
            if complete
            else None
        )
        variant_summaries[variant] = {
            "category_wiscores": category_scores,
            "category_counts": {key: len(value) for key, value in categories.items()},
            "overall_wiscore": overall,
            "all_six_categories_present": complete,
        }
    return {
        "protocol": "WISE legacy GPT-4o WiScore",
        "official_implementation": OFFICIAL_IMPLEMENTATION,
        "formula": "(0.7*consistency + 0.2*realism + 0.1*aesthetic_quality)/2",
        "selected": len(items),
        "completed": completed,
        "variants": variant_summaries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        data_root=DEFAULT_BENCHMARK_ROOT / "WISE",
        results_dir=DEFAULT_RESULTS_ROOT / "wise_results",
        output_dir=DEFAULT_RESULTS_ROOT / "wise_results" / "evaluation",
    )
    add_judge_arguments(parser, default_workers=8)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS)
    parser.add_argument("--variant", choices=VARIANTS, default="original")
    args = parser.parse_args(argv)
    try:
        results_dir = args.results_dir.expanduser().resolve()
        items = _load_items(
            args.data_root.expanduser().resolve(),
            ResultResolver(results_dir),
            domains=args.domains or list(DOMAINS),
            variant=args.variant,
        )
        items = select_items(items, args)
        report = coverage_report(
            "WISE",
            items,
            results_dir=results_dir,
            extra={
                "protocol": "legacy",
                "official_implementation": OFFICIAL_IMPLEMENTATION,
            },
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
            response = judge.evaluate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=EVALUATION_PROMPT.format(
                    prompt=item["prompt"], explanation=item["explanation"]
                ),
                image_urls=(image_to_data_url(item["result_path"]),),
                max_tokens=300,
            )
            scores = _parse_scores(response.text)
            return {
                "status": "ok",
                "sample_id": item["sample_id"],
                "prompt_id": item["prompt_id"],
                "variant": item["variant"],
                "category": item["category"],
                "subcategory": item["subcategory"],
                "result_path": str(item["result_path"]),
                **scores,
                "wiscore": (
                    0.7 * scores["consistency"]
                    + 0.2 * scores["realism"]
                    + 0.1 * scores["aesthetic_quality"]
                )
                / 2.0,
                "raw_response": response.text,
                "usage": dict(response.usage),
            }

        run_threaded(
            with_results,
            workers=args.workers,
            function=evaluate,
            on_result=store.update,
        )
        summary = _summary(items, store.records)
        summary["judge"] = {
            "model": settings.model,
            "base_url": settings.base_url,
            "wire_api": settings.wire_api,
            "official_reference_model": "gpt-4o-2024-05-13",
        }
        atomic_write_json(output_dir / "summary.json", summary)
        official_rows = [
            {
                "prompt_id": record["prompt_id"],
                "variant": record["variant"],
                "Subcategory": record.get("subcategory"),
                **{key: record[key] for key in REQUIRED_SCORES},
            }
            for record in store.records.values()
            if record.get("status") == "ok"
        ]
        atomic_write_jsonl(
            output_dir / "scores.jsonl",
            sorted(official_rows, key=lambda row: (row["variant"], row["prompt_id"])),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        failures = sum(
            store.records.get(str(item["sample_id"]), {}).get("status") != "ok"
            for item in items
        )
        return 2 if failures else 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[WISE evaluation] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
