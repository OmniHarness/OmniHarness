"""Evaluate GenEval2 with the official Qwen3-VL Soft-TIFA protocol."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation_runner import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_RESULTS_ROOT,
    EvaluationError,
    ResultResolver,
    add_common_arguments,
    atomic_write_json,
    coverage_report,
    exit_code_for_coverage,
    load_evaluation_records,
    geometric_mean,
    mean,
    natural_key,
    read_jsonl,
    select_items,
)


OFFICIAL_IMPLEMENTATION = "https://github.com/facebookresearch/GenEval2/blob/main/evaluation.py"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
METHODS = ("vqascore", "tifa", "soft_tifa_am", "soft_tifa_gm")
SKILLS = ("object", "attribute", "count", "position", "verb")
NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def _load_items(data_root: Path, resolver: ResultResolver) -> list[dict[str, Any]]:
    rows = read_jsonl(data_root / "geneval2_data.jsonl")
    items: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample_id = f"{index:04d}"
        result = resolver.find(
            (
                sample_id,
                f"{index:05d}",
                f"geneval2_{sample_id}",
                f"geneval2-{sample_id}",
            ),
            extensions=IMAGE_EXTENSIONS,
        )
        items.append(
            {
                "sample_id": sample_id,
                "row_index": index,
                "prompt": str(row.get("prompt", "")),
                "atom_count": int(row.get("atom_count", 0)),
                "vqa_list": list(row.get("vqa_list", [])),
                "skills": list(row.get("skills", [])),
                "result_path": result,
            }
        )
    return items


class SoftTIFAEvaluator:
    def __init__(self, model_name: str, device_map: str) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise EvaluationError(
                "GenEval2 requires torch and transformers==4.57.0"
            ) from exc
        self.torch = torch
        print(f"Loading official GenEval2 judge: {model_name}")
        self.processor = AutoProcessor.from_pretrained(
            model_name, torch_dtype="auto", device_map=device_map
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", device_map=device_map
        )

    def _send(
        self,
        prompt: str,
        image_path: Path,
        answer_list: Sequence[str] | None = None,
    ) -> tuple[str, float | None]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        probabilities = self.torch.nn.functional.softmax(outputs.scores[0], dim=-1)
        answer_probability = None
        if answer_list:
            answer_probability = 0.0
            for answer in answer_list:
                token_id = self.processor.tokenizer.encode(answer)[0]
                answer_probability += probabilities[0, token_id].item()
        prediction = self.processor.batch_decode(
            [self.torch.argmax(probabilities)]
        )[0]
        return prediction, answer_probability

    @staticmethod
    def _answers(question: str, answer: str) -> list[str]:
        if question.startswith("How many"):
            numeric = NUMBER_WORDS.get(answer, "other")
            return [
                answer,
                answer.capitalize(),
                " " + answer,
                " " + answer.capitalize(),
                numeric,
                " " + numeric,
            ]
        return ["Yes", "yes", " yes", " Yes"]

    def evaluate(self, item: Mapping[str, Any], method: str) -> dict[str, Any]:
        image_path = item["result_path"]
        if method == "vqascore":
            question = f'Does this image show "{item["prompt"]}"? Answer the question with Yes or No.'
            _, probability = self._send(
                question, image_path, ["Yes", "yes", " yes", " Yes"]
            )
            atom_scores = [float(probability or 0.0)]
        else:
            atom_scores = []
            predictions: list[str] = []
            for question, answer in item["vqa_list"]:
                answers = self._answers(str(question), str(answer))
                prediction, probability = self._send(
                    f"{question} Answer in one word.", image_path, answers
                )
                predictions.append(prediction)
                if method == "tifa":
                    atom_scores.append(float(prediction.lower() in answers))
                else:
                    atom_scores.append(float(probability or 0.0))
        prompt_score = (
            geometric_mean(atom_scores)
            if method == "soft_tifa_gm"
            else float(mean(atom_scores) or 0.0)
        )
        return {
            "status": "ok",
            "sample_id": item["sample_id"],
            "row_index": item["row_index"],
            "prompt": item["prompt"],
            "atom_count": item["atom_count"],
            "skills": item["skills"],
            "result_path": str(image_path),
            "method": method,
            "atom_scores": atom_scores,
            "prompt_score": prompt_score,
        }


def _summary(
    items: list[dict[str, Any]], records: Mapping[str, Any], method: str
) -> dict[str, Any]:
    completed = [
        records.get(str(item["sample_id"]), {})
        for item in items
        if records.get(str(item["sample_id"]), {}).get("status") == "ok"
    ]
    skill_scores: dict[str, list[float]] = defaultdict(list)
    atomicity_scores: dict[str, list[float]] = defaultdict(list)
    for record in completed:
        for skill, score in zip(record.get("skills", []), record["atom_scores"]):
            skill_scores[str(skill)].append(float(score))
        atomicity_scores[str(record["atom_count"])].append(
            float(record["prompt_score"])
        )
    skill_means = {
        skill: mean(values) for skill, values in sorted(skill_scores.items())
    }
    # The paper reports an unweighted arithmetic mean of all five skill scores.
    overall = (
        100.0 * sum(float(skill_means[skill]) for skill in SKILLS) / len(SKILLS)
        if all(skill_means.get(skill) is not None for skill in SKILLS)
        and method != "vqascore"
        else None
    )
    return {
        "protocol": "GenEval2 official Qwen3-VL Soft-TIFA",
        "official_implementation": OFFICIAL_IMPLEMENTATION,
        "method": method,
        "selected": len(items),
        "completed": len(completed),
        "overall_score_percent": overall,
        "prompt_score_percent": 100.0
        * float(mean(float(record["prompt_score"]) for record in completed) or 0.0),
        "skill_scores": skill_means,
        "atomicity_scores": {
            atomicity: mean(values)
            for atomicity, values in sorted(
                atomicity_scores.items(), key=lambda item: int(item[0])
            )
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        data_root=DEFAULT_BENCHMARK_ROOT / "GenEval2",
        results_dir=DEFAULT_RESULTS_ROOT / "geneval2_results",
        output_dir=DEFAULT_RESULTS_ROOT / "geneval2_results" / "evaluation",
    )
    parser.add_argument("--method", choices=METHODS, default="soft_tifa_am")
    parser.add_argument(
        "--judge-model", default="Qwen/Qwen3-VL-8B-Instruct"
    )
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args(argv)
    try:
        results_dir = args.results_dir.expanduser().resolve()
        items = select_items(
            _load_items(
                args.data_root.expanduser().resolve(), ResultResolver(results_dir)
            ),
            args,
        )
        report = coverage_report(
            "GenEval2",
            items,
            results_dir=results_dir,
            extra={
                "official_implementation": OFFICIAL_IMPLEMENTATION,
                "method": args.method,
                "judge_model": args.judge_model,
            },
        )
        if args.validate_only:
            return exit_code_for_coverage(report, strict=args.strict_completeness)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = output_dir / "per_sample.json"
        records = load_evaluation_records(cache_path, overwrite=args.overwrite)
        evaluator = SoftTIFAEvaluator(args.judge_model, args.device_map)
        for position, item in enumerate(items, start=1):
            sample_id = str(item["sample_id"])
            if item["result_path"] is None:
                record = {"status": "missing_result", "sample_id": sample_id}
            else:
                try:
                    record = evaluator.evaluate(item, args.method)
                except Exception as exc:
                    record = {
                        "status": "error",
                        "sample_id": sample_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            records[sample_id] = record
            atomic_write_json(
                cache_path,
                {key: records[key] for key in sorted(records, key=natural_key)},
            )
            print(f"[{position}/{len(items)}] {sample_id}: {record['status']}")
        summary = _summary(items, records, args.method)
        summary["judge_model"] = args.judge_model
        summary["device_map"] = args.device_map
        atomic_write_json(output_dir / "summary.json", summary)
        score_lists = [
            records[str(item["sample_id"])]["atom_scores"]
            for item in items
            if records.get(str(item["sample_id"]), {}).get("status") == "ok"
        ]
        atomic_write_json(output_dir / "score_lists.json", score_lists)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        failures = sum(
            records.get(str(item["sample_id"]), {}).get("status") != "ok"
            for item in items
        )
        return 2 if failures else 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[GenEval2 evaluation] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
