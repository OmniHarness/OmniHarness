"""Evaluate ReasonEdit with its official CLIP and preservation metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

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
    mean,
    natural_key,
    select_items,
)


OFFICIAL_IMPLEMENTATION = "https://github.com/TencentARC/SmartEdit/blob/main/test/metrics_evaluation.py"
CLIP_SUFFIX = re.compile(r"\s+CLIP:\s*(.*)$", re.I)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
OFFICIAL_CATEGORIES = (
    "1-Left-Right",
    "2-Relative-Size",
    "3-Mirror",
    "4-Color",
    "5-Multiple-Objects",
    "6-Reasoning",
)
SUPPLEMENTAL_CATEGORY = "7-Add-supp"


def _load_items(data_root: Path, resolver: ResultResolver) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for category_root in sorted(
        (path for path in data_root.iterdir() if path.is_dir()),
        key=lambda path: natural_key(path.name),
    ):
        text_files = tuple(category_root.glob("*.txt"))
        if len(text_files) != 1:
            raise EvaluationError(
                f"ReasonEdit category needs exactly one text file: {category_root.name}"
            )
        lines = [
            line.strip()
            for line in text_files[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for index, line in enumerate(lines, start=1):
            match = CLIP_SUFFIX.search(line)
            clip_target = match.group(1).strip() if match else None
            instruction = line[: match.start()].strip() if match else line
            sample_id = f"{category_root.name}-{index:03d}"
            result = resolver.find(
                (
                    sample_id,
                    f"reasonedit_{sample_id}",
                    f"{category_root.name}_{index:03d}",
                    f"{index:03d}",
                ),
                extensions=IMAGE_EXTENSIONS,
                category_hints=(category_root.name,),
            )
            original = (category_root / f"{index:03d}.png").resolve()
            mask = (category_root / f"{index:03d}_mask.jpg").resolve()
            if not original.is_file():
                raise EvaluationError(f"missing ReasonEdit original image: {original}")
            official = category_root.name in OFFICIAL_CATEGORIES
            if official and (not mask.is_file() or not clip_target):
                raise EvaluationError(
                    f"incomplete official ReasonEdit annotation: {sample_id}"
                )
            items.append(
                {
                    "sample_id": sample_id,
                    "category": category_root.name,
                    "index": index,
                    "instruction": instruction,
                    "clip_target": clip_target,
                    "original_path": original,
                    "mask_path": mask if mask.is_file() else None,
                    "result_path": result,
                    "official_evaluable": official,
                }
            )
    return items


class OfficialMetrics:
    def __init__(self, requested_device: str) -> None:
        try:
            import torch
            from torchmetrics.image import (
                PeakSignalNoiseRatio,
                StructuralSimilarityIndexMeasure,
            )
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            from torchmetrics.multimodal import CLIPScore
            from torchmetrics.regression import MeanSquaredError
        except ImportError as exc:
            raise EvaluationError(
                "ReasonEdit evaluation requires torch, torchvision, and torchmetrics"
            ) from exc
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise EvaluationError("CUDA was requested but is unavailable")
        self.torch = torch
        self.device = requested_device
        self.clip = CLIPScore(
            model_name_or_path="openai/clip-vit-large-patch14"
        ).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(
            self.device
        )
        self.mse = MeanSquaredError().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

    def evaluate(self, item: dict[str, Any]) -> dict[str, Any]:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise EvaluationError("ReasonEdit evaluation requires numpy and Pillow") from exc
        resampling = Image.Resampling
        original = Image.open(item["original_path"]).convert("RGB").resize(
            (256, 256), resample=resampling.BICUBIC
        )
        edited = Image.open(item["result_path"]).convert("RGB").resize(
            (256, 256), resample=resampling.BICUBIC
        )
        mask_image = Image.open(item["mask_path"]).convert("L").resize(
            (256, 256), resample=resampling.NEAREST
        )
        mask_array = np.asarray(mask_image)
        y_coord, x_coord = np.where(mask_array)
        if not len(x_coord):
            raise EvaluationError(f"empty evaluation mask: {item['mask_path']}")
        left, right = int(np.min(x_coord)), int(np.max(x_coord))
        top, bottom = int(np.min(y_coord)), int(np.max(y_coord))
        crop = np.asarray(edited.crop((left, top, right, bottom)))
        crop_tensor = self.torch.tensor(crop).permute(2, 0, 1).to(self.device)
        clip_score = self.clip(crop_tensor, str(item["clip_target"])).cpu().item()

        background_mask = 1.0 - mask_array.astype(np.float32) / 255.0
        background_mask = np.repeat(background_mask[:, :, None], 3, axis=2)
        original_array = np.asarray(original).astype(np.float32) / 255.0
        edited_array = np.asarray(edited).astype(np.float32) / 255.0
        original_array *= background_mask
        edited_array *= background_mask
        original_tensor = (
            self.torch.tensor(original_array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        edited_tensor = (
            self.torch.tensor(edited_array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        psnr = self.psnr(original_tensor, edited_tensor).cpu().item()
        lpips = self.lpips(
            original_tensor * 2 - 1, edited_tensor * 2 - 1
        ).cpu().item()
        mse = self.mse(
            original_tensor.squeeze(0).contiguous(),
            edited_tensor.squeeze(0).contiguous(),
        ).cpu().item()
        ssim = self.ssim(original_tensor, edited_tensor).cpu().item()
        return {
            "status": "ok",
            "sample_id": item["sample_id"],
            "category": item["category"],
            "instruction": item["instruction"],
            "clip_target": item["clip_target"],
            "result_path": str(item["result_path"]),
            "mask_path": str(item["mask_path"]),
            "psnr_unedited": psnr,
            "lpips_unedited": lpips,
            "mse_unedited": mse,
            "ssim_unedited": ssim,
            "clip_edited_region": clip_score,
        }


def _summary(items: list[dict[str, Any]], records: dict[str, Any]) -> dict[str, Any]:
    metric_names = (
        "psnr_unedited",
        "lpips_unedited",
        "mse_unedited",
        "ssim_unedited",
        "clip_edited_region",
    )
    categories: dict[str, dict[str, Any]] = {}
    for category in OFFICIAL_CATEGORIES:
        category_records = [
            records.get(str(item["sample_id"]), {})
            for item in items
            if item["category"] == category
            and records.get(str(item["sample_id"]), {}).get("status") == "ok"
        ]
        categories[category] = {
            "count": len(category_records),
            **{
                metric: mean(float(record[metric]) for record in category_records)
                for metric in metric_names
            },
        }
    understanding_records = [
        records.get(str(item["sample_id"]), {})
        for item in items
        if item["category"] in OFFICIAL_CATEGORIES[:5]
        and records.get(str(item["sample_id"]), {}).get("status") == "ok"
    ]
    reasoning_records = [
        records.get(str(item["sample_id"]), {})
        for item in items
        if item["category"] == "6-Reasoning"
        and records.get(str(item["sample_id"]), {}).get("status") == "ok"
    ]

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(rows),
            **{
                metric: mean(float(record[metric]) for record in rows)
                for metric in metric_names
            },
        }

    supplemental = [item for item in items if item["category"] == SUPPLEMENTAL_CATEGORY]
    return {
        "protocol": "ReasonEdit official SmartEdit metrics at 256x256",
        "official_implementation": OFFICIAL_IMPLEMENTATION,
        "mask_role": "evaluation_only",
        "categories": categories,
        "understanding_total": aggregate(understanding_records),
        "reasoning_total": aggregate(reasoning_records),
        "supplemental_excluded": {
            "category": SUPPLEMENTAL_CATEGORY,
            "count": len(supplemental),
            "reason": "the official script publishes no mask/CLIP metric for Add-supp",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        data_root=DEFAULT_BENCHMARK_ROOT / "ReasonEdit",
        results_dir=DEFAULT_RESULTS_ROOT / "reasonedit_results",
        output_dir=DEFAULT_RESULTS_ROOT / "reasonedit_results" / "evaluation",
    )
    parser.add_argument(
        "--categories", nargs="+", choices=OFFICIAL_CATEGORIES + (SUPPLEMENTAL_CATEGORY,)
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    try:
        results_dir = args.results_dir.expanduser().resolve()
        items = _load_items(
            args.data_root.expanduser().resolve(), ResultResolver(results_dir)
        )
        if args.categories:
            items = [item for item in items if item["category"] in args.categories]
        items = select_items(items, args)
        evaluable = [item for item in items if item["official_evaluable"]]
        report = coverage_report(
            "ReasonEdit",
            evaluable,
            results_dir=results_dir,
            extra={
                "official_implementation": OFFICIAL_IMPLEMENTATION,
                "loaded_total_including_supplemental": len(items),
                "excluded_supplemental": len(items) - len(evaluable),
                "mask_is_generation_input": False,
                "mask_is_evaluation_annotation": True,
            },
        )
        if args.validate_only:
            return exit_code_for_coverage(report, strict=args.strict_completeness)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        records_path = output_dir / "per_sample.json"
        existing = load_evaluation_records(records_path, overwrite=args.overwrite)
        metrics = OfficialMetrics(args.device)
        for position, item in enumerate(evaluable, start=1):
            sample_id = str(item["sample_id"])
            if item["result_path"] is None:
                existing[sample_id] = {"status": "missing_result", "sample_id": sample_id}
            else:
                try:
                    existing[sample_id] = metrics.evaluate(item)
                except Exception as exc:
                    existing[sample_id] = {
                        "status": "error",
                        "sample_id": sample_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            atomic_write_json(
                records_path,
                {key: existing[key] for key in sorted(existing, key=natural_key)},
            )
            print(f"[{position}/{len(evaluable)}] {sample_id}: {existing[sample_id]['status']}")
        summary = _summary(items, existing)
        summary["device"] = metrics.device
        atomic_write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        failures = sum(
            existing.get(str(item["sample_id"]), {}).get("status") != "ok"
            for item in evaluable
        )
        return 2 if failures else 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ReasonEdit evaluation] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
