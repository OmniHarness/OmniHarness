"""Evaluate flat OmniHarness GenEval outputs with the official detector metric."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from evaluation_runner import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_RESULTS_ROOT,
    REPOSITORY_ROOT,
    EvaluationError,
    ResultResolver,
    add_common_arguments,
    atomic_write_json,
    atomic_write_jsonl,
    coverage_report,
    exit_code_for_coverage,
    load_evaluation_records,
    mean,
    read_jsonl,
    select_items,
)


OFFICIAL_IMPLEMENTATION = "https://github.com/djghosh13/geneval/blob/main/evaluation/evaluate_images.py"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
COLORS = (
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
)
CLASSNAMES = tuple(
    line.strip()
    for line in """person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
computer mouse
tv remote
computer keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush""".splitlines()
)


def _load_items(data_root: Path, resolver: ResultResolver) -> list[dict[str, Any]]:
    rows = read_jsonl(data_root / "evaluation_metadata.jsonl")
    items: list[dict[str, Any]] = []
    for index, metadata in enumerate(rows):
        sample_id = f"{index:04d}"
        result = resolver.find(
            (
                sample_id,
                f"{index:05d}",
                f"geneval_{sample_id}",
                f"geneval-{sample_id}",
            ),
            extensions=IMAGE_EXTENSIONS,
        )
        items.append(
            {
                "sample_id": sample_id,
                "row_index": index,
                "metadata": metadata,
                "result_path": result,
            }
        )
    return items


class OfficialGenEval:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import mmdet
            import numpy as np
            import open_clip
            import torch
            from clip_benchmark.metrics import zeroshot_classification as zsc
            from mmdet.apis import inference_detector, init_detector
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise EvaluationError(
                "GenEval requires mmdet 2.x, mmcv-full, torch, open-clip-torch, "
                "clip-benchmark, numpy, and Pillow"
            ) from exc
        major = str(getattr(mmdet, "__version__", "")).split(".")[0]
        if major and major != "2":
            raise EvaluationError(
                f"official GenEval requires mmdet 2.x, found {mmdet.__version__}"
            )
        if args.device == "cuda" and not torch.cuda.is_available():
            raise EvaluationError("official GenEval evaluation requires an available CUDA GPU")
        self.np = np
        self.torch = torch
        self.Image = Image
        self.ImageOps = ImageOps
        self.inference_detector = inference_detector
        self.zsc = zsc
        self.zsc.tqdm = lambda iterator, *unused_args, **unused_kwargs: iterator
        config_path = args.model_config
        if config_path is None:
            config_path = Path(mmdet.__file__).resolve().parent / (
                "../configs/mask2former/"
                "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
            )
        config_path = Path(config_path).expanduser().resolve()
        checkpoint = (
            args.model_path.expanduser().resolve()
            / f"{args.detector_name}.pth"
        )
        if not config_path.is_file():
            raise EvaluationError(f"Mask2Former config is missing: {config_path}")
        if not checkpoint.is_file():
            raise EvaluationError(f"Mask2Former checkpoint is missing: {checkpoint}")
        print(f"Loading Mask2Former from {checkpoint}")
        started = time.monotonic()
        self.detector = init_detector(
            str(config_path), str(checkpoint), device=args.device
        )
        self.clip_model, _, self.transform = open_clip.create_model_and_transforms(
            args.clip_model, pretrained="openai", device=args.device
        )
        self.tokenizer = open_clip.get_tokenizer(args.clip_model)
        self.device = args.device
        self.threshold = args.threshold
        self.counting_threshold = args.counting_threshold
        self.max_objects = args.max_objects
        self.nms_threshold = args.max_overlap
        self.position_threshold = args.position_threshold
        self.color_classifiers: dict[str, Any] = {}
        print(f"Loaded GenEval models in {time.monotonic() - started:.1f}s")

    @staticmethod
    def _iou(box_a: Any, box_b: Any) -> float:
        def area(box: Any) -> Any:
            return max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)

        intersection = area(
            [
                max(box_a[0], box_b[0]),
                max(box_a[1], box_b[1]),
                min(box_a[2], box_b[2]),
                min(box_a[3], box_b[3]),
            ]
        )
        union = area(box_a) + area(box_b) - intersection
        return float(intersection / union) if union else 0.0

    def _relative_position(self, obj_a: Any, obj_b: Any) -> set[str]:
        boxes = self.np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = self.np.abs(self.np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        revised = self.np.maximum(
            self.np.abs(offset) - self.position_threshold * (dim_a + dim_b), 0
        ) * self.np.sign(offset)
        if self.np.all(self.np.abs(revised) < 1e-3):
            return set()
        dx, dy = revised / self.np.linalg.norm(offset)
        relations: set[str] = set()
        if dx < -0.5:
            relations.add("left of")
        if dx > 0.5:
            relations.add("right of")
        if dy < -0.5:
            relations.add("above")
        if dy > 0.5:
            relations.add("below")
        return relations

    def _classify_colors(
        self, image: Any, objects: list[Any], classname: str
    ) -> list[str]:
        if classname not in self.color_classifiers:
            self.color_classifiers[classname] = self.zsc.zero_shot_classifier(
                self.clip_model,
                self.tokenizer,
                list(COLORS),
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    "a photo of a {c} object",
                ],
                self.device,
            )
        parent = self

        class ImageCrops(self.torch.utils.data.Dataset):
            def __len__(self) -> int:
                return len(objects)

            def __getitem__(self, index: int) -> tuple[Any, int]:
                box, mask = objects[index]
                crop_source = image
                if mask is not None:
                    blank = parent.Image.new("RGB", image.size, color="#999")
                    crop_source = parent.Image.composite(
                        image, blank, parent.Image.fromarray(mask)
                    )
                return parent.transform(crop_source.crop(box[:4])), 0

        loader = self.torch.utils.data.DataLoader(
            ImageCrops(), batch_size=16, num_workers=0
        )
        with self.torch.no_grad():
            prediction, unused_targets = self.zsc.run_classification(
                self.clip_model,
                self.color_classifiers[classname],
                loader,
                self.device,
            )
        return [COLORS[index.item()] for index in prediction.argmax(1)]

    def _evaluate_requirements(
        self, image: Any, objects: Mapping[str, list[Any]], metadata: Mapping[str, Any]
    ) -> tuple[bool, str]:
        correct = True
        reasons: list[str] = []
        matched_groups: list[list[Any] | None] = []
        for requirement in metadata.get("include", []):
            classname = requirement["class"]
            matched = True
            found = objects.get(classname, [])[: requirement["count"]]
            if len(found) < requirement["count"]:
                correct = matched = False
                reasons.append(
                    f"expected {classname}>={requirement['count']}, found {len(found)}"
                )
            elif "color" in requirement:
                colors = self._classify_colors(image, found, classname)
                if colors.count(requirement["color"]) < requirement["count"]:
                    correct = matched = False
                    reasons.append(
                        f"expected {requirement['color']} {classname}>="
                        f"{requirement['count']}, found {colors.count(requirement['color'])}"
                    )
            if "position" in requirement and matched:
                expected_relation, target_group = requirement["position"]
                target = matched_groups[target_group]
                if target is None:
                    correct = matched = False
                    reasons.append(
                        f"no target for {classname} to be {expected_relation}"
                    )
                else:
                    for obj in found:
                        for target_obj in target:
                            relations = self._relative_position(obj, target_obj)
                            if expected_relation not in relations:
                                correct = matched = False
                                reasons.append(
                                    f"expected {classname} {expected_relation} target, "
                                    f"found {' and '.join(relations)} target"
                                )
                                break
                        if not matched:
                            break
            matched_groups.append(found if matched else None)
        for requirement in metadata.get("exclude", []):
            classname = requirement["class"]
            if len(objects.get(classname, [])) >= requirement["count"]:
                correct = False
                reasons.append(
                    f"expected {classname}<{requirement['count']}, "
                    f"found {len(objects[classname])}"
                )
        return correct, "\n".join(reasons)

    def evaluate(self, item: Mapping[str, Any]) -> dict[str, Any]:
        path = Path(item["result_path"])
        metadata = item["metadata"]
        detector_result = self.inference_detector(self.detector, str(path))
        boxes = detector_result[0] if isinstance(detector_result, tuple) else detector_result
        segmentation = (
            detector_result[1]
            if isinstance(detector_result, tuple) and len(detector_result) > 1
            else None
        )
        image = self.ImageOps.exif_transpose(self.Image.open(path)).convert("RGB")
        detected: dict[str, list[Any]] = {}
        confidence = (
            self.counting_threshold
            if metadata["tag"] == "counting"
            else self.threshold
        )
        for class_index, classname in enumerate(CLASSNAMES):
            ordering = self.np.argsort(boxes[class_index][:, 4])[::-1]
            ordering = ordering[boxes[class_index][ordering, 4] > confidence]
            remaining = ordering[: self.max_objects].tolist()
            found: list[Any] = []
            while remaining:
                best = remaining.pop(0)
                mask = None if segmentation is None else segmentation[class_index][best]
                found.append((boxes[class_index][best], mask))
                remaining = [
                    candidate
                    for candidate in remaining
                    if self.nms_threshold == 1
                    or self._iou(boxes[class_index][best], boxes[class_index][candidate])
                    < self.nms_threshold
                ]
            if found:
                detected[classname] = found
        correct, reason = self._evaluate_requirements(image, detected, metadata)
        return {
            "status": "ok",
            "sample_id": item["sample_id"],
            "filename": str(path),
            "tag": metadata["tag"],
            "prompt": metadata["prompt"],
            "correct": bool(correct),
            "reason": reason,
            "metadata": dict(metadata),
            "details": {
                key: [box.tolist() for box, unused_mask in value]
                for key, value in detected.items()
            },
        }


def _summary(items: list[dict[str, Any]], records: Mapping[str, Any]) -> dict[str, Any]:
    completed = [
        records.get(str(item["sample_id"]), {})
        for item in items
        if records.get(str(item["sample_id"]), {}).get("status") == "ok"
    ]
    tag_scores: dict[str, list[float]] = defaultdict(list)
    for record in completed:
        tag_scores[str(record["tag"])].append(float(bool(record["correct"])))
    task_scores = [float(mean(values) or 0.0) for values in tag_scores.values()]
    return {
        "protocol": "GenEval official Mask2Former + CLIP color classifier",
        "official_implementation": OFFICIAL_IMPLEMENTATION,
        "selected": len(items),
        "completed": len(completed),
        "correct_images": sum(bool(record["correct"]) for record in completed),
        "correct_image_rate": mean(
            float(bool(record["correct"])) for record in completed
        ),
        "task_breakdown": {
            tag: {"count": len(values), "score": mean(values)}
            for tag, values in tag_scores.items()
        },
        "overall_score_average_over_tasks": mean(task_scores),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        data_root=DEFAULT_BENCHMARK_ROOT / "GenEval",
        results_dir=DEFAULT_RESULTS_ROOT / "geneval_results",
        output_dir=DEFAULT_RESULTS_ROOT / "geneval_results" / "evaluation",
    )
    parser.add_argument("--model-config", type=Path)
    parser.add_argument(
        "--model-path", type=Path, default=REPOSITORY_ROOT / "resources" / "models" / "geneval"
    )
    parser.add_argument(
        "--detector-name",
        default="mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco",
    )
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--counting-threshold", type=float, default=0.9)
    parser.add_argument("--max-objects", type=int, default=16)
    parser.add_argument("--max-overlap", type=float, default=1.0)
    parser.add_argument("--position-threshold", type=float, default=0.1)
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
            "GenEval",
            items,
            results_dir=results_dir,
            extra={"official_implementation": OFFICIAL_IMPLEMENTATION},
        )
        if args.validate_only:
            return exit_code_for_coverage(report, strict=args.strict_completeness)
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = output_dir / "per_sample.json"
        records = load_evaluation_records(cache_path, overwrite=args.overwrite)
        evaluator = OfficialGenEval(args)
        for position, item in enumerate(items, start=1):
            sample_id = str(item["sample_id"])
            if item["result_path"] is None:
                record = {"status": "missing_result", "sample_id": sample_id}
            else:
                try:
                    record = evaluator.evaluate(item)
                except Exception as exc:
                    record = {
                        "status": "error",
                        "sample_id": sample_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            records[sample_id] = record
            atomic_write_json(cache_path, records)
            print(f"[{position}/{len(items)}] {sample_id}: {record['status']}")
        summary = _summary(items, records)
        summary["detector"] = {
            "name": args.detector_name,
            "model_path": str(args.model_path.expanduser().resolve()),
            "model_config": (
                str(args.model_config.expanduser().resolve())
                if args.model_config is not None
                else "mmdet-2.x bundled config"
            ),
            "clip_model": args.clip_model,
            "device": args.device,
            "threshold": args.threshold,
            "counting_threshold": args.counting_threshold,
            "max_objects": args.max_objects,
            "max_overlap": args.max_overlap,
            "position_threshold": args.position_threshold,
        }
        atomic_write_json(output_dir / "summary.json", summary)
        official_rows = [
            {
                "filename": record["filename"],
                "tag": record["tag"],
                "prompt": record["prompt"],
                "correct": record["correct"],
                "reason": record["reason"],
                "metadata": json.dumps(record["metadata"]),
                "details": json.dumps(record["details"]),
            }
            for record in records.values()
            if record.get("status") == "ok"
        ]
        atomic_write_jsonl(output_dir / "results.jsonl", official_rows)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        failures = sum(
            records.get(str(item["sample_id"]), {}).get("status") != "ok"
            for item in items
        )
        return 2 if failures else 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[GenEval evaluation] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
