"""Evaluate ComfyBench outputs with its official GPT-4o five-modality rubric."""

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
    coverage_report,
    exit_code_for_coverage,
    image_to_data_url,
    load_judge_settings,
    read_json,
    run_threaded,
    sample_video,
    select_items,
)


OFFICIAL_IMPLEMENTATION = "https://github.com/xxyQwQ/ComfyBench/blob/main/script/evaluation.py"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".gif", ".mov", ".webm", ".avi")
MODALITIES = ("t2i", "i2i", "t2v", "i2v", "v2v")
CATEGORIES = ("vanilla", "complex", "creative")
TAG_PATTERN = re.compile(r"<(?P<tag>analysis|judgment)>(?P<text>.*?)</(?P=tag)>", re.I | re.S)


PROMPTS = {
    "t2i": """You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of a text-to-image generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The given image is the generation result, with an actual resolution of {result_resolution}.

First, analyze whether the generation result meets each key point in the instruction. Enclose your analysis in the <analysis> tag. For example: <analysis>There is a cat in an astronaut suit, which is consistent with the instruction. The wall is white, which is different from the "green wall" in the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.""",
    "i2i": """You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of an image-to-image generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The first image is the input reference, with an actual resolution of {reference_resolution}. The second image is the generation result, with an actual resolution of {result_resolution}.

First, analyze whether the generation result meets each key point in the instruction based on the input reference. Enclose your analysis in the <analysis> tag. For example: <analysis>The generation result keeps the structure of the input reference, but the car is not removed, which is not consistent with the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.""",
    "t2v": """You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of a text-to-video generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The given {result_frame_count} images are the frames sampled from the generation result, with an actual resolution of {result_resolution}, duration of {result_duration} seconds and {result_frame_rate} frames per second.

First, analyze whether the generation result meets each key point in the instruction. Enclose your analysis in the <analysis> tag. For example: <analysis>There is a walking robot, which is consistent with the instruction. However, the scene is a street, which is different from the "forest" in the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.""",
    "i2v": """You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of an image-to-video generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The first image is the input reference, with an actual resolution of {reference_resolution}. The remaining {result_frame_count} images are the frames sampled from the generation result, with an actual resolution of {result_resolution}, duration of {result_duration} seconds and {result_frame_rate} frames per second.

First, analyze whether the generation result meets each key point in the instruction based on the input reference. Enclose your analysis in the <analysis> tag. For example: <analysis>The generation result contains a moving car, which is consistent with the instruction. However, it fails to follow the style of the input reference.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.""",
    "v2v": """You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of a video-to-video generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The first {reference_frame_count} images are the frames sampled from the input reference, with an actual resolution of {reference_resolution}, duration of {reference_duration} seconds and {reference_frame_rate} frames per second. The remaining {result_frame_count} images are the frames sampled from the generation result, with an actual resolution of {result_resolution}, duration of {result_duration} seconds and {result_frame_rate} frames per second.

First, analyze whether the generation result meets each key point in the instruction based on the input reference. Enclose your analysis in the <analysis> tag. For example: <analysis>The generation result improves the resolution of the input reference. However, it fails to convert the input inference into an oil painting style, which is not consistent with the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.""",
}


def _image_resolution(path: Path) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise EvaluationError("Pillow is required for ComfyBench evaluation") from exc
    with Image.open(path) as image:
        width, height = image.size
    return f"{width}x{height}"


def _parse_answer(text: str) -> tuple[str, str]:
    fields = {
        match.group("tag").lower(): match.group("text").strip()
        for match in TAG_PATTERN.finditer(text)
    }
    judgment = fields.get("judgment", "")
    if judgment.lower() not in {"true", "false"}:
        raise EvaluationError("judge answer lacks a valid <judgment>True|False</judgment>")
    return fields.get("analysis", ""), judgment.title()


def _load_items(data_root: Path, resolver: ResultResolver) -> list[dict[str, Any]]:
    payload = read_json(data_root / "complete.json")
    if not isinstance(payload, Mapping):
        raise EvaluationError("ComfyBench complete.json must contain an object")
    items: list[dict[str, Any]] = []
    for sample_id, raw in sorted(payload.items(), key=lambda item: int(item[0])):
        if not isinstance(raw, Mapping):
            raise EvaluationError(f"invalid ComfyBench row: {sample_id}")
        modality = str(raw.get("modality", "")).lower()
        if modality not in MODALITIES:
            raise EvaluationError(f"unsupported ComfyBench modality: {modality}")
        extensions = IMAGE_EXTENSIONS if modality.endswith("i") else VIDEO_EXTENSIONS
        result = resolver.find(
            (str(sample_id), f"comfybench_{sample_id}"), extensions=extensions
        )
        resource_value = raw.get("resource")
        resource = None
        if resource_value not in (None, "", "None"):
            resource = (data_root / "resource" / str(resource_value)).resolve()
            if not resource.is_file():
                raise EvaluationError(f"missing ComfyBench resource: {resource}")
        items.append(
            {
                "sample_id": str(sample_id),
                "name": raw.get("name"),
                "category": str(raw.get("category", "")),
                "modality": modality,
                "instruction": str(raw.get("instruction", "")).strip(),
                "resource_path": resource,
                "result_path": result,
            }
        )
    return items


def _prepare_judge_input(item: Mapping[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    modality = str(item["modality"])
    result = item["result_path"]
    resource = item["resource_path"]
    values: dict[str, Any] = {"instruction": item["instruction"]}
    image_urls: list[str] = []
    media_metadata: dict[str, Any] = {}
    if modality == "t2i":
        values["result_resolution"] = _image_resolution(result)
        image_urls.append(image_to_data_url(result))
    elif modality == "i2i":
        values["reference_resolution"] = _image_resolution(resource)
        values["result_resolution"] = _image_resolution(result)
        image_urls.extend((image_to_data_url(resource), image_to_data_url(result)))
    elif modality == "t2v":
        result_frames, result_meta = sample_video(result)
        values.update(
            result_frame_count=len(result_frames),
            result_resolution=f"{result_meta['width']}x{result_meta['height']}",
            result_duration=f"{result_meta['duration']:.2f}",
            result_frame_rate=f"{result_meta['frame_rate']:g}",
        )
        image_urls.extend(result_frames)
        media_metadata["result"] = result_meta
    elif modality == "i2v":
        result_frames, result_meta = sample_video(result)
        values.update(
            reference_resolution=_image_resolution(resource),
            result_frame_count=len(result_frames),
            result_resolution=f"{result_meta['width']}x{result_meta['height']}",
            result_duration=f"{result_meta['duration']:.2f}",
            result_frame_rate=f"{result_meta['frame_rate']:g}",
        )
        image_urls.append(image_to_data_url(resource))
        image_urls.extend(result_frames)
        media_metadata["result"] = result_meta
    else:
        reference_frames, reference_meta = sample_video(resource)
        result_frames, result_meta = sample_video(result)
        values.update(
            reference_frame_count=len(reference_frames),
            reference_resolution=f"{reference_meta['width']}x{reference_meta['height']}",
            reference_duration=f"{reference_meta['duration']:.2f}",
            reference_frame_rate=f"{reference_meta['frame_rate']:g}",
            result_frame_count=len(result_frames),
            result_resolution=f"{result_meta['width']}x{result_meta['height']}",
            result_duration=f"{result_meta['duration']:.2f}",
            result_frame_rate=f"{result_meta['frame_rate']:g}",
        )
        image_urls.extend(reference_frames)
        image_urls.extend(result_frames)
        media_metadata.update(reference=reference_meta, result=result_meta)
    return PROMPTS[modality].format(**values), image_urls, media_metadata


def _summarize(items: list[dict[str, Any]], records: Mapping[str, Any]) -> dict[str, Any]:
    groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "outputs_available": 0, "resolved": 0}
    )
    modalities: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "outputs_available": 0, "resolved": 0}
    )
    for item in items:
        record = records.get(str(item["sample_id"]), {})
        groups[str(item["category"])]["total"] += 1
        modalities[str(item["modality"])]["total"] += 1
        if item["result_path"] is not None:
            groups[str(item["category"])]["outputs_available"] += 1
            modalities[str(item["modality"])]["outputs_available"] += 1
        if record.get("resolved") is True:
            groups[str(item["category"])]["resolved"] += 1
            modalities[str(item["modality"])]["resolved"] += 1
    total = len(items)
    outputs_available = sum(item["result_path"] is not None for item in items)
    resolved = sum(
        records.get(str(item["sample_id"]), {}).get("resolved") is True
        for item in items
    )
    return {
        "protocol": "ComfyBench official GPT-4o resolve-rate rubric",
        "official_implementation": OFFICIAL_IMPLEMENTATION,
        "total": total,
        "outputs_available": outputs_available,
        "resolved": resolved,
        "output_coverage_rate": outputs_available / total if total else 0.0,
        "resolve_rate": resolved / total if total else 0.0,
        "by_category": dict(groups),
        "by_modality": dict(modalities),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(
        parser,
        data_root=DEFAULT_BENCHMARK_ROOT / "ComfyBench",
        results_dir=DEFAULT_RESULTS_ROOT / "comfybench_results",
        output_dir=DEFAULT_RESULTS_ROOT / "comfybench_results" / "evaluation",
    )
    add_judge_arguments(parser, default_workers=4)
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES)
    args = parser.parse_args(argv)
    try:
        data_root = args.data_root.expanduser().resolve()
        results_dir = args.results_dir.expanduser().resolve()
        resolver = ResultResolver(results_dir)
        items = _load_items(data_root, resolver)
        if args.modalities:
            items = [item for item in items if item["modality"] in args.modalities]
        if args.categories:
            items = [item for item in items if item["category"] in args.categories]
        items = select_items(items, args)
        report = coverage_report(
            "ComfyBench",
            items,
            results_dir=results_dir,
            extra={
                "official_implementation": OFFICIAL_IMPLEMENTATION,
                "video_tasks_are_standard_tasks": True,
            },
        )
        if args.validate_only:
            return exit_code_for_coverage(report, strict=args.strict_completeness)
        settings = load_judge_settings(args)
        judge = VisionJudge(settings)
        output_dir = args.output_dir.expanduser().resolve()
        store = JsonEvaluationStore(output_dir / "per_sample.json", overwrite=args.overwrite)
        pending = store.pending(items, overwrite=args.overwrite)
        pending_with_results = [item for item in pending if item["result_path"] is not None]
        for item in pending:
            if item["result_path"] is None:
                store.update(
                    str(item["sample_id"]),
                    {"status": "missing_result", "sample_id": item["sample_id"]},
                )

        def evaluate(item: dict[str, Any]) -> dict[str, Any]:
            prompt, image_urls, media_metadata = _prepare_judge_input(item)
            response = judge.evaluate(
                system_prompt="", user_prompt=prompt, image_urls=image_urls
            )
            analysis, judgment = _parse_answer(response.text)
            return {
                "status": "ok",
                "sample_id": item["sample_id"],
                "category": item["category"],
                "modality": item["modality"],
                "instruction": item["instruction"],
                "result_path": str(item["result_path"]),
                "analysis": analysis,
                "judgment": judgment,
                "resolved": judgment == "True",
                "raw_response": response.text,
                "usage": dict(response.usage),
                "media_metadata": media_metadata,
            }

        run_threaded(
            pending_with_results,
            workers=args.workers,
            function=evaluate,
            on_result=store.update,
        )
        summary = _summarize(items, store.records)
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
        print(f"[ComfyBench evaluation] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
