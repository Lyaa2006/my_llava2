#!/usr/bin/env python3
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


TASK_SPECS = [
    ("task1", "ImageNet-R", "ImageNet-R.json"),
    ("task2", "ArxivQA", "ArxivQA.json"),
    ("task3", "VizWiz", "VizWiz.json"),
    ("task4", "IconQA", "IconQA.json"),
    ("task5", "CLEVR-Math", "CLEVR-Math.json"),
    ("task6", "Flickr30k", "Flickr30k.json"),
]


@dataclass
class TaskManifest:
    task_id: str
    task_name: str
    data_config: str
    train_path: str
    train_folder: str
    sample_count_hint: int


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _count_samples(json_path: Path) -> int:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data)


def build_manifest(
    checkpoint_dir: Path,
    data_config_root: Path,
    output_dir: Path,
    model_config_path: Path,
) -> Dict:
    adapter_config = _load_json(checkpoint_dir / "adapter_config.json")
    model_config = _load_json(model_config_path)

    task_entries: List[TaskManifest] = []
    for task_id, task_name, config_name in TASK_SPECS:
        cfg_path = (data_config_root / config_name).resolve()
        cfg = _load_json(cfg_path)
        train_path = Path(cfg["train_path"]).resolve()
        train_folder = Path(cfg["train_folder"]).resolve()
        task_entries.append(
            TaskManifest(
                task_id=task_id,
                task_name=task_name,
                data_config=str(cfg_path),
                train_path=str(train_path),
                train_folder=str(train_folder),
                sample_count_hint=_count_samples(train_path),
            )
        )

    return {
        "mode": "interface_only",
        "summary": (
            "This manifest stages task6 RegLoRA -> HiDESC anchor extraction. "
            "The actual GPU extraction job is intentionally not started here."
        ),
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "planned_anchor_cache_path": str(output_dir / "anchors.pt"),
        "planned_meta_path": str(output_dir / "meta.json"),
        "checkpoint_adapter_config": adapter_config,
        "model_config": model_config,
        "tasks": [asdict(entry) for entry in task_entries],
        "recommended_runtime": {
            "python_bin": "/home/lyaa/miniconda3/envs/MCITlib/bin/python3",
            "note": (
                "Use the HiDESC-compatible environment when you later implement "
                "or run the actual extractor."
            ),
        },
        "expected_output_schema": {
            "anchors.pt": [
                "image_anchors",
                "text_anchors",
                "image_sums",
                "text_sums",
                "image_boundary",
                "text_boundary",
                "task_ids",
                "task_names",
                "data_configs",
                "data_paths",
                "image_folders",
                "sample_counts",
                "used_indices",
                "model_config",
                "prompt_version",
                "image_aspect_ratio",
                "data_key",
                "image_folder_key",
            ],
            "meta.json": [
                "task_ids",
                "task_names",
                "data_configs",
                "data_paths",
                "image_folders",
                "sample_counts",
                "prompt_version",
                "image_aspect_ratio",
                "model_config",
            ],
        },
        "notes": [
            (
                "RegLoRA task6 checkpoints do not already contain HiDESC "
                "anchor tensors, so extraction must be performed before "
                "checkpoint injection."
            ),
            (
                "This script prepares the manifest and output layout only. "
                "It does not start any GPU workload."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage a task6 RegLoRA->HiDESC anchor-cache extraction manifest "
            "without launching GPU work."
        )
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--data-config-root",
        default="/mnt/lyaa/MCITlib/configs/data_configs/UCIT",
    )
    parser.add_argument(
        "--model-config",
        default="/mnt/lyaa/MCITlib/configs/model_configs/internvl.json",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    data_config_root = Path(args.data_config_root).resolve()
    model_config_path = Path(args.model_config).resolve()

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir does not exist: {checkpoint_dir}")
    if not (checkpoint_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Missing adapter_config.json in checkpoint dir: {checkpoint_dir}"
        )
    if not data_config_root.is_dir():
        raise FileNotFoundError(f"Data config root does not exist: {data_config_root}")
    if not model_config_path.is_file():
        raise FileNotFoundError(f"Model config does not exist: {model_config_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        checkpoint_dir=checkpoint_dir,
        data_config_root=data_config_root,
        output_dir=output_dir,
        model_config_path=model_config_path,
    )
    manifest_path = output_dir / "extraction_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps({"manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
