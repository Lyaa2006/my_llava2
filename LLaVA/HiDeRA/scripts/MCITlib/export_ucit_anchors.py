#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
WORKSPACE_ROOT = SCRIPT_PATH.parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llava import conversation as conversation_lib
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.multimodal_encoder.builder import build_text_tower, build_vision_tower
from llava.train.train_MOE import DataArguments, DataCollatorForSupervisedDataset, LazySupervisedDataset


DEFAULT_TASK_CONFIGS = [
    ("task1", "ImageNet-R", WORKSPACE_ROOT / "configs/data_configs/UCIT/ImageNet-R.json"),
    ("task2", "ArxivQA", WORKSPACE_ROOT / "configs/data_configs/UCIT/ArxivQA.json"),
    ("task3", "VizWiz", WORKSPACE_ROOT / "configs/data_configs/UCIT/VizWiz.json"),
    ("task4", "IconQA", WORKSPACE_ROOT / "configs/data_configs/UCIT/IconQA.json"),
    ("task5", "CLEVR-Math", WORKSPACE_ROOT / "configs/data_configs/UCIT/CLEVR-Math.json"),
    ("task6", "Flickr30k", WORKSPACE_ROOT / "configs/data_configs/UCIT/Flickr30k.json"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Offline export UCIT task anchors.")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=WORKSPACE_ROOT / "configs/model_configs/llava.json",
        help="Path to the LLaVA model config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save anchors.pt and meta.json.",
    )
    parser.add_argument(
        "--data-configs",
        type=Path,
        nargs="*",
        default=None,
        help="Optional custom UCIT data config JSONs. Default order is task1-task6.",
    )
    parser.add_argument(
        "--task-names",
        nargs="*",
        default=None,
        help="Optional human-readable task names aligned with --data-configs.",
    )
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help="Optional task ids aligned with --data-configs, e.g. task1 task2 ...",
    )
    parser.add_argument(
        "--data-key",
        default="train_path",
        help="Dataset JSON field to use inside each data config.",
    )
    parser.add_argument(
        "--image-folder-key",
        default="train_folder",
        help="Image folder field to use inside each data config.",
    )
    parser.add_argument(
        "--prompt-version",
        default="v1",
        help="Conversation template version. Matches the training scripts by default.",
    )
    parser.add_argument(
        "--model-max-length",
        type=int,
        default=2048,
        help="Tokenizer max length used when preparing prompts.",
    )
    parser.add_argument(
        "--image-aspect-ratio",
        default="pad",
        help="Image aspect ratio preprocessing mode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for offline feature extraction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=None,
        help="Optional cap per task. Omit to use all samples.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("first", "random"),
        default="first",
        help="How to choose samples when --max-samples-per-task is set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for optional subsampling.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for CLIP vision/text towers.",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "fp32"),
        default="fp16",
        help="Feature extractor dtype on GPU.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r") as handle:
        return json.load(handle)


def resolve_dtype(dtype_name: str, device: torch.device):
    if device.type != "cuda":
        return torch.float32
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    return torch.float16


def configure_tokenizer_and_conversation(model_name: str, prompt_version: str, model_max_length: int):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    if prompt_version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[prompt_version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    return tokenizer


def build_feature_extractors(model_config, device: torch.device, dtype: torch.dtype):
    cfg = SimpleNamespace(
        mm_vision_tower=model_config["vision_tower"],
        vision_tower=model_config["vision_tower"],
        mm_text_tower=model_config["text_tower"],
        text_tower=model_config["text_tower"],
        mm_vision_select_layer=-2,
        mm_text_select_layer=-1,
        mm_vision_select_feature="patch",
    )
    vision_tower = build_vision_tower(cfg)
    text_tower = build_text_tower(cfg)
    vision_tower.to(device=device, dtype=dtype)
    text_tower.to(device=device, dtype=dtype)
    vision_tower.eval()
    text_tower.eval()
    clip_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_config["text_tower"],
        model_max_length=77,
        padding_side="right",
        use_fast=True,
    )
    return vision_tower, text_tower, clip_tokenizer


def maybe_subset_dataset(dataset, max_samples, sample_mode, seed):
    if max_samples is None or max_samples >= len(dataset):
        return dataset, None
    if sample_mode == "first":
        indices = list(range(max_samples))
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(dataset)), k=max_samples))
    return Subset(dataset, indices), indices


def decode_clip_inputs(input_ids: torch.Tensor, tokenizer):
    input_pad = np.where(
        input_ids.cpu().numpy() != IMAGE_TOKEN_INDEX,
        input_ids.cpu().numpy(),
        tokenizer.pad_token_id,
    )
    decoded_inputs = tokenizer.batch_decode(input_pad, skip_special_tokens=True)
    decoded_hidden_inputs = ["\n".join(decoded_input.split("\n")[1:]) for decoded_input in decoded_inputs]
    return [decoded_input.split(" ASSISTANT")[0] for decoded_input in decoded_hidden_inputs]


def compute_task_anchor(
    task_id,
    task_name,
    data_config_path: Path,
    tokenizer,
    clip_tokenizer,
    vision_tower,
    text_tower,
    args,
):
    data_config = load_json(data_config_path)
    data_path = Path(data_config[args.data_key])
    image_folder = data_config[args.image_folder_key]

    data_args = DataArguments(
        data_path=str(data_path),
        memory_data_path=None,
        lazy_preprocess=True,
        is_multimodal=True,
        image_folder=image_folder,
        image_aspect_ratio=args.image_aspect_ratio,
    )
    data_args.image_processor = vision_tower.image_processor
    data_args.mm_use_im_start_end = False

    dataset = LazySupervisedDataset(
        data_path=str(data_path),
        tokenizer=tokenizer,
        data_args=data_args,
    )
    dataset, used_indices = maybe_subset_dataset(
        dataset,
        args.max_samples_per_task,
        args.sample_mode,
        args.seed + len(task_name),
    )
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collator,
    )

    image_sum = None
    text_sum = None
    sample_count = 0

    progress = tqdm(data_loader, desc=f"{task_id}:{task_name}", leave=False)
    for batch in progress:
        images = batch["images"].to(device=vision_tower.device, dtype=vision_tower.dtype, non_blocking=True)
        with torch.no_grad():
            image_features, _ = vision_tower(images)

        clip_inputs = decode_clip_inputs(batch["input_ids"], tokenizer)
        clip_text_inputs = clip_tokenizer(
            clip_inputs,
            padding="longest",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_features = text_tower(clip_text_inputs)

        image_features = image_features.float().cpu()
        text_features = text_features.float().cpu()
        batch_image_sum = image_features.sum(dim=0)
        batch_text_sum = text_features.sum(dim=0)
        if image_sum is None:
            image_sum = batch_image_sum
            text_sum = batch_text_sum
        else:
            image_sum += batch_image_sum
            text_sum += batch_text_sum
        sample_count += image_features.shape[0]
        progress.set_postfix(samples=sample_count)

    if sample_count == 0:
        raise ValueError(f"No samples found for {task_name} ({data_config_path}).")

    image_anchor = (image_sum / sample_count).unsqueeze(0)
    text_anchor = (text_sum / sample_count).unsqueeze(0)
    return {
        "task_id": task_id,
        "task_name": task_name,
        "data_config": str(data_config_path),
        "data_path": str(data_path),
        "image_folder": image_folder,
        "sample_count": sample_count,
        "used_indices": used_indices,
        "image_sum": image_sum,
        "text_sum": text_sum,
        "image_anchor": image_anchor,
        "text_anchor": text_anchor,
    }


def resolve_task_specs(args):
    if args.data_configs is None or len(args.data_configs) == 0:
        return [
            {"task_id": task_id, "task_name": task_name, "data_config": data_config}
            for task_id, task_name, data_config in DEFAULT_TASK_CONFIGS
        ]

    task_count = len(args.data_configs)
    if args.task_names is not None and len(args.task_names) != task_count:
        raise ValueError("--task-names must align with --data-configs.")
    if args.task_ids is not None and len(args.task_ids) != task_count:
        raise ValueError("--task-ids must align with --data-configs.")

    specs = []
    for idx, data_config in enumerate(args.data_configs):
        task_name = args.task_names[idx] if args.task_names is not None else data_config.stem
        task_id = args.task_ids[idx] if args.task_ids is not None else f"task{idx + 1}"
        specs.append({"task_id": task_id, "task_name": task_name, "data_config": data_config})
    return specs


def save_outputs(results, args, model_config):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_anchors": torch.stack([result["image_anchor"] for result in results], dim=0),
        "text_anchors": torch.stack([result["text_anchor"] for result in results], dim=0),
        "image_sums": torch.stack([result["image_sum"] for result in results], dim=0),
        "text_sums": torch.stack([result["text_sum"] for result in results], dim=0),
        "image_boundary": torch.tensor([[result["sample_count"]] for result in results], dtype=torch.float32),
        "text_boundary": torch.tensor([[result["sample_count"]] for result in results], dtype=torch.float32),
        "task_ids": [result["task_id"] for result in results],
        "task_names": [result["task_name"] for result in results],
        "data_configs": [result["data_config"] for result in results],
        "data_paths": [result["data_path"] for result in results],
        "image_folders": [result["image_folder"] for result in results],
        "sample_counts": [result["sample_count"] for result in results],
        "used_indices": [result["used_indices"] for result in results],
        "model_config": model_config,
        "prompt_version": args.prompt_version,
        "image_aspect_ratio": args.image_aspect_ratio,
        "data_key": args.data_key,
        "image_folder_key": args.image_folder_key,
    }
    torch.save(payload, args.output_dir / "anchors.pt")

    meta = {
        "task_ids": payload["task_ids"],
        "task_names": payload["task_names"],
        "data_configs": payload["data_configs"],
        "data_paths": payload["data_paths"],
        "image_folders": payload["image_folders"],
        "sample_counts": payload["sample_counts"],
        "prompt_version": payload["prompt_version"],
        "image_aspect_ratio": payload["image_aspect_ratio"],
        "model_config": model_config,
    }
    with open(args.output_dir / "meta.json", "w") as handle:
        json.dump(meta, handle, indent=2)


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_config = load_json(args.model_config)
    task_specs = resolve_task_specs(args)

    tokenizer = configure_tokenizer_and_conversation(
        model_name=model_config["model_name"],
        prompt_version=args.prompt_version,
        model_max_length=args.model_max_length,
    )
    vision_tower, text_tower, clip_tokenizer = build_feature_extractors(
        model_config=model_config,
        device=device,
        dtype=dtype,
    )

    results = []
    for spec in task_specs:
        results.append(
            compute_task_anchor(
                task_id=spec["task_id"],
                task_name=spec["task_name"],
                data_config_path=Path(spec["data_config"]),
                tokenizer=tokenizer,
                clip_tokenizer=clip_tokenizer,
                vision_tower=vision_tower,
                text_tower=text_tower,
                args=args,
            )
        )

    save_outputs(results, args, model_config)
    print(f"Saved anchor cache to {args.output_dir}")
    print(f"Anchor tensor shape: {tuple(torch.stack([result['image_anchor'] for result in results], dim=0).shape)}")


if __name__ == "__main__":
    main()
