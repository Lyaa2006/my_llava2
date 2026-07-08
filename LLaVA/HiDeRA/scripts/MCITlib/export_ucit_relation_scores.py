#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llava import conversation as conversation_lib
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.multimodal_encoder.builder import build_text_tower, build_vision_tower
from llava.train.train_MOE import DataArguments, DataCollatorForSupervisedDataset, LazySupervisedDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Offline export UCIT relation scores from cached task anchors.")
    parser.add_argument(
        "--anchor-cache",
        type=Path,
        required=True,
        help="Path to anchor cache directory or anchors.pt file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save relation_scores.pt and meta.json.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for offline relation simulation. Use the training batch size to stay close to full training.",
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
        help="Optional cap per task for faster debugging. Omit to use all samples.",
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
    parser.add_argument("--routing-image-weight", type=float, default=0.5)
    parser.add_argument("--routing-text-weight", type=float, default=0.5)
    parser.add_argument("--routing-history-weight", type=float, default=0.15)
    parser.add_argument("--routing-temperature", type=float, default=0.1)
    parser.add_argument("--routing-top-k", type=int, default=2)
    parser.add_argument("--routing-min-similarity", type=float, default=-1.0)
    parser.add_argument("--routing-prior-momentum", type=float, default=0.8)
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Include the current task anchor itself as a relation candidate when exporting completed-task caches.",
    )
    return parser.parse_args()


def resolve_anchor_cache_path(anchor_cache_arg: Path):
    if anchor_cache_arg.is_dir():
        return anchor_cache_arg / "anchors.pt"
    return anchor_cache_arg


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


def compose_relation_logits(image_scores, text_scores, history_scores, args):
    return (
        float(args.routing_image_weight) * image_scores
        + float(args.routing_text_weight) * text_scores
        + float(args.routing_history_weight) * history_scores
    )


def mask_relation_logits(relation_logits, min_similarity):
    if min_similarity <= -1.0:
        return relation_logits
    masked_relation_logits = torch.where(
        relation_logits >= min_similarity,
        relation_logits,
        torch.full_like(relation_logits, float("-inf")),
    )
    if not torch.isfinite(masked_relation_logits).any():
        return relation_logits
    return masked_relation_logits


def build_sparse_relation_weights(logits, top_k, temperature):
    if logits.numel() == 0:
        return logits
    top_k = int(top_k)
    if top_k <= 0:
        top_k = logits.numel()
    top_k = min(top_k, logits.numel())
    if top_k < logits.numel():
        top_values, top_indices = torch.topk(logits, k=top_k)
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(0, top_indices, top_values)
    else:
        masked_logits = logits
    temperature = max(float(temperature), 1e-6)
    return F.softmax(masked_logits / temperature, dim=0)


def build_dataset(data_path, image_folder, image_processor, tokenizer, image_aspect_ratio, max_samples, sample_mode, seed):
    data_args = DataArguments(
        data_path=data_path,
        memory_data_path=None,
        lazy_preprocess=True,
        is_multimodal=True,
        image_folder=image_folder,
        image_aspect_ratio=image_aspect_ratio,
    )
    data_args.image_processor = image_processor
    data_args.mm_use_im_start_end = False
    dataset = LazySupervisedDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    return maybe_subset_dataset(dataset, max_samples, sample_mode, seed)


def simulate_task_relation(
    task_idx,
    payload,
    tokenizer,
    clip_tokenizer,
    vision_tower,
    text_tower,
    task_relation_scores,
    expert_usage_prior,
    args,
):
    data_path = payload["data_paths"][task_idx]
    image_folder = payload["image_folders"][task_idx]
    dataset, used_indices = build_dataset(
        data_path=data_path,
        image_folder=image_folder,
        image_processor=vision_tower.image_processor,
        tokenizer=tokenizer,
        image_aspect_ratio=payload["image_aspect_ratio"],
        max_samples=args.max_samples_per_task,
        sample_mode=args.sample_mode,
        seed=args.seed + task_idx,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
    )

    candidate_count = task_idx + 1 if args.include_self else task_idx
    if candidate_count <= 0:
        return {
            "used_indices": used_indices,
            "batch_count": 1,
            "sample_count": len(dataset),
            "mean_relation_weights": torch.ones(1, dtype=torch.float32),
            "last_relation_weights": torch.ones(1, dtype=torch.float32),
            "candidate_count": 1,
        }

    candidate_image_anchors = F.normalize(
        payload["image_anchors"][:candidate_count].float().squeeze(1),
        dim=1,
    )
    candidate_text_anchors = F.normalize(
        payload["text_anchors"][:candidate_count].float().squeeze(1),
        dim=1,
    )

    weight_sum = torch.zeros(candidate_count, dtype=torch.float32)
    last_relation_weights = torch.zeros(candidate_count, dtype=torch.float32)
    batch_count = 0
    sample_count = 0

    progress = tqdm(
        data_loader,
        desc=f"{payload['task_ids'][task_idx]}:{payload['task_names'][task_idx]}",
        leave=False,
    )
    for batch in progress:
        images = batch["images"].to(device=vision_tower.device, dtype=vision_tower.dtype, non_blocking=True)
        with torch.no_grad():
            image_guide_features, _ = vision_tower(images)

        clip_inputs = decode_clip_inputs(batch["input_ids"], tokenizer)
        clip_text_inputs = clip_tokenizer(
            clip_inputs,
            padding="longest",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_guide_features = text_tower(clip_text_inputs)

        current_image_summary = F.normalize(image_guide_features.float().mean(dim=0), dim=0)
        current_text_summary = F.normalize(text_guide_features.float().mean(dim=0), dim=0)

        image_scores = torch.matmul(candidate_image_anchors, current_image_summary.cpu())
        text_scores = torch.matmul(candidate_text_anchors, current_text_summary.cpu())
        history_scores = task_relation_scores[task_idx, :candidate_count].detach().float()
        relation_logits = compose_relation_logits(image_scores, text_scores, history_scores, args)
        relation_logits = mask_relation_logits(relation_logits, args.routing_min_similarity)
        relation_weights = build_sparse_relation_weights(
            relation_logits,
            top_k=min(int(args.routing_top_k), candidate_count),
            temperature=args.routing_temperature,
        )

        momentum = float(args.routing_prior_momentum)
        task_relation_scores[task_idx, :candidate_count] = (
            momentum * task_relation_scores[task_idx, :candidate_count]
            + (1.0 - momentum) * relation_weights
        )
        expert_usage_prior[:candidate_count] = (
            momentum * expert_usage_prior[:candidate_count]
            + (1.0 - momentum) * relation_weights
        )

        weight_sum += relation_weights
        last_relation_weights = relation_weights
        batch_count += 1
        sample_count += int(image_guide_features.shape[0])
        progress.set_postfix(batches=batch_count, samples=sample_count)

    if batch_count == 0:
        raise ValueError(f"No samples found for task index {task_idx}.")

    return {
        "used_indices": used_indices,
        "batch_count": batch_count,
        "sample_count": sample_count,
        "mean_relation_weights": weight_sum / batch_count,
        "last_relation_weights": last_relation_weights,
        "candidate_count": candidate_count,
    }


def save_outputs(payload, task_relation_scores, expert_usage_prior, task_means, task_lasts, task_batch_counts, task_sample_counts, used_indices, args, anchor_cache_path):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "task_relation_scores": task_relation_scores,
        "expert_usage_prior": expert_usage_prior,
        "task_mean_relation_weights": task_means,
        "task_last_relation_weights": task_lasts,
        "task_batch_counts": task_batch_counts,
        "task_sample_counts": task_sample_counts,
        "used_indices": used_indices,
        "task_ids": payload["task_ids"],
        "task_names": payload["task_names"],
        "data_paths": payload["data_paths"],
        "image_folders": payload["image_folders"],
        "anchor_cache_path": str(anchor_cache_path),
        "routing_config": {
            "routing_image_weight": args.routing_image_weight,
            "routing_text_weight": args.routing_text_weight,
            "routing_history_weight": args.routing_history_weight,
            "routing_temperature": args.routing_temperature,
            "routing_top_k": args.routing_top_k,
            "routing_min_similarity": args.routing_min_similarity,
            "routing_prior_momentum": args.routing_prior_momentum,
            "include_self": args.include_self,
        },
    }
    torch.save(result, args.output_dir / "relation_scores.pt")

    meta = {
        "task_ids": payload["task_ids"],
        "task_names": payload["task_names"],
        "task_batch_counts": task_batch_counts.tolist(),
        "task_sample_counts": task_sample_counts.tolist(),
        "anchor_cache_path": str(anchor_cache_path),
        "routing_config": result["routing_config"],
    }
    with open(args.output_dir / "meta.json", "w") as handle:
        json.dump(meta, handle, indent=2)


def main():
    args = parse_args()
    anchor_cache_path = resolve_anchor_cache_path(args.anchor_cache)
    payload = torch.load(anchor_cache_path, map_location="cpu")

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_config = payload["model_config"]
    prompt_version = payload.get("prompt_version", "v1")

    tokenizer = configure_tokenizer_and_conversation(
        model_name=model_config["model_name"],
        prompt_version=prompt_version,
        model_max_length=2048,
    )
    vision_tower, text_tower, clip_tokenizer = build_feature_extractors(
        model_config=model_config,
        device=device,
        dtype=dtype,
    )

    num_tasks = len(payload["task_ids"])
    task_relation_scores = torch.zeros((num_tasks, num_tasks), dtype=torch.float32)
    expert_usage_prior = torch.zeros(num_tasks, dtype=torch.float32)
    task_means = torch.zeros((num_tasks, num_tasks), dtype=torch.float32)
    task_lasts = torch.zeros((num_tasks, num_tasks), dtype=torch.float32)
    task_batch_counts = torch.zeros(num_tasks, dtype=torch.int64)
    task_sample_counts = torch.zeros(num_tasks, dtype=torch.int64)
    used_indices = [None for _ in range(num_tasks)]

    start_task_idx = 0 if args.include_self else 1
    for task_idx in range(start_task_idx, num_tasks):
        task_result = simulate_task_relation(
            task_idx=task_idx,
            payload=payload,
            tokenizer=tokenizer,
            clip_tokenizer=clip_tokenizer,
            vision_tower=vision_tower,
            text_tower=text_tower,
            task_relation_scores=task_relation_scores,
            expert_usage_prior=expert_usage_prior,
            args=args,
        )
        candidate_count = task_result["candidate_count"]
        task_means[task_idx, :candidate_count] = task_result["mean_relation_weights"]
        task_lasts[task_idx, :candidate_count] = task_result["last_relation_weights"]
        task_batch_counts[task_idx] = task_result["batch_count"]
        task_sample_counts[task_idx] = task_result["sample_count"]
        used_indices[task_idx] = task_result["used_indices"]

    save_outputs(
        payload=payload,
        task_relation_scores=task_relation_scores,
        expert_usage_prior=expert_usage_prior,
        task_means=task_means,
        task_lasts=task_lasts,
        task_batch_counts=task_batch_counts,
        task_sample_counts=task_sample_counts,
        used_indices=used_indices,
        args=args,
        anchor_cache_path=anchor_cache_path,
    )
    print(f"Saved relation cache to {args.output_dir}")
    print(f"Relation score tensor shape: {tuple(task_relation_scores.shape)}")


if __name__ == "__main__":
    main()
