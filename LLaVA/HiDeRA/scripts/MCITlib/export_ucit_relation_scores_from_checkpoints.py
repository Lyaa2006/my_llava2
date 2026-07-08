#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F


DEFAULT_TASK_IDS = ["task1", "task2", "task3", "task4", "task5", "task6"]
DEFAULT_TASK_NAMES = [
    "ImageNet-R",
    "ArxivQA",
    "VizWiz",
    "IconQA",
    "CLEVR-Math",
    "Flickr30k",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export UCIT relation scores directly from anchors saved in HiDE checkpoints."
    )
    parser.add_argument(
        "--checkpoint-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Ordered HiDE checkpoint directories, e.g. Task1 ... Task6.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory to save per-stage relation score caches.",
    )
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=DEFAULT_TASK_IDS,
        help="Task ids aligned with the continual task order.",
    )
    parser.add_argument(
        "--task-names",
        nargs="*",
        default=DEFAULT_TASK_NAMES,
        help="Task names aligned with the continual task order.",
    )
    parser.add_argument(
        "--max-task-slots",
        type=int,
        default=10,
        help="Maximum persistent task slots stored in the checkpoint state.",
    )
    parser.add_argument("--routing-image-weight", type=float, default=0.5)
    parser.add_argument("--routing-text-weight", type=float, default=0.5)
    parser.add_argument(
        "--routing-history-weight",
        type=float,
        default=0.0,
        help="History is disabled by default because this exporter is anchor-only.",
    )
    parser.add_argument("--routing-temperature", type=float, default=0.1)
    parser.add_argument("--routing-top-k", type=int, default=2)
    parser.add_argument("--routing-min-similarity", type=float, default=-1.0)
    parser.add_argument(
        "--include-self",
        dest="include_self",
        action="store_true",
        default=True,
        help="Include self expert when recomputing each stage relation matrix.",
    )
    parser.add_argument(
        "--exclude-self",
        dest="include_self",
        action="store_false",
        help="Exclude self expert from each row.",
    )
    return parser.parse_args()


def infer_stage_task_count(checkpoint_dir: Path, fallback_count: int):
    match = re.search(r"(?:Task|task)(\d+)", checkpoint_dir.name)
    if match:
        return int(match.group(1))
    return fallback_count


def load_state(checkpoint_dir: Path):
    state_path = checkpoint_dir / "non_lora_trainables.bin"
    if not state_path.exists():
        raise FileNotFoundError(f"Checkpoint state not found: {state_path}")
    return torch.load(state_path, map_location="cpu"), state_path


def load_anchor_series(state_dict, prefix: str, max_task_slots: int):
    values = []
    for slot_idx in range(max_task_slots):
        key = f"{prefix}.{slot_idx}"
        if key not in state_dict:
            break
        values.append(state_dict[key].detach().float().clone())
    if not values:
        raise KeyError(f"No checkpoint entries found for prefix '{prefix}'.")
    return values


def load_boundaries(state_dict, prefix: str, max_task_slots: int):
    values = []
    for slot_idx in range(max_task_slots):
        key = f"{prefix}.{slot_idx}"
        if key not in state_dict:
            break
        values.append(int(round(float(state_dict[key].detach().float().view(-1)[0]))))
    return values


def mask_relation_logits(relation_logits: torch.Tensor, min_similarity: float):
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


def build_sparse_relation_weights(logits: torch.Tensor, top_k: int, temperature: float):
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


def compose_relation_logits(image_scores, text_scores, history_scores, args):
    return (
        float(args.routing_image_weight) * image_scores
        + float(args.routing_text_weight) * text_scores
        + float(args.routing_history_weight) * history_scores
    )


def compute_relation_payload(state_dict, checkpoint_dir: Path, active_task_count: int, args):
    image_anchors = load_anchor_series(state_dict, "base_model.model.image_anchors", args.max_task_slots)
    text_anchors = load_anchor_series(state_dict, "base_model.model.text_anchors", args.max_task_slots)
    image_boundaries = load_boundaries(state_dict, "base_model.model.image_boundary", args.max_task_slots)
    text_boundaries = load_boundaries(state_dict, "base_model.model.text_boundary", args.max_task_slots)

    active_task_count = min(active_task_count, len(image_anchors), len(text_anchors), len(args.task_ids), len(args.task_names))
    if active_task_count <= 0:
        raise ValueError(f"No active tasks detected for checkpoint: {checkpoint_dir}")

    active_image_anchors = torch.stack(
        [F.normalize(anchor.float().squeeze(0), dim=0) for anchor in image_anchors[:active_task_count]],
        dim=0,
    )
    active_text_anchors = torch.stack(
        [F.normalize(anchor.float().squeeze(0), dim=0) for anchor in text_anchors[:active_task_count]],
        dim=0,
    )

    image_scores = torch.matmul(active_image_anchors, active_image_anchors.T)
    text_scores = torch.matmul(active_text_anchors, active_text_anchors.T)
    history_scores = torch.zeros((active_task_count, active_task_count), dtype=torch.float32)

    task_relation_scores = torch.zeros((active_task_count, active_task_count), dtype=torch.float32)
    for task_idx in range(active_task_count):
        candidate_count = active_task_count if args.include_self else task_idx
        if candidate_count <= 0:
            task_relation_scores[task_idx, 0] = 1.0
            continue
        row_logits = compose_relation_logits(
            image_scores[task_idx, :candidate_count],
            text_scores[task_idx, :candidate_count],
            history_scores[task_idx, :candidate_count],
            args,
        )
        row_logits = mask_relation_logits(row_logits, args.routing_min_similarity)
        task_relation_scores[task_idx, :candidate_count] = build_sparse_relation_weights(
            row_logits,
            top_k=min(int(args.routing_top_k), candidate_count),
            temperature=args.routing_temperature,
        )

    expert_usage_prior = task_relation_scores.mean(dim=0)
    task_ids = args.task_ids[:active_task_count]
    task_names = args.task_names[:active_task_count]
    task_sample_counts = [
        max(image_boundaries[idx], text_boundaries[idx]) if idx < len(image_boundaries) and idx < len(text_boundaries) else 0
        for idx in range(active_task_count)
    ]

    return {
        "task_ids": task_ids,
        "task_names": task_names,
        "task_relation_scores": task_relation_scores,
        "expert_usage_prior": expert_usage_prior,
        "task_mean_relation_weights": task_relation_scores.clone(),
        "task_last_relation_weights": task_relation_scores.clone(),
        "task_batch_counts": [1] * active_task_count,
        "task_sample_counts": task_sample_counts,
        "used_indices": [None] * active_task_count,
        "image_anchors": torch.stack(image_anchors[:active_task_count], dim=0),
        "text_anchors": torch.stack(text_anchors[:active_task_count], dim=0),
    }


def save_stage_outputs(output_dir: Path, payload: dict, checkpoint_dir: Path, state_path: Path, active_task_count: int, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "task_ids": payload["task_ids"],
        "task_names": payload["task_names"],
        "task_relation_scores": payload["task_relation_scores"],
        "expert_usage_prior": payload["expert_usage_prior"],
        "task_mean_relation_weights": payload["task_mean_relation_weights"],
        "task_last_relation_weights": payload["task_last_relation_weights"],
        "task_batch_counts": payload["task_batch_counts"],
        "task_sample_counts": payload["task_sample_counts"],
        "used_indices": payload["used_indices"],
        "source_checkpoint": str(checkpoint_dir),
        "source_state_path": str(state_path),
        "routing_config": {
            "routing_image_weight": args.routing_image_weight,
            "routing_text_weight": args.routing_text_weight,
            "routing_history_weight": args.routing_history_weight,
            "routing_temperature": args.routing_temperature,
            "routing_top_k": args.routing_top_k,
            "routing_min_similarity": args.routing_min_similarity,
            "include_self": args.include_self,
            "anchor_only": True,
        },
    }
    torch.save(result, output_dir / "relation_scores.pt")

    meta = {
        "task_ids": payload["task_ids"],
        "task_names": payload["task_names"],
        "active_task_count": active_task_count,
        "source_checkpoint": str(checkpoint_dir),
        "source_state_path": str(state_path),
        "task_sample_counts": payload["task_sample_counts"],
        "routing_config": result["routing_config"],
    }
    with open(output_dir / "meta.json", "w") as handle:
        json.dump(meta, handle, indent=2)


def main():
    args = parse_args()
    if len(args.task_ids) != len(args.task_names):
        raise ValueError("--task-ids and --task-names must have the same length.")

    manifest = []
    for stage_idx, checkpoint_dir in enumerate(args.checkpoint_dirs, start=1):
        state_dict, state_path = load_state(checkpoint_dir)
        active_task_count = infer_stage_task_count(checkpoint_dir, fallback_count=stage_idx)
        payload = compute_relation_payload(state_dict, checkpoint_dir, active_task_count, args)
        stage_task_id = payload["task_ids"][-1]
        output_dir = args.output_root / stage_task_id
        save_stage_outputs(output_dir, payload, checkpoint_dir, state_path, active_task_count, args)
        manifest.append(
            {
                "stage_task_id": stage_task_id,
                "active_task_count": active_task_count,
                "source_checkpoint": str(checkpoint_dir),
                "output_dir": str(output_dir),
            }
        )
        print(f"[saved] {stage_task_id}: {output_dir}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    with open(args.output_root / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
