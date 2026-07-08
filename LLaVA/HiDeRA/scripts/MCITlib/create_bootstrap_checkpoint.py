#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Create a bootstrap checkpoint by injecting anchor/relation caches.")
    parser.add_argument("--source-checkpoint", type=Path, required=True, help="Source HiDE/HiDeRA checkpoint directory.")
    parser.add_argument("--anchor-cache", type=Path, required=True, help="Path to anchor cache directory or anchors.pt.")
    parser.add_argument("--relation-cache", type=Path, required=True, help="Path to relation cache directory or relation_scores.pt.")
    parser.add_argument("--output-checkpoint", type=Path, required=True, help="Output checkpoint directory.")
    parser.add_argument("--max-task-slots", type=int, default=10, help="Number of persistent task slots in model state.")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it exists.")
    return parser.parse_args()


def resolve_cache_file(path: Path, filename: str):
    if path.is_dir():
        return path / filename
    return path


def ensure_file(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def load_payload(path: Path):
    return torch.load(path, map_location="cpu")


def pad_relation_tensor(source_tensor: torch.Tensor, target_shape):
    padded = torch.zeros(target_shape, dtype=source_tensor.dtype)
    slices = tuple(slice(0, dim) for dim in source_tensor.shape)
    padded[slices] = source_tensor
    return padded


def update_slot_series(state_dict, prefix, values, max_slots):
    for slot_idx in range(max_slots):
        key = f"{prefix}.{slot_idx}"
        if key not in state_dict:
            continue
        target_tensor = state_dict[key]
        if slot_idx < len(values):
            replacement = values[slot_idx].to(dtype=target_tensor.dtype)
            if tuple(replacement.shape) != tuple(target_tensor.shape):
                replacement = replacement.reshape(target_tensor.shape)
            state_dict[key] = replacement.clone()


def main():
    args = parse_args()
    source_ckpt = args.source_checkpoint
    output_ckpt = args.output_checkpoint
    anchor_cache = resolve_cache_file(args.anchor_cache, "anchors.pt")
    relation_cache = resolve_cache_file(args.relation_cache, "relation_scores.pt")

    ensure_file(source_ckpt / "non_lora_trainables.bin", "Source non_lora_trainables.bin")
    ensure_file(source_ckpt / "adapter_model.bin", "Source adapter_model.bin")
    ensure_file(anchor_cache, "Anchor cache")
    ensure_file(relation_cache, "Relation cache")

    if output_ckpt.exists():
        if not args.force:
            raise FileExistsError(f"Output checkpoint already exists: {output_ckpt}. Use --force to overwrite.")
        shutil.rmtree(output_ckpt)

    shutil.copytree(source_ckpt, output_ckpt)

    state_path = output_ckpt / "non_lora_trainables.bin"
    state_dict = torch.load(state_path, map_location="cpu")
    anchor_payload = load_payload(anchor_cache)
    relation_payload = load_payload(relation_cache)

    update_slot_series(
        state_dict,
        "base_model.model.image_anchors",
        list(anchor_payload["image_anchors"]),
        args.max_task_slots,
    )
    update_slot_series(
        state_dict,
        "base_model.model.text_anchors",
        list(anchor_payload["text_anchors"]),
        args.max_task_slots,
    )
    update_slot_series(
        state_dict,
        "base_model.model.image_boundary",
        list(anchor_payload["image_boundary"]),
        args.max_task_slots,
    )
    update_slot_series(
        state_dict,
        "base_model.model.text_boundary",
        list(anchor_payload["text_boundary"]),
        args.max_task_slots,
    )

    relation_key = "base_model.model.task_relation_scores"
    if relation_key in state_dict:
        state_dict[relation_key] = pad_relation_tensor(
            relation_payload["task_relation_scores"],
            state_dict[relation_key].shape,
        ).to(dtype=state_dict[relation_key].dtype)
    else:
        state_dict[relation_key] = pad_relation_tensor(
            relation_payload["task_relation_scores"],
            (args.max_task_slots, args.max_task_slots),
        )

    prior_key = "base_model.model.expert_usage_prior"
    if prior_key in state_dict:
        state_dict[prior_key] = pad_relation_tensor(
            relation_payload["expert_usage_prior"],
            state_dict[prior_key].shape,
        ).to(dtype=state_dict[prior_key].dtype)
    else:
        state_dict[prior_key] = pad_relation_tensor(
            relation_payload["expert_usage_prior"],
            (args.max_task_slots,),
        )

    torch.save(state_dict, state_path)

    bootstrap_meta = {
        "source_checkpoint": str(source_ckpt),
        "anchor_cache": str(anchor_cache),
        "relation_cache": str(relation_cache),
        "output_checkpoint": str(output_ckpt),
        "task_ids": anchor_payload.get("task_ids", []),
        "task_names": anchor_payload.get("task_names", []),
    }
    with open(output_ckpt / "bootstrap_meta.json", "w") as handle:
        json.dump(bootstrap_meta, handle, indent=2)

    print(f"Bootstrap checkpoint created at {output_ckpt}")


if __name__ == "__main__":
    main()
