#!/usr/bin/env python3
import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


ROLE_CONFIG = {
    "routing_image_weight": 0.5,
    "routing_text_weight": 0.5,
    "routing_history_weight": 0.15,
    "routing_temperature": 0.1,
    "routing_min_similarity": -1.0,
    "routing_prior_momentum": 0.8,
    "role_top_k": 2,
    "role_birth_threshold": 0.70,
    "role_assignment_top_k": 1,
    "role_assignment_min_similarity": 0.75,
    "role_assignment_margin": 0.10,
    "role_member_top_k": 2,
    "routing_role_prior_weight": 0.1,
    "routing_role_member_weight": 0.5,
    "routing_role_size_penalty": 0.20,
}


@dataclass
class WorkerJob:
    source_dir: str
    output_root: str
    device: str


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_normalize(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim > 1:
        tensor = tensor.squeeze(0)
    tensor = tensor.float()
    norm = torch.linalg.norm(tensor)
    if not torch.isfinite(norm) or float(norm.item()) <= 0.0:
        return torch.zeros_like(tensor, dtype=torch.float32)
    return F.normalize(tensor, dim=0)


def _compose_relation_logits(
    image_scores: torch.Tensor,
    text_scores: torch.Tensor,
    history_scores: torch.Tensor,
) -> torch.Tensor:
    return (
        float(ROLE_CONFIG["routing_image_weight"]) * image_scores
        + float(ROLE_CONFIG["routing_text_weight"]) * text_scores
        + float(ROLE_CONFIG["routing_history_weight"]) * history_scores
    )


def _mask_relation_logits(relation_logits: torch.Tensor) -> torch.Tensor:
    min_similarity = float(ROLE_CONFIG["routing_min_similarity"])
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


def _role_member_tasks(
    task_role_membership: torch.Tensor,
    role_id: int,
    active_experts: int,
) -> List[int]:
    memberships = task_role_membership[:active_experts, role_id].detach().float()
    return [idx for idx in range(active_experts) if float(memberships[idx].item()) > 0.0]


def _score_tasks(
    image_anchors: Sequence[torch.Tensor],
    text_anchors: Sequence[torch.Tensor],
    expert_usage_prior: torch.Tensor,
    task_indices: Sequence[int],
    image_anchor: torch.Tensor,
    text_anchor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if not task_indices:
        return torch.empty(0, device=device, dtype=torch.float32)
    image_bank = torch.stack(
        [_safe_normalize(image_anchors[idx]).to(device) for idx in task_indices],
        dim=0,
    )
    text_bank = torch.stack(
        [_safe_normalize(text_anchors[idx]).to(device) for idx in task_indices],
        dim=0,
    )
    image_scores = torch.matmul(image_bank, image_anchor.to(device))
    text_scores = torch.matmul(text_bank, text_anchor.to(device))
    history_scores = expert_usage_prior[list(task_indices)].detach().float().to(device)
    return _mask_relation_logits(
        _compose_relation_logits(image_scores, text_scores, history_scores)
    )


def _score_roles(
    image_anchors: Sequence[torch.Tensor],
    text_anchors: Sequence[torch.Tensor],
    expert_logits: torch.Tensor,
    role_image_prototypes: Sequence[torch.Tensor],
    role_text_prototypes: Sequence[torch.Tensor],
    role_usage_prior: torch.Tensor,
    role_task_count: torch.Tensor,
    task_role_membership: torch.Tensor,
    active_roles: int,
    active_experts: int,
    image_anchor: torch.Tensor,
    text_anchor: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if active_roles == 0:
        empty = torch.empty(0, device=device, dtype=torch.float32)
        return empty, empty

    role_image_bank = torch.stack(
        [_safe_normalize(role_image_prototypes[idx]).to(device) for idx in range(active_roles)],
        dim=0,
    )
    role_text_bank = torch.stack(
        [_safe_normalize(role_text_prototypes[idx]).to(device) for idx in range(active_roles)],
        dim=0,
    )
    prototype_scores = (
        float(ROLE_CONFIG["routing_image_weight"]) * torch.matmul(role_image_bank, image_anchor.to(device))
        + float(ROLE_CONFIG["routing_text_weight"]) * torch.matmul(role_text_bank, text_anchor.to(device))
    )

    role_scores: List[torch.Tensor] = []
    member_top_k = int(ROLE_CONFIG["role_member_top_k"])
    member_weight = float(ROLE_CONFIG["routing_role_member_weight"])
    prior_weight = float(ROLE_CONFIG["routing_role_prior_weight"])
    size_penalty = float(ROLE_CONFIG.get("routing_role_size_penalty", 0.0))
    role_priors = role_usage_prior[:active_roles].detach().float().to(device)
    role_sizes = torch.as_tensor(role_task_count[:active_roles], device=device, dtype=torch.float32)

    for role_id in range(active_roles):
        member_tasks = _role_member_tasks(task_role_membership, role_id, active_experts)
        if not member_tasks:
            member_score = torch.tensor(0.0, device=device, dtype=torch.float32)
        else:
            top_k = min(max(member_top_k, 1), len(member_tasks))
            member_values = torch.topk(expert_logits[member_tasks], k=top_k).values
            member_score = member_values.mean()
        role_scores.append(
            prototype_scores[role_id]
            + member_weight * member_score
            + prior_weight * role_priors[role_id]
            - size_penalty * torch.log1p(role_sizes[role_id].clamp_min(0.0))
        )
    return _mask_relation_logits(torch.stack(role_scores, dim=0)), prototype_scores


def _assign_roles_from_sample(
    image_anchors: Sequence[torch.Tensor],
    text_anchors: Sequence[torch.Tensor],
    expert_logits: torch.Tensor,
    role_image_prototypes: Sequence[torch.Tensor],
    role_text_prototypes: Sequence[torch.Tensor],
    role_usage_prior: torch.Tensor,
    role_task_count: torch.Tensor,
    task_role_membership: torch.Tensor,
    active_roles: int,
    active_experts: int,
    image_anchor: torch.Tensor,
    text_anchor: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int], bool]:
    role_scores, prototype_scores = _score_roles(
        image_anchors=image_anchors,
        text_anchors=text_anchors,
        expert_logits=expert_logits,
        role_image_prototypes=role_image_prototypes,
        role_text_prototypes=role_text_prototypes,
        role_usage_prior=role_usage_prior,
        role_task_count=role_task_count,
        task_role_membership=task_role_membership,
        active_roles=active_roles,
        active_experts=active_experts,
        image_anchor=image_anchor,
        text_anchor=text_anchor,
        device=device,
    )
    if role_scores.numel() == 0:
        return torch.empty(0, device=device), [], True
    top_value, top_index = torch.topk(role_scores, k=1)
    best_score = float(top_value[0].item())
    best_role_id = int(top_index[0].item())
    best_prototype_score = float(prototype_scores[best_role_id].item())
    birth_threshold = float(ROLE_CONFIG["role_birth_threshold"])
    prototype_threshold = float(ROLE_CONFIG.get("role_assignment_min_similarity", birth_threshold))
    score_margin = float(ROLE_CONFIG.get("role_assignment_margin", 0.0))

    if best_score < birth_threshold or best_prototype_score < prototype_threshold:
        return torch.empty(0, device=device), [], True

    if role_scores.numel() > 1 and score_margin > 0.0:
        top2_values = torch.topk(role_scores, k=2).values
        if (
            float((top2_values[0] - top2_values[1]).item()) < score_margin
            and best_prototype_score < prototype_threshold + score_margin
        ):
            return torch.empty(0, device=device), [], True

    top_m = min(
        max(1, int(ROLE_CONFIG.get("role_assignment_top_k", 1))),
        role_scores.numel(),
    )
    top_values, top_indices = torch.topk(role_scores, k=top_m)
    if top_m == 1:
        membership = torch.ones(1, device=device, dtype=torch.float32)
    else:
        membership = F.softmax(
            top_values / max(float(ROLE_CONFIG["routing_temperature"]), 1e-6),
            dim=0,
        )
    return membership, top_indices.tolist(), False


def _build_role_state(
    image_anchors: Sequence[torch.Tensor],
    text_anchors: Sequence[torch.Tensor],
    completed_task_count: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    max_task_slots = len(image_anchors)
    max_role_slots = max_task_slots

    expert_usage_prior = torch.zeros(max_task_slots, dtype=torch.float32, device=device)
    role_image_prototypes = [
        torch.zeros_like(image_anchors[0], dtype=image_anchors[0].dtype, device=device)
        for _ in range(max_role_slots)
    ]
    role_text_prototypes = [
        torch.zeros_like(text_anchors[0], dtype=text_anchors[0].dtype, device=device)
        for _ in range(max_role_slots)
    ]
    role_task_count = torch.zeros(max_role_slots, dtype=torch.float32, device=device)
    role_usage_prior = torch.zeros(max_role_slots, dtype=torch.float32, device=device)
    task_role_membership = torch.zeros(
        (max_task_slots, max_role_slots),
        dtype=torch.float32,
        device=device,
    )
    active_role_count = 0

    for task_id in range(completed_task_count):
        task_image_anchor = _safe_normalize(image_anchors[task_id]).to(device)
        task_text_anchor = _safe_normalize(text_anchors[task_id]).to(device)

        if task_id == 0 and active_role_count == 0:
            role_id = 0
            role_image_prototypes[role_id] = task_image_anchor.unsqueeze(0).to(role_image_prototypes[role_id].dtype)
            role_text_prototypes[role_id] = task_text_anchor.unsqueeze(0).to(role_text_prototypes[role_id].dtype)
            role_task_count[role_id] = max(1.0, float(role_task_count[role_id].item()))
            role_usage_prior[role_id] = max(1.0, float(role_usage_prior[role_id].item()))
            task_role_membership[task_id].zero_()
            task_role_membership[task_id, role_id] = 1.0
            active_role_count = 1
            expert_usage_prior[task_id] = max(1.0, float(expert_usage_prior[task_id].item()))
            continue

        expert_logits = _score_tasks(
            image_anchors=image_anchors,
            text_anchors=text_anchors,
            expert_usage_prior=expert_usage_prior,
            task_indices=list(range(task_id)),
            image_anchor=task_image_anchor,
            text_anchor=task_text_anchor,
            device=device,
        )
        membership, candidate_roles, role_birth = _assign_roles_from_sample(
            image_anchors=image_anchors,
            text_anchors=text_anchors,
            expert_logits=expert_logits,
            role_image_prototypes=role_image_prototypes,
            role_text_prototypes=role_text_prototypes,
            role_usage_prior=role_usage_prior,
            role_task_count=role_task_count,
            task_role_membership=task_role_membership,
            active_roles=active_role_count,
            active_experts=task_id,
            image_anchor=task_image_anchor,
            text_anchor=task_text_anchor,
            device=device,
        )

        if active_role_count == 0 or role_birth or not candidate_roles:
            role_id = min(active_role_count, max_role_slots - 1)
            role_image_prototypes[role_id] = task_image_anchor.unsqueeze(0).to(role_image_prototypes[role_id].dtype)
            role_text_prototypes[role_id] = task_text_anchor.unsqueeze(0).to(role_text_prototypes[role_id].dtype)
            role_task_count[role_id] = max(1.0, float(role_task_count[role_id].item()))
            role_usage_prior[role_id] = max(1.0, float(role_usage_prior[role_id].item()))
            task_role_membership[task_id].zero_()
            task_role_membership[task_id, role_id] = 1.0
            active_role_count = max(active_role_count, role_id + 1)
        else:
            momentum = float(ROLE_CONFIG["routing_prior_momentum"])
            task_role_membership[task_id].zero_()
            for local_idx, role_id in enumerate(candidate_roles):
                weight = float(membership[local_idx].detach().item())
                if weight <= 0.0:
                    continue
                count = float(role_task_count[role_id].detach().item())
                updated_count = count + weight
                updated_image = (
                    count * _safe_normalize(role_image_prototypes[role_id].detach())
                    + weight * task_image_anchor
                ) / max(updated_count, 1e-6)
                updated_text = (
                    count * _safe_normalize(role_text_prototypes[role_id].detach())
                    + weight * task_text_anchor
                ) / max(updated_count, 1e-6)
                role_image_prototypes[role_id] = updated_image.unsqueeze(0).to(role_image_prototypes[role_id].dtype)
                role_text_prototypes[role_id] = updated_text.unsqueeze(0).to(role_text_prototypes[role_id].dtype)
                role_task_count[role_id] = updated_count
                role_usage_prior[role_id] = (
                    momentum * role_usage_prior[role_id].detach().float()
                    + (1.0 - momentum) * weight
                )
                task_role_membership[task_id, role_id] = weight

        expert_usage_prior[task_id] = max(1.0, float(expert_usage_prior[task_id].item()))

    return {
        "expert_usage_prior": expert_usage_prior.cpu(),
        "role_image_prototypes": [tensor.cpu() for tensor in role_image_prototypes],
        "role_text_prototypes": [tensor.cpu() for tensor in role_text_prototypes],
        "role_task_count": role_task_count.cpu(),
        "role_usage_prior": role_usage_prior.cpu(),
        "task_role_membership": task_role_membership.cpu(),
        "active_role_count": torch.tensor([float(active_role_count)], dtype=torch.float32),
    }


def _detect_prefix(state_dict: Dict[str, torch.Tensor]) -> str:
    for key in state_dict:
        if key.endswith("image_anchors.0"):
            return key[: -len("image_anchors.0")]
    raise KeyError("Failed to locate anchor prefix in non_lora_trainables.bin")


def _collect_anchor_series(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    name: str,
) -> List[torch.Tensor]:
    series: List[Tuple[int, torch.Tensor]] = []
    for key, value in state_dict.items():
        if not key.startswith(prefix + name + "."):
            continue
        index = int(key.rsplit(".", 1)[1])
        series.append((index, value))
    if not series:
        raise KeyError(f"Missing {name} entries in non_lora_trainables.bin")
    series.sort(key=lambda item: item[0])
    return [value for _, value in series]


def _read_completed_task_count(checkpoint_dir: Path) -> int:
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if adapter_config_path.is_file():
        with adapter_config_path.open("r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        if "cur_task" in adapter_config:
            return int(adapter_config["cur_task"]) + 1

    stem = checkpoint_dir.name
    if stem.startswith("Task"):
        digits = []
        for ch in stem[4:]:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if digits:
            return int("".join(digits))
    raise ValueError(f"Failed to infer completed task count from {checkpoint_dir}")


def _copy_checkpoint_dir(source_dir: Path, target_dir: Path) -> None:
    shutil.copytree(source_dir, target_dir, symlinks=False)


def _convert_one(job: WorkerJob) -> Dict[str, object]:
    source_dir = Path(job.source_dir).resolve()
    target_dir = Path(job.output_root).resolve() / source_dir.name
    if target_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint dir: {target_dir}")

    device = torch.device(job.device)
    if device.type == "cuda":
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

    _copy_checkpoint_dir(source_dir, target_dir)

    non_lora_path = target_dir / "non_lora_trainables.bin"
    state_dict = torch.load(non_lora_path, map_location="cpu")
    prefix = _detect_prefix(state_dict)
    image_anchors = _collect_anchor_series(state_dict, prefix, "image_anchors")
    text_anchors = _collect_anchor_series(state_dict, prefix, "text_anchors")
    completed_task_count = _read_completed_task_count(source_dir)

    role_state = _build_role_state(
        image_anchors=image_anchors,
        text_anchors=text_anchors,
        completed_task_count=completed_task_count,
        device=device,
    )

    state_dict[prefix + "expert_usage_prior"] = role_state["expert_usage_prior"]
    state_dict[prefix + "role_task_count"] = role_state["role_task_count"]
    state_dict[prefix + "role_usage_prior"] = role_state["role_usage_prior"]
    state_dict[prefix + "task_role_membership"] = role_state["task_role_membership"]
    state_dict[prefix + "active_role_count"] = role_state["active_role_count"]

    for idx, tensor in enumerate(role_state["role_image_prototypes"]):
        state_dict[f"{prefix}role_image_prototypes.{idx}"] = tensor
    for idx, tensor in enumerate(role_state["role_text_prototypes"]):
        state_dict[f"{prefix}role_text_prototypes.{idx}"] = tensor

    torch.save(state_dict, non_lora_path)

    active_role_count = int(role_state["active_role_count"].item())
    summary = {
        "checkpoint_name": source_dir.name,
        "source_dir": str(source_dir),
        "output_dir": str(target_dir),
        "completed_task_count": completed_task_count,
        "active_role_count": active_role_count,
        "device": str(device),
        "converted_at_utc": _timestamp(),
    }
    with (target_dir / "hidesc_role_conversion_meta.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _chunk_jobs(checkpoint_dirs: Sequence[Path], gpu_ids: Sequence[str]) -> List[List[WorkerJob]]:
    worker_count = max(1, len(gpu_ids))
    chunks: List[List[WorkerJob]] = [[] for _ in range(worker_count)]
    for idx, checkpoint_dir in enumerate(checkpoint_dirs):
        worker_idx = idx % worker_count
        chunks[worker_idx].append(
            WorkerJob(
                source_dir=str(checkpoint_dir),
                output_root="",
                device=f"cuda:{gpu_ids[worker_idx]}" if gpu_ids[worker_idx] else "cpu",
            )
        )
    return chunks


def _worker_convert(jobs: Sequence[WorkerJob]) -> List[Dict[str, object]]:
    results = []
    for job in jobs:
        results.append(_convert_one(job))
    return results


def _discover_checkpoint_dirs(source_root: Path, checkpoint_names: Sequence[str]) -> List[Path]:
    if checkpoint_names:
        checkpoint_dirs = [source_root / name for name in checkpoint_names]
    else:
        checkpoint_dirs = sorted(
            path
            for path in source_root.iterdir()
            if path.is_dir() and (path / "non_lora_trainables.bin").is_file()
        )
    missing = [str(path) for path in checkpoint_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint directories: {missing}")
    return checkpoint_dirs


def _copy_description_caches(source_root: Path, output_root: Path) -> None:
    source_cache_dir = source_root / "description_caches"
    if not source_cache_dir.is_dir():
        return
    target_cache_dir = output_root / "description_caches"
    if target_cache_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache dir: {target_cache_dir}")
    shutil.copytree(source_cache_dir, target_cache_dir, symlinks=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert HiDeCL checkpoint directories into HiDESC-compatible checkpoints "
            "by reconstructing role memory from saved anchors."
        )
    )
    parser.add_argument("--source-root", required=True, help="Root directory containing HiDeCL task checkpoints.")
    parser.add_argument("--output-root", required=True, help="New root directory to write converted HiDESC checkpoints.")
    parser.add_argument(
        "--gpu-ids",
        default="0,1",
        help="Comma-separated CUDA device ids used for parallel conversion workers. Use cpu for CPU-only conversion.",
    )
    parser.add_argument(
        "--checkpoint-names",
        nargs="*",
        default=[],
        help="Optional checkpoint directory names to convert. Defaults to all task checkpoints under source root.",
    )
    parser.add_argument(
        "--copy-description-caches",
        action="store_true",
        help="Also copy source_root/description_caches into the new output root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse existing output root: {output_root}")

    checkpoint_dirs = _discover_checkpoint_dirs(source_root, args.checkpoint_names)
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found under {source_root}")

    gpu_tokens = [token.strip() for token in args.gpu_ids.split(",") if token.strip()]
    if not gpu_tokens:
        gpu_tokens = ["cpu"]
    if len(gpu_tokens) == 1 and gpu_tokens[0].lower() == "cpu":
        gpu_tokens = [""]

    output_root.mkdir(parents=True, exist_ok=False)

    job_chunks = _chunk_jobs(checkpoint_dirs, gpu_tokens)
    for chunk in job_chunks:
        for idx, job in enumerate(chunk):
            chunk[idx] = WorkerJob(
                source_dir=job.source_dir,
                output_root=str(output_root),
                device=job.device,
            )

    results: List[Dict[str, object]] = []
    if len(job_chunks) == 1:
        results.extend(_worker_convert(job_chunks[0]))
    else:
        max_workers = len(job_chunks)
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            for worker_result in executor.map(_worker_convert, job_chunks):
                results.extend(worker_result)

    if args.copy_description_caches:
        _copy_description_caches(source_root, output_root)

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "converted_at_utc": _timestamp(),
        "gpu_ids": gpu_tokens,
        "copied_description_caches": bool(args.copy_description_caches),
        "checkpoint_count": len(results),
        "checkpoints": sorted(results, key=lambda item: item["checkpoint_name"]),
    }
    with (output_root / "hidesc_conversion_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
