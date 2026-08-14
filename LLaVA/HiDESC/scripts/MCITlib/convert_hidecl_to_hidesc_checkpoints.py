#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

LEGACY_STATE_KEYS = (
    "image_anchors.",
    "image_boundary.",
    "role_image_prototypes.",
)


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
    tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
    norm = torch.linalg.norm(tensor)
    if not torch.isfinite(norm) or float(norm.item()) <= 0.0:
        return torch.zeros_like(tensor, dtype=torch.float32)
    return F.normalize(tensor, dim=0)


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    mean = scores.mean()
    std = scores.std(unbiased=False)
    return (scores - mean) / std.clamp_min(1e-6)


def _compose_relation_logits(
    spectral_scores: torch.Tensor,
    text_scores: torch.Tensor,
    history_scores: torch.Tensor,
) -> torch.Tensor:
    logits = (
        float(ROLE_CONFIG["routing_image_weight"]) * spectral_scores
        + float(ROLE_CONFIG["routing_text_weight"]) * text_scores
        + float(ROLE_CONFIG["routing_history_weight"]) * history_scores
    )
    min_similarity = float(ROLE_CONFIG["routing_min_similarity"])
    if min_similarity > -1.0:
        logits = torch.where(
            logits >= min_similarity,
            logits,
            torch.full_like(logits, float("-inf")),
        )
    if not torch.isfinite(logits).any():
        return spectral_scores
    return logits


def _role_member_tasks(
    task_role_membership: torch.Tensor,
    role_id: int,
    active_experts: int,
) -> List[int]:
    membership = torch.nan_to_num(
        task_role_membership[:active_experts, role_id].detach().float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return [idx for idx in range(active_experts) if float(membership[idx].item()) > 0.0]


def _score_tasks(
    spectral_image_anchors: Sequence[torch.Tensor],
    text_anchors: Sequence[torch.Tensor],
    expert_usage_prior: torch.Tensor,
    task_indices: Sequence[int],
    spectral_anchor: torch.Tensor,
    text_anchor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if not task_indices:
        return torch.empty(0, device=device, dtype=torch.float32)
    spectral_bank = torch.stack(
        [_safe_normalize(spectral_image_anchors[idx]).to(device) for idx in task_indices],
        dim=0,
    )
    text_bank = torch.stack(
        [_safe_normalize(text_anchors[idx]).to(device) for idx in task_indices],
        dim=0,
    )
    spectral_scores = _normalize_scores(torch.matmul(spectral_bank, spectral_anchor.to(device)))
    text_scores = _normalize_scores(torch.matmul(text_bank, text_anchor.to(device)))
    history_scores = _normalize_scores(
        expert_usage_prior[list(task_indices)].detach().float().to(device)
    )
    return _compose_relation_logits(spectral_scores, text_scores, history_scores)


def _score_roles(
    role_spectral_prototypes: Sequence[torch.Tensor],
    role_text_prototypes: Sequence[torch.Tensor],
    role_usage_prior: torch.Tensor,
    role_task_count: torch.Tensor,
    task_role_membership: torch.Tensor,
    expert_logits: torch.Tensor,
    active_roles: int,
    active_experts: int,
    spectral_anchor: torch.Tensor,
    text_anchor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if active_roles == 0:
        return torch.empty(0, device=device, dtype=torch.float32)
    role_spectral_bank = torch.stack(
        [_safe_normalize(role_spectral_prototypes[idx]).to(device) for idx in range(active_roles)],
        dim=0,
    )
    role_text_bank = torch.stack(
        [_safe_normalize(role_text_prototypes[idx]).to(device) for idx in range(active_roles)],
        dim=0,
    )
    prototype_scores = (
        float(ROLE_CONFIG["routing_image_weight"])
        * torch.matmul(role_spectral_bank, spectral_anchor.to(device))
        + float(ROLE_CONFIG["routing_text_weight"])
        * torch.matmul(role_text_bank, text_anchor.to(device))
    )
    member_top_k = int(ROLE_CONFIG["role_member_top_k"])
    member_weight = float(ROLE_CONFIG["routing_role_member_weight"])
    prior_weight = float(ROLE_CONFIG["routing_role_prior_weight"])
    size_penalty = float(ROLE_CONFIG["routing_role_size_penalty"])
    role_priors = role_usage_prior[:active_roles].detach().float().to(device)
    role_sizes = role_task_count[:active_roles].detach().float().to(device)
    scores = []
    for role_id in range(active_roles):
        member_tasks = _role_member_tasks(task_role_membership, role_id, active_experts)
        if member_tasks:
            top_k = min(max(member_top_k, 1), len(member_tasks))
            member_score = torch.topk(expert_logits[member_tasks], k=top_k).values.mean()
        else:
            member_score = torch.tensor(0.0, device=device, dtype=torch.float32)
        scores.append(
            prototype_scores[role_id]
            + member_weight * member_score
            + prior_weight * role_priors[role_id]
            - size_penalty * torch.log1p(role_sizes[role_id].clamp_min(0.0))
        )
    return torch.stack(scores, dim=0)


def _assign_roles_from_sample(
    role_spectral_prototypes: Sequence[torch.Tensor],
    role_text_prototypes: Sequence[torch.Tensor],
    role_usage_prior: torch.Tensor,
    role_task_count: torch.Tensor,
    task_role_membership: torch.Tensor,
    expert_logits: torch.Tensor,
    active_roles: int,
    active_experts: int,
    spectral_anchor: torch.Tensor,
    text_anchor: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int], bool]:
    role_scores = _score_roles(
        role_spectral_prototypes=role_spectral_prototypes,
        role_text_prototypes=role_text_prototypes,
        role_usage_prior=role_usage_prior,
        role_task_count=role_task_count,
        task_role_membership=task_role_membership,
        expert_logits=expert_logits,
        active_roles=active_roles,
        active_experts=active_experts,
        spectral_anchor=spectral_anchor,
        text_anchor=text_anchor,
        device=device,
    )
    if role_scores.numel() == 0:
        return torch.empty(0, device=device, dtype=torch.float32), [], True
    best_score = float(torch.topk(role_scores, k=1).values[0].item())
    if best_score < float(ROLE_CONFIG["role_birth_threshold"]):
        return torch.empty(0, device=device, dtype=torch.float32), [], True
    top_m = min(max(1, int(ROLE_CONFIG["role_assignment_top_k"])), role_scores.numel())
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
    spectral_image_anchors: Sequence[torch.Tensor],
    text_anchors: Sequence[torch.Tensor],
    completed_task_count: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    max_task_slots = len(spectral_image_anchors)
    max_role_slots = max_task_slots
    expert_usage_prior = torch.zeros(max_task_slots, dtype=torch.float32, device=device)
    role_spectral_prototypes = [
        torch.zeros_like(spectral_image_anchors[0], dtype=spectral_image_anchors[0].dtype, device=device)
        for _ in range(max_role_slots)
    ]
    role_text_prototypes = [
        torch.zeros_like(text_anchors[0], dtype=text_anchors[0].dtype, device=device)
        for _ in range(max_role_slots)
    ]
    role_task_count = torch.zeros(max_role_slots, dtype=torch.float32, device=device)
    role_usage_prior = torch.zeros(max_role_slots, dtype=torch.float32, device=device)
    task_role_membership = torch.zeros(
        (max_task_slots, max_role_slots), dtype=torch.float32, device=device
    )
    active_role_count = 0

    for task_id in range(min(completed_task_count, max_task_slots)):
        task_spectral_anchor = _safe_normalize(spectral_image_anchors[task_id]).to(device)
        task_text_anchor = _safe_normalize(text_anchors[task_id]).to(device)
        if task_id == 0 and active_role_count == 0:
            role_spectral_prototypes[0] = task_spectral_anchor.unsqueeze(0).to(role_spectral_prototypes[0].dtype)
            role_text_prototypes[0] = task_text_anchor.unsqueeze(0).to(role_text_prototypes[0].dtype)
            role_task_count[0] = 1.0
            role_usage_prior[0] = 1.0
            task_role_membership[task_id, 0] = 1.0
            expert_usage_prior[task_id] = 1.0
            active_role_count = 1
            continue

        expert_logits = _score_tasks(
            spectral_image_anchors=spectral_image_anchors,
            text_anchors=text_anchors,
            expert_usage_prior=expert_usage_prior,
            task_indices=list(range(task_id)),
            spectral_anchor=task_spectral_anchor,
            text_anchor=task_text_anchor,
            device=device,
        )
        membership, candidate_roles, role_birth = _assign_roles_from_sample(
            role_spectral_prototypes=role_spectral_prototypes,
            role_text_prototypes=role_text_prototypes,
            role_usage_prior=role_usage_prior,
            role_task_count=role_task_count,
            task_role_membership=task_role_membership,
            expert_logits=expert_logits,
            active_roles=active_role_count,
            active_experts=task_id,
            spectral_anchor=task_spectral_anchor,
            text_anchor=task_text_anchor,
            device=device,
        )

        if active_role_count == 0 or role_birth or not candidate_roles:
            role_id = min(active_role_count, max_role_slots - 1)
            role_spectral_prototypes[role_id] = task_spectral_anchor.unsqueeze(0).to(role_spectral_prototypes[role_id].dtype)
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
                weight = float(membership[local_idx].item())
                if weight <= 0.0:
                    continue
                count = float(role_task_count[role_id].item())
                updated_count = count + weight
                updated_spectral = (
                    count * _safe_normalize(role_spectral_prototypes[role_id])
                    + weight * task_spectral_anchor
                ) / max(updated_count, 1e-6)
                updated_text = (
                    count * _safe_normalize(role_text_prototypes[role_id])
                    + weight * task_text_anchor
                ) / max(updated_count, 1e-6)
                role_spectral_prototypes[role_id] = updated_spectral.unsqueeze(0).to(role_spectral_prototypes[role_id].dtype)
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
        "role_spectral_prototypes": [tensor.cpu() for tensor in role_spectral_prototypes],
        "role_text_prototypes": [tensor.cpu() for tensor in role_text_prototypes],
        "role_task_count": role_task_count.cpu(),
        "role_usage_prior": role_usage_prior.cpu(),
        "task_role_membership": task_role_membership.cpu(),
        "active_role_count": torch.tensor([float(active_role_count)], dtype=torch.float32),
    }


def _detect_prefix(state_dict: Dict[str, torch.Tensor]) -> str:
    candidates = ("spectral_image_anchors.0", "text_anchors.0", "image_anchors.0")
    for suffix in candidates:
        for key in state_dict:
            if key.endswith(suffix):
                return key[: -len(suffix)]
    raise KeyError("Failed to locate HiDESC state prefix in non_lora_trainables.bin")


def _collect_series(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    name: str,
) -> List[torch.Tensor]:
    series: List[Tuple[int, torch.Tensor]] = []
    for key, value in state_dict.items():
        if key.startswith(prefix + name + "."):
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


def _purge_legacy_image_state(state_dict: Dict[str, torch.Tensor], prefix: str) -> None:
    remove_keys = []
    for key in state_dict:
        for suffix in LEGACY_STATE_KEYS:
            if key.startswith(prefix + suffix):
                remove_keys.append(key)
                break
    for key in remove_keys:
        state_dict.pop(key, None)


def _convert_one(job: WorkerJob) -> Dict[str, object]:
    source_dir = Path(job.source_dir).resolve()
    target_dir = Path(job.output_root).resolve() / source_dir.name
    if target_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint dir: {target_dir}")

    device = torch.device(job.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    _copy_checkpoint_dir(source_dir, target_dir)
    non_lora_path = target_dir / "non_lora_trainables.bin"
    state_dict = torch.load(non_lora_path, map_location="cpu")
    prefix = _detect_prefix(state_dict)
    spectral_image_anchors = _collect_series(state_dict, prefix, "spectral_image_anchors")
    text_anchors = _collect_series(state_dict, prefix, "text_anchors")
    completed_task_count = _read_completed_task_count(source_dir)

    role_state = _build_role_state(
        spectral_image_anchors=spectral_image_anchors,
        text_anchors=text_anchors,
        completed_task_count=completed_task_count,
        device=device,
    )
    _purge_legacy_image_state(state_dict, prefix)
    state_dict[prefix + "expert_usage_prior"] = role_state["expert_usage_prior"]
    state_dict[prefix + "role_task_count"] = role_state["role_task_count"]
    state_dict[prefix + "role_usage_prior"] = role_state["role_usage_prior"]
    state_dict[prefix + "task_role_membership"] = role_state["task_role_membership"]
    state_dict[prefix + "active_role_count"] = role_state["active_role_count"]
    for idx, tensor in enumerate(role_state["role_spectral_prototypes"]):
        state_dict[f"{prefix}role_spectral_prototypes.{idx}"] = tensor
    for idx, tensor in enumerate(role_state["role_text_prototypes"]):
        state_dict[f"{prefix}role_text_prototypes.{idx}"] = tensor
    torch.save(state_dict, non_lora_path)

    summary = {
        "checkpoint_name": source_dir.name,
        "source_dir": str(source_dir),
        "output_dir": str(target_dir),
        "completed_task_count": completed_task_count,
        "active_role_count": int(role_state["active_role_count"].item()),
        "device": str(device),
        "converted_at_utc": _timestamp(),
        "conversion_mode": "spectral_only",
    }
    with (target_dir / "hidesc_conversion_meta.json").open("w", encoding="utf-8") as f:
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
    return [_convert_one(job) for job in jobs]


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
            "Convert HiDeCL checkpoint directories into clean HiDESC checkpoints "
            "using spectral image anchors only."
        )
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--checkpoint-names", nargs="*", default=[])
    parser.add_argument("--copy-description-caches", action="store_true")
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
        gpu_tokens = [""]
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
        with ProcessPoolExecutor(
            max_workers=len(job_chunks),
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
        "conversion_mode": "spectral_only",
        "checkpoints": sorted(results, key=lambda item: item["checkpoint_name"]),
    }
    with (output_root / "hidesc_conversion_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
