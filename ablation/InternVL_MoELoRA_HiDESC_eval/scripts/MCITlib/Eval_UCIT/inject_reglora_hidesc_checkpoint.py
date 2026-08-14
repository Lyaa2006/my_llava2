#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


ROLE_CONFIG = {
    "routing_image_weight": 0.5,
    "routing_text_weight": 0.5,
    "routing_history_weight": 0.0,
    "routing_temperature": 0.1,
    "routing_min_similarity": -1.0,
    "routing_prior_momentum": 0.8,
    "role_birth_threshold": 0.55,
    "role_assignment_top_k": 1,
    "role_member_top_k": 2,
    "routing_role_prior_weight": 0.0,
    "routing_role_member_weight": 0.2,
    "routing_role_size_penalty": 0.30,
}


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
    image_scores: torch.Tensor,
    text_scores: torch.Tensor,
    history_scores: torch.Tensor,
) -> torch.Tensor:
    logits = (
        float(ROLE_CONFIG["routing_image_weight"]) * image_scores
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
        return image_scores
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
    image_scores = _normalize_scores(torch.matmul(image_bank, image_anchor.to(device)))
    text_scores = _normalize_scores(torch.matmul(text_bank, text_anchor.to(device)))
    history_scores = _normalize_scores(
        expert_usage_prior[list(task_indices)].detach().float().to(device)
    )
    return _compose_relation_logits(image_scores, text_scores, history_scores)


def _score_roles(
    role_image_prototypes: Sequence[torch.Tensor],
    role_text_prototypes: Sequence[torch.Tensor],
    role_usage_prior: torch.Tensor,
    role_task_count: torch.Tensor,
    task_role_membership: torch.Tensor,
    expert_logits: torch.Tensor,
    active_roles: int,
    active_experts: int,
    image_anchor: torch.Tensor,
    text_anchor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if active_roles == 0:
        return torch.empty(0, device=device, dtype=torch.float32)
    role_image_bank = torch.stack(
        [_safe_normalize(role_image_prototypes[idx]).to(device) for idx in range(active_roles)],
        dim=0,
    )
    role_text_bank = torch.stack(
        [_safe_normalize(role_text_prototypes[idx]).to(device) for idx in range(active_roles)],
        dim=0,
    )
    prototype_scores = (
        float(ROLE_CONFIG["routing_image_weight"])
        * torch.matmul(role_image_bank, image_anchor.to(device))
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


def _assign_roles(
    role_image_prototypes: Sequence[torch.Tensor],
    role_text_prototypes: Sequence[torch.Tensor],
    role_usage_prior: torch.Tensor,
    role_task_count: torch.Tensor,
    task_role_membership: torch.Tensor,
    expert_logits: torch.Tensor,
    active_roles: int,
    active_experts: int,
    image_anchor: torch.Tensor,
    text_anchor: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int], bool]:
    role_scores = _score_roles(
        role_image_prototypes=role_image_prototypes,
        role_text_prototypes=role_text_prototypes,
        role_usage_prior=role_usage_prior,
        role_task_count=role_task_count,
        task_role_membership=task_role_membership,
        expert_logits=expert_logits,
        active_roles=active_roles,
        active_experts=active_experts,
        image_anchor=image_anchor,
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
        max_task_slots, max_role_slots, dtype=torch.float32, device=device
    )
    active_role_count = 0

    for task_id in range(completed_task_count):
        task_image_anchor = _safe_normalize(image_anchors[task_id]).to(device)
        task_text_anchor = _safe_normalize(text_anchors[task_id]).to(device)
        if task_id == 0:
            role_image_prototypes[0] = task_image_anchor.unsqueeze(0).to(role_image_prototypes[0].dtype)
            role_text_prototypes[0] = task_text_anchor.unsqueeze(0).to(role_text_prototypes[0].dtype)
            role_task_count[0] = 1.0
            role_usage_prior[0] = 1.0
            task_role_membership[task_id, 0] = 1.0
            expert_usage_prior[task_id] = 1.0
            active_role_count = 1
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
        membership, candidate_roles, role_birth = _assign_roles(
            role_image_prototypes=role_image_prototypes,
            role_text_prototypes=role_text_prototypes,
            role_usage_prior=role_usage_prior,
            role_task_count=role_task_count,
            task_role_membership=task_role_membership,
            expert_logits=expert_logits,
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
                weight = float(membership[local_idx].item())
                if weight <= 0.0:
                    continue
                count = float(role_task_count[role_id].item())
                updated_count = count + weight
                updated_image = (
                    count * _safe_normalize(role_image_prototypes[role_id])
                    + weight * task_image_anchor
                ) / max(updated_count, 1e-6)
                updated_text = (
                    count * _safe_normalize(role_text_prototypes[role_id])
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


def _resolve_output_dir(args: argparse.Namespace, checkpoint_dir: Path) -> Path:
    if args.inplace:
        return checkpoint_dir
    if args.output_dir is None:
        raise ValueError("Provide --inplace or --output-dir.")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and args.fail_if_present:
        raise FileExistsError(f"Refusing to overwrite existing output dir: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(checkpoint_dir, output_dir, symlinks=False)
    return output_dir


def _write_adapter_compat_config(adapter_config_path: Path) -> Dict:
    with adapter_config_path.open("r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    adapter_config["peft_type"] = "MOE_LORA_HiDe"
    adapter_config["task_type"] = "CAUSAL_LM_HiDe"
    adapter_config.setdefault("fan_in_fan_out", False)
    adapter_config.setdefault("init_lora_weights", True)
    adapter_config.setdefault("layers_pattern", None)
    adapter_config.setdefault("layers_to_transform", None)
    adapter_config.setdefault("modules_to_save", None)
    adapter_config.setdefault("revision", None)
    with adapter_config_path.open("w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)
    return adapter_config


def _load_anchor_bundle(anchor_cache_path: Path) -> Dict[str, torch.Tensor]:
    bundle = torch.load(anchor_cache_path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise TypeError(f"Anchor cache must be a dict: {anchor_cache_path}")
    if "image_anchors" not in bundle or "text_anchors" not in bundle:
        raise KeyError(
            f"Anchor cache must contain image_anchors/text_anchors: {anchor_cache_path}"
        )
    return bundle


def _apply_role_config_overrides(args: argparse.Namespace) -> Dict[str, float]:
    overrides = {
        "routing_image_weight": args.routing_image_weight,
        "routing_text_weight": args.routing_text_weight,
        "routing_history_weight": args.routing_history_weight,
        "routing_temperature": args.routing_temperature,
        "routing_min_similarity": args.routing_min_similarity,
        "routing_prior_momentum": args.routing_prior_momentum,
        "role_birth_threshold": args.role_birth_threshold,
        "role_assignment_top_k": args.role_assignment_top_k,
        "role_member_top_k": args.role_member_top_k,
        "routing_role_prior_weight": args.routing_role_prior_weight,
        "routing_role_member_weight": args.routing_role_member_weight,
        "routing_role_size_penalty": args.routing_role_size_penalty,
    }
    ROLE_CONFIG.update(overrides)
    return dict(ROLE_CONFIG)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inject HiDESC-compatible anchor and role memory tensors into a "
            "copied RegLoRA checkpoint directory."
        )
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--anchor-cache", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--fail-if-present", action="store_true")
    parser.add_argument(
        "--state-prefix",
        default="base_model.model.",
        help=(
            "Prefix used when writing HiDESC tensors into non_lora_trainables.bin. "
            "Defaults to the prefix used by existing HiDESC UCIT checkpoints."
        ),
    )
    parser.add_argument("--routing-image-weight", type=float, default=ROLE_CONFIG["routing_image_weight"])
    parser.add_argument("--routing-text-weight", type=float, default=ROLE_CONFIG["routing_text_weight"])
    parser.add_argument("--routing-history-weight", type=float, default=ROLE_CONFIG["routing_history_weight"])
    parser.add_argument("--routing-temperature", type=float, default=ROLE_CONFIG["routing_temperature"])
    parser.add_argument("--routing-min-similarity", type=float, default=ROLE_CONFIG["routing_min_similarity"])
    parser.add_argument("--routing-prior-momentum", type=float, default=ROLE_CONFIG["routing_prior_momentum"])
    parser.add_argument("--role-birth-threshold", type=float, default=ROLE_CONFIG["role_birth_threshold"])
    parser.add_argument("--role-assignment-top-k", type=int, default=ROLE_CONFIG["role_assignment_top_k"])
    parser.add_argument("--role-member-top-k", type=int, default=ROLE_CONFIG["role_member_top_k"])
    parser.add_argument("--routing-role-prior-weight", type=float, default=ROLE_CONFIG["routing_role_prior_weight"])
    parser.add_argument("--routing-role-member-weight", type=float, default=ROLE_CONFIG["routing_role_member_weight"])
    parser.add_argument("--routing-role-size-penalty", type=float, default=ROLE_CONFIG["routing_role_size_penalty"])
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    anchor_cache_path = Path(args.anchor_cache).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir does not exist: {checkpoint_dir}")
    if not anchor_cache_path.is_file():
        raise FileNotFoundError(f"Anchor cache does not exist: {anchor_cache_path}")
    role_config_used = _apply_role_config_overrides(args)

    target_dir = _resolve_output_dir(args, checkpoint_dir)
    non_lora_path = target_dir / "non_lora_trainables.bin"
    adapter_config_path = target_dir / "adapter_config.json"
    if not non_lora_path.is_file():
        raise FileNotFoundError(f"Missing non_lora_trainables.bin: {non_lora_path}")
    if not adapter_config_path.is_file():
        raise FileNotFoundError(f"Missing adapter_config.json: {adapter_config_path}")

    state_dict = torch.load(non_lora_path, map_location="cpu")
    anchor_bundle = _load_anchor_bundle(anchor_cache_path)

    image_anchors = [tensor for tensor in anchor_bundle["image_anchors"]]
    text_anchors = [tensor for tensor in anchor_bundle["text_anchors"]]
    image_boundary = anchor_bundle.get("image_boundary")
    text_boundary = anchor_bundle.get("text_boundary")

    completed_task_count = _read_completed_task_count(checkpoint_dir)
    role_state = _build_role_state(
        image_anchors=image_anchors,
        text_anchors=text_anchors,
        completed_task_count=completed_task_count,
        device=torch.device("cpu"),
    )

    prefix = args.state_prefix
    for idx, tensor in enumerate(text_anchors):
        state_dict[f"{prefix}text_anchors.{idx}"] = tensor.cpu()
    for idx, tensor in enumerate(image_anchors):
        state_dict[f"{prefix}spectral_image_anchors.{idx}"] = tensor.cpu()
        state_dict[f"{prefix}image_anchors.{idx}"] = tensor.cpu()
    if text_boundary is not None:
        for idx, tensor in enumerate(text_boundary):
            state_dict[f"{prefix}text_boundary.{idx}"] = tensor.cpu()
    if image_boundary is not None:
        for idx, tensor in enumerate(image_boundary):
            state_dict[f"{prefix}spectral_image_boundary.{idx}"] = tensor.cpu()
            state_dict[f"{prefix}image_boundary.{idx}"] = tensor.cpu()

    state_dict[prefix + "expert_usage_prior"] = role_state["expert_usage_prior"]
    state_dict[prefix + "role_task_count"] = role_state["role_task_count"]
    state_dict[prefix + "role_usage_prior"] = role_state["role_usage_prior"]
    state_dict[prefix + "task_role_membership"] = role_state["task_role_membership"]
    state_dict[prefix + "active_role_count"] = role_state["active_role_count"]
    for idx, tensor in enumerate(role_state["role_image_prototypes"]):
        state_dict[f"{prefix}role_spectral_prototypes.{idx}"] = tensor
        state_dict[f"{prefix}role_image_prototypes.{idx}"] = tensor
    for idx, tensor in enumerate(role_state["role_text_prototypes"]):
        state_dict[f"{prefix}role_text_prototypes.{idx}"] = tensor

    torch.save(state_dict, non_lora_path)
    adapter_config = _write_adapter_compat_config(adapter_config_path)

    summary = {
        "source_checkpoint_dir": str(checkpoint_dir),
        "target_dir": str(target_dir),
        "anchor_cache": str(anchor_cache_path),
        "completed_task_count": completed_task_count,
        "state_prefix": prefix,
        "role_config_used": role_config_used,
        "adapter_config_after_injection": adapter_config,
        "notes": [
            "The LoRA adapter weights are kept intact.",
            (
                "adapter_config.json is rewritten to the HiDESC PEFT type so the "
                "later eval wrapper can target the HiDESC runtime."
            ),
            (
                "InternVL ablation defaults are tuned to encourage a 3-role UCIT split "
                "under the current Task6 anchor geometry."
            ),
        ],
    }
    with (target_dir / "reglora_hidesc_injection_meta.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
