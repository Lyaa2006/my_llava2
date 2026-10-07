#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def _safe_normalize(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim > 1:
        tensor = tensor.squeeze(0)
    tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
    norm = torch.linalg.norm(tensor)
    if not torch.isfinite(norm) or float(norm.item()) <= 0.0:
        return torch.zeros_like(tensor, dtype=torch.float32)
    return F.normalize(tensor, dim=0)


def _detect_prefix(state_dict: Dict[str, torch.Tensor]) -> str:
    for suffix in ("spectral_image_anchors.0", "text_anchors.0"):
        for key in state_dict:
            if key.endswith(suffix):
                return key[: -len(suffix)]
    raise KeyError("Failed to locate HiDESC prefix in non_lora_trainables.bin")


def _collect_series(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    name: str,
) -> List[torch.Tensor]:
    items: List[Tuple[int, torch.Tensor]] = []
    for key, value in state_dict.items():
        if key.startswith(prefix + name + "."):
            items.append((int(key.rsplit(".", 1)[1]), value))
    if not items:
        raise KeyError(f"Missing {name} entries in non_lora_trainables.bin")
    items.sort(key=lambda item: item[0])
    return [value for _, value in items]


def _rebuild_role_spectral_prototypes(
    spectral_image_anchors: List[torch.Tensor],
    task_role_membership: torch.Tensor,
    role_task_count: torch.Tensor,
) -> List[torch.Tensor]:
    max_roles = int(task_role_membership.shape[1])
    rebuilt: List[torch.Tensor] = []
    for role_id in range(max_roles):
        member_weights = torch.nan_to_num(
            task_role_membership[:, role_id].detach().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        active_members = torch.nonzero(member_weights > 0.0, as_tuple=False).flatten().tolist()
        if not active_members:
            rebuilt.append(torch.zeros_like(spectral_image_anchors[0]))
            continue
        weighted = []
        for task_id in active_members:
            weighted.append(
                float(member_weights[task_id].item())
                * _safe_normalize(spectral_image_anchors[task_id]).unsqueeze(0)
            )
        prototype = torch.stack(weighted, dim=0).sum(dim=0)
        if float(role_task_count[role_id].item()) <= 0.0:
            role_task_count[role_id] = float(len(active_members))
        rebuilt.append(_safe_normalize(prototype).unsqueeze(0))
    return rebuilt


def _resolve_output_dir(args: argparse.Namespace, checkpoint_dir: Path) -> Path:
    if args.inplace:
        return checkpoint_dir
    if args.output_root is None:
        raise ValueError("Provide --inplace or --output-root.")
    output_dir = Path(args.output_root).resolve() / checkpoint_dir.name
    if output_dir.exists() and args.fail_if_present:
        raise FileExistsError(f"Refusing to overwrite existing output dir: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(checkpoint_dir, output_dir, symlinks=False)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild HiDESC spectral role memory from spectral task anchors."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--fail-if-present", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir does not exist: {checkpoint_dir}")

    target_dir = _resolve_output_dir(args, checkpoint_dir)
    non_lora_path = target_dir / "non_lora_trainables.bin"
    if not non_lora_path.is_file():
        raise FileNotFoundError(f"Missing non_lora_trainables.bin: {non_lora_path}")

    state_dict = torch.load(non_lora_path, map_location="cpu")
    prefix = _detect_prefix(state_dict)
    spectral_image_anchors = _collect_series(state_dict, prefix, "spectral_image_anchors")
    task_role_membership = state_dict[prefix + "task_role_membership"].clone()
    role_task_count = state_dict[prefix + "role_task_count"].clone()
    rebuilt = _rebuild_role_spectral_prototypes(
        spectral_image_anchors=spectral_image_anchors,
        task_role_membership=task_role_membership,
        role_task_count=role_task_count,
    )

    for idx, tensor in enumerate(rebuilt):
        state_dict[f"{prefix}role_spectral_prototypes.{idx}"] = tensor.to(
            state_dict.get(
                f"{prefix}role_spectral_prototypes.{idx}",
                tensor,
            ).dtype
        )
    state_dict[prefix + "role_task_count"] = role_task_count
    torch.save(state_dict, non_lora_path)

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "target_dir": str(target_dir),
        "role_count": len(rebuilt),
        "rebuilt_from": [
            "spectral_image_anchors",
            "task_role_membership",
            "role_task_count",
        ],
    }
    with (target_dir / "hidesc_spectral_role_rebuild_meta.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
