#!/usr/bin/env python3
"""Inject an offline HiDESC prototype cache into copied LoRA checkpoints."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import torch


MAX_TASK_COUNT = 6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--role-bank",
        default=None,
        help="Optional role-bank payload produced by materialize_internvl_hidesc_role_bank.py.",
    )
    parser.add_argument(
        "--checkpoint-suffix",
        default="llava_lora",
        help="Task directory suffix inside the output root, e.g. llava_lora or internvl_hidesc.",
    )
    parser.add_argument(
        "--task-count",
        type=int,
        default=MAX_TASK_COUNT,
        help=f"Number of tasks to inject (1-{MAX_TASK_COUNT}).",
    )
    for task_id in range(1, MAX_TASK_COUNT + 1):
        parser.add_argument(f"--task{task_id}-source")
    return parser.parse_args()


def _find_prefix(state_dict: Dict[str, torch.Tensor]) -> str:
    suffix = "spectral_image_anchors.0"
    for key in state_dict:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    legacy_suffix = "image_anchors.0"
    for key in state_dict:
        if key.endswith(legacy_suffix):
            raise ValueError(
                "Checkpoint only contains legacy image_anchors.* tensors. "
                "This script injects spectral image prototype caches and requires "
                "spectral_image_anchors.* slots."
            )
    raise KeyError("Checkpoint does not contain spectral_image_anchors.0")


def _require_keys(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    names: Iterable[str],
) -> None:
    missing = [name for name in names if prefix + name not in state_dict]
    if missing:
        raise KeyError(
            "Checkpoint is missing required HiDESC tensors: "
            + ", ".join(missing[:8])
        )


def _as_row(value: torch.Tensor, expected_dim: int, label: str) -> torch.Tensor:
    if value.ndim != 1 or value.shape[0] != expected_dim:
        raise ValueError(
            f"{label} must have shape [{expected_dim}], got {tuple(value.shape)}"
        )
    return value.unsqueeze(0)


def _copy_cache_tensor(
    state_dict: Dict[str, torch.Tensor],
    key: str,
    value: torch.Tensor,
) -> None:
    if key not in state_dict:
        raise KeyError(f"Missing checkpoint tensor: {key}")
    target = state_dict[key]
    if tuple(target.shape) != tuple(value.shape):
        raise ValueError(
            f"Shape mismatch for {key}: checkpoint={tuple(target.shape)}, "
            f"cache={tuple(value.shape)}"
        )
    state_dict[key] = value.to(dtype=target.dtype)


def _copy_role_bank_tensor(
    state_dict: Dict[str, torch.Tensor],
    key: str,
    value: torch.Tensor,
) -> None:
    if key not in state_dict:
        raise KeyError(f"Missing checkpoint tensor: {key}")
    target = state_dict[key]
    if value.ndim == 0:
        value = value.reshape(1)
    if target.ndim == 1:
        if value.ndim != 1 or value.shape[0] > target.shape[0]:
            raise ValueError(
                f"Role-bank vector mismatch for {key}: checkpoint={tuple(target.shape)}, "
                f"role_bank={tuple(value.shape)}"
            )
        padded = torch.zeros_like(target)
        padded[: value.shape[0]] = value.to(dtype=target.dtype)
        state_dict[key] = padded
        return
    if target.ndim == 2:
        if value.ndim != 2 or value.shape[0] > target.shape[0] or value.shape[1] > target.shape[1]:
            raise ValueError(
                f"Role-bank matrix mismatch for {key}: checkpoint={tuple(target.shape)}, "
                f"role_bank={tuple(value.shape)}"
            )
        padded = torch.zeros_like(target)
        padded[: value.shape[0], : value.shape[1]] = value.to(dtype=target.dtype)
        state_dict[key] = padded
        return
    raise ValueError(f"Unsupported checkpoint tensor rank for {key}: {tuple(target.shape)}")


def _inject_role_bank(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
    role_bank: Dict[str, object],
) -> Dict[str, object]:
    active_roles = int(role_bank.get("active_role_count", 0))
    if active_roles <= 0:
        raise ValueError("Role bank must declare a positive active_role_count")

    role_image = role_bank["role_spectral_prototypes"]
    role_text = role_bank["role_text_prototypes"]
    membership = role_bank["task_role_membership"]
    role_task_count = role_bank["role_task_count"]
    role_usage_prior = role_bank["role_usage_prior"]

    if not isinstance(role_image, torch.Tensor) or not isinstance(role_text, torch.Tensor):
        raise TypeError("Role bank prototypes must be tensors")
    if not isinstance(membership, torch.Tensor):
        raise TypeError("Role bank membership must be a tensor")
    if not isinstance(role_task_count, torch.Tensor) or not isinstance(role_usage_prior, torch.Tensor):
        raise TypeError("Role bank counts and priors must be tensors")

    for role_id in range(active_roles):
        _copy_role_bank_tensor(
            state_dict,
            f"{prefix}role_spectral_prototypes.{role_id}",
            role_image[role_id].unsqueeze(0),
        )
        _copy_role_bank_tensor(
            state_dict,
            f"{prefix}role_text_prototypes.{role_id}",
            role_text[role_id].unsqueeze(0),
        )

    for key, value in list(state_dict.items()):
        if key.startswith(prefix + "role_spectral_prototypes.") and not any(
            key == f"{prefix}role_spectral_prototypes.{role_id}" for role_id in range(active_roles)
        ):
            state_dict[key] = torch.zeros_like(value)
        elif key.startswith(prefix + "role_text_prototypes.") and not any(
            key == f"{prefix}role_text_prototypes.{role_id}" for role_id in range(active_roles)
        ):
            state_dict[key] = torch.zeros_like(value)

    _copy_role_bank_tensor(state_dict, f"{prefix}role_task_count", role_task_count)
    _copy_role_bank_tensor(state_dict, f"{prefix}role_usage_prior", role_usage_prior)
    _copy_role_bank_tensor(state_dict, f"{prefix}task_role_membership", membership)

    active_role_key = f"{prefix}active_role_count"
    if active_role_key not in state_dict:
        raise KeyError(f"Missing checkpoint tensor: {active_role_key}")
    state_dict[active_role_key] = torch.tensor(
        [float(active_roles)], dtype=state_dict[active_role_key].dtype
    )

    return {
        "path": str(role_bank.get("source_path", "")),
        "active_role_count": active_roles,
        "groups": role_bank.get("groups"),
    }


def _clear_role_memory(state_dict: Dict[str, torch.Tensor], prefix: str) -> None:
    for key, value in list(state_dict.items()):
        if key.startswith(prefix + "role_spectral_prototypes."):
            state_dict[key] = torch.zeros_like(value)
        elif key.startswith(prefix + "role_text_prototypes."):
            state_dict[key] = torch.zeros_like(value)

    for name in (
        "role_task_count",
        "role_usage_prior",
        "task_role_membership",
        "active_role_count",
    ):
        key = prefix + name
        if key in state_dict:
            state_dict[key] = torch.zeros_like(state_dict[key])


def _inject_checkpoint(
    source_dir: Path,
    target_dir: Path,
    prototypes: torch.Tensor,
    text_anchors: torch.Tensor,
    counts: List[int],
    cache_path: Path,
    cache_metadata: Dict[str, object],
    role_bank_path: Path | None = None,
) -> Dict[str, object]:
    shutil.copytree(source_dir, target_dir, symlinks=False)
    non_lora_path = target_dir / "non_lora_trainables.bin"
    if not non_lora_path.is_file():
        raise FileNotFoundError(f"Missing non_lora_trainables.bin: {non_lora_path}")

    state_dict = torch.load(non_lora_path, map_location="cpu")
    try:
        prefix = _find_prefix(state_dict)
    except (KeyError, ValueError):
        # The original InternVL training artifact stores only the projector in
        # non_lora_trainables.bin.  Create the routing slots in the same naming
        # convention used by the model loader; the model will coerce these
        # tensors against its registered slot sizes at load time.
        prefix = "base_model.model.model."
        for task_id in range(len(counts)):
            state_dict.setdefault(
                f"{prefix}spectral_image_anchors.{task_id}",
                torch.zeros(1, prototypes.shape[-1], dtype=prototypes.dtype),
            )
            state_dict.setdefault(
                f"{prefix}spectral_image_boundary.{task_id}",
                torch.zeros(1, dtype=torch.float32),
            )
            state_dict.setdefault(
                f"{prefix}text_anchors.{task_id}",
                torch.zeros(1, text_anchors.shape[-1], dtype=text_anchors.dtype),
            )
            state_dict.setdefault(
                f"{prefix}text_boundary.{task_id}",
                torch.zeros(1, dtype=torch.float32),
            )
        for role_id in range(int(role_bank_path is not None) * 4 or len(counts)):
            state_dict.setdefault(
                f"{prefix}role_spectral_prototypes.{role_id}",
                torch.zeros(1, prototypes.shape[-1], dtype=prototypes.dtype),
            )
            state_dict.setdefault(
                f"{prefix}role_text_prototypes.{role_id}",
                torch.zeros(1, text_anchors.shape[-1], dtype=text_anchors.dtype),
            )
        state_dict.setdefault(f"{prefix}role_task_count", torch.zeros(len(counts)))
        state_dict.setdefault(f"{prefix}role_usage_prior", torch.zeros(len(counts)))
        state_dict.setdefault(
            f"{prefix}task_role_membership", torch.zeros(len(counts), len(counts))
        )
        state_dict.setdefault(f"{prefix}active_role_count", torch.zeros(1))
        state_dict.setdefault(f"{prefix}expert_usage_prior", torch.zeros(len(counts)))
    required = []
    task_count = len(counts)
    for task_id in range(task_count):
        required.extend(
            [
                f"spectral_image_anchors.{task_id}",
                f"spectral_image_boundary.{task_id}",
                f"text_anchors.{task_id}",
                f"text_boundary.{task_id}",
            ]
        )
    _require_keys(state_dict, prefix, required)

    for task_id in range(task_count):
        image_key = f"{prefix}spectral_image_anchors.{task_id}"
        image_value = _as_row(
            prototypes[task_id],
            state_dict[image_key].shape[-1],
            f"cache prototypes[{task_id}]",
        )
        _copy_cache_tensor(state_dict, image_key, image_value)

        image_boundary_key = f"{prefix}spectral_image_boundary.{task_id}"
        image_boundary = torch.tensor(
            [float(counts[task_id])],
            dtype=state_dict[image_boundary_key].dtype,
        )
        _copy_cache_tensor(state_dict, image_boundary_key, image_boundary)

        text_key = f"{prefix}text_anchors.{task_id}"
        text_value = _as_row(
            text_anchors[task_id],
            state_dict[text_key].shape[-1],
            f"cache text_anchors[{task_id}]",
        )
        _copy_cache_tensor(state_dict, text_key, text_value)

        text_boundary_key = f"{prefix}text_boundary.{task_id}"
        text_boundary = torch.tensor(
            [float(counts[task_id])],
            dtype=state_dict[text_boundary_key].dtype,
        )
        _copy_cache_tensor(state_dict, text_boundary_key, text_boundary)

    role_memory = "cleared_for_runtime_rebuild"
    if role_bank_path is None:
        # The old checkpoint role bank was built from the old task anchors. Clear it
        # so the new routing config rebuilds role memory from the injected cache.
        _clear_role_memory(state_dict, prefix)
    else:
        role_bank = torch.load(role_bank_path, map_location="cpu")
        if not isinstance(role_bank, dict):
            raise TypeError(f"Expected a role-bank dictionary in {role_bank_path}")
        role_bank["source_path"] = str(role_bank_path)
        role_memory = _inject_role_bank(state_dict, prefix, role_bank)
    torch.save(state_dict, non_lora_path)

    meta = {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "prototype_cache": str(cache_path),
        "cache_metadata": cache_metadata,
        "injected_task_count": task_count,
        "role_memory": role_memory,
    }
    if role_bank_path is not None:
        meta["role_bank_path"] = str(role_bank_path)
    with (target_dir / "hidesc_offline_prototype_injection_meta.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(meta, handle, indent=2)
    return meta


def main() -> None:
    args = _parse_args()
    cache_path = Path(args.prototype_cache).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"Prototype cache does not exist: {cache_path}")
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output root: {output_root}"
        )

    payload = torch.load(cache_path, map_location="cpu")
    prototypes = payload.get("prototypes", payload.get("image_anchors"))
    if prototypes is None:
        raise KeyError("Prototype cache must contain 'prototypes' or 'image_anchors'")
    text_anchors = payload["text_anchors"]
    raw_counts = payload.get("counts", payload.get("sample_counts"))
    if raw_counts is None:
        raise KeyError("Prototype cache must contain 'counts' or 'sample_counts'")
    counts = [int(value) for value in raw_counts]
    metadata = payload.get("metadata", {})
    role_bank_path = Path(args.role_bank).expanduser().resolve() if args.role_bank else None
    task_count = int(args.task_count)
    if not 1 <= task_count <= MAX_TASK_COUNT:
        raise ValueError(f"--task-count must be in [1, {MAX_TASK_COUNT}]")
    if metadata.get("benchmark") not in (None, "ucit", "dcl"):
        raise ValueError(f"Unsupported cache metadata={metadata}")
    if prototypes.ndim != 2 or prototypes.shape[0] != task_count:
        raise ValueError(
            f"Expected prototypes with {task_count} rows, "
            f"got {tuple(prototypes.shape)}"
        )
    if tuple(text_anchors.shape) != (task_count, 768):
        raise ValueError(
            f"Expected text_anchors shape {(task_count, 768)}, "
            f"got {tuple(text_anchors.shape)}"
        )
    if len(counts) != task_count:
        raise ValueError(f"Expected {task_count} counts, got {len(counts)}")
    if role_bank_path is not None and not role_bank_path.is_file():
        raise FileNotFoundError(f"Role bank does not exist: {role_bank_path}")

    output_root.mkdir(parents=True, exist_ok=False)
    missing_sources = [
        task_id
        for task_id in range(1, task_count + 1)
        if getattr(args, f"task{task_id}_source") is None
    ]
    if missing_sources:
        raise FileNotFoundError(
            "Missing required task source arguments: "
            + ", ".join(f"--task{task_id}-source" for task_id in missing_sources)
        )
    source_paths = [
        Path(getattr(args, f"task{task_id}_source")).expanduser().resolve()
        for task_id in range(1, task_count + 1)
    ]
    summaries = []
    for task_id, source_dir in enumerate(source_paths, start=1):
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Task{task_id} source does not exist: {source_dir}")
        target_dir = output_root / f"Task{task_id}_{args.checkpoint_suffix}"
        summaries.append(
            _inject_checkpoint(
                source_dir=source_dir,
                target_dir=target_dir,
                prototypes=prototypes,
                text_anchors=text_anchors,
                counts=counts,
                cache_path=cache_path,
                cache_metadata=metadata,
                role_bank_path=role_bank_path,
            )
        )

    manifest = {
        "output_root": str(output_root),
        "prototype_cache": str(cache_path),
        "cache_metadata": metadata,
        "counts": counts,
        "checkpoints": summaries,
    }
    with (output_root / "hidesc_offline_prototype_injection_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
