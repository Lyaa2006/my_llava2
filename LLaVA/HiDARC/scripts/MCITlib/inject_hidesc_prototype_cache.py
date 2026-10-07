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
                "This script injects spectral prototype caches (6144-d image "
                "prototypes) and requires checkpoints that already expose "
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
) -> Dict[str, object]:
    shutil.copytree(source_dir, target_dir, symlinks=False)
    non_lora_path = target_dir / "non_lora_trainables.bin"
    if not non_lora_path.is_file():
        raise FileNotFoundError(f"Missing non_lora_trainables.bin: {non_lora_path}")

    state_dict = torch.load(non_lora_path, map_location="cpu")
    prefix = _find_prefix(state_dict)
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

    # The old checkpoint role bank was built from the old task anchors. Clear it
    # so the new routing config rebuilds role memory from the injected cache.
    _clear_role_memory(state_dict, prefix)
    torch.save(state_dict, non_lora_path)

    meta = {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "prototype_cache": str(cache_path),
        "cache_metadata": cache_metadata,
        "injected_task_count": task_count,
        "role_memory": "cleared_for_runtime_rebuild",
    }
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
    prototypes = payload["prototypes"]
    text_anchors = payload["text_anchors"]
    counts = [int(value) for value in payload["counts"]]
    metadata = payload.get("metadata", {})
    task_count = int(args.task_count)
    if not 1 <= task_count <= MAX_TASK_COUNT:
        raise ValueError(f"--task-count must be in [1, {MAX_TASK_COUNT}]")
    if metadata.get("benchmark") not in (None, "ucit", "dcl"):
        raise ValueError(f"Unsupported cache metadata={metadata}")
    if tuple(prototypes.shape) != (task_count, 6144):
        raise ValueError(
            f"Expected prototypes shape {(task_count, 6144)}, "
            f"got {tuple(prototypes.shape)}"
        )
    if tuple(text_anchors.shape) != (task_count, 768):
        raise ValueError(
            f"Expected text_anchors shape {(task_count, 768)}, "
            f"got {tuple(text_anchors.shape)}"
        )
    if len(counts) != task_count:
        raise ValueError(f"Expected {task_count} counts, got {len(counts)}")

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
        target_dir = output_root / f"Task{task_id}_llava_lora"
        summaries.append(
            _inject_checkpoint(
                source_dir=source_dir,
                target_dir=target_dir,
                prototypes=prototypes,
                text_anchors=text_anchors,
                counts=counts,
                cache_path=cache_path,
                cache_metadata=metadata,
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
