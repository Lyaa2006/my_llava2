#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "docs" / "hierarchical_boundary_sweeps"
_RUNTIME_IMPORTS: Optional[SimpleNamespace] = None

MODEL_SPECS = {
    "llava": {
        "config_path": REPO_ROOT / "configs" / "model_configs" / "llava.json",
        "backend_root": REPO_ROOT / "LLaVA" / "LoRA-FT",
        "conv_mode": "llava_v1",
        "display_name": "LLaVA",
    },
    "internvl": {
        "config_path": REPO_ROOT / "configs" / "model_configs" / "internvl.json",
        "backend_root": REPO_ROOT / "InternVL" / "CL-MoE",
        "conv_mode": "llava_v1",
        "display_name": "InternVL",
    },
}

BENCHMARK_SPECS = {
    "ucit": {
        "config_dir": REPO_ROOT / "configs" / "data_configs" / "UCIT",
        "tasks": ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k"],
        "config_files": {
            "ImageNet-R": "ImageNet-R.json",
            "ArxivQA": "ArxivQA.json",
            "VizWiz": "VizWiz.json",
            "IconQA": "IconQA.json",
            "CLEVR": "CLEVR-Math.json",
            "Flickr30k": "Flickr30k.json",
        },
        "reasoning_tasks": ["ArxivQA", "IconQA", "CLEVR"],
        "style_tasks": ["ImageNet-R", "ArxivQA", "IconQA", "CLEVR", "VizWiz", "Flickr30k"],
        "display_name": "UCIT",
    },
    "acl": {
        "config_dir": REPO_ROOT / "configs" / "data_configs" / "MLLM-ACL",
        "tasks": ["APP", "Math", "OCR", "VP"],
        "config_files": {task: f"{task}.json" for task in ["APP", "Math", "OCR", "VP"]},
        "reasoning_tasks": ["APP", "Math", "OCR", "VP"],
        "style_tasks": ["APP", "Math", "OCR", "VP"],
        "display_name": "MLLM-ACL",
    },
    "dcl": {
        "config_dir": REPO_ROOT / "configs" / "data_configs" / "MLLM-DCL",
        "tasks": ["AD", "Fin", "Med", "RS", "Sci"],
        "config_files": {task: f"{task}.json" for task in ["AD", "Fin", "Med", "RS", "Sci"]},
        "reasoning_tasks": ["AD", "Fin", "Med", "RS", "Sci"],
        "style_tasks": ["AD", "Fin", "Med", "RS", "Sci"],
        "display_name": "MLLM-DCL",
    },
}

PLOT_COLORS = {
    "reasoning": "#d1495b",
    "task": "#2f6690",
    "style_logit": "#3e8914",
    "handoff": "#edae49",
}


@dataclass
class Sample:
    task: str
    question_id: str
    image_path: str
    text: str
    answer: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_runtime_imports() -> SimpleNamespace:
    global _RUNTIME_IMPORTS
    if _RUNTIME_IMPORTS is None:
        import numpy as np
        import torch
        from PIL import Image
        from tqdm import tqdm
        from transformers import AutoConfig, AutoTokenizer

        _RUNTIME_IMPORTS = SimpleNamespace(
            np=np,
            torch=torch,
            Image=Image,
            tqdm=tqdm,
            AutoConfig=AutoConfig,
            AutoTokenizer=AutoTokenizer,
        )
        globals()["np"] = np
        globals()["torch"] = torch
        globals()["Image"] = Image
        globals()["tqdm"] = tqdm
        globals()["AutoConfig"] = AutoConfig
        globals()["AutoTokenizer"] = AutoTokenizer
    return _RUNTIME_IMPORTS


def set_seed(seed: int) -> None:
    ensure_runtime_imports()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_name_list(raw_values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not raw_values:
        return None
    values: List[str] = []
    for item in raw_values:
        values.extend(part.strip() for part in item.split(",") if part.strip())
    return values or None


def load_model_runtime_spec(model_key: str) -> Dict[str, str]:
    model_spec = MODEL_SPECS[model_key]
    config = load_json(model_spec["config_path"])
    return {
        "model_key": model_key,
        "model_path": config["model_name"],
        "vision_tower_path": config["vision_tower"],
        "backend_root": str(model_spec["backend_root"]),
        "conv_mode": model_spec["conv_mode"],
        "display_name": model_spec["display_name"],
    }


def load_task_definitions(benchmark_key: str, task_names: Sequence[str]) -> Dict[str, Dict[str, str]]:
    spec = BENCHMARK_SPECS[benchmark_key]
    task_defs: Dict[str, Dict[str, str]] = {}
    for task in task_names:
        config_filename = spec["config_files"].get(task, f"{task}.json")
        config_path = spec["config_dir"] / config_filename
        config = load_json(config_path)
        test_path = config.get("test_path") or config.get("train_path")
        image_root = config.get("test_folder") or config.get("train_folder") or ""
        task_defs[task] = {
            "config_path": str(config_path),
            "test_path": test_path,
            "image_root": image_root,
        }
    return task_defs


def resolve_image_path(raw_image_path: str, image_root: str) -> str:
    image_path = Path(raw_image_path)
    if image_path.is_absolute():
        return str(image_path)
    return str(Path(image_root) / image_path)


def load_task_samples(
    benchmark_key: str,
    task_names: Sequence[str],
    samples_per_task: int,
    seed: int,
) -> Dict[str, List[Sample]]:
    rng = random.Random(seed)
    task_defs = load_task_definitions(benchmark_key, task_names)
    task_samples: Dict[str, List[Sample]] = {}
    for task in task_names:
        task_def = task_defs[task]
        raw_samples = load_json(Path(task_def["test_path"]))
        if samples_per_task < len(raw_samples):
            raw_samples = rng.sample(raw_samples, samples_per_task)
        task_samples[task] = [
            Sample(
                task=task,
                question_id=str(item.get("question_id", "")),
                image_path=resolve_image_path(str(item["image"]), task_def["image_root"]),
                text=str(item.get("text", "")).strip(),
                answer="" if item.get("answer") is None else str(item.get("answer", "")).strip(),
            )
            for item in raw_samples
        ]
    return task_samples


def strip_existing_instruction(text: str) -> str:
    cleaned = text.strip()
    patterns = (
        r"Answer with the option.*$",
        r"Answer the question using a single word or phrase\.\s*$",
        r"Answer using a single word or short phrase\.\s*$",
        r"Answer using a short phrase\.\s*$",
        r"Answer in one complete sentence\.\s*$",
        r"Generate a brief caption for the image\.\s*$",
        r"Respond with a single word or phrase\.\s*$",
    )
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    return cleaned


def has_explicit_option_labels(text: str) -> bool:
    return bool(re.search(r"(^|\n)\s*([A-H]|[0-9]+)[\.\)]\s+\S", text))


def build_style_variants(sample: Sample) -> Dict[str, str]:
    base = strip_existing_instruction(sample.text) or sample.text.strip()
    variants: Dict[str, str] = {}

    if has_explicit_option_labels(sample.text):
        variants["label"] = f"{base}\nAnswer with the option id only."
        variants["option_text"] = f"{base}\nAnswer with the exact option text only. Do not output the option id."
        variants["sentence"] = (
            f"{base}\nAnswer in one complete sentence using the exact option content. Do not output the option id."
        )
        return variants

    variants["phrase"] = f"{base}\nAnswer using a short phrase."
    variants["sentence"] = f"{base}\nAnswer in one complete sentence."
    return variants


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return 1.0 - (num / np.clip(den, 1e-8, None))


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(np.sum(exp, axis=-1, keepdims=True), 1e-8, None)


def js_divergence(logits_a: np.ndarray, logits_b: np.ndarray) -> np.ndarray:
    p = softmax_np(logits_a)
    q = softmax_np(logits_b)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(np.clip(p, 1e-8, None)) - np.log(np.clip(m, 1e-8, None))), axis=-1)
    kl_qm = np.sum(q * (np.log(np.clip(q, 1e-8, None)) - np.log(np.clip(m, 1e-8, None))), axis=-1)
    return 0.5 * (kl_pm + kl_qm)


def smooth_curve(values: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def normalize_curve(values: np.ndarray) -> np.ndarray:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.zeros_like(values)
    vmin = float(finite_values.min())
    vmax = float(finite_values.max())
    if math.isclose(vmin, vmax):
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def fisher_task_separation(layer_reps: np.ndarray, labels: List[str]) -> np.ndarray:
    unique_labels = sorted(set(labels))
    num_layers = layer_reps.shape[1]
    scores = np.zeros(num_layers, dtype=np.float32)
    for layer_idx in range(num_layers):
        layer_matrix = layer_reps[:, layer_idx, :]
        global_mean = layer_matrix.mean(axis=0)
        between = 0.0
        within = 0.0
        for label in unique_labels:
            indices = [idx for idx, cur in enumerate(labels) if cur == label]
            group = layer_matrix[indices]
            group_mean = group.mean(axis=0)
            between += float(group.shape[0]) * float(np.sum((group_mean - global_mean) ** 2))
            within += float(np.sum((group - group_mean) ** 2))
        scores[layer_idx] = between / max(within, 1e-8)
    return scores


def choose_deep_middle_boundary(
    reasoning_curve: np.ndarray,
    task_curve: np.ndarray,
    search_start_layer: int,
    search_end_layer: int,
) -> Tuple[int, np.ndarray]:
    layer_count = len(reasoning_curve)
    raw_scores = np.full(layer_count, -1e9, dtype=np.float32)
    search_start = max(1, search_start_layer)
    search_end = min(search_end_layer, layer_count - 1)
    for idx in range(search_start - 1, search_end):
        closeness = 1.0 - abs(float(reasoning_curve[idx]) - float(task_curve[idx]))
        reasoning_rise = max(
            0.0,
            float(reasoning_curve[min(idx + 1, layer_count - 1)]) - float(reasoning_curve[max(idx - 1, 0)]),
        )
        task_fall = max(
            0.0,
            float(task_curve[max(idx - 1, 0)]) - float(task_curve[min(idx + 1, layer_count - 1)]),
        )
        raw_scores[idx] = 0.5 * closeness + 0.25 * reasoning_rise + 0.25 * task_fall
    boundary = int(np.argmax(raw_scores)) + 1
    boundary = max(1, min(boundary, layer_count - 1))
    return boundary, normalize_curve(raw_scores)


def choose_mid_shallow_boundary(style_logit_curve: np.ndarray, deep_boundary: int, min_search_layer: int) -> int:
    layer_count = len(style_logit_curve)
    search_start = min(max(deep_boundary + 1, min_search_layer, 1), layer_count)
    if search_start >= layer_count:
        return layer_count
    tail_curve = style_logit_curve[search_start - 1 :]
    tail_layers = np.arange(search_start, layer_count + 1)
    return int(tail_layers[int(np.argmax(tail_curve))])


def plot_curves(
    output_path: Path,
    title: str,
    layers: np.ndarray,
    reasoning_curve: np.ndarray,
    task_curve: np.ndarray,
    style_logit_curve: np.ndarray,
    deep_boundary: int,
    mid_shallow_boundary: int,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11.2, 6.0), dpi=180)
    ax.plot(layers, reasoning_curve, color=PLOT_COLORS["reasoning"], linewidth=2.4, label="Reasoning Signal")
    ax.plot(layers, task_curve, color=PLOT_COLORS["task"], linewidth=2.4, label="Task Separation")
    ax.plot(layers, style_logit_curve, color=PLOT_COLORS["style_logit"], linewidth=2.6, label="Style Logit Signal")

    ax.axvspan(1, deep_boundary, color=PLOT_COLORS["reasoning"], alpha=0.08)
    ax.axvspan(deep_boundary, mid_shallow_boundary, color=PLOT_COLORS["task"], alpha=0.08)
    ax.axvspan(mid_shallow_boundary, layers[-1], color=PLOT_COLORS["style_logit"], alpha=0.08)
    ax.axvline(deep_boundary, linestyle="--", color="#444444", linewidth=1.4)
    ax.axvline(mid_shallow_boundary, linestyle="--", color="#444444", linewidth=1.4)
    ax.text(deep_boundary, 1.03, f"b1 = {deep_boundary}", ha="center", va="bottom", fontsize=10)
    ax.text(mid_shallow_boundary, 1.03, f"b2 = {mid_shallow_boundary}", ha="center", va="bottom", fontsize=10)
    ax.set_xlim(1, layers[-1])
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized Score")
    ax.set_title(title)
    ax.legend(frameon=True, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def summarize_metric(values: Sequence[float]) -> Dict[str, float]:
    ordered = [float(v) for v in values]
    if not ordered:
        return {}
    return {
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "min": float(min(ordered)),
        "max": float(max(ordered)),
        "std": float(statistics.pstdev(ordered)) if len(ordered) > 1 else 0.0,
    }


def import_llava_backend(backend_root: str) -> SimpleNamespace:
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model import LlavaLlamaForCausalLM
    from llava.utils import disable_torch_init

    return SimpleNamespace(
        DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN=DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN=DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX=IMAGE_TOKEN_INDEX,
        LlavaLlamaForCausalLM=LlavaLlamaForCausalLM,
        conv_templates=conv_templates,
        tokenizer_image_token=tokenizer_image_token,
        disable_torch_init=disable_torch_init,
    )


def build_prompt(question: str, mm_use_im_start_end: bool, conv_mode: str, api: SimpleNamespace) -> str:
    if mm_use_im_start_end:
        user_prompt = f"{api.DEFAULT_IM_START_TOKEN}{api.DEFAULT_IMAGE_TOKEN}{api.DEFAULT_IM_END_TOKEN}\n{question}"
    else:
        user_prompt = f"{api.DEFAULT_IMAGE_TOKEN}\n{question}"
    conv = api.conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def infer_image_size(image_processor) -> int:
    size = getattr(image_processor, "crop_size", None)
    if isinstance(size, dict):
        for key in ("height", "width", "shortest_edge"):
            if key in size:
                return int(size[key])
    if isinstance(size, int):
        return int(size)
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        for key in ("height", "width", "shortest_edge"):
            if key in size:
                return int(size[key])
    if isinstance(size, int):
        return int(size)
    return 336


def load_image_tensor(image_path: str, image_processor, device: str, blank: bool = False) -> torch.Tensor:
    image_size = infer_image_size(image_processor)
    if blank:
        image = Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
    else:
        image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return pixel_values.unsqueeze(0).to(device=device, dtype=torch.float16)


def last_token_hidden_stack(hidden_states: Sequence[torch.Tensor]) -> np.ndarray:
    vectors = []
    for layer_hidden in hidden_states[1:]:
        vec = layer_hidden[0, -1].detach().float().cpu()
        vec = vec / vec.norm(p=2).clamp_min(1e-6)
        vectors.append(vec.numpy())
    return np.stack(vectors, axis=0)


def last_token_logits_stack(model, hidden_states: Sequence[torch.Tensor]) -> np.ndarray:
    norm = model.get_model().norm
    lm_head = model.lm_head
    per_layer_logits = []
    for layer_hidden in hidden_states[1:]:
        token_hidden = layer_hidden[:, -1:, :]
        normed = norm(token_hidden)
        logits = lm_head(normed)[0, 0].detach().float().cpu().numpy()
        per_layer_logits.append(logits)
    return np.stack(per_layer_logits, axis=0)


def run_single_analysis(args: argparse.Namespace) -> Path:
    ensure_runtime_imports()
    model_spec = load_model_runtime_spec(args.model)
    benchmark_spec = BENCHMARK_SPECS[args.benchmark]
    reasoning_tasks = parse_name_list(args.reasoning_tasks) or benchmark_spec["reasoning_tasks"]
    style_tasks = parse_name_list(args.style_tasks) or benchmark_spec["style_tasks"]
    all_tasks = benchmark_spec["tasks"]

    api = import_llava_backend(model_spec["backend_root"])

    ensure_dir(Path(args.output_dir))
    set_seed(args.seed)
    api.disable_torch_init()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRITON_CACHE_DIR", str(Path(args.output_dir) / "triton_cache"))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot initialize CUDA in this environment. "
            "Check the active conda environment, CUDA driver visibility, and GPU allocation. "
            f"torch.version.cuda={torch.version.cuda!r}, device={args.device!r}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_spec["model_path"], use_fast=False, local_files_only=True)
    cfg = AutoConfig.from_pretrained(model_spec["model_path"], local_files_only=True)
    cfg.mm_vision_tower = model_spec["vision_tower_path"]
    cfg.vision_tower = model_spec["vision_tower_path"]
    if not hasattr(cfg, "mm_vision_select_layer"):
        cfg.mm_vision_select_layer = -2
    if hasattr(cfg, "mm_text_tower"):
        delattr(cfg, "mm_text_tower")
    if hasattr(cfg, "text_tower"):
        delattr(cfg, "text_tower")

    model = api.LlavaLlamaForCausalLM.from_pretrained(
        model_spec["model_path"],
        low_cpu_mem_usage=True,
        config=cfg,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model.to(device=args.device, dtype=torch.float16)
    model.eval()

    vision_tower = model.get_vision_tower()
    if vision_tower is None:
        raise RuntimeError("Failed to initialize local vision tower.")
    vision_tower.vision_tower_name = model_spec["vision_tower_path"]
    vision_tower.load_model()
    vision_tower.to(device=args.device, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)

    stage1_samples = load_task_samples(args.benchmark, all_tasks, args.stage1_samples_per_task, args.seed)
    style_samples = load_task_samples(args.benchmark, style_tasks, args.style_samples_per_task, args.seed + 17)

    all_reps = []
    all_labels = []
    reasoning_distances = []
    logit_style_all = []
    skipped_style_samples = 0

    progress_total = sum(len(v) for v in stage1_samples.values()) + sum(len(v) for v in style_samples.values())
    progress = tqdm(total=progress_total, desc=f"{args.model}-{args.benchmark}-seed{args.seed}")

    with torch.inference_mode():
        for task, samples in stage1_samples.items():
            for sample in samples:
                image_tensor = load_image_tensor(sample.image_path, image_processor, args.device)
                prompt = build_prompt(sample.text, mm_use_im_start_end, model_spec["conv_mode"], api)
                input_ids = api.tokenizer_image_token(
                    prompt,
                    tokenizer,
                    api.IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                ).unsqueeze(0).to(args.device)
                outputs = model(
                    input_ids=input_ids,
                    images=image_tensor,
                    output_hidden_states=True,
                    return_dict=True,
                )
                pooled = last_token_hidden_stack(outputs.hidden_states)
                all_reps.append(pooled)
                all_labels.append(task)

                if task in reasoning_tasks:
                    blank_tensor = load_image_tensor(sample.image_path, image_processor, args.device, blank=True)
                    blank_outputs = model(
                        input_ids=input_ids,
                        images=blank_tensor,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    blank_pooled = last_token_hidden_stack(blank_outputs.hidden_states)
                    reasoning_distances.append(cosine_distance(pooled, blank_pooled))
                progress.update(1)

        for task, samples in style_samples.items():
            for sample in samples:
                variants = build_style_variants(sample)
                if len(variants) < 2:
                    skipped_style_samples += 1
                    progress.update(1)
                    continue
                image_tensor = load_image_tensor(sample.image_path, image_processor, args.device)
                logit_stacks = {}
                for variant_name, variant_text in variants.items():
                    prompt = build_prompt(variant_text, mm_use_im_start_end, model_spec["conv_mode"], api)
                    input_ids = api.tokenizer_image_token(
                        prompt,
                        tokenizer,
                        api.IMAGE_TOKEN_INDEX,
                        return_tensors="pt",
                    ).unsqueeze(0).to(args.device)
                    outputs = model(
                        input_ids=input_ids,
                        images=image_tensor,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    logit_stacks[variant_name] = last_token_logits_stack(model, outputs.hidden_states)

                names = list(logit_stacks.keys())
                logit_pairs = []
                for idx in range(len(names)):
                    for jdx in range(idx + 1, len(names)):
                        logit_pairs.append(js_divergence(logit_stacks[names[idx]], logit_stacks[names[jdx]]))
                if logit_pairs:
                    logit_style_all.append(np.mean(np.stack(logit_pairs, axis=0), axis=0))
                progress.update(1)

    progress.close()

    if not reasoning_distances:
        raise RuntimeError("No reasoning samples were collected. Check reasoning task configuration.")
    if not logit_style_all:
        raise RuntimeError("No style samples were collected. Check style task configuration.")

    layer_reps = np.stack(all_reps, axis=0)
    reasoning_curve = normalize_curve(smooth_curve(np.mean(np.stack(reasoning_distances, axis=0), axis=0), window=3))
    task_curve = normalize_curve(smooth_curve(fisher_task_separation(layer_reps, all_labels), window=3))
    style_logit_curve = normalize_curve(smooth_curve(np.mean(np.stack(logit_style_all, axis=0), axis=0), window=3))

    default_search_end = max(args.b1_search_end_layer, int(math.ceil(len(reasoning_curve) * 0.5)))
    deep_boundary, handoff_curve = choose_deep_middle_boundary(
        reasoning_curve,
        task_curve,
        args.b1_search_start_layer,
        default_search_end,
    )
    mid_shallow_boundary = choose_mid_shallow_boundary(style_logit_curve, deep_boundary, args.min_search_layer)

    layer_count = len(reasoning_curve)
    deep_boundary = max(1, min(deep_boundary, layer_count - 1))
    mid_shallow_boundary = max(deep_boundary + 1, min(mid_shallow_boundary, layer_count))

    output_dir = Path(args.output_dir)
    report_path = output_dir / "boundary_report.json"
    figure_path = output_dir / "boundary_curves.png"
    layers = np.arange(1, layer_count + 1)

    figure_created = True
    try:
        plot_curves(
            figure_path,
            f"{model_spec['display_name']} on {benchmark_spec['display_name']} (seed={args.seed})",
            layers,
            reasoning_curve,
            task_curve,
            style_logit_curve,
            deep_boundary,
            mid_shallow_boundary,
        )
    except ImportError:
        figure_created = False

    report = {
        "model": args.model,
        "benchmark": args.benchmark,
        "seed": args.seed,
        "model_display_name": model_spec["display_name"],
        "benchmark_display_name": benchmark_spec["display_name"],
        "recommended_boundaries": {
            "b1": deep_boundary,
            "b2": mid_shallow_boundary,
            "b1_ratio": deep_boundary / layer_count,
            "b2_ratio": mid_shallow_boundary / layer_count,
            "layer_count": layer_count,
            "deep_layers": [1, deep_boundary],
            "middle_layers": [deep_boundary + 1, mid_shallow_boundary - 1],
            "shallow_layers": [mid_shallow_boundary, layer_count],
        },
        "tasks": {
            "all_tasks": list(all_tasks),
            "reasoning_tasks": list(reasoning_tasks),
            "style_tasks": list(style_tasks),
        },
        "sampling": {
            "stage1_samples_per_task": args.stage1_samples_per_task,
            "style_samples_per_task": args.style_samples_per_task,
            "style_skipped_samples": skipped_style_samples,
        },
        "search": {
            "b1_search_start_layer": args.b1_search_start_layer,
            "b1_search_end_layer": default_search_end,
            "min_search_layer": args.min_search_layer,
        },
        "curves": {
            "reasoning_signal": reasoning_curve.tolist(),
            "task_separation": task_curve.tolist(),
            "style_logit_signal": style_logit_curve.tolist(),
            "deep_middle_handoff_score": handoff_curve.tolist(),
        },
        "artifacts": {
            "figure_path": str(figure_path) if figure_created else None,
            "figure_created": figure_created,
            "report_path": str(report_path),
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report_path


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_reports(output_root: Path, models: Sequence[str], benchmarks: Sequence[str], seeds: Sequence[int]) -> None:
    per_seed_rows: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}

    for model in models:
        for benchmark in benchmarks:
            for seed in seeds:
                report_path = output_root / model / benchmark / f"seed_{seed}" / "boundary_report.json"
                if not report_path.exists():
                    continue
                report = load_json(report_path)
                grouped.setdefault((model, benchmark), []).append(report)
                boundaries = report["recommended_boundaries"]
                per_seed_rows.append(
                    {
                        "model": model,
                        "benchmark": benchmark,
                        "seed": seed,
                        "layer_count": boundaries["layer_count"],
                        "b1": boundaries["b1"],
                        "b2": boundaries["b2"],
                        "b1_ratio": boundaries["b1_ratio"],
                        "b2_ratio": boundaries["b2_ratio"],
                        "report_path": str(report_path),
                    }
                )

    aggregate_rows: List[Dict[str, object]] = []
    aggregate_json = {"per_seed": per_seed_rows, "aggregates": {}}
    for (model, benchmark), reports in sorted(grouped.items()):
        b1_values = [report["recommended_boundaries"]["b1"] for report in reports]
        b2_values = [report["recommended_boundaries"]["b2"] for report in reports]
        b1_ratio_values = [report["recommended_boundaries"]["b1_ratio"] for report in reports]
        b2_ratio_values = [report["recommended_boundaries"]["b2_ratio"] for report in reports]
        summary = {
            "model": model,
            "benchmark": benchmark,
            "num_runs": len(reports),
            "layer_count": reports[0]["recommended_boundaries"]["layer_count"],
            "b1": summarize_metric(b1_values),
            "b2": summarize_metric(b2_values),
            "b1_ratio": summarize_metric(b1_ratio_values),
            "b2_ratio": summarize_metric(b2_ratio_values),
            "seeds": [report["seed"] for report in reports],
        }
        aggregate_json["aggregates"][f"{model}::{benchmark}"] = summary
        aggregate_rows.append(
            {
                "model": model,
                "benchmark": benchmark,
                "num_runs": len(reports),
                "layer_count": summary["layer_count"],
                "b1_mean": summary["b1"]["mean"],
                "b1_median": summary["b1"]["median"],
                "b1_std": summary["b1"]["std"],
                "b2_mean": summary["b2"]["mean"],
                "b2_median": summary["b2"]["median"],
                "b2_std": summary["b2"]["std"],
                "b1_ratio_mean": summary["b1_ratio"]["mean"],
                "b1_ratio_median": summary["b1_ratio"]["median"],
                "b2_ratio_mean": summary["b2_ratio"]["mean"],
                "b2_ratio_median": summary["b2_ratio"]["median"],
            }
        )

    write_csv(
        output_root / "boundary_summary_per_seed.csv",
        per_seed_rows,
        ["model", "benchmark", "seed", "layer_count", "b1", "b2", "b1_ratio", "b2_ratio", "report_path"],
    )
    write_csv(
        output_root / "boundary_summary_aggregated.csv",
        aggregate_rows,
        [
            "model",
            "benchmark",
            "num_runs",
            "layer_count",
            "b1_mean",
            "b1_median",
            "b1_std",
            "b2_mean",
            "b2_median",
            "b2_std",
            "b1_ratio_mean",
            "b1_ratio_median",
            "b2_ratio_mean",
            "b2_ratio_median",
        ],
    )
    with open(output_root / "boundary_summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_json, f, indent=2)


def run_sweep(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    ensure_dir(output_root)
    models = args.models
    benchmarks = args.benchmarks
    seeds = args.seeds

    for model in models:
        if model not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {model}")
    for benchmark in benchmarks:
        if benchmark not in BENCHMARK_SPECS:
            raise ValueError(f"Unknown benchmark: {benchmark}")

    script_path = Path(__file__).resolve()
    for model in models:
        for benchmark in benchmarks:
            for seed in seeds:
                combo_dir = output_root / model / benchmark / f"seed_{seed}"
                report_path = combo_dir / "boundary_report.json"
                if report_path.exists() and not args.overwrite:
                    continue
                ensure_dir(combo_dir)
                child_cmd = [
                    sys.executable,
                    str(script_path),
                    "--single-run",
                    "--model",
                    model,
                    "--benchmark",
                    benchmark,
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(combo_dir),
                    "--device",
                    args.device,
                    "--stage1-samples-per-task",
                    str(args.stage1_samples_per_task),
                    "--style-samples-per-task",
                    str(args.style_samples_per_task),
                    "--min-search-layer",
                    str(args.min_search_layer),
                    "--b1-search-start-layer",
                    str(args.b1_search_start_layer),
                    "--b1-search-end-layer",
                    str(args.b1_search_end_layer),
                ]
                if args.reasoning_tasks:
                    child_cmd.extend(["--reasoning-tasks", *args.reasoning_tasks])
                if args.style_tasks:
                    child_cmd.extend(["--style-tasks", *args.style_tasks])
                subprocess.run(child_cmd, check=True, cwd=str(REPO_ROOT))

    aggregate_reports(output_root, models, benchmarks, seeds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run hierarchical boundary analysis for multiple model/benchmark/seed combinations.",
    )
    parser.add_argument("--single-run", action="store_true", help="Internal flag for one model-benchmark-seed run.")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument("--benchmark", choices=sorted(BENCHMARK_SPECS.keys()))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--models", nargs="+", default=["llava", "internvl"])
    parser.add_argument("--benchmarks", nargs="+", default=["ucit", "acl", "dcl"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--stage1-samples-per-task", type=int, default=128)
    parser.add_argument("--style-samples-per-task", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min-search-layer", type=int, default=20)
    parser.add_argument("--b1-search-start-layer", type=int, default=2)
    parser.add_argument("--b1-search-end-layer", type=int, default=16)
    parser.add_argument("--reasoning-tasks", nargs="+", default=None)
    parser.add_argument("--style-tasks", nargs="+", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.single_run:
        if not args.model or not args.benchmark or args.output_dir is None:
            parser.error("--single-run requires --model, --benchmark, and --output-dir.")
        run_single_analysis(args)
        return

    run_sweep(args)


if __name__ == "__main__":
    main()
