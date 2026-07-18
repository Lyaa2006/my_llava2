#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAVA_ROOT = os.path.join(REPO_ROOT, "LLaVA", "LoRA-FT")
if LLAVA_ROOT not in sys.path:
    sys.path.insert(0, LLAVA_ROOT)

from llava.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import tokenizer_image_token  # noqa: E402
from llava.model import LlavaLlamaForCausalLM  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402


DEFAULT_MODEL_PATH = "/mnt/lyaa/my_llava/llava-v1.5-7b"
DEFAULT_VISION_TOWER_PATH = "/mnt/lyaa/my_llava/clip-vit-large-patch14-336"
DEFAULT_UCIT_ROOT = "/mnt/lyaa/my_llava/UCIT"
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "docs", "hierarchical_boundary_outputs")

TASK_FILES = {
    "ImageNet-R": "ImageNet-R/test_3000.json",
    "ArxivQA": "ArxivQA/test_3000.json",
    "VizWiz": "VizWiz/test_3000.json",
    "IconQA": "IconQA/test_3000.json",
    "CLEVR": "CLEVR/test_3000.json",
    "Flickr30k": "Flickr30k/test_3000.json",
}

REASONING_TASKS = ("ArxivQA", "IconQA", "CLEVR")
STYLE_TASKS = ("ImageNet-R", "ArxivQA", "IconQA", "CLEVR")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_task_samples(ucit_root: str, task_names: Sequence[str], samples_per_task: int, seed: int) -> Dict[str, List[Sample]]:
    rng = random.Random(seed)
    task_samples: Dict[str, List[Sample]] = {}
    for task in task_names:
        json_path = os.path.join(ucit_root, TASK_FILES[task])
        with open(json_path, "r", encoding="utf-8") as f:
            raw_samples = json.load(f)
        if samples_per_task < len(raw_samples):
            raw_samples = rng.sample(raw_samples, samples_per_task)
        task_samples[task] = [
            Sample(
                task=task,
                question_id=str(item["question_id"]),
                image_path=os.path.join(ucit_root, "datasets", item["image"]),
                text=item["text"].strip(),
                answer=str(item.get("answer", "")).strip(),
            )
            for item in raw_samples
        ]
    return task_samples


def build_prompt(question: str, mm_use_im_start_end: bool, conv_mode: str) -> str:
    if mm_use_im_start_end:
        user_prompt = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{question}"
    else:
        user_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_image_tensor(image_path: str, image_processor, device: str, blank: bool = False, image_size: int = 336) -> torch.Tensor:
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


def strip_existing_instruction(text: str) -> str:
    text = re.sub(r"Answer with the option.*$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    text = re.sub(r"Answer the question using a single word or phrase\.\s*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"Generate a brief caption for the image\.\s*$", "", text, flags=re.IGNORECASE).strip()
    return text


def has_explicit_option_labels(text: str) -> bool:
    return bool(re.search(r"(^|\n)\s*([A-D]|[0-9])[\.\)]\s+\S", text))


def build_style_variants(sample: Sample) -> Dict[str, str]:
    base = strip_existing_instruction(sample.text)
    variants: Dict[str, str] = {}

    if sample.task == "ImageNet-R":
        variants["phrase"] = f"{base}\nAnswer using a single word or short phrase."
        variants["sentence"] = f"{base}\nAnswer in one complete sentence."
        return variants

    if sample.task == "CLEVR":
        variants["phrase"] = f"{base}\nAnswer using a short phrase."
        variants["sentence"] = f"{base}\nAnswer in one complete sentence."
        return variants

    if sample.task in ("ArxivQA", "IconQA") and has_explicit_option_labels(sample.text):
        variants["label"] = f"{base}\nAnswer with the option id only."
        variants["option_text"] = f"{base}\nAnswer with the exact option text only. Do not output the option id."
        variants["sentence"] = f"{base}\nAnswer in one complete sentence using the exact option content. Do not output the option id."
        return variants

    variants["phrase"] = f"{base}\nAnswer using a short phrase."
    variants["sentence"] = f"{base}\nAnswer in one complete sentence."
    return variants


def choose_deep_middle_boundary(reasoning_curve: np.ndarray, task_curve: np.ndarray) -> Tuple[int, np.ndarray]:
    layer_count = len(reasoning_curve)
    raw_scores = np.full(layer_count, -1e9, dtype=np.float32)
    search_start = 2
    search_end = min(16, layer_count - 1)
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
    output_path: str,
    layers: np.ndarray,
    reasoning_curve: np.ndarray,
    task_curve: np.ndarray,
    style_logit_curve: np.ndarray,
    deep_boundary: int,
    mid_shallow_boundary: int,
) -> None:
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
    ax.set_title("Hierarchical Boundary Analysis on UCIT")
    ax.legend(frameon=True, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_report(
    output_path: str,
    deep_boundary: int,
    mid_shallow_boundary: int,
    reasoning_curve: np.ndarray,
    task_curve: np.ndarray,
    style_logit_curve: np.ndarray,
    handoff_curve: np.ndarray,
    config: Dict[str, int | str | None],
) -> None:
    report = {
        "recommended_boundaries": {
            "deep_middle_boundary": deep_boundary,
            "middle_shallow_boundary": mid_shallow_boundary,
            "deep_layers": [1, deep_boundary],
            "middle_layers": [deep_boundary + 1, mid_shallow_boundary - 1],
            "shallow_layers": [mid_shallow_boundary, len(reasoning_curve)],
        },
        "config": config,
        "curves": {
            "reasoning_signal": reasoning_curve.tolist(),
            "task_separation": task_curve.tolist(),
            "style_logit_signal": style_logit_curve.tolist(),
            "deep_middle_handoff_score": handoff_curve.tolist(),
        },
        "notes": [
            "This experiment uses plain base LLaVA forward only.",
            "Deep-middle boundary is selected by a handoff score: curve closeness plus reasoning rise and task decay.",
            "Middle-shallow boundary is selected as the late-layer peak of the style logit signal.",
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified analysis for deep-middle and middle-shallow boundaries on base LLaVA.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vision-tower-path", type=str, default=DEFAULT_VISION_TOWER_PATH)
    parser.add_argument("--ucit-root", type=str, default=DEFAULT_UCIT_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage1-samples-per-task", type=int, default=32)
    parser.add_argument("--style-samples-per-task", type=int, default=32)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-search-layer", type=int, default=20)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)
    disable_torch_init()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, local_files_only=True)
    cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    cfg.mm_vision_tower = args.vision_tower_path
    if not hasattr(cfg, "mm_vision_select_layer"):
        cfg.mm_vision_select_layer = -2
    if hasattr(cfg, "mm_text_tower"):
        delattr(cfg, "mm_text_tower")

    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path,
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
    vision_tower.vision_tower_name = args.vision_tower_path
    vision_tower.load_model()
    vision_tower.to(device=args.device, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)

    stage1_samples = load_task_samples(args.ucit_root, tuple(TASK_FILES.keys()), args.stage1_samples_per_task, args.seed)
    style_samples = load_task_samples(args.ucit_root, STYLE_TASKS, args.style_samples_per_task, args.seed + 17)

    all_reps = []
    all_labels = []
    reasoning_distances = []
    logit_style_all = []

    progress_total = sum(len(v) for v in stage1_samples.values()) + sum(len(v) for v in style_samples.values())
    progress = tqdm(total=progress_total, desc="Hierarchical boundary analysis")

    with torch.inference_mode():
        for task, samples in stage1_samples.items():
            for sample in samples:
                image_tensor = load_image_tensor(sample.image_path, image_processor, args.device)
                prompt = build_prompt(sample.text, mm_use_im_start_end, args.conv_mode)
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
                outputs = model(
                    input_ids=input_ids,
                    images=image_tensor,
                    output_hidden_states=True,
                    return_dict=True,
                )
                pooled = last_token_hidden_stack(outputs.hidden_states)
                all_reps.append(pooled)
                all_labels.append(task)

                if task in REASONING_TASKS:
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
                    progress.update(1)
                    continue
                image_tensor = load_image_tensor(sample.image_path, image_processor, args.device)
                logit_stacks = {}
                for variant_name, variant_text in variants.items():
                    prompt = build_prompt(variant_text, mm_use_im_start_end, args.conv_mode)
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
                    outputs = model(
                        input_ids=input_ids,
                        images=image_tensor,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    logit_stacks[variant_name] = last_token_logits_stack(model, outputs.hidden_states)

                names = list(logit_stacks.keys())
                logit_pairs = []
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        logit_pairs.append(js_divergence(logit_stacks[names[i]], logit_stacks[names[j]]))
                if logit_pairs:
                    logit_style_all.append(np.mean(np.stack(logit_pairs, axis=0), axis=0))
                progress.update(1)

    progress.close()

    layer_reps = np.stack(all_reps, axis=0)
    reasoning_curve = normalize_curve(smooth_curve(np.mean(np.stack(reasoning_distances, axis=0), axis=0), window=3))
    task_curve = normalize_curve(smooth_curve(fisher_task_separation(layer_reps, all_labels), window=3))
    style_logit_curve = normalize_curve(smooth_curve(np.mean(np.stack(logit_style_all, axis=0), axis=0), window=3))

    deep_boundary, handoff_curve = choose_deep_middle_boundary(reasoning_curve, task_curve)
    mid_shallow_boundary = choose_mid_shallow_boundary(style_logit_curve, deep_boundary, args.min_search_layer)

    layer_count = len(reasoning_curve)
    deep_boundary = max(1, min(deep_boundary, layer_count - 1))
    mid_shallow_boundary = max(deep_boundary + 1, min(mid_shallow_boundary, layer_count))

    layers = np.arange(1, layer_count + 1)
    report_path = os.path.join(args.output_dir, "hierarchical_boundary_report.json")
    figure_path = os.path.join(args.output_dir, "hierarchical_boundary_curves.png")
    save_report(
        report_path,
        deep_boundary,
        mid_shallow_boundary,
        reasoning_curve,
        task_curve,
        style_logit_curve,
        handoff_curve,
        {
            "model_path": args.model_path,
            "vision_tower_path": args.vision_tower_path,
            "ucit_root": args.ucit_root,
            "stage1_samples_per_task": args.stage1_samples_per_task,
            "style_samples_per_task": args.style_samples_per_task,
            "seed": args.seed,
            "min_search_layer": args.min_search_layer,
        },
    )
    plot_curves(
        figure_path,
        layers,
        reasoning_curve,
        task_curve,
        style_logit_curve,
        deep_boundary,
        mid_shallow_boundary,
    )

    print(f"Deep-middle boundary: layer {deep_boundary}")
    print(f"Middle-shallow boundary: layer {mid_shallow_boundary}")
    print(f"Saved report to: {report_path}")
    print(f"Saved figure to: {figure_path}")


if __name__ == "__main__":
    main()
