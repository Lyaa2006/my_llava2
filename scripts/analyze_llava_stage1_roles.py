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
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token  # noqa: E402
from llava.model import LlavaLlamaForCausalLM  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402


DEFAULT_MODEL_PATH = "/mnt/lyaa/my_llava/llava-v1.5-7b"
DEFAULT_UCIT_ROOT = "/mnt/lyaa/my_llava/UCIT"
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "docs", "stage1_outputs")
DEFAULT_CLIP_TOWER_PATH = "/mnt/lyaa/my_llava/clip-vit-large-patch14-336"

TASK_FILES = {
    "ImageNet-R": "ImageNet-R/test_3000.json",
    "ArxivQA": "ArxivQA/test_3000.json",
    "VizWiz": "VizWiz/test_3000.json",
    "IconQA": "IconQA/test_3000.json",
    "CLEVR": "CLEVR/test_3000.json",
    "Flickr30k": "Flickr30k/test_3000.json",
}

REASONING_TASKS = ("ArxivQA", "IconQA", "CLEVR")
STYLE_TASKS = ("VizWiz", "Flickr30k", "ImageNet-R")
PLOT_COLORS = {
    "reasoning": "#d1495b",
    "task": "#2f6690",
    "style": "#edae49",
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


def load_task_samples(ucit_root: str, samples_per_task: int, seed: int) -> Dict[str, List[Sample]]:
    rng = random.Random(seed)
    task_samples: Dict[str, List[Sample]] = {}
    for task, rel_path in TASK_FILES.items():
        json_path = os.path.join(ucit_root, rel_path)
        with open(json_path, "r", encoding="utf-8") as f:
            raw_samples = json.load(f)
        if samples_per_task < len(raw_samples):
            raw_samples = rng.sample(raw_samples, samples_per_task)
        samples = []
        for item in raw_samples:
            samples.append(
                Sample(
                    task=task,
                    question_id=str(item["question_id"]),
                    image_path=os.path.join(ucit_root, "datasets", item["image"]),
                    text=item["text"].strip(),
                    answer=str(item.get("answer", "")).strip(),
                )
            )
        task_samples[task] = samples
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


def load_image_tensor(
    image_path: str,
    image_processor,
    device: torch.device,
    image_size: int,
    blank: bool = False,
) -> torch.Tensor:
    if blank:
        image = Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
    else:
        image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return pixel_values.unsqueeze(0).to(device=device, dtype=torch.float16)


def pool_hidden_states(hidden_states: Sequence[torch.Tensor]) -> np.ndarray:
    pooled = []
    for layer_hidden in hidden_states[1:]:
        vec = layer_hidden[0, -1].detach().float().cpu()
        vec = vec / vec.norm(p=2).clamp_min(1e-6)
        pooled.append(vec.numpy())
    return np.stack(pooled, axis=0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return 1.0 - (num / np.clip(den, 1e-8, None))


def smooth_curve(values: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def normalize_curve(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def rewrite_style_prompt(task: str, text: str, variant: str) -> str:
    if variant == "original":
        return text

    stripped = text.strip()
    if task in ("VizWiz", "Flickr30k"):
        stem = re.sub(r"Generate a brief caption for the image\.\s*$", "", stripped).strip()
        if variant == "short":
            return f"{stem}\nDescribe the scene in 4 to 7 words."
        return f"{stem}\nDescribe the scene in one complete sentence."

    if "Answer with the option" in stripped:
        stem = re.sub(r"Answer with the option.*$", "", stripped).strip()
        if variant == "short":
            return f"{stem}\nReply with only the option id."
        return f"{stem}\nState the final choice as 'Option X' and stop."

    if "Answer the question using a single word or phrase." in stripped:
        stem = stripped.replace("Answer the question using a single word or phrase.", "").strip()
        if variant == "short":
            return f"{stem}\nAnswer using a short noun phrase only."
        return f"{stem}\nAnswer in one complete sentence."

    if variant == "short":
        return f"{stripped}\nReply as briefly as possible."
    return f"{stripped}\nReply in one complete sentence."


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
            group = layer_matrix[[idx for idx, cur in enumerate(labels) if cur == label]]
            group_mean = group.mean(axis=0)
            between += float(group.shape[0]) * float(np.sum((group_mean - global_mean) ** 2))
            within += float(np.sum((group - group_mean) ** 2))
        scores[layer_idx] = between / max(within, 1e-8)
    return scores


def choose_boundaries(
    reasoning_curve: np.ndarray,
    task_curve: np.ndarray,
    style_curve: np.ndarray,
) -> Tuple[int, int]:
    num_layers = len(reasoning_curve)
    reasoning_peak = int(np.argmax(reasoning_curve))

    b1 = None
    for idx in range(reasoning_peak + 1, num_layers - 2):
        if task_curve[idx] >= reasoning_curve[idx]:
            b1 = idx
            break
    if b1 is None:
        b1 = int(np.argmax(task_curve - reasoning_curve))
    b1 = max(1, min(b1, num_layers - 3))

    b2 = None
    for idx in range(b1 + 1, num_layers - 1):
        if style_curve[idx] >= task_curve[idx]:
            b2 = idx
            break
    if b2 is None:
        tail = style_curve[b1 + 1 :] - task_curve[b1 + 1 :]
        b2 = b1 + 1 + int(np.argmax(tail))
    b2 = max(b1 + 1, min(b2, num_layers - 1))
    return b1 + 1, b2 + 1


def plot_main_curves(
    layers: np.ndarray,
    reasoning_curve: np.ndarray,
    task_curve: np.ndarray,
    style_curve: np.ndarray,
    b1: int,
    b2: int,
    output_path: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=180)
    ax.plot(layers, reasoning_curve, color=PLOT_COLORS["reasoning"], linewidth=2.8, label="Reasoning Signal")
    ax.plot(layers, task_curve, color=PLOT_COLORS["task"], linewidth=2.8, label="Task Separation")
    ax.plot(layers, style_curve, color=PLOT_COLORS["style"], linewidth=2.8, label="Style Sensitivity")

    ax.axvspan(1, b1, color=PLOT_COLORS["reasoning"], alpha=0.08)
    ax.axvspan(b1, b2, color=PLOT_COLORS["task"], alpha=0.08)
    ax.axvspan(b2, layers[-1], color=PLOT_COLORS["style"], alpha=0.08)
    ax.axvline(b1, linestyle="--", color="#444444", linewidth=1.4)
    ax.axvline(b2, linestyle="--", color="#444444", linewidth=1.4)
    ax.text(b1, 1.03, f"b1 = {b1}", ha="center", va="bottom", fontsize=10)
    ax.text(b2, 1.03, f"b2 = {b2}", ha="center", va="bottom", fontsize=10)

    ax.set_xlim(1, layers[-1])
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized Score")
    ax.set_title("Stage 1 Layer Role Identification on UCIT")
    ax.legend(frameon=True, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_reasoning_heatmap(
    per_task_reasoning: Dict[str, np.ndarray],
    b1: int,
    b2: int,
    output_path: str,
) -> None:
    plt.style.use("seaborn-v0_8-white")
    tasks = list(per_task_reasoning.keys())
    matrix = np.stack([per_task_reasoning[task] for task in tasks], axis=0)

    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=180)
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(np.arange(1, matrix.shape[1] + 1))
    ax.set_xlabel("Layer")
    ax.set_title("Image Contribution by Task")
    ax.axvline(b1 - 0.5, linestyle="--", color="white", linewidth=1.2)
    ax.axvline(b2 - 0.5, linestyle="--", color="white", linewidth=1.2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Cosine Distance vs Blank Image")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_report(
    output_path: str,
    b1: int,
    b2: int,
    reasoning_curve: np.ndarray,
    task_curve: np.ndarray,
    style_curve: np.ndarray,
    args,
) -> None:
    report = {
        "recommended_boundaries": {
            "boundary_1": b1,
            "boundary_2": b2,
            "deep_layers": [1, b1],
            "middle_layers": [b1 + 1, b2],
            "shallow_layers": [b2 + 1, len(reasoning_curve)],
        },
        "assumptions": {
            "reasoning_tasks": list(REASONING_TASKS),
            "style_tasks": list(STYLE_TASKS),
        },
        "config": {
            "model_path": args.model_path,
            "ucit_root": args.ucit_root,
            "samples_per_task": args.samples_per_task,
            "conv_mode": args.conv_mode,
            "seed": args.seed,
        },
        "curves": {
            "reasoning_signal": reasoning_curve.tolist(),
            "task_separation": task_curve.tolist(),
            "style_sensitivity": style_curve.tolist(),
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert-free stage-1 layer role identification for LLaVA on UCIT.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--ucit-root", type=str, default=DEFAULT_UCIT_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vision-tower-path", type=str, default=DEFAULT_CLIP_TOWER_PATH)
    parser.add_argument("--samples-per-task", type=int, default=12)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-image-size", type=int, default=336)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)
    disable_torch_init()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    model_name = get_model_name_from_path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        local_files_only=True,
    )
    cfg = AutoConfig.from_pretrained(
        args.model_path,
        local_files_only=True,
    )
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

    vision_tower = model.get_vision_tower()
    if vision_tower is None:
        raise RuntimeError("Failed to initialize the local vision tower.")
    vision_tower.vision_tower_name = args.vision_tower_path
    vision_tower.load_model()
    vision_tower.to(device=args.device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    model.eval()
    device = next(model.parameters()).device
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)

    task_samples = load_task_samples(args.ucit_root, args.samples_per_task, args.seed)

    all_reps = []
    all_labels = []
    style_distances = []
    reasoning_distances = []
    per_task_reasoning = {}

    task_order = list(TASK_FILES.keys())
    total_steps = sum(len(task_samples[task]) for task in task_order)
    progress = tqdm(total=total_steps, desc="Encoding UCIT samples")

    with torch.inference_mode():
        for task in task_order:
            task_reasoning_curves = []
            for sample in task_samples[task]:
                image_tensor = load_image_tensor(
                    sample.image_path,
                    image_processor,
                    device,
                    image_size=args.max_image_size,
                    blank=False,
                )
                prompt = build_prompt(sample.text, mm_use_im_start_end, args.conv_mode)
                input_ids = tokenizer_image_token(
                    prompt,
                    tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                ).unsqueeze(0).to(device)

                outputs = model(
                    input_ids=input_ids,
                    images=image_tensor,
                    output_hidden_states=True,
                    return_dict=True,
                )
                pooled = pool_hidden_states(outputs.hidden_states)
                all_reps.append(pooled)
                all_labels.append(task)

                style_variants = ["short", "long"]
                for variant in style_variants:
                    style_prompt = build_prompt(
                        rewrite_style_prompt(task, sample.text, variant),
                        mm_use_im_start_end,
                        args.conv_mode,
                    )
                    style_input_ids = tokenizer_image_token(
                        style_prompt,
                        tokenizer,
                        IMAGE_TOKEN_INDEX,
                        return_tensors="pt",
                    ).unsqueeze(0).to(device)
                    style_outputs = model(
                        input_ids=style_input_ids,
                        images=image_tensor,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    style_pooled = pool_hidden_states(style_outputs.hidden_states)
                    style_distances.append(cosine_distance(pooled, style_pooled))

                blank_tensor = load_image_tensor(
                    sample.image_path,
                    image_processor,
                    device,
                    image_size=args.max_image_size,
                    blank=True,
                )
                blank_outputs = model(
                    input_ids=input_ids,
                    images=blank_tensor,
                    output_hidden_states=True,
                    return_dict=True,
                )
                blank_pooled = pool_hidden_states(blank_outputs.hidden_states)
                image_distance = cosine_distance(pooled, blank_pooled)
                task_reasoning_curves.append(image_distance)
                if task in REASONING_TASKS:
                    reasoning_distances.append(image_distance)

                progress.update(1)

            per_task_reasoning[task] = np.mean(np.stack(task_reasoning_curves, axis=0), axis=0)

    progress.close()

    layer_reps = np.stack(all_reps, axis=0)
    task_curve_raw = fisher_task_separation(layer_reps, all_labels)
    style_curve_raw = np.mean(np.stack(style_distances, axis=0), axis=0)
    reasoning_curve_raw = np.mean(np.stack(reasoning_distances, axis=0), axis=0)

    reasoning_curve = normalize_curve(smooth_curve(reasoning_curve_raw))
    task_curve = normalize_curve(smooth_curve(task_curve_raw))
    style_curve = normalize_curve(smooth_curve(style_curve_raw))

    num_layers = len(reasoning_curve)
    layers = np.arange(1, num_layers + 1)
    b1, b2 = choose_boundaries(reasoning_curve, task_curve, style_curve)

    plot_main_curves(
        layers,
        reasoning_curve,
        task_curve,
        style_curve,
        b1,
        b2,
        os.path.join(args.output_dir, "stage1_boundary_curves.png"),
    )
    plot_reasoning_heatmap(
        per_task_reasoning,
        b1,
        b2,
        os.path.join(args.output_dir, "stage1_image_contribution_heatmap.png"),
    )
    save_report(
        os.path.join(args.output_dir, "stage1_boundary_report.json"),
        b1,
        b2,
        reasoning_curve,
        task_curve,
        style_curve,
        args,
    )

    print(f"Recommended boundary 1: layer {b1}")
    print(f"Recommended boundary 2: layer {b2}")
    print(f"Saved figures and report to: {args.output_dir}")


if __name__ == "__main__":
    main()
