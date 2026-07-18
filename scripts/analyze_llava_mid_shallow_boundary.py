#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

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
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "docs", "mid_shallow_outputs")

TASK_FILES = {
    "ImageNet-R": "ImageNet-R/test_3000.json",
    "ArxivQA": "ArxivQA/test_3000.json",
    "IconQA": "IconQA/test_3000.json",
    "CLEVR": "CLEVR/test_3000.json",
}


@dataclass
class Sample:
    task: str
    question_id: str
    image_path: str
    text: str


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
    samples_by_task: Dict[str, List[Sample]] = {}
    for task, rel_path in TASK_FILES.items():
        file_path = os.path.join(ucit_root, rel_path)
        with open(file_path, "r", encoding="utf-8") as f:
            raw_samples = json.load(f)
        if samples_per_task < len(raw_samples):
            raw_samples = rng.sample(raw_samples, samples_per_task)
        samples_by_task[task] = [
            Sample(
                task=task,
                question_id=str(item["question_id"]),
                image_path=os.path.join(ucit_root, "datasets", item["image"]),
                text=item["text"].strip(),
            )
            for item in raw_samples
        ]
    return samples_by_task


def build_prompt(question: str, mm_use_im_start_end: bool, conv_mode: str) -> str:
    if mm_use_im_start_end:
        user_prompt = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{question}"
    else:
        user_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_image_tensor(image_path: str, image_processor, device: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return pixel_values.unsqueeze(0).to(device=device, dtype=torch.float16)


def last_token_hidden_stack(hidden_states: Sequence[torch.Tensor]) -> np.ndarray:
    pooled = []
    for layer_hidden in hidden_states[1:]:
        vec = layer_hidden[0, -1].detach().float().cpu()
        vec = vec / vec.norm(p=2).clamp_min(1e-6)
        pooled.append(vec.numpy())
    return np.stack(pooled, axis=0)


def last_token_logits_stack(model, hidden_states: Sequence[torch.Tensor]) -> np.ndarray:
    per_layer_logits = []
    norm = model.get_model().norm
    lm_head = model.lm_head
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
    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


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


def select_boundary(style_curve: np.ndarray, deep_boundary: int, min_search_layer: int) -> int:
    layer_count = len(style_curve)
    search_start = min(max(deep_boundary + 1, min_search_layer, 1), layer_count)
    if search_start >= layer_count:
        return layer_count

    tail_curve = style_curve[search_start - 1 :]
    tail_layers = np.arange(search_start, layer_count + 1)

    depth_prior = np.linspace(0.45, 1.0, num=len(tail_curve), dtype=np.float32)
    weighted = tail_curve * depth_prior
    peak = float(weighted.max())
    threshold = 0.85 * peak

    window = 4
    for idx in range(0, max(1, len(weighted) - window + 1)):
        local = weighted[idx : idx + window]
        if np.all(local >= 0.80 * peak) and weighted[idx] >= threshold:
            return int(tail_layers[idx])

    return int(tail_layers[int(np.argmax(weighted))])


def save_report(
    output_path: str,
    boundary: int,
    deep_boundary: int,
    min_search_layer: int,
    hidden_curve: np.ndarray,
    logit_curve: np.ndarray,
    combined_curve: np.ndarray,
    per_task_curves: Dict[str, List[float]],
    usable_counts: Dict[str, int],
    args,
) -> None:
    report = {
        "recommended_middle_shallow_boundary": boundary,
        "fixed_deep_middle_boundary": deep_boundary,
        "min_search_layer": min_search_layer,
        "config": {
            "model_path": args.model_path,
            "vision_tower_path": args.vision_tower_path,
            "ucit_root": args.ucit_root,
            "samples_per_task": args.samples_per_task,
            "conv_mode": args.conv_mode,
            "seed": args.seed,
        },
        "hidden_style_curve": hidden_curve.tolist(),
        "logit_style_curve": logit_curve.tolist(),
        "combined_style_curve": combined_curve.tolist(),
        "per_task_combined_curve": per_task_curves,
        "usable_sample_count": usable_counts,
        "notes": [
            "This experiment uses plain base LLaVA forward only.",
            "The main shallow metric is the first-answer-token logit divergence estimated with a layer-wise logit lens.",
            "The auxiliary metric is the hidden-state divergence at the last prompt token, directly adjacent to answer generation.",
            "Boundary search is constrained to late layers after the fixed deep-middle boundary.",
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the middle-shallow boundary using output-style control prompts.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vision-tower-path", type=str, default=DEFAULT_VISION_TOWER_PATH)
    parser.add_argument("--ucit-root", type=str, default=DEFAULT_UCIT_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-task", type=int, default=16)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deep-boundary", type=int, default=10)
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
    task_samples = load_task_samples(args.ucit_root, args.samples_per_task, args.seed)

    all_hidden_distances = []
    all_logit_divergences = []
    per_task_hidden: Dict[str, List[np.ndarray]] = {task: [] for task in TASK_FILES}
    per_task_logits: Dict[str, List[np.ndarray]] = {task: [] for task in TASK_FILES}
    usable_counts = {task: 0 for task in TASK_FILES}

    total = sum(len(samples) for samples in task_samples.values())
    progress = tqdm(total=total, desc="Analyzing mid-shallow boundary")

    with torch.inference_mode():
        for task, samples in task_samples.items():
            for sample in samples:
                variants = build_style_variants(sample)
                if len(variants) < 2:
                    progress.update(1)
                    continue

                image_tensor = load_image_tensor(sample.image_path, image_processor, args.device)
                hidden_stacks = {}
                logit_stacks = {}
                for variant_name, variant_text in variants.items():
                    prompt = build_prompt(variant_text, mm_use_im_start_end, args.conv_mode)
                    input_ids = tokenizer_image_token(
                        prompt,
                        tokenizer,
                        IMAGE_TOKEN_INDEX,
                        return_tensors="pt",
                    ).unsqueeze(0).to(args.device)
                    outputs = model(
                        input_ids=input_ids,
                        images=image_tensor,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_stacks[variant_name] = last_token_hidden_stack(outputs.hidden_states)
                    logit_stacks[variant_name] = last_token_logits_stack(model, outputs.hidden_states)

                names = list(hidden_stacks.keys())
                pairwise_hidden = []
                pairwise_logits = []
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        pairwise_hidden.append(cosine_distance(hidden_stacks[names[i]], hidden_stacks[names[j]]))
                        pairwise_logits.append(js_divergence(logit_stacks[names[i]], logit_stacks[names[j]]))
                if not pairwise_hidden or not pairwise_logits:
                    progress.update(1)
                    continue

                mean_hidden = np.mean(np.stack(pairwise_hidden, axis=0), axis=0)
                mean_logits = np.mean(np.stack(pairwise_logits, axis=0), axis=0)

                all_hidden_distances.append(mean_hidden)
                all_logit_divergences.append(mean_logits)
                per_task_hidden[task].append(mean_hidden)
                per_task_logits[task].append(mean_logits)
                usable_counts[task] += 1
                progress.update(1)

    progress.close()

    if not all_hidden_distances or not all_logit_divergences:
        raise RuntimeError("No usable samples were found for middle-shallow analysis.")

    hidden_curve = normalize_curve(smooth_curve(np.mean(np.stack(all_hidden_distances, axis=0), axis=0), window=3))
    logit_curve = normalize_curve(smooth_curve(np.mean(np.stack(all_logit_divergences, axis=0), axis=0), window=3))
    combined_curve = normalize_curve(smooth_curve(0.35 * hidden_curve + 0.65 * logit_curve, window=3))
    boundary = select_boundary(combined_curve, args.deep_boundary, args.min_search_layer)

    per_task_curves: Dict[str, List[float]] = {}
    for task in TASK_FILES:
        if per_task_hidden[task] and per_task_logits[task]:
            hidden_curve_task = normalize_curve(
                smooth_curve(np.mean(np.stack(per_task_hidden[task], axis=0), axis=0), window=3)
            )
            logit_curve_task = normalize_curve(
                smooth_curve(np.mean(np.stack(per_task_logits[task], axis=0), axis=0), window=3)
            )
            combined_task = normalize_curve(smooth_curve(0.35 * hidden_curve_task + 0.65 * logit_curve_task, window=3))
            per_task_curves[task] = combined_task.tolist()
        else:
            per_task_curves[task] = []

    report_path = os.path.join(args.output_dir, "mid_shallow_boundary_report.json")
    save_report(
        report_path,
        boundary,
        args.deep_boundary,
        args.min_search_layer,
        hidden_curve,
        logit_curve,
        combined_curve,
        per_task_curves,
        usable_counts,
        args,
    )

    print(f"Recommended middle-shallow boundary: layer {boundary}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
