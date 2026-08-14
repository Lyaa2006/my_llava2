#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from stage1_protocol_lib import ensure_dir, load_json, read_jsonl, write_json


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_FAMILY = os.environ.get("STAGE1_MODEL_FAMILY", "llava").lower()
MODEL_ROOT_NAME = "InternVL" if MODEL_FAMILY in {"internvl", "intern-vl"} else "LLaVA"
LLAVA_ROOT = REPO_ROOT / MODEL_ROOT_NAME / "MR-LoRA"
if str(LLAVA_ROOT) not in sys.path:
    sys.path.insert(0, str(LLAVA_ROOT))

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402


STYLE_METRIC_NAME = "expanded_style_vocab_js"

STYLE_TOKEN_TEXTS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "yes",
    "no",
    "Yes",
    "No",
    "true",
    "false",
    "True",
    "False",
    ".",
    ",",
    ":",
    ";",
    "-",
    "The",
    "There",
    "It",
    "This",
    "An",
    "is",
    "are",
    "shows",
    "contains",
    "appears",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Stage 1 boundary windows with real model forward passes.")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--probe-jsonl", type=str, required=True)
    parser.add_argument("--split-manifest", type=str, default=None)
    parser.add_argument("--split-type-filter", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument(
        "--style-aggregation",
        type=str,
        default="raw",
        choices=["raw", "cluster_balanced"],
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--min-consecutive", type=int, default=2)
    parser.add_argument("--min-segment-width", type=int, default=4)
    parser.add_argument(
        "--boundary-strategy",
        type=str,
        default="ordered",
        choices=["ordered", "transition", "windowed_transition", "b1_only_crossover"],
    )
    parser.add_argument("--b1-search-low", type=int, default=10)
    parser.add_argument("--b1-search-high", type=int, default=20)
    parser.add_argument("--transition-anchor-offset", type=int, default=2)
    parser.add_argument("--transition-margin", type=float, default=0.03)
    parser.add_argument("--transition-min-run", type=int, default=3)
    parser.add_argument("--late-preference", type=float, default=0.01)
    parser.add_argument("--min-transition-support", type=float, default=0.01)
    parser.add_argument("--min-pre-reasoning-margin", type=float, default=-0.08)
    parser.add_argument("--b1-band-score-tolerance", type=float, default=0.03)
    parser.add_argument("--b1-band-min-width", type=int, default=3)
    parser.add_argument("--b1-band-max-width", type=int, default=5)
    parser.add_argument("--max-splits", type=int, default=0)
    return parser


def load_model(model_config_path: Path, device: str):
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[stage1] CUDA unavailable, falling back to cpu for smoke/runtime.")
        device = "cpu"
    cfg = load_json(model_config_path)
    model_path = cfg["model_name"]
    model_base = cfg.get("model_base")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_path,
        device_map={"": device},
        device=device,
    )
    model.eval()
    return tokenizer, model, image_processor, device


def build_prompt(question_text: str, tokenizer, conv_mode: str) -> str:
    model_config = getattr(tokenizer, "model_max_length", None)
    mm_use_im_start_end = False
    user_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{question_text}"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_image_tensor(image_path: str, image_processor, model, device: str, blank: bool = False) -> torch.Tensor:
    image_size = getattr(image_processor, "crop_size", None)
    if isinstance(image_size, dict):
        image_size = image_size.get("height") or image_size.get("width") or image_size.get("shortest_edge")
    if not isinstance(image_size, int):
        image_size = 336
    if blank:
        image = Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
    else:
        image = Image.open(image_path).convert("RGB")
    processed = process_images([image], image_processor, model.config)
    if isinstance(processed, list):
        processed = torch.stack(processed, dim=0)
    return processed.to(device=device, dtype=torch.float16)


def tokenize_prompt(prompt: str, tokenizer, device: str) -> torch.Tensor:
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids.to(device=device)


def build_style_token_ids(tokenizer) -> List[int]:
    token_ids = set()
    for tok in STYLE_TOKEN_TEXTS:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.add(int(ids[0]))
    for extra in (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id):
        if extra is not None:
            token_ids.add(int(extra))
    return sorted(token_ids)


def hidden_stack_from_outputs(outputs) -> np.ndarray:
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("Model did not return hidden states.")
    stack = []
    for layer_hidden in hidden_states[1:]:
        vec = layer_hidden[0, -1].detach().float().cpu().numpy()
        stack.append(vec)
    return np.stack(stack, axis=0)


def logits_stack_from_hidden(model, hidden_stack: np.ndarray, token_ids: Sequence[int]) -> np.ndarray:
    norm = model.model.norm
    lm_head = model.lm_head
    per_layer = []
    token_ids = list(token_ids)
    for layer_hidden in hidden_stack:
        tensor = torch.from_numpy(layer_hidden).to(device=next(model.parameters()).device, dtype=torch.float16)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        normed = norm(tensor)
        logits = lm_head(normed)[0, 0]
        if token_ids:
            logits = logits[token_ids]
        per_layer.append(logits.detach().float().cpu().numpy())
    return np.stack(per_layer, axis=0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-8:
        return 0.0
    return 1.0 - (num / den)


def softmax_np(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(np.sum(exp, axis=-1, keepdims=True), 1e-8, None)


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    p = softmax_np(a)
    q = softmax_np(b)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(np.clip(p, 1e-8, None)) - np.log(np.clip(m, 1e-8, None))))
    kl_qm = np.sum(q * (np.log(np.clip(q, 1e-8, None)) - np.log(np.clip(m, 1e-8, None))))
    return float(0.5 * (kl_pm + kl_qm))


def smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def normalize(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    vmin = float(finite.min())
    vmax = float(finite.max())
    if math.isclose(vmin, vmax):
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def consecutive_positive_run(values: np.ndarray, min_consecutive: int) -> Optional[int]:
    run = 0
    for idx, value in enumerate(values):
        if value > 0:
            run += 1
            if run >= min_consecutive:
                return idx - min_consecutive + 1
        else:
            run = 0
    return None


def fisher_separation(features: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    if features.size == 0:
        return np.zeros((0,), dtype=np.float32)
    unique_labels = sorted(set(labels))
    num_layers = features.shape[1]
    scores = np.zeros(num_layers, dtype=np.float32)
    for layer_idx in range(num_layers):
        layer_matrix = features[:, layer_idx, :]
        global_mean = layer_matrix.mean(axis=0)
        between = 0.0
        within = 0.0
        for label in unique_labels:
            indices = [i for i, cur in enumerate(labels) if cur == label]
            if not indices:
                continue
            group = layer_matrix[indices]
            group_mean = group.mean(axis=0)
            between += float(group.shape[0]) * float(np.sum((group_mean - global_mean) ** 2))
            within += float(np.sum((group - group_mean) ** 2))
        scores[layer_idx] = between / max(within, 1e-8)
    return scores


def ordered_segmentation_boundaries(
    reasoning: np.ndarray,
    objective: np.ndarray,
    style: np.ndarray,
    min_segment_width: int,
) -> Dict[str, object]:
    layer_count = len(reasoning)
    if layer_count == 0:
        return {"b1": 1, "b2": 2, "profiles": {}, "segment_scores": {}}

    reasoning_norm = smooth(normalize(reasoning))
    objective_norm = smooth(normalize(objective))
    style_norm = smooth(normalize(style))

    min_width = max(2, min(int(min_segment_width), max(2, layer_count // 3)))
    if (3 * min_width) > layer_count:
        min_width = max(1, layer_count // 4)

    best_score = None
    best = None
    for b1 in range(min_width, layer_count - (2 * min_width) + 1):
        for b2 in range(b1 + min_width, layer_count - min_width + 1):
            early_r = reasoning_norm[:b1]
            early_o = objective_norm[:b1]
            early_s = style_norm[:b1]
            mid_r = reasoning_norm[b1:b2]
            mid_o = objective_norm[b1:b2]
            mid_s = style_norm[b1:b2]
            late_r = reasoning_norm[b2:]
            late_o = objective_norm[b2:]
            late_s = style_norm[b2:]

            early_score = float(np.mean(early_r - np.maximum(early_o, early_s)))
            middle_score = float(np.mean(mid_o - np.maximum(mid_r, mid_s)))
            late_score = float(np.mean(late_s - np.maximum(late_r, late_o)))
            total = early_score + middle_score + late_score
            if best_score is None or total > best_score:
                best_score = total
                best = {
                    "b1": b1,
                    "b2": b2,
                    "segment_scores": {
                        "early_reasoning_margin": early_score,
                        "middle_objective_margin": middle_score,
                        "late_style_margin": late_score,
                        "total": total,
                    },
                }

    if best is None:
        b1 = max(1, layer_count // 3)
        b2 = max(b1 + 1, (2 * layer_count) // 3)
        best = {
            "b1": b1,
            "b2": min(layer_count, b2),
            "segment_scores": {
                "early_reasoning_margin": 0.0,
                "middle_objective_margin": 0.0,
                "late_style_margin": 0.0,
                "total": 0.0,
            },
        }

    return {
        "b1": int(best["b1"]),
        "b2": int(best["b2"]),
        "profiles": {
            "reasoning_norm": reasoning_norm.tolist(),
            "objective_norm": objective_norm.tolist(),
            "style_norm": style_norm.tolist(),
        },
        "segment_scores": best["segment_scores"],
    }


def transition_first_boundary(
    reasoning_norm: np.ndarray,
    objective_norm: np.ndarray,
    min_segment_width: int,
    transition_anchor_offset: int,
    transition_margin: float,
    transition_min_run: int,
) -> int:
    layer_count = len(reasoning_norm)
    if layer_count == 0:
        return 1

    peak_idx = int(np.argmax(reasoning_norm))
    stop = max(min_segment_width, layer_count - (2 * min_segment_width))
    start = max(min_segment_width, peak_idx + max(1, int(transition_anchor_offset)))
    start = min(start, stop)

    dominance = smooth(reasoning_norm - objective_norm)
    for b1 in range(start, stop + 1):
        run = dominance[b1 : b1 + max(1, int(transition_min_run))]
        if run.size < max(1, int(transition_min_run)):
            break
        if np.all(run <= float(transition_margin)):
            return int(b1)

    post = dominance[start : stop + 1]
    if post.size == 0:
        return int(start)

    below = np.where(post <= float(transition_margin))[0]
    if below.size > 0:
        return int(start + int(below[0]))
    return int(start + int(np.argmin(post)))


def transition_segmentation_boundaries(
    reasoning: np.ndarray,
    objective: np.ndarray,
    style: np.ndarray,
    min_segment_width: int,
    transition_anchor_offset: int,
    transition_margin: float,
    transition_min_run: int,
) -> Dict[str, object]:
    layer_count = len(reasoning)
    if layer_count == 0:
        return {"b1": 1, "b2": 2, "profiles": {}, "segment_scores": {}}

    reasoning_norm = smooth(normalize(reasoning))
    objective_norm = smooth(normalize(objective))
    style_norm = smooth(normalize(style))

    min_width = max(2, min(int(min_segment_width), max(2, layer_count // 3)))
    if (3 * min_width) > layer_count:
        min_width = max(1, layer_count // 4)

    b1 = transition_first_boundary(
        reasoning_norm=reasoning_norm,
        objective_norm=objective_norm,
        min_segment_width=min_width,
        transition_anchor_offset=transition_anchor_offset,
        transition_margin=transition_margin,
        transition_min_run=transition_min_run,
    )
    b1 = max(min_width, min(b1, layer_count - (2 * min_width)))

    best_score = None
    best = None
    for b2 in range(b1 + min_width, layer_count - min_width + 1):
        early_r = reasoning_norm[:b1]
        early_o = objective_norm[:b1]
        early_s = style_norm[:b1]
        mid_r = reasoning_norm[b1:b2]
        mid_o = objective_norm[b1:b2]
        mid_s = style_norm[b1:b2]
        late_r = reasoning_norm[b2:]
        late_o = objective_norm[b2:]
        late_s = style_norm[b2:]

        early_score = float(np.mean(early_r - np.maximum(early_o, early_s))) if len(early_r) else 0.0
        middle_score = float(np.mean(mid_o - np.maximum(mid_r, mid_s)))
        late_score = float(np.mean(late_s - np.maximum(late_r, late_o)))
        total = early_score + middle_score + late_score
        if best_score is None or total > best_score:
            best_score = total
            best = {
                "b1": b1,
                "b2": b2,
                "segment_scores": {
                    "early_reasoning_margin": early_score,
                    "middle_objective_margin": middle_score,
                    "late_style_margin": late_score,
                    "total": total,
                },
            }

    if best is None:
        b2 = max(b1 + 1, (2 * layer_count) // 3)
        best = {
            "b1": b1,
            "b2": min(layer_count, b2),
            "segment_scores": {
                "early_reasoning_margin": 0.0,
                "middle_objective_margin": 0.0,
                "late_style_margin": 0.0,
                "total": 0.0,
            },
        }

    return {
        "b1": int(best["b1"]),
        "b2": int(best["b2"]),
        "profiles": {
            "reasoning_norm": reasoning_norm.tolist(),
            "objective_norm": objective_norm.tolist(),
            "style_norm": style_norm.tolist(),
        },
        "segment_scores": best["segment_scores"],
    }


def windowed_transition_segmentation_boundaries(
    reasoning: np.ndarray,
    objective: np.ndarray,
    style: np.ndarray,
    min_segment_width: int,
    b1_search_low: int,
    b1_search_high: int,
    transition_margin: float,
    transition_min_run: int,
    late_preference: float,
    min_transition_support: float,
) -> Dict[str, object]:
    layer_count = len(reasoning)
    if layer_count == 0:
        return {"b1": 1, "b2": 2, "profiles": {}, "segment_scores": {}}

    reasoning_norm = smooth(normalize(reasoning))
    objective_norm = smooth(normalize(objective))
    style_norm = smooth(normalize(style))

    min_width = max(2, min(int(min_segment_width), max(2, layer_count // 3)))
    if (3 * min_width) > layer_count:
        min_width = max(1, layer_count // 4)

    low = max(int(b1_search_low), min_width)
    high = min(int(b1_search_high), layer_count - (2 * min_width))
    if high < low:
        low = min_width
        high = max(low, layer_count - (2 * min_width))

    dominance = smooth(objective_norm - reasoning_norm - 0.25 * style_norm)

    scored: List[Tuple[float, int, int, Dict[str, float]]] = []
    for b1 in range(low, high + 1):
        run = dominance[b1 : b1 + max(1, int(transition_min_run))]
        if run.size < max(1, int(transition_min_run)):
            continue
        run_mean = float(np.mean(run))
        if run_mean < -float(transition_margin):
            continue
        best_b2 = None
        best_score = None
        best_scores = None
        for b2 in range(b1 + min_width, layer_count - min_width + 1):
            early_r = reasoning_norm[:b1]
            early_o = objective_norm[:b1]
            early_s = style_norm[:b1]
            mid_r = reasoning_norm[b1:b2]
            mid_o = objective_norm[b1:b2]
            mid_s = style_norm[b1:b2]
            late_r = reasoning_norm[b2:]
            late_o = objective_norm[b2:]
            late_s = style_norm[b2:]

            early_score = float(np.mean(early_r - np.maximum(early_o, early_s))) if len(early_r) else 0.0
            middle_score = float(np.mean(mid_o - np.maximum(mid_r, mid_s))) if len(mid_o) else -1e9
            late_score = float(np.mean(late_s - np.maximum(late_r, late_o))) if len(late_s) else -1e9
            total = early_score + middle_score + late_score
            total += float(late_preference) * float(b1 - low)
            if best_score is None or total > best_score:
                best_score = total
                best_b2 = b2
                best_scores = {
                    "early_reasoning_margin": early_score,
                    "middle_objective_margin": middle_score,
                    "late_style_margin": late_score,
                    "transition_support": run_mean,
                    "total": total,
                }
        if best_score is not None and best_b2 is not None and best_scores is not None:
            scored.append((float(best_score), int(b1), int(best_b2), best_scores))

    if not scored:
        b1 = low
        b2 = max(b1 + min_width, min(layer_count, b1 + 2 * min_width))
        return {
            "b1": int(b1),
            "b2": int(b2),
            "status": "fallback_no_candidate",
            "profiles": {
                "reasoning_norm": reasoning_norm.tolist(),
                "objective_norm": objective_norm.tolist(),
                "style_norm": style_norm.tolist(),
            },
            "segment_scores": {
                "early_reasoning_margin": 0.0,
                "middle_objective_margin": 0.0,
                "late_style_margin": 0.0,
                "transition_support": 0.0,
                "total": 0.0,
            },
            "validity": {
                "is_valid_b1": False,
                "reasons": ["no_candidate_in_search_window"],
            },
            "search_window": {"b1_low": int(low), "b1_high": int(high)},
        }

    best_score = max(item[0] for item in scored)
    survivors = [item for item in scored if item[0] >= best_score - float(transition_margin)]
    if not survivors:
        survivors = scored
    _, b1, b2, scores = max(survivors, key=lambda item: (item[1], item[0]))

    validity_reasons: List[str] = []
    if scores["early_reasoning_margin"] <= 0:
        validity_reasons.append("early_reasoning_margin_nonpositive")
    if scores["middle_objective_margin"] <= 0:
        validity_reasons.append("middle_objective_margin_nonpositive")
    if scores["transition_support"] < float(min_transition_support):
        validity_reasons.append("transition_support_too_small")
    if scores["total"] <= 0:
        validity_reasons.append("total_score_nonpositive")

    return {
        "b1": int(b1),
        "b2": int(b2),
        "status": "candidate",
        "profiles": {
            "reasoning_norm": reasoning_norm.tolist(),
            "objective_norm": objective_norm.tolist(),
            "style_norm": style_norm.tolist(),
        },
        "segment_scores": scores,
        "validity": {
            "is_valid_b1": len(validity_reasons) == 0,
            "reasons": validity_reasons,
        },
        "search_window": {"b1_low": int(low), "b1_high": int(high)},
    }


def b1_only_crossover_boundaries(
    reasoning: np.ndarray,
    objective: np.ndarray,
    style: np.ndarray,
    min_segment_width: int,
    b1_search_low: int,
    b1_search_high: int,
    transition_min_run: int,
    min_transition_support: float,
    late_preference: float,
    min_pre_reasoning_margin: float,
    b1_band_score_tolerance: float,
    b1_band_min_width: int,
    b1_band_max_width: int,
) -> Dict[str, object]:
    layer_count = len(reasoning)
    if layer_count == 0:
        return {"b1": 1, "b2": 2, "profiles": {}, "segment_scores": {}, "validity": {"is_valid_b1": False, "reasons": ["empty_curve"]}}

    reasoning_norm = smooth(normalize(reasoning))
    objective_norm = smooth(normalize(objective))
    style_norm = smooth(normalize(style))

    min_width = max(2, min(int(min_segment_width), max(2, layer_count // 3)))
    if (3 * min_width) > layer_count:
        min_width = max(1, layer_count // 4)

    low = max(int(b1_search_low), min_width)
    high = min(int(b1_search_high), layer_count - (2 * min_width))
    if high < low:
        low = min_width
        high = max(low, layer_count - (2 * min_width))

    run_width = max(1, int(transition_min_run))
    candidates: List[Tuple[float, int, Dict[str, float]]] = []
    for b1 in range(low, high + 1):
        pre_start = max(0, b1 - run_width)
        pre = reasoning_norm[pre_start:b1] - objective_norm[pre_start:b1]
        post = objective_norm[b1 : b1 + run_width] - reasoning_norm[b1 : b1 + run_width]
        if pre.size < run_width or post.size < run_width:
            continue
        pre_margin = float(np.mean(pre))
        post_margin = float(np.mean(post))
        boundary_contrast = post_margin + pre_margin
        local_score = min(pre_margin, post_margin) + boundary_contrast
        local_score += float(late_preference) * float(b1 - low)
        candidates.append(
            (
                local_score,
                int(b1),
                {
                    "local_pre_reasoning_margin": pre_margin,
                    "local_post_objective_margin": post_margin,
                    "boundary_contrast": boundary_contrast,
                },
            )
        )

    if not candidates:
        b1 = low
        b2 = max(b1 + min_width, min(layer_count, b1 + 2 * min_width))
        return {
            "b1": int(b1),
            "b2": int(b2),
            "status": "fallback_no_candidate",
            "profiles": {
                "reasoning_norm": reasoning_norm.tolist(),
                "objective_norm": objective_norm.tolist(),
                "style_norm": style_norm.tolist(),
            },
            "segment_scores": {
                "local_pre_reasoning_margin": 0.0,
                "local_post_objective_margin": 0.0,
                "boundary_contrast": 0.0,
                "middle_objective_margin": 0.0,
                "late_style_margin": 0.0,
                "total": 0.0,
            },
            "validity": {
                "is_valid_b1": False,
                "reasons": ["no_candidate_in_search_window"],
            },
            "search_window": {"b1_low": int(low), "b1_high": int(high)},
        }

    viable_candidates = [
        candidate
        for candidate in candidates
        if candidate[2]["local_post_objective_margin"] >= float(min_transition_support)
        and candidate[2]["local_pre_reasoning_margin"] >= float(min_pre_reasoning_margin)
    ]
    candidate_pool = viable_candidates if viable_candidates else candidates
    best_local_score, b1, local_scores = max(candidate_pool, key=lambda item: (item[0], item[1]))

    band_min_width = max(1, int(b1_band_min_width))
    band_max_width = max(band_min_width, int(b1_band_max_width))
    band_tolerance = max(0.0, float(b1_band_score_tolerance))
    band_survivors = [
        candidate
        for candidate in candidate_pool
        if candidate[0] >= (best_local_score - band_tolerance)
    ]
    band_points = sorted({int(candidate[1]) for candidate in band_survivors})
    if not band_points:
        band_points = [int(b1)]

    def choose_b1_band(points: Sequence[int], center: int) -> Tuple[int, int, int]:
        if not points:
            return int(center), int(center), 1
        point_set = set(int(point) for point in points)
        best_band: Optional[Tuple[int, int, int, int, float]] = None
        for width in range(band_min_width, band_max_width + 1):
            start_low = max(low, center - width + 1)
            start_high = min(center, high - width + 1)
            if start_high < start_low:
                continue
            for start in range(start_low, start_high + 1):
                end = start + width - 1
                if center < start or center > end:
                    continue
                support = sum(1 for layer_idx in range(start, end + 1) if layer_idx in point_set)
                distance = abs(((start + end) / 2.0) - center)
                score = (support, width, -distance)
                if best_band is None or score > (best_band[0], best_band[1], best_band[2]):
                    best_band = (support, width, int(start), int(end), float(distance))
        if best_band is not None:
            _, width, start, end, _ = best_band
            return start, end, width

        width = min(band_max_width, max(band_min_width, 1))
        half = width // 2
        start = max(low, center - half)
        end = min(high, start + width - 1)
        start = max(low, end - width + 1)
        return int(start), int(end), int(end - start + 1)

    band_low, band_high, band_width = choose_b1_band(band_points, int(b1))

    best_b2 = None
    best_tail_score = None
    best_tail_scores = None
    for b2 in range(b1 + min_width, layer_count - min_width + 1):
        mid_r = reasoning_norm[b1:b2]
        mid_o = objective_norm[b1:b2]
        mid_s = style_norm[b1:b2]
        late_r = reasoning_norm[b2:]
        late_o = objective_norm[b2:]
        late_s = style_norm[b2:]

        middle_score = float(np.mean(mid_o - np.maximum(mid_r, mid_s))) if len(mid_o) else -1e9
        late_score = float(np.mean(late_s - np.maximum(late_r, late_o))) if len(late_s) else -1e9
        tail_score = middle_score + late_score
        if best_tail_score is None or tail_score > best_tail_score:
            best_tail_score = tail_score
            best_b2 = b2
            best_tail_scores = {
                "middle_objective_margin": middle_score,
                "late_style_margin": late_score,
            }

    if best_b2 is None or best_tail_scores is None:
        best_b2 = max(b1 + min_width, min(layer_count, b1 + 2 * min_width))
        best_tail_scores = {"middle_objective_margin": 0.0, "late_style_margin": 0.0}

    reasons: List[str] = []
    if not viable_candidates:
        reasons.append("no_viable_crossover_band_in_search_window")
    if band_width < band_min_width:
        reasons.append("b1_band_too_narrow")

    segment_scores = {
        "local_pre_reasoning_margin": local_scores["local_pre_reasoning_margin"],
        "local_post_objective_margin": local_scores["local_post_objective_margin"],
        "boundary_contrast": local_scores["boundary_contrast"],
        "middle_objective_margin": best_tail_scores["middle_objective_margin"],
        "late_style_margin": best_tail_scores["late_style_margin"],
        "band_candidate_count": float(len(band_survivors)),
        "band_width": float(band_width),
        "total": float(best_local_score + best_tail_scores["middle_objective_margin"] + best_tail_scores["late_style_margin"]),
    }

    return {
        "b1": int(b1),
        "b2": int(best_b2),
        "status": "band_candidate" if viable_candidates else "candidate_without_band",
        "boundary_windows": {
            "b1": [int(band_low), int(band_high)],
            "b2": [int(best_b2), int(best_b2)],
        },
        "profiles": {
            "reasoning_norm": reasoning_norm.tolist(),
            "objective_norm": objective_norm.tolist(),
            "style_norm": style_norm.tolist(),
        },
        "segment_scores": segment_scores,
        "validity": {
            "is_valid_b1": len(reasons) == 0,
            "reasons": reasons,
        },
        "search_window": {"b1_low": int(low), "b1_high": int(high)},
    }


def cache_path(cache_dir: Path, sample_id: str) -> Path:
    safe = sample_id.replace("/", "_").replace("::", "__")
    return cache_dir / f"{safe}.npz"


def ensure_sample_cache(
    sample: Dict[str, object],
    tokenizer,
    model,
    image_processor,
    device: str,
    conv_mode: str,
    cache_dir: Path,
    style_token_ids: Sequence[int],
    overwrite_cache: bool,
) -> Path:
    path = cache_path(cache_dir, str(sample["sample_id"]))
    if path.exists() and not overwrite_cache:
        cached = np.load(path, allow_pickle=False)
        hidden = cached["hidden"]
        blank_hidden = cached["blank_hidden"] if "blank_hidden" in cached.files else np.zeros((0, 0), dtype=np.float16)
        style_logits = cached["style_logits"] if "style_logits" in cached.files else np.zeros((hidden.shape[0], 0), dtype=np.float16)
        style_metric_name = str(cached["style_metric_name"][0]) if "style_metric_name" in cached.files else ""
        style_token_count = int(cached["style_token_count"][0]) if "style_token_count" in cached.files else int(style_logits.shape[-1])
        expected_count = len(style_token_ids)
        if style_metric_name == STYLE_METRIC_NAME and style_token_count == expected_count:
            return path
        refreshed_style_logits = logits_stack_from_hidden(model, hidden.astype(np.float32), style_token_ids)
        np.savez_compressed(
            path,
            hidden=hidden.astype(np.float16),
            blank_hidden=blank_hidden.astype(np.float16),
            style_logits=refreshed_style_logits.astype(np.float16),
            style_token_count=np.asarray([expected_count], dtype=np.int32),
            style_metric_name=np.asarray([STYLE_METRIC_NAME]),
        )
        return path

    prompt = build_prompt(str(sample["prompt_text"]), tokenizer, conv_mode)
    input_ids = tokenize_prompt(prompt, tokenizer, device)
    image_tensor = load_image_tensor(str(sample["image_path"]), image_processor, model, device, blank=False)
    blank_tensor = load_image_tensor(str(sample["image_path"]), image_processor, model, device, blank=True)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            images=image_tensor,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = hidden_stack_from_outputs(outputs)

        blank_hidden = np.zeros((0, 0), dtype=np.float16)
        if str(sample.get("probe_family")) == "reasoning":
            blank_outputs = model(
                input_ids=input_ids,
                images=blank_tensor,
                output_hidden_states=True,
                return_dict=True,
            )
            blank_hidden = hidden_stack_from_outputs(blank_outputs)

        style_logits = logits_stack_from_hidden(model, hidden, style_token_ids)

    np.savez_compressed(
        path,
        hidden=hidden.astype(np.float16),
        blank_hidden=blank_hidden.astype(np.float16),
        style_logits=style_logits.astype(np.float16),
        style_token_count=np.asarray([len(style_token_ids)], dtype=np.int32),
        style_metric_name=np.asarray([STYLE_METRIC_NAME]),
    )
    return path


def load_cache(cache_dir: Path, sample_id: str) -> Dict[str, np.ndarray]:
    path = cache_path(cache_dir, sample_id)
    payload = np.load(path, allow_pickle=False)
    return {name: payload[name] for name in payload.files}


def compute_split_curves(
    samples: Sequence[Dict[str, object]],
    cache_dir: Path,
    split_records: Sequence[Dict[str, object]],
    style_aggregation: str,
) -> Dict[str, object]:
    sample_map = {str(sample["sample_id"]): sample for sample in samples}
    reasoning_ids = [str(s["sample_id"]) for s in split_records if str(s.get("probe_family")) == "reasoning"]
    objective_ids = [str(s["sample_id"]) for s in split_records if str(s.get("objective_label", "")).startswith("objective_")]

    if not split_records:
        return {}

    first_cache = load_cache(cache_dir, str(split_records[0]["sample_id"]))
    layer_count = int(first_cache["hidden"].shape[0])

    reasoning_features = []
    reasoning_labels = []
    for sample_id in reasoning_ids:
        cache = load_cache(cache_dir, sample_id)
        reasoning_features.append(cache["hidden"].astype(np.float32))
        reasoning_labels.append(str(sample_map[sample_id]["objective_label"]))
    if reasoning_features:
        reasoning_curve = fisher_separation(np.stack(reasoning_features, axis=0), reasoning_labels)
        reasoning_counts = np.full(layer_count, len(reasoning_features), dtype=np.int32)
    else:
        reasoning_curve = np.zeros(layer_count, dtype=np.float32)
        reasoning_counts = np.zeros(layer_count, dtype=np.int32)

    objective_features = []
    objective_labels = []
    for sample_id in objective_ids:
        cache = load_cache(cache_dir, sample_id)
        objective_features.append(cache["hidden"].astype(np.float32))
        objective_labels.append(str(sample_map[sample_id]["objective_label"]))
    if objective_features:
        objective_features_arr = np.stack(objective_features, axis=0)
        objective_curve = fisher_separation(objective_features_arr, objective_labels)
        objective_counts = np.full(layer_count, len(objective_features), dtype=np.int32)
    else:
        objective_curve = np.zeros(layer_count, dtype=np.float32)
        objective_counts = np.zeros(layer_count, dtype=np.int32)

    style_curve = np.zeros(layer_count, dtype=np.float32)
    style_counts = np.zeros(layer_count, dtype=np.int32)
    groups: Dict[str, List[str]] = defaultdict(list)
    for sample_id in [str(s["sample_id"]) for s in split_records if str(s.get("probe_family")) == "style"]:
        groups[str(sample_map[sample_id]["paired_group_id"])].append(sample_id)
    style_group_curves: List[np.ndarray] = []
    cluster_to_group_curves: Dict[str, List[np.ndarray]] = defaultdict(list)
    style_group_counts_by_cluster: Dict[str, int] = defaultdict(int)
    for group_ids in groups.values():
        if len(group_ids) < 2:
            continue
        caches = {sid: load_cache(cache_dir, sid) for sid in group_ids}
        group_curve = np.zeros(layer_count, dtype=np.float32)
        group_counts = np.zeros(layer_count, dtype=np.int32)
        for layer_idx in range(layer_count):
            logits = [caches[sid]["style_logits"][layer_idx].astype(np.float32) for sid in group_ids]
            for i in range(len(logits)):
                for j in range(i + 1, len(logits)):
                    pair_js = js_divergence(logits[i], logits[j])
                    group_curve[layer_idx] += pair_js
                    group_counts[layer_idx] += 1
                    style_curve[layer_idx] += pair_js
                    style_counts[layer_idx] += 1
        group_curve = group_curve / np.clip(group_counts, 1, None)
        style_group_curves.append(group_curve)
        cluster_name = str(sample_map[group_ids[0]].get("prompt_cluster", "unknown"))
        cluster_to_group_curves[cluster_name].append(group_curve)
        style_group_counts_by_cluster[cluster_name] += 1
    style_curve = style_curve / np.clip(style_counts, 1, None)

    if style_group_curves and style_aggregation == "cluster_balanced":
        cluster_means = [np.mean(group_curves, axis=0) for group_curves in cluster_to_group_curves.values() if group_curves]
        if cluster_means:
            style_curve = np.mean(cluster_means, axis=0).astype(np.float32)

    if not reasoning_features or not objective_features or int(style_counts.sum()) == 0:
        return {}

    return {
        "layer_count": layer_count,
        "curves": {
            "reasoning_signal": reasoning_curve.tolist(),
            "objective_signal": objective_curve.tolist(),
            "style_signal": style_curve.tolist(),
        },
        "diagnostics": {
            "reasoning": {
                "per_layer_sample_counts": reasoning_counts.tolist(),
            },
            "objective": {
                "per_layer_sample_counts": objective_counts.tolist(),
            },
            "style": {
                "per_layer_pair_counts": style_counts.tolist(),
                "metric_name": STYLE_METRIC_NAME,
                "style_token_count": len(STYLE_TOKEN_TEXTS),
                "aggregation_mode": style_aggregation,
                "style_group_count": len(style_group_curves),
                "style_group_counts_by_cluster": dict(sorted(style_group_counts_by_cluster.items())),
            },
        },
        "sample_counts": {
            "reasoning": len(reasoning_ids),
            "objective": len(objective_ids),
            "style_groups": len(groups),
        },
    }


def build_report(
    model_key: str,
    split_name: str,
    split_type: str,
    held_out_value: str,
    split_records: Sequence[Dict[str, object]],
    curves_payload: Dict[str, object],
    output_path: Path,
    seed_note: str,
    min_segment_width: int,
    boundary_strategy: str,
    split_type_filter: Sequence[str],
    b1_search_low: int,
    b1_search_high: int,
    transition_anchor_offset: int,
    transition_margin: float,
    transition_min_run: int,
    min_transition_support: float,
    late_preference: float,
    min_pre_reasoning_margin: float,
    b1_band_score_tolerance: float,
    b1_band_min_width: int,
    b1_band_max_width: int,
) -> Dict[str, object]:
    reasoning = np.asarray(curves_payload["curves"]["reasoning_signal"], dtype=np.float32)
    objective = np.asarray(curves_payload["curves"]["objective_signal"], dtype=np.float32)
    style = np.asarray(curves_payload["curves"]["style_signal"], dtype=np.float32)
    if boundary_strategy == "transition":
        boundaries = transition_segmentation_boundaries(
            reasoning,
            objective,
            style,
            min_segment_width=min_segment_width,
            transition_anchor_offset=transition_anchor_offset,
            transition_margin=transition_margin,
            transition_min_run=transition_min_run,
        )
    elif boundary_strategy == "windowed_transition":
        boundaries = windowed_transition_segmentation_boundaries(
            reasoning,
            objective,
            style,
            min_segment_width=min_segment_width,
            b1_search_low=b1_search_low,
            b1_search_high=b1_search_high,
            transition_margin=transition_margin,
            transition_min_run=transition_min_run,
            late_preference=late_preference,
            min_transition_support=min_transition_support,
        )
    elif boundary_strategy == "b1_only_crossover":
        boundaries = b1_only_crossover_boundaries(
            reasoning,
            objective,
            style,
            min_segment_width=min_segment_width,
            b1_search_low=b1_search_low,
            b1_search_high=b1_search_high,
            transition_min_run=transition_min_run,
            min_transition_support=min_transition_support,
            late_preference=late_preference,
            min_pre_reasoning_margin=min_pre_reasoning_margin,
            b1_band_score_tolerance=b1_band_score_tolerance,
            b1_band_min_width=b1_band_min_width,
            b1_band_max_width=b1_band_max_width,
        )
    else:
        boundaries = ordered_segmentation_boundaries(reasoning, objective, style, min_segment_width=min_segment_width)
    layer_count = int(curves_payload["layer_count"])
    report = {
        "model": model_key,
        "dataset": held_out_value,
        "split_name": split_name,
        "split_type": split_type,
        "seed_note": seed_note,
        "boundary_strategy": boundary_strategy,
        "style_aggregation": str(curves_payload["diagnostics"]["style"].get("aggregation_mode", "raw")),
        "split_type_filter": list(split_type_filter),
        "sample_count": len(split_records),
        "layer_count": layer_count,
        "status": str(boundaries.get("status", "ok")),
        "recommended_boundaries": {
            "b1": int(boundaries["b1"]),
            "b2": int(boundaries["b2"]),
            "b1_ratio": float(boundaries["b1"] / layer_count),
            "b2_ratio": float(boundaries["b2"] / layer_count),
        },
        "boundary_windows": boundaries.get(
            "boundary_windows",
            {
                "b1": [int(boundaries["b1"]), int(boundaries["b1"])],
                "b2": [int(boundaries["b2"]), int(boundaries["b2"])],
            },
        ),
        "curves": {
            "reasoning_signal": curves_payload["curves"]["reasoning_signal"],
            "objective_signal": curves_payload["curves"]["objective_signal"],
            "style_signal": curves_payload["curves"]["style_signal"],
            "reasoning_signal_norm": boundaries["profiles"]["reasoning_norm"],
            "objective_signal_norm": boundaries["profiles"]["objective_norm"],
            "style_signal_norm": boundaries["profiles"]["style_norm"],
        },
        "diagnostics": curves_payload["diagnostics"],
        "segmentation": boundaries["segment_scores"],
        "validity": boundaries.get("validity", {"is_valid_b1": True, "reasons": []}),
        "sample_counts": curves_payload["sample_counts"],
        "artifacts": {
            "report_path": str(output_path),
        },
    }
    if "search_window" in boundaries:
        report["search_window"] = boundaries["search_window"]
    write_json(output_path, report)
    return report


def main() -> None:
    args = build_parser().parse_args()
    disable_torch_init()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (output_dir / "cache")
    ensure_dir(cache_dir)

    model_config = Path(args.model_config)
    probe_jsonl = Path(args.probe_jsonl)
    split_manifest = Path(args.split_manifest) if args.split_manifest else None

    tokenizer, model, image_processor, runtime_device = load_model(model_config, args.device)
    style_token_ids = build_style_token_ids(tokenizer)
    samples = read_jsonl(probe_jsonl)

    for sample in tqdm(samples, desc="cache", leave=False):
        ensure_sample_cache(
            sample=sample,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            device=runtime_device,
            conv_mode=args.conv_mode,
            cache_dir=cache_dir,
            style_token_ids=style_token_ids,
            overwrite_cache=args.overwrite_cache,
        )

    if split_manifest and split_manifest.exists():
        split_payload = load_json(split_manifest)
        splits = split_payload.get("splits", [])
    else:
        splits = [
            {
                "split_name": "all",
                "split_type": "all",
                "held_out_value": "all",
                "test_ids": [str(sample["sample_id"]) for sample in samples],
            }
        ]

    split_type_filters = {s.strip() for s in str(args.split_type_filter).split(",") if s.strip()}
    if split_type_filters:
        splits = [split for split in splits if str(split.get("split_type")) in split_type_filters]

    reports: List[Dict[str, object]] = []
    for idx, split in enumerate(tqdm(splits, desc="splits", leave=False)):
        if args.max_splits and idx >= args.max_splits:
            break
        test_ids = [str(x) for x in split.get("test_ids", [])]
        if not test_ids:
            continue
        split_records = [sample for sample in samples if str(sample["sample_id"]) in set(test_ids)]
        if not split_records:
            continue
        curves_payload = compute_split_curves(
            samples,
            cache_dir,
            split_records,
            style_aggregation=args.style_aggregation,
        )
        if not curves_payload:
            continue
        split_output_dir = output_dir / split["split_name"].replace("::", "__")
        ensure_dir(split_output_dir)
        report = build_report(
            model_key="llava" if "llava" in str(model_config).lower() else model_config.stem,
            split_name=str(split["split_name"]),
            split_type=str(split["split_type"]),
            held_out_value=str(split.get("held_out_value", split.get("split_name"))),
            split_records=split_records,
            curves_payload=curves_payload,
            output_path=split_output_dir / "boundary_report.json",
            seed_note=f"cache={cache_dir.name}",
            min_segment_width=args.min_segment_width,
            boundary_strategy=args.boundary_strategy,
            split_type_filter=sorted(split_type_filters),
            b1_search_low=args.b1_search_low,
            b1_search_high=args.b1_search_high,
            transition_anchor_offset=args.transition_anchor_offset,
            transition_margin=args.transition_margin,
            transition_min_run=args.transition_min_run,
            min_transition_support=args.min_transition_support,
            late_preference=args.late_preference,
            min_pre_reasoning_margin=args.min_pre_reasoning_margin,
            b1_band_score_tolerance=args.b1_band_score_tolerance,
            b1_band_min_width=args.b1_band_min_width,
            b1_band_max_width=args.b1_band_max_width,
        )
        reports.append(report)

    if reports:
        b1_values = [r["recommended_boundaries"]["b1"] for r in reports]
        b2_values = [r["recommended_boundaries"]["b2"] for r in reports]
        valid_reports = [r for r in reports if bool(r.get("validity", {}).get("is_valid_b1", True))]
        b1_window_lows = [int(r["boundary_windows"]["b1"][0]) for r in reports]
        b1_window_highs = [int(r["boundary_windows"]["b1"][1]) for r in reports]
        b2_window_lows = [int(r["boundary_windows"]["b2"][0]) for r in reports]
        b2_window_highs = [int(r["boundary_windows"]["b2"][1]) for r in reports]
        summary = {
            "num_reports": len(reports),
            "num_valid_b1_reports": len(valid_reports),
            "reports": [
                {
                    "model": r["model"],
                    "dataset": r["dataset"],
                    "split_name": r["split_name"],
                    "split_type": r["split_type"],
                    "status": r.get("status", "ok"),
                    "b1": r["recommended_boundaries"]["b1"],
                    "b2": r["recommended_boundaries"]["b2"],
                    "is_valid_b1": bool(r.get("validity", {}).get("is_valid_b1", True)),
                    "validity_reasons": r.get("validity", {}).get("reasons", []),
                    "b1_ratio": r["recommended_boundaries"]["b1_ratio"],
                    "b2_ratio": r["recommended_boundaries"]["b2_ratio"],
                    "boundary_windows": r["boundary_windows"],
                    "sample_count": r["sample_count"],
                    "report_path": r["artifacts"]["report_path"],
                }
                for r in reports
            ],
            "aggregate": {
                "b1_mean": float(np.mean(b1_values)),
                "b1_std": float(np.std(b1_values)),
                "b2_mean": float(np.mean(b2_values)),
                "b2_std": float(np.std(b2_values)),
                "b1_median": float(np.median(b1_values)),
                "b2_median": float(np.median(b2_values)),
                "valid_b1_mean": float(np.mean([r["recommended_boundaries"]["b1"] for r in valid_reports])) if valid_reports else None,
                "valid_b1_std": float(np.std([r["recommended_boundaries"]["b1"] for r in valid_reports])) if valid_reports else None,
                "boundary_windows": {
                    "b1": [int(np.percentile(b1_window_lows, 25)), int(np.percentile(b1_window_highs, 75))],
                    "b2": [int(np.percentile(b2_window_lows, 25)), int(np.percentile(b2_window_highs, 75))],
                },
                "valid_boundary_windows": {
                    "b1": [int(np.percentile([int(r["boundary_windows"]["b1"][0]) for r in valid_reports], 25)), int(np.percentile([int(r["boundary_windows"]["b1"][1]) for r in valid_reports], 75))]
                    if valid_reports
                    else None,
                    "b2": [int(np.percentile([int(r["boundary_windows"]["b2"][0]) for r in valid_reports], 25)), int(np.percentile([int(r["boundary_windows"]["b2"][1]) for r in valid_reports], 75))]
                    if valid_reports
                    else None,
                },
            },
        }
    else:
        summary = {"num_reports": 0, "reports": []}

    write_json(output_dir / "stage1_boundary_summary.json", summary)
    print(output_dir / "stage1_boundary_summary.json")


if __name__ == "__main__":
    main()
