#!/usr/bin/env python3
import argparse
import gc
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer


SCRIPT_DIR = os.path.dirname(__file__)
HI_DE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
MCITLIB_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../.."))
if HI_DE_ROOT not in sys.path:
    sys.path.insert(0, HI_DE_ROOT)

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
from HiDe.peft import PeftModel  # noqa: E402


DEFAULT_BASE_MODEL_PATH = "/mnt/lyaa/my_llava/llava-v1.5-7b"
DEFAULT_VISION_TOWER_PATH = "/mnt/lyaa/my_llava/clip-vit-large-patch14-336"
DEFAULT_UCIT_ROOT = "/mnt/lyaa/my_llava/UCIT"
# Source of truth: logs/train_UCIT_full_20260703_211225.log
DEFAULT_CHECKPOINT_ROOT = "/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe"
DEFAULT_OUTPUT_DIR = os.path.join(MCITLIB_ROOT, "docs", "experiment2_template_alignment_v21")
DEFAULT_CONV_MODE = "llava_v1"
DEFAULT_CONTENT_CLAUSE = (
    "objects, attributes, shapes, colors, textures, scene context, visible text, and spatial relations."
)
DEFAULT_PROMPT_VARIANTS = {
    "canonical": f"Describe the image using visual evidence: {DEFAULT_CONTENT_CLAUSE}",
    "variant_a": (
        "As a vision analyst, produce an observation report grounded only in visible evidence: "
        f"{DEFAULT_CONTENT_CLAUSE}"
    ),
    "variant_b": (
        "Report only what is visually present in the image, avoid speculation, and organize the answer by these categories: "
        f"{DEFAULT_CONTENT_CLAUSE}"
    ),
    "variant_c": (
        "Provide a scene audit based strictly on observable details and cover the following visual aspects: "
        f"{DEFAULT_CONTENT_CLAUSE}"
    ),
}

TASK_DATA_FILES = {
    1: ("ImageNet-R", "ImageNet-R/train.json"),
    2: ("ArxivQA", "ArxivQA/train_4w.json"),
    3: ("VizWiz", "VizWiz/train.json"),
    4: ("IconQA", "IconQA/train.json"),
    5: ("CLEVR", "CLEVR/train_4w.json"),
    6: ("Flickr30k", "Flickr30k/train_brief_4w.json"),
}

ASSISTANT_PREFIX_TOKENS = ("▁A", "SS", "IST", "ANT", ":")
PUNCT_TOKENS = {",", ".", ":", ";", "?", "!"}
CONTENT_STOPWORDS = {"and"}

PLOT_COLORS = {
    "before": "#2f6690",
    "after": "#d1495b",
    "scatter": "#3e8914",
    "line": "#edae49",
    "template": "#4f772d",
    "content": "#7c3aed",
}


@dataclass
class Sample:
    task_id: int
    task_name: str
    image_path: str
    cache_key: str


@dataclass
class PromptState:
    states: torch.Tensor
    token_pieces: List[str]
    content_mask: torch.Tensor
    template_mask: torch.Tensor
    word_labels: List[str]
    word_spans: List[List[int]]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(message: str, mm_use_im_start_end: bool, conv_mode: str) -> str:
    if mm_use_im_start_end:
        user_prompt = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{message}"
    else:
        user_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{message}"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_task_samples(ucit_root: str, task_id: int, samples_per_task: int, seed: int) -> List[Sample]:
    if task_id not in TASK_DATA_FILES:
        raise ValueError(f"Unsupported task id: {task_id}")
    task_name, rel_path = TASK_DATA_FILES[task_id]
    data_path = os.path.join(ucit_root, rel_path)
    with open(data_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)
    rng = random.Random(seed + task_id * 13)
    if samples_per_task > 0 and samples_per_task < len(raw_samples):
        raw_samples = rng.sample(raw_samples, samples_per_task)
    samples = []
    for idx, item in enumerate(raw_samples):
        samples.append(
            Sample(
                task_id=task_id,
                task_name=task_name,
                image_path=os.path.join(ucit_root, "datasets", item["image"]),
                cache_key=f"{task_name}_{idx:08d}",
            )
        )
    return samples


def load_image_tensor(image_path: str, image_processor, device: str, dtype: torch.dtype) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return pixel_values.unsqueeze(0).to(device=device, dtype=dtype)


def normalize_piece(piece: str) -> str:
    piece = piece.replace("▁", "").lower()
    return "".join(ch for ch in piece if ch.isalnum())


def safe_token_pieces(tokenizer, token_ids: Sequence[int]) -> List[str]:
    pieces: List[str] = []
    special_token_ids = set(tokenizer.all_special_ids)
    vocab_size = len(tokenizer)
    for token_id in token_ids:
        if token_id < 0 or token_id >= vocab_size or token_id in special_token_ids:
            pieces.append("")
            continue
        try:
            pieces.append(tokenizer.convert_ids_to_tokens(int(token_id)))
        except (IndexError, OverflowError):
            pieces.append("")
    return pieces


def find_assistant_start(token_pieces: Sequence[str]) -> int:
    cur_len = len(token_pieces)
    if cur_len >= len(ASSISTANT_PREFIX_TOKENS):
        suffix = token_pieces[-len(ASSISTANT_PREFIX_TOKENS):]
        if tuple(suffix) == ASSISTANT_PREFIX_TOKENS:
            return cur_len - len(ASSISTANT_PREFIX_TOKENS)
    return cur_len


def build_content_and_template_masks(window_pieces: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    content_mask = torch.zeros(len(window_pieces), dtype=torch.bool)
    template_mask = torch.zeros(len(window_pieces), dtype=torch.bool)

    content_start = None
    for idx in range(len(window_pieces) - 1, -1, -1):
        if window_pieces[idx] == ":":
            content_start = idx + 1
            break
    if content_start is None:
        content_start = len(window_pieces)

    for idx, piece in enumerate(window_pieces):
        if piece in PUNCT_TOKENS or not piece:
            continue
        token_norm = normalize_piece(piece)
        if not token_norm:
            continue
        if idx >= content_start and token_norm not in CONTENT_STOPWORDS:
            content_mask[idx] = True
        else:
            template_mask[idx] = True
    return content_mask, template_mask


def group_content_word_spans(window_pieces: Sequence[str], content_mask: torch.Tensor) -> Tuple[List[str], List[List[int]]]:
    labels: List[str] = []
    spans: List[List[int]] = []
    for idx, piece in enumerate(window_pieces):
        if not bool(content_mask[idx]):
            continue
        clean_piece = piece.replace("▁", "")
        if piece.startswith("▁") or not spans:
            labels.append(clean_piece)
            spans.append([idx])
        else:
            labels[-1] += clean_piece
            spans[-1].append(idx)
    return labels, spans


def extract_prompt_state(
    model,
    tokenizer,
    image_tensor: torch.Tensor,
    prompt_text: str,
    mm_use_im_start_end: bool,
    conv_mode: str,
    hidden_layer: int,
    max_tokens: int,
    device: str,
) -> PromptState:
    prompt = build_prompt(prompt_text, mm_use_im_start_end, conv_mode)
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    outputs = model(
        input_ids=input_ids,
        images=image_tensor,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states[hidden_layer]

    cur_len = int(attention_mask[0].long().sum().item())
    token_ids = input_ids[0, :cur_len].tolist()
    token_pieces = safe_token_pieces(tokenizer, token_ids)
    assistant_start = find_assistant_start(token_pieces)
    desc_end = assistant_start
    desc_start = max(0, desc_end - max_tokens)

    window_states = hidden_states[0, desc_start:desc_end].detach().float().cpu()
    window_pieces = token_pieces[desc_start:desc_end]
    content_mask, template_mask = build_content_and_template_masks(window_pieces)
    word_labels, word_spans = group_content_word_spans(window_pieces, content_mask)
    return PromptState(
        states=window_states,
        token_pieces=window_pieces,
        content_mask=content_mask,
        template_mask=template_mask,
        word_labels=word_labels,
        word_spans=word_spans,
    )


def rms_normalize_hidden(hidden_states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = hidden_states.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(eps).sqrt()
    return hidden_states.float() / rms


def mean_masked_drift(before_state: PromptState, after_state: PromptState, mask_name: str) -> float:
    shared_len = min(before_state.states.shape[0], after_state.states.shape[0])
    if shared_len == 0:
        return 0.0
    before_states = rms_normalize_hidden(before_state.states[:shared_len])
    after_states = rms_normalize_hidden(after_state.states[:shared_len])
    before_mask = getattr(before_state, mask_name)[:shared_len]
    after_mask = getattr(after_state, mask_name)[:shared_len]
    mask = before_mask & after_mask
    if not torch.any(mask):
        return 0.0
    drift = (after_states - before_states).norm(dim=-1)
    return float(drift[mask].mean().item())


def span_mean(hidden_states: torch.Tensor, span: Sequence[int]) -> torch.Tensor:
    return hidden_states[list(span)].mean(dim=0)


def compute_word_alignment(reference_state: PromptState, variant_state: PromptState) -> Tuple[List[str], List[float]]:
    ref_norm = rms_normalize_hidden(reference_state.states)
    var_norm = rms_normalize_hidden(variant_state.states)
    word_count = min(len(reference_state.word_spans), len(variant_state.word_spans))
    labels: List[str] = []
    values: List[float] = []
    for idx in range(word_count):
        ref_span = reference_state.word_spans[idx]
        var_span = variant_state.word_spans[idx]
        ref_vec = span_mean(ref_norm, ref_span)
        var_vec = span_mean(var_norm, var_span)
        cosine = torch.nn.functional.cosine_similarity(ref_vec.unsqueeze(0), var_vec.unsqueeze(0), dim=-1)
        labels.append(reference_state.word_labels[idx])
        values.append(float(cosine.item()))
    return labels, values


def build_word_embedding_matrix(state: PromptState) -> torch.Tensor:
    norm_states = rms_normalize_hidden(state.states)
    word_vectors = [span_mean(norm_states, span) for span in state.word_spans]
    if not word_vectors:
        return torch.zeros((0, norm_states.shape[-1]), dtype=torch.float32)
    matrix = torch.stack(word_vectors, dim=0)
    matrix = torch.nn.functional.normalize(matrix, dim=-1)
    return matrix


def compute_retrieval_metrics(reference_state: PromptState, variant_state: PromptState) -> Tuple[List[str], List[float], float, float]:
    ref_matrix = build_word_embedding_matrix(reference_state)
    var_matrix = build_word_embedding_matrix(variant_state)
    word_count = min(ref_matrix.shape[0], var_matrix.shape[0], len(reference_state.word_labels))
    if word_count == 0:
        return [], [], 0.0, 0.0

    ref_matrix = ref_matrix[:word_count]
    var_matrix = var_matrix[:word_count]
    sim_matrix = ref_matrix @ var_matrix.T
    diag = sim_matrix.diag()
    row_pred = sim_matrix.argmax(dim=1)
    row_hits = row_pred.eq(torch.arange(word_count))
    hit_list = [float(x.item()) for x in row_hits.float()]

    incorrect_mask = ~torch.eye(word_count, dtype=torch.bool)
    if word_count > 1:
        incorrect_scores = sim_matrix.masked_fill(~incorrect_mask, float("-inf"))
        best_incorrect = incorrect_scores.max(dim=1).values
        best_incorrect = torch.where(
            torch.isfinite(best_incorrect),
            best_incorrect,
            torch.zeros_like(best_incorrect),
        )
    else:
        best_incorrect = torch.zeros_like(diag)
    margin = (diag - best_incorrect).mean()
    accuracy = row_hits.float().mean()
    return reference_state.word_labels[:word_count], hit_list, float(accuracy.item()), float(margin.item())


def average_alignment_scores(score_lists: Sequence[List[float]]) -> List[float]:
    if not score_lists:
        return []
    min_len = min(len(scores) for scores in score_lists)
    stacked = np.asarray([scores[:min_len] for scores in score_lists], dtype=np.float32)
    return stacked.mean(axis=0).tolist()


def mean_scalar(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x_arr = np.asarray(xs, dtype=np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def collect_prompt_state_sets(
    model,
    tokenizer,
    image_processor,
    samples: Sequence[Sample],
    prompt_variants: Dict[str, str],
    mm_use_im_start_end: bool,
    conv_mode: str,
    hidden_layer: int,
    max_tokens: int,
    device: str,
    dtype: torch.dtype,
) -> List[Dict[str, PromptState]]:
    state_sets: List[Dict[str, PromptState]] = []
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            image_tensor = load_image_tensor(sample.image_path, image_processor, device, dtype)
            variant_states: Dict[str, PromptState] = {}
            for name, prompt_text in prompt_variants.items():
                variant_states[name] = extract_prompt_state(
                    model=model,
                    tokenizer=tokenizer,
                    image_tensor=image_tensor,
                    prompt_text=prompt_text,
                    mm_use_im_start_end=mm_use_im_start_end,
                    conv_mode=conv_mode,
                    hidden_layer=hidden_layer,
                    max_tokens=max_tokens,
                    device=device,
                )
            state_sets.append(variant_states)
    return state_sets


def load_model_bundle(
    model_path: str,
    base_model_path: Optional[str],
    vision_tower_path: str,
    device: str,
    model_dtype: torch.dtype,
    num_task: int,
):
    checkpoint_dir = os.path.abspath(os.path.expanduser(model_path))
    is_checkpoint_dir = os.path.isdir(checkpoint_dir)
    has_lora_adapter = is_checkpoint_dir and (
        os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin"))
        or os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))
    )
    has_non_lora_weights = is_checkpoint_dir and os.path.exists(
        os.path.join(checkpoint_dir, "non_lora_trainables.bin")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path if has_lora_adapter and base_model_path is not None else checkpoint_dir,
        use_fast=False,
        local_files_only=True,
    )
    config_source = checkpoint_dir if (is_checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "config.json"))) else (base_model_path or checkpoint_dir)
    cfg = AutoConfig.from_pretrained(config_source, local_files_only=True)
    cfg.mm_vision_tower = vision_tower_path
    cfg.mm_text_tower = vision_tower_path
    if not hasattr(cfg, "mm_vision_select_layer"):
        cfg.mm_vision_select_layer = -2
    if not hasattr(cfg, "mm_vision_select_feature"):
        cfg.mm_vision_select_feature = "patch"
    if not hasattr(cfg, "mm_text_select_layer"):
        cfg.mm_text_select_layer = -1

    if has_lora_adapter and base_model_path is not None:
        model = LlavaLlamaForCausalLM.from_pretrained(
            base_model_path,
            low_cpu_mem_usage=True,
            config=cfg,
            torch_dtype=model_dtype,
            local_files_only=True,
        )
        if has_non_lora_weights:
            non_lora_path = os.path.join(checkpoint_dir, "non_lora_trainables.bin")
            non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
            non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)
        model = PeftModel.from_pretrained(model, checkpoint_dir)
        model = model.merge_and_unload()
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(
            checkpoint_dir,
            low_cpu_mem_usage=True,
            config=cfg,
            torch_dtype=model_dtype,
            local_files_only=True,
        )

    if hasattr(model, "set_tokenizer"):
        model.set_tokenizer(tokenizer)
    if hasattr(model, "set_clip_tokenizer"):
        clip_tokenizer = AutoTokenizer.from_pretrained(
            vision_tower_path,
            cache_dir=None,
            model_max_length=77,
            padding_side="right",
            use_fast=True,
            local_files_only=True,
        )
        model.set_clip_tokenizer(clip_tokenizer)
    if hasattr(model, "set_eval"):
        model.set_eval(num_task)

    model.to(device=device, dtype=model_dtype)
    vision_tower = model.get_vision_tower()
    if vision_tower is None:
        raise RuntimeError(f"Failed to load vision tower from {model_path}")
    vision_tower.vision_tower_name = vision_tower_path
    if not vision_tower.is_loaded:
        vision_tower.load_model()
        vision_tower.to(device=device, dtype=model_dtype)

    text_tower = model.get_text_tower() if hasattr(model, "get_text_tower") else None
    if text_tower is not None:
        text_tower.text_tower_name = vision_tower_path
        if not text_tower.is_loaded:
            text_tower.load_model()
        text_tower.to(device=device, dtype=model_dtype)

    model.eval()
    image_processor = vision_tower.image_processor
    mm_use_im_start_end = bool(getattr(model.config, "mm_use_im_start_end", False))
    return tokenizer, model, image_processor, mm_use_im_start_end


def task_checkpoint_paths(checkpoint_root: str, task_id: int, base_model_path: str) -> Tuple[str, str]:
    current = os.path.join(checkpoint_root, f"Task{task_id}_llava_lora")
    previous = base_model_path if task_id == 1 else os.path.join(checkpoint_root, f"Task{task_id - 1}_llava_lora")
    return previous, current


def summarize_task_metrics(
    task_id: int,
    task_name: str,
    samples: Sequence[Sample],
    before_sets: Sequence[Dict[str, PromptState]],
    after_sets: Sequence[Dict[str, PromptState]],
) -> Dict[str, object]:
    sample_alignment_before: List[float] = []
    sample_alignment_after: List[float] = []
    sample_alignment_drop: List[float] = []
    sample_retrieval_before: List[float] = []
    sample_retrieval_after: List[float] = []
    sample_retrieval_drop: List[float] = []
    sample_margin_before: List[float] = []
    sample_margin_after: List[float] = []
    sample_margin_drop: List[float] = []
    sample_template_drift: List[float] = []
    sample_content_drift: List[float] = []
    word_before_scores: Dict[str, List[float]] = {}
    word_after_scores: Dict[str, List[float]] = {}
    word_before_hits: Dict[str, List[float]] = {}
    word_after_hits: Dict[str, List[float]] = {}

    for before_states, after_states in zip(before_sets, after_sets):
        before_canonical = before_states["canonical"]
        after_canonical = after_states["canonical"]
        before_variant_scores: List[List[float]] = []
        after_variant_scores: List[List[float]] = []
        before_variant_hits: List[List[float]] = []
        after_variant_hits: List[List[float]] = []
        before_variant_accuracy: List[float] = []
        after_variant_accuracy: List[float] = []
        before_variant_margin: List[float] = []
        after_variant_margin: List[float] = []

        for variant_name in ("variant_a", "variant_b", "variant_c"):
            before_labels, before_scores = compute_word_alignment(before_canonical, before_states[variant_name])
            after_labels, after_scores = compute_word_alignment(after_canonical, after_states[variant_name])
            before_variant_scores.append(before_scores)
            after_variant_scores.append(after_scores)

            for label, value in zip(before_labels, before_scores):
                word_before_scores.setdefault(label, []).append(value)
            for label, value in zip(after_labels, after_scores):
                word_after_scores.setdefault(label, []).append(value)

            hit_labels_before, before_hits, before_acc, before_margin = compute_retrieval_metrics(
                before_canonical,
                before_states[variant_name],
            )
            hit_labels_after, after_hits, after_acc, after_margin = compute_retrieval_metrics(
                after_canonical,
                after_states[variant_name],
            )
            before_variant_hits.append(before_hits)
            after_variant_hits.append(after_hits)
            before_variant_accuracy.append(before_acc)
            after_variant_accuracy.append(after_acc)
            before_variant_margin.append(before_margin)
            after_variant_margin.append(after_margin)

            for label, value in zip(hit_labels_before, before_hits):
                word_before_hits.setdefault(label, []).append(value)
            for label, value in zip(hit_labels_after, after_hits):
                word_after_hits.setdefault(label, []).append(value)

        before_mean_scores = average_alignment_scores(before_variant_scores)
        after_mean_scores = average_alignment_scores(after_variant_scores)
        before_mean_hits = average_alignment_scores(before_variant_hits)
        after_mean_hits = average_alignment_scores(after_variant_hits)
        before_alignment = mean_scalar(before_mean_scores)
        after_alignment = mean_scalar(after_mean_scores)
        sample_alignment_before.append(before_alignment)
        sample_alignment_after.append(after_alignment)
        sample_alignment_drop.append(before_alignment - after_alignment)
        before_retrieval = mean_scalar(before_variant_accuracy)
        after_retrieval = mean_scalar(after_variant_accuracy)
        sample_retrieval_before.append(before_retrieval)
        sample_retrieval_after.append(after_retrieval)
        sample_retrieval_drop.append(before_retrieval - after_retrieval)
        before_margin_value = mean_scalar(before_variant_margin)
        after_margin_value = mean_scalar(after_variant_margin)
        sample_margin_before.append(before_margin_value)
        sample_margin_after.append(after_margin_value)
        sample_margin_drop.append(before_margin_value - after_margin_value)
        sample_template_drift.append(mean_masked_drift(before_canonical, after_canonical, "template_mask"))
        sample_content_drift.append(mean_masked_drift(before_canonical, after_canonical, "content_mask"))

    word_labels = list(word_before_scores.keys())
    word_alignment_before_mean = [mean_scalar(word_before_scores[label]) for label in word_labels]
    word_alignment_after_mean = [mean_scalar(word_after_scores.get(label, [])) for label in word_labels]
    word_retrieval_before_mean = [mean_scalar(word_before_hits.get(label, [])) for label in word_labels]
    word_retrieval_after_mean = [mean_scalar(word_after_hits.get(label, [])) for label in word_labels]

    return {
        "task_id": task_id,
        "task_name": task_name,
        "label": f"Task {task_id} / {task_name}",
        "num_samples": len(samples),
        "sample_alignment_before": sample_alignment_before,
        "sample_alignment_after": sample_alignment_after,
        "sample_alignment_drop": sample_alignment_drop,
        "sample_retrieval_before": sample_retrieval_before,
        "sample_retrieval_after": sample_retrieval_after,
        "sample_retrieval_drop": sample_retrieval_drop,
        "sample_margin_before": sample_margin_before,
        "sample_margin_after": sample_margin_after,
        "sample_margin_drop": sample_margin_drop,
        "sample_template_drift": sample_template_drift,
        "sample_content_drift": sample_content_drift,
        "alignment_before_mean": mean_scalar(sample_alignment_before),
        "alignment_after_mean": mean_scalar(sample_alignment_after),
        "alignment_drop_mean": mean_scalar(sample_alignment_drop),
        "retrieval_before_mean": mean_scalar(sample_retrieval_before),
        "retrieval_after_mean": mean_scalar(sample_retrieval_after),
        "retrieval_drop_mean": mean_scalar(sample_retrieval_drop),
        "margin_before_mean": mean_scalar(sample_margin_before),
        "margin_after_mean": mean_scalar(sample_margin_after),
        "margin_drop_mean": mean_scalar(sample_margin_drop),
        "template_drift_mean": mean_scalar(sample_template_drift),
        "content_drift_mean": mean_scalar(sample_content_drift),
        "template_alignment_drop_corr": correlation(sample_template_drift, sample_alignment_drop),
        "template_retrieval_drop_corr": correlation(sample_template_drift, sample_retrieval_drop),
        "template_margin_drop_corr": correlation(sample_template_drift, sample_margin_drop),
        "word_labels": word_labels,
        "word_alignment_before_mean": word_alignment_before_mean,
        "word_alignment_after_mean": word_alignment_after_mean,
        "word_retrieval_before_mean": word_retrieval_before_mean,
        "word_retrieval_after_mean": word_retrieval_after_mean,
    }


def aggregate_word_alignment(task_reports: Sequence[Dict[str, object]], key: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    if not task_reports:
        return [], np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    word_labels = task_reports[0]["word_labels"]
    before = np.asarray([task[f"word_{key}_before_mean"] for task in task_reports], dtype=np.float32)
    after = np.asarray([task[f"word_{key}_after_mean"] for task in task_reports], dtype=np.float32)
    return word_labels, before.mean(axis=0), after.mean(axis=0)


def plot_results(output_path: str, task_reports: Sequence[Dict[str, object]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15, 10.5), dpi=180)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])

    ax1 = fig.add_subplot(grid[0, 0])
    idx = np.arange(len(task_reports))
    width = 0.28
    before_data = [np.asarray(task["sample_retrieval_before"], dtype=np.float32) for task in task_reports]
    after_data = [np.asarray(task["sample_retrieval_after"], dtype=np.float32) for task in task_reports]
    before_pos = idx - width / 2
    after_pos = idx + width / 2
    before_parts = ax1.violinplot(before_data, positions=before_pos, widths=width, showmeans=True, showmedians=False, showextrema=False)
    after_parts = ax1.violinplot(after_data, positions=after_pos, widths=width, showmeans=True, showmedians=False, showextrema=False)
    for body in before_parts["bodies"]:
        body.set_facecolor(PLOT_COLORS["before"])
        body.set_edgecolor(PLOT_COLORS["before"])
        body.set_alpha(0.55)
    for body in after_parts["bodies"]:
        body.set_facecolor(PLOT_COLORS["after"])
        body.set_edgecolor(PLOT_COLORS["after"])
        body.set_alpha(0.55)
    for part in ("cmeans",):
        if part in before_parts:
            before_parts[part].set_color(PLOT_COLORS["before"])
        if part in after_parts:
            after_parts[part].set_color(PLOT_COLORS["after"])
    ax1.set_xticks(idx)
    ax1.set_xticklabels([task["label"] for task in task_reports], rotation=20, ha="right")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("Content-Word Retrieval Accuracy")
    ax1.set_title("2.1 Strong-Template Retrieval Stability")
    ax1.legend(
        handles=[
            Patch(facecolor=PLOT_COLORS["before"], edgecolor=PLOT_COLORS["before"], alpha=0.55, label="Before Training"),
            Patch(facecolor=PLOT_COLORS["after"], edgecolor=PLOT_COLORS["after"], alpha=0.55, label="After Training"),
        ],
        loc="lower left",
        frameon=True,
    )

    ax2 = fig.add_subplot(grid[0, 1])
    pooled_x: List[float] = []
    pooled_y: List[float] = []
    color_map = plt.cm.get_cmap("tab10", len(task_reports))
    for task_idx, task in enumerate(task_reports):
        x = np.asarray(task["sample_template_drift"], dtype=np.float32)
        y = np.asarray(task["sample_retrieval_drop"], dtype=np.float32)
        pooled_x.extend(x.tolist())
        pooled_y.extend(y.tolist())
        ax2.scatter(x, y, s=18, alpha=0.65, color=color_map(task_idx), label=task["task_name"])
    if len(pooled_x) >= 2:
        x_arr = np.asarray(pooled_x, dtype=np.float32)
        y_arr = np.asarray(pooled_y, dtype=np.float32)
        slope, intercept = np.polyfit(x_arr, y_arr, deg=1)
        line_x = np.linspace(float(x_arr.min()), float(x_arr.max()), 100)
        line_y = slope * line_x + intercept
        ax2.plot(line_x, line_y, color=PLOT_COLORS["line"], linewidth=2.0, label="Linear Fit")
        corr_value = correlation(pooled_x, pooled_y)
        ax2.text(
            0.03,
            0.97,
            f"Pearson r = {corr_value:.3f}",
            transform=ax2.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
        )
    ax2.set_xlabel("Canonical Template Drift (Before -> After)")
    ax2.set_ylabel("Retrieval Drop (Before - After)")
    ax2.set_title("Template Drift vs. Content-Word Retrieval Loss")
    ax2.legend(loc="best", fontsize=8, frameon=True)

    ax3 = fig.add_subplot(grid[1, 0])
    word_labels, before_word_mean, after_word_mean = aggregate_word_alignment(task_reports, "alignment")
    x = np.arange(len(word_labels))
    bar_width = 0.38
    ax3.bar(x - bar_width / 2, before_word_mean, width=bar_width, color=PLOT_COLORS["before"], alpha=0.85, label="Before Training")
    ax3.bar(x + bar_width / 2, after_word_mean, width=bar_width, color=PLOT_COLORS["after"], alpha=0.85, label="After Training")
    ax3.set_xticks(x)
    ax3.set_xticklabels(word_labels, rotation=20, ha="right")
    ax3.set_ylim(0.0, 1.05)
    ax3.set_ylabel("Mean Content-Word Alignment")
    ax3.set_title("Token-Level Cosine Alignment")
    ax3.legend(loc="lower left", frameon=True)

    ax4 = fig.add_subplot(grid[1, 1])
    word_labels_rt, before_word_rt, after_word_rt = aggregate_word_alignment(task_reports, "retrieval")
    x_rt = np.arange(len(word_labels_rt))
    ax4.bar(x_rt - bar_width / 2, before_word_rt, width=bar_width, color=PLOT_COLORS["before"], alpha=0.85, label="Before Training")
    ax4.bar(x_rt + bar_width / 2, after_word_rt, width=bar_width, color=PLOT_COLORS["after"], alpha=0.85, label="After Training")
    ax4.set_xticks(x_rt)
    ax4.set_xticklabels(word_labels_rt, rotation=20, ha="right")
    ax4.set_ylim(0.0, 1.05)
    ax4.set_ylabel("Mean Retrieval Hit Rate")
    ax4.set_title("Token-Level Retrieval Accuracy")
    ax4.legend(loc="lower left", frameon=True)

    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_report(output_path: str, report: Dict[str, object]) -> None:
    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether template-token drift hurts semantic token alignment.")
    parser.add_argument("--base-model-path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--vision-tower-path", type=str, default=DEFAULT_VISION_TOWER_PATH)
    parser.add_argument("--ucit-root", type=str, default=DEFAULT_UCIT_ROOT)
    parser.add_argument("--checkpoint-root", type=str, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--samples-per-task", type=int, default=64)
    parser.add_argument("--description-hidden-layer", type=int, default=-2)
    parser.add_argument("--description-max-tokens", type=int, default=56)
    parser.add_argument("--conv-mode", type=str, default=DEFAULT_CONV_MODE)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)
    disable_torch_init()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    runtime_device = args.device
    if runtime_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable; falling back to CPU.")
        runtime_device = "cpu"
    model_dtype = torch.float16 if runtime_device.startswith("cuda") else torch.float32

    report = {
        "config": {
            "base_model_path": args.base_model_path,
            "vision_tower_path": args.vision_tower_path,
            "ucit_root": args.ucit_root,
            "checkpoint_root": args.checkpoint_root,
            "samples_per_task": args.samples_per_task,
            "description_hidden_layer": args.description_hidden_layer,
            "description_max_tokens": args.description_max_tokens,
            "experiment_version": "2.1",
            "prompt_variants": DEFAULT_PROMPT_VARIANTS,
            "seed": args.seed,
        },
        "tasks": [],
    }

    task_reports: List[Dict[str, object]] = []
    for task_id in args.task_ids:
        task_name, rel_path = TASK_DATA_FILES[task_id]
        data_path = os.path.join(args.ucit_root, rel_path)
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Missing data file: {data_path}")

        samples = load_task_samples(args.ucit_root, task_id, args.samples_per_task, args.seed)
        prev_path, cur_path = task_checkpoint_paths(args.checkpoint_root, task_id, args.base_model_path)
        if not os.path.isdir(cur_path) and not os.path.isfile(cur_path):
            raise FileNotFoundError(f"Missing current checkpoint: {cur_path}")
        if task_id > 1 and not os.path.isdir(prev_path):
            raise FileNotFoundError(f"Missing previous checkpoint: {prev_path}")

        print(f"[Task {task_id}] loading previous model: {prev_path}")
        prev_base = None if task_id == 1 else args.base_model_path
        prev_tokenizer, prev_model, prev_image_processor, prev_mm_use_im_start_end = load_model_bundle(
            prev_path,
            prev_base,
            args.vision_tower_path,
            runtime_device,
            model_dtype,
            task_id - 1 if task_id > 1 else 1,
        )
        before_sets = collect_prompt_state_sets(
            prev_model,
            prev_tokenizer,
            prev_image_processor,
            samples,
            DEFAULT_PROMPT_VARIANTS,
            prev_mm_use_im_start_end,
            args.conv_mode,
            args.description_hidden_layer,
            args.description_max_tokens,
            runtime_device,
            model_dtype,
        )
        del prev_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[Task {task_id}] loading current model: {cur_path}")
        cur_tokenizer, cur_model, cur_image_processor, cur_mm_use_im_start_end = load_model_bundle(
            cur_path,
            args.base_model_path,
            args.vision_tower_path,
            runtime_device,
            model_dtype,
            task_id,
        )
        after_sets = collect_prompt_state_sets(
            cur_model,
            cur_tokenizer,
            cur_image_processor,
            samples,
            DEFAULT_PROMPT_VARIANTS,
            cur_mm_use_im_start_end,
            args.conv_mode,
            args.description_hidden_layer,
            args.description_max_tokens,
            runtime_device,
            model_dtype,
        )
        del cur_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        task_report = summarize_task_metrics(task_id, task_name, samples, before_sets, after_sets)
        task_report["data_path"] = data_path
        task_report["previous_checkpoint"] = prev_path
        task_report["current_checkpoint"] = cur_path
        task_reports.append(task_report)
        print(
            f"[Task {task_id}] align_before={task_report['alignment_before_mean']:.4f} "
            f"align_after={task_report['alignment_after_mean']:.4f} "
            f"retr_before={task_report['retrieval_before_mean']:.4f} "
            f"retr_after={task_report['retrieval_after_mean']:.4f} "
            f"template_drift={task_report['template_drift_mean']:.4f} "
            f"retr_corr={task_report['template_retrieval_drop_corr']:.4f}"
        )

    report["tasks"] = task_reports
    report_path = os.path.join(args.output_dir, "template_alignment_report.json")
    figure_path = os.path.join(args.output_dir, "template_alignment_curves.png")
    save_report(report_path, report)
    plot_results(figure_path, task_reports)

    print(f"Saved report to: {report_path}")
    print(f"Saved figure to: {figure_path}")


if __name__ == "__main__":
    main()
