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
from HiDe.peft import PeftModel, WEIGHTS_NAME  # noqa: E402


DEFAULT_BASE_MODEL_PATH = "/mnt/lyaa/my_llava/llava-v1.5-7b"
DEFAULT_VISION_TOWER_PATH = "/mnt/lyaa/my_llava/clip-vit-large-patch14-336"
DEFAULT_UCIT_ROOT = "/mnt/lyaa/my_llava/UCIT"
DEFAULT_CHECKPOINT_ROOT = "/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe"
DEFAULT_OUTPUT_DIR = os.path.join(MCITLIB_ROOT, "docs", "experiment2_description_drift")
DEFAULT_DESCRIPTION_PROMPT = (
    "Describe the image using visual evidence: objects, attributes, shapes, colors, "
    "textures, scene context, visible text, and spatial relations."
)
DEFAULT_CONV_MODE = "llava_v1"
DESCRIPTION_KEY_TERMS = (
    "object",
    "objects",
    "attribute",
    "attributes",
    "shape",
    "shapes",
    "color",
    "colors",
    "texture",
    "textures",
    "scene",
    "text",
    "spatial",
    "relation",
    "relations",
)
DESCRIPTION_KEY_STOPWORDS = {
    "and",
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "in",
    "on",
    "at",
    "from",
    "by",
    "with",
    "using",
    "visual",
    "evidence",
}
ASSISTANT_PREFIX_TOKENS = ("▁A", "SS", "IST", "ANT", ":")
PUNCT_TOKENS = {",", ".", ":", ";", "?", "!"}

TASK_DATA_FILES = {
    1: ("ImageNet-R", "ImageNet-R/train.json"),
    2: ("ArxivQA", "ArxivQA/train_4w.json"),
    3: ("VizWiz", "VizWiz/train.json"),
    4: ("IconQA", "IconQA/train.json"),
    5: ("CLEVR", "CLEVR/train_4w.json"),
    6: ("Flickr30k", "Flickr30k/train_brief_4w.json"),
}

PLOT_COLORS = {
    "curve": "#2f6690",
    "key": "#d1495b",
    "template": "#4f772d",
    "ratio": "#edae49",
}


@dataclass
class Sample:
    task_id: int
    task_name: str
    image_path: str
    cache_key: str


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


def resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


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
        image_rel = item["image"]
        samples.append(
            Sample(
                task_id=task_id,
                task_name=task_name,
                image_path=os.path.join(ucit_root, "datasets", image_rel),
                cache_key=f"{task_name}_{idx:08d}",
            )
        )
    return samples


def load_image_tensor(image_path: str, image_processor, device: str, dtype: torch.dtype) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return pixel_values.unsqueeze(0).to(device=device, dtype=dtype)


def build_description_key_mask(
    input_ids: torch.Tensor,
    tokenizer,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    key_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    special_token_ids = set(tokenizer.all_special_ids)
    vocab_size = len(tokenizer)

    def normalize_piece(piece: str) -> str:
        piece = piece.replace("▁", "").lower()
        return "".join(ch for ch in piece if ch.isalnum())

    for batch_idx in range(input_ids.shape[0]):
        cur_len = int(attention_mask[batch_idx].long().sum().item())
        token_ids = input_ids[batch_idx, :cur_len].tolist()
        token_pieces = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= vocab_size or token_id in special_token_ids:
                token_pieces.append("")
                continue
            try:
                token_pieces.append(tokenizer.convert_ids_to_tokens(int(token_id)))
            except (IndexError, OverflowError):
                token_pieces.append("")

        content_start = next(
            (
                i + 2
                for i in range(cur_len - 1)
                if token_pieces[i] == "▁evidence" and token_pieces[i + 1] == ":"
            ),
            None,
        )
        assistant_start = None
        if cur_len >= len(ASSISTANT_PREFIX_TOKENS):
            suffix = token_pieces[-len(ASSISTANT_PREFIX_TOKENS):]
            if suffix == list(ASSISTANT_PREFIX_TOKENS):
                assistant_start = cur_len - len(ASSISTANT_PREFIX_TOKENS)

        if content_start is not None:
            content_end = assistant_start if assistant_start is not None else cur_len
            for token_idx in range(content_start, content_end):
                token_piece = token_pieces[token_idx]
                if token_piece in PUNCT_TOKENS:
                    continue
                token_norm = normalize_piece(token_piece)
                if not token_norm or token_norm in DESCRIPTION_KEY_STOPWORDS:
                    continue
                key_mask[batch_idx, token_idx] = True

        if not torch.any(key_mask[batch_idx, :cur_len]):
            for token_idx in range(cur_len):
                token_id = int(input_ids[batch_idx, token_idx].item())
                if token_id < 0 or token_id >= vocab_size or token_id in special_token_ids:
                    continue
                try:
                    token_text = tokenizer.decode([token_id], skip_special_tokens=True).strip().lower()
                except OverflowError:
                    continue
                token_text = "".join(ch for ch in token_text if ch.isalnum())
                if not token_text:
                    continue
                if any(term in token_text or token_text in term for term in DESCRIPTION_KEY_TERMS):
                    key_mask[batch_idx, token_idx] = True
        if not torch.any(key_mask[batch_idx, :cur_len]):
            key_mask[batch_idx, :cur_len] = attention_mask[batch_idx, :cur_len].bool()
    return key_mask


def rms_normalize_hidden(hidden_states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = hidden_states.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(eps).sqrt()
    return hidden_states.float() / rms


def select_description_tokens(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    key_mask: torch.Tensor,
    max_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_len = int(attention_mask.long().sum().item())
    start_idx = max(0, seq_len - max_tokens)
    return hidden_states[0, start_idx:seq_len].detach().float().cpu(), key_mask[0, start_idx:seq_len].detach().cpu()


def collect_description_states(
    model,
    tokenizer,
    image_processor,
    samples: Sequence[Sample],
    description_prompt: str,
    mm_use_im_start_end: bool,
    conv_mode: str,
    description_hidden_layer: int,
    description_max_tokens: int,
    device: str,
    dtype: torch.dtype,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    state_list: List[torch.Tensor] = []
    key_mask_list: List[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            prompt = build_prompt(description_prompt, mm_use_im_start_end, conv_mode)
            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(device)
            attention_mask = input_ids.ne(tokenizer.pad_token_id)
            key_mask = build_description_key_mask(input_ids, tokenizer, attention_mask)
            image_tensor = load_image_tensor(sample.image_path, image_processor, device, dtype)
            outputs = model(
                input_ids=input_ids,
                images=image_tensor,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states[description_hidden_layer]
            desc_states, desc_key_mask = select_description_tokens(
                hidden_states,
                attention_mask,
                key_mask,
                description_max_tokens,
            )
            state_list.append(desc_states)
            key_mask_list.append(desc_key_mask)
    return state_list, key_mask_list


def pad_sequence_list(
    tensors: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not tensors:
        raise ValueError("tensors must be non-empty")
    max_len = max(t.shape[0] for t in tensors)
    hidden_size = tensors[0].shape[-1]
    padded_states = torch.zeros((len(tensors), max_len, hidden_size), dtype=torch.float32)
    padded_masks = torch.zeros((len(tensors), max_len), dtype=torch.bool)
    for idx, (state, mask) in enumerate(zip(tensors, masks)):
        seq_len = state.shape[0]
        padded_states[idx, :seq_len] = state.float()
        padded_masks[idx, :seq_len] = mask.bool()
    return padded_states, padded_masks


def compute_task_metrics(
    prev_states: Sequence[torch.Tensor],
    cur_states: Sequence[torch.Tensor],
    key_masks: Sequence[torch.Tensor],
) -> Dict[str, object]:
    shared_len = min(min(t.shape[0] for t in prev_states), min(t.shape[0] for t in cur_states))
    prev_trimmed = [t[:shared_len] for t in prev_states]
    cur_trimmed = [t[:shared_len] for t in cur_states]
    key_trimmed = [m[:shared_len] for m in key_masks]

    valid_trimmed = [torch.ones(shared_len, dtype=torch.bool) for _ in prev_trimmed]
    prev_tensor, _ = pad_sequence_list(prev_trimmed, valid_trimmed)
    cur_tensor, valid_mask = pad_sequence_list(cur_trimmed, valid_trimmed)
    key_tensor = torch.stack([m[:shared_len] for m in key_trimmed], dim=0)

    drift = (rms_normalize_hidden(cur_tensor) - rms_normalize_hidden(prev_tensor)).norm(dim=-1)
    total_curve_sum = drift.sum(dim=0)
    total_curve_count = valid_mask.float().sum(dim=0).clamp_min(1.0)
    token_curve = (total_curve_sum / total_curve_count).tolist()

    sample_key_mean: List[float] = []
    sample_template_mean: List[float] = []
    sample_focus_ratio: List[float] = []
    sample_total_mean: List[float] = []
    sample_key_mass: List[float] = []

    for sample_idx in range(drift.shape[0]):
        token_change = drift[sample_idx]
        valid = valid_mask[sample_idx]
        key_mask = key_tensor[sample_idx] & valid
        if not torch.any(key_mask):
            key_mask = valid.clone()
        non_key_mask = valid & (~key_mask)
        total_change = token_change[valid].sum().clamp_min(1e-6)
        key_mass = token_change[key_mask].sum() / total_change
        key_density = token_change[key_mask].mean() if torch.any(key_mask) else token_change.new_zeros(())
        template_density = token_change[non_key_mask].mean() if torch.any(non_key_mask) else token_change.new_zeros(())
        sample_key_mean.append(float(key_density.item()))
        sample_template_mean.append(float(template_density.item()))
        sample_focus_ratio.append(float((template_density / key_density.clamp_min(1e-6)).item()))
        sample_total_mean.append(float((token_change[valid].mean()).item()))
        sample_key_mass.append(float(key_mass.item()))

    return {
        "token_curve": token_curve,
        "key_position_mask": key_tensor.any(dim=0).tolist(),
        "sample_key_means": sample_key_mean,
        "sample_template_means": sample_template_mean,
        "key_mean": float(np.mean(sample_key_mean)),
        "template_mean": float(np.mean(sample_template_mean)),
        "nonkey_mean": float(np.mean(sample_template_mean)),
        "focus_ratio": float(np.mean(sample_focus_ratio)),
        "total_mean": float(np.mean(sample_total_mean)),
        "key_mass": float(np.mean(sample_key_mass)),
    }


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


def plot_results(output_path: str, task_reports: Sequence[Dict[str, object]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.5), dpi=180, gridspec_kw={"height_ratios": [1.7, 1.1]})

    ax = axes[0]
    task_labels = [task["label"] for task in task_reports]
    idx = np.arange(len(task_reports))
    width = 0.28
    key_data = [np.asarray(task["sample_key_means"], dtype=np.float32) for task in task_reports]
    template_data = [np.asarray(task["sample_template_means"], dtype=np.float32) for task in task_reports]
    key_pos = idx - width / 2
    template_pos = idx + width / 2
    key_parts = ax.violinplot(key_data, positions=key_pos, widths=width, showmeans=True, showmedians=False, showextrema=False)
    template_parts = ax.violinplot(template_data, positions=template_pos, widths=width, showmeans=True, showmedians=False, showextrema=False)
    for body in key_parts["bodies"]:
        body.set_facecolor(PLOT_COLORS["key"])
        body.set_edgecolor(PLOT_COLORS["key"])
        body.set_alpha(0.55)
    for body in template_parts["bodies"]:
        body.set_facecolor(PLOT_COLORS["template"])
        body.set_edgecolor(PLOT_COLORS["template"])
        body.set_alpha(0.55)
    for part in ("cmeans",):
        if part in key_parts:
            key_parts[part].set_color(PLOT_COLORS["key"])
        if part in template_parts:
            template_parts[part].set_color(PLOT_COLORS["template"])
    ax.set_xticks(idx)
    ax.set_xticklabels(task_labels, rotation=20, ha="right")
    ax.set_ylabel("Token Drift")
    ax.set_title("Structure-Aware Key vs. Template Drift Distribution")
    ax.legend(
        handles=[
            Patch(facecolor=PLOT_COLORS["key"], edgecolor=PLOT_COLORS["key"], alpha=0.55, label="Key Token"),
            Patch(facecolor=PLOT_COLORS["template"], edgecolor=PLOT_COLORS["template"], alpha=0.55, label="Template Token"),
        ],
        loc="upper right",
        frameon=True,
    )

    ax2 = axes[1]
    max_len = max(len(task["token_curve"]) for task in task_reports)
    x = np.arange(1, max_len + 1)
    aggregate = np.zeros(max_len, dtype=np.float32)
    counts = np.zeros(max_len, dtype=np.float32)
    key_mask = np.zeros(max_len, dtype=bool)
    for task in task_reports:
        curve = np.asarray(task["token_curve"], dtype=np.float32)
        mask = np.asarray(task["key_position_mask"], dtype=bool)
        aggregate[: len(curve)] += curve
        counts[: len(curve)] += 1.0
        key_mask[: len(mask)] |= mask
    aggregate = aggregate / np.clip(counts, 1.0, None)
    ax2.plot(x[: len(aggregate)], aggregate, color=PLOT_COLORS["curve"], linewidth=2.5, label="Mean Drift")
    ax2.scatter(x[key_mask], aggregate[key_mask], color=PLOT_COLORS["key"], s=22, label="Key Token")
    ax2.scatter(x[~key_mask], aggregate[~key_mask], color=PLOT_COLORS["template"], s=14, alpha=0.65, label="Template Token")
    ax2.set_title("Token Drift Curve with Structure-Aware Key Tokens")
    ax2.set_xlabel("Description Token Position")
    ax2.set_ylabel("Mean Drift")
    ax2.set_xlim(1, max_len)
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_report(output_path: str, report: Dict[str, object]) -> None:
    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze description hidden-state drift before vs after HiDe training.")
    parser.add_argument("--base-model-path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--vision-tower-path", type=str, default=DEFAULT_VISION_TOWER_PATH)
    parser.add_argument("--ucit-root", type=str, default=DEFAULT_UCIT_ROOT)
    parser.add_argument("--checkpoint-root", type=str, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--samples-per-task", type=int, default=64)
    parser.add_argument("--description-prompt", type=str, default=DEFAULT_DESCRIPTION_PROMPT)
    parser.add_argument("--description-hidden-layer", type=int, default=-2)
    parser.add_argument("--description-max-tokens", type=int, default=32)
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
            "key_mask_strategy": "structure_aware",
            "seed": args.seed,
        },
        "tasks": [],
    }

    task_reports: List[Dict[str, object]] = []
    for task_id in args.task_ids:
        if task_id not in TASK_DATA_FILES:
            raise ValueError(f"Unsupported task id: {task_id}")
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
        prev_states, prev_key_masks = collect_description_states(
            prev_model,
            prev_tokenizer,
            prev_image_processor,
            samples,
            args.description_prompt,
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
        cur_states, _ = collect_description_states(
            cur_model,
            cur_tokenizer,
            cur_image_processor,
            samples,
            args.description_prompt,
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

        task_metrics = compute_task_metrics(prev_states, cur_states, prev_key_masks)
        label = f"Task {task_id} / {task_name}"
        task_report = {
            "task_id": task_id,
            "task_name": task_name,
            "label": label,
            "data_path": data_path,
            "previous_checkpoint": prev_path,
            "current_checkpoint": cur_path,
            "num_samples": len(samples),
            **task_metrics,
        }
        task_reports.append(task_report)
        print(
            f"[Task {task_id}] key={task_metrics['key_mean']:.4f} "
            f"template={task_metrics['template_mean']:.4f} "
            f"ratio={task_metrics['focus_ratio']:.4f}"
        )

    report["tasks"] = task_reports
    report_path = os.path.join(args.output_dir, "description_drift_report.json")
    figure_path = os.path.join(args.output_dir, "description_drift_curves.png")
    save_report(report_path, report)
    plot_results(figure_path, task_reports)

    print(f"Saved report to: {report_path}")
    print(f"Saved figure to: {figure_path}")


if __name__ == "__main__":
    main()
