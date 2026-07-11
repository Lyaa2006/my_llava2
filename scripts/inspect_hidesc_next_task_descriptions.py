#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIDE_ROOT = PROJECT_ROOT / "LLaVA" / "HiDe"

if str(HIDE_ROOT) not in sys.path:
    sys.path.insert(0, str(HIDE_ROOT))

from llava.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates  # noqa: E402
from llava.mm_utils import (  # noqa: E402
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402


DESCRIPTION_PROMPT = (
    "Describe the image using visual evidence: objects, attributes, shapes, "
    "colors, textures, scene context, visible text, and spatial relations."
)

DEFAULT_BASE_MODEL = "/mnt/lyaa/my_llava/llava-v1.5-7b"
DEFAULT_VISION_TOWER = "/mnt/lyaa/my_llava/clip-vit-large-patch14-336"
DEFAULT_TEXT_TOWER = "/mnt/lyaa/my_llava/clip-vit-large-patch14-336"
DEFAULT_IMAGE_ROOT = "/mnt/lyaa/my_llava/UCIT/datasets"

TRANSITIONS = [
    {
        "name": "Task1_to_Task2_ArxivQA",
        "checkpoint": "/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe/Task1_llava_lora",
        "dataset": "/mnt/lyaa/my_llava/UCIT/ArxivQA/train_4w.json",
    },
    {
        "name": "Task2_to_Task3_VizWiz",
        "checkpoint": "/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe/Task2_llava_lora",
        "dataset": "/mnt/lyaa/my_llava/UCIT/VizWiz/train.json",
    },
    {
        "name": "Task3_to_Task4_IconQA",
        "checkpoint": "/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe/Task3_llava_lora",
        "dataset": "/mnt/lyaa/my_llava/UCIT/IconQA/train.json",
    },
    {
        "name": "Task4_to_Task5_CLEVR",
        "checkpoint": "/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe/Task4_llava_lora",
        "dataset": "/mnt/lyaa/my_llava/UCIT/CLEVR/train_4w.json",
    },
    {
        "name": "Task5_to_Task6_Flickr30k",
        "checkpoint": "/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe/Task5_llava_lora",
        "dataset": "/mnt/lyaa/my_llava/UCIT/Flickr30k/train_brief_4w.json",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-transition", type=int, default=1)
    parser.add_argument(
        "--model-source",
        choices=("checkpoint", "base"),
        default="checkpoint",
        help="Use task-specific checkpoints or the shared base model for all transitions.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--vision-tower", default=DEFAULT_VISION_TOWER)
    parser.add_argument("--text-tower", default=DEFAULT_TEXT_TOWER)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--num-task", type=int, default=6)
    return parser.parse_args()


def load_dataset(path):
    with open(path, "r") as f:
        return json.load(f)


def strip_question_text(raw_text):
    text = raw_text.replace("<image>", "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def sample_entries(dataset, image_root, sample_count, rng):
    candidates = []
    for idx, item in enumerate(dataset):
        image_rel = item.get("image")
        if not image_rel:
            continue
        image_path = os.path.join(image_root, image_rel)
        if not os.path.exists(image_path):
            continue
        user_turn = None
        for turn in item.get("conversations", []):
            if turn.get("from") == "human":
                user_turn = turn.get("value", "")
                break
        candidates.append(
            {
                "dataset_index": idx,
                "image_rel": image_rel,
                "image_path": image_path,
                "source_prompt": strip_question_text(user_turn or ""),
            }
        )
    if len(candidates) < sample_count:
        raise ValueError(f"Not enough image samples in dataset, requested {sample_count}, found {len(candidates)}")
    return rng.sample(candidates, sample_count)


def infer_conv_mode(model_name):
    lowered = model_name.lower()
    if "llama-2" in lowered:
        return "llava_llama_2"
    if "v1" in lowered:
        return "llava_v1"
    if "mpt" in lowered:
        return "mpt"
    return "llava_v0"


def resolve_model_spec(args, transition):
    if args.model_source == "base":
        return {
            "label": args.base_model,
            "model_path": args.base_model,
            "model_base": None,
        }
    return {
        "label": transition["checkpoint"],
        "model_path": transition["checkpoint"],
        "model_base": args.base_model,
    }


def build_base_config_override(args):
    cfg = AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    cfg.mm_vision_tower = args.vision_tower
    cfg.vision_tower = args.vision_tower
    cfg.mm_text_tower = args.text_tower
    cfg.text_tower = args.text_tower
    if getattr(cfg, "mm_text_select_layer", None) is None:
        cfg.mm_text_select_layer = -2
    return cfg


def setup_base_model_runtime(model, tokenizer, args):
    clip_tokenizer = AutoTokenizer.from_pretrained(
        args.text_tower,
        cache_dir=None,
        model_max_length=77,
        padding_side="right",
        use_fast=True,
        local_files_only=True,
    )
    model.set_clip_tokenizer(clip_tokenizer)
    model.set_tokenizer(tokenizer)
    model.set_eval(args.num_task)


def generate_description(model, tokenizer, image_processor, image_path, query, conv_mode, temperature, max_new_tokens):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = Image.open(image_path).convert("RGB")
    device = next(model.parameters()).device
    images_tensor = process_images([image], image_processor, model.config).to(device=device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=temperature > 0,
            temperature=temperature,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    output_text = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    if output_text.endswith(stop_str):
        output_text = output_text[: -len(stop_str)].strip()
    return output_text


def main():
    args = parse_args()
    disable_torch_init()
    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    loaded_model_spec = None
    loaded_bundle = None

    for transition in TRANSITIONS:
        dataset = load_dataset(transition["dataset"])
        samples = sample_entries(dataset, args.image_root, args.samples_per_transition, rng)
        model_spec = resolve_model_spec(args, transition)
        model_name = get_model_name_from_path(model_spec["model_path"])
        conv_mode = infer_conv_mode(model_name)

        if loaded_model_spec != model_spec["label"]:
            print(f"[load] model from {model_spec['label']}", flush=True)
            if loaded_bundle is not None:
                del loaded_bundle
                torch.cuda.empty_cache()
            load_kwargs = {
                "text_tower": args.text_tower,
                "num_task": args.num_task,
                "device_map": "auto",
            }
            if args.model_source == "base":
                load_kwargs["config"] = build_base_config_override(args)
            tokenizer, model, image_processor, _ = load_pretrained_model(
                model_spec["model_path"],
                model_spec["model_base"],
                model_name,
                **load_kwargs,
            )
            if args.model_source == "base":
                setup_base_model_runtime(model, tokenizer, args)
            model.eval()
            loaded_bundle = (tokenizer, model, image_processor)
            loaded_model_spec = model_spec["label"]
        else:
            tokenizer, model, image_processor = loaded_bundle

        for sample in samples:
            print(f"[gen] {transition['name']} idx={sample['dataset_index']} image={sample['image_rel']}", flush=True)
            description = generate_description(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image_path=sample["image_path"],
                query=DESCRIPTION_PROMPT,
                conv_mode=conv_mode,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
            all_results.append(
                {
                    "transition": transition["name"],
                    "checkpoint": transition["checkpoint"],
                    "model_source": args.model_source,
                    "model_path_used": model_spec["model_path"],
                    "dataset": transition["dataset"],
                    "dataset_index": sample["dataset_index"],
                    "image_rel": sample["image_rel"],
                    "source_prompt": sample["source_prompt"],
                    "description_prompt": DESCRIPTION_PROMPT,
                    "generated_description": description,
                }
            )

    if loaded_bundle is not None:
        del loaded_bundle
        torch.cuda.empty_cache()

    with open(output_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {len(all_results)} samples to {output_path}", flush=True)


if __name__ == "__main__":
    main()
