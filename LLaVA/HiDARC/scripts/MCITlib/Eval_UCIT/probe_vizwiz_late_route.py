"""Probe late one-hot expert choices on a small random VizWiz subset."""

import argparse
import json
import os
import random
import types

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.eval.CoIN.coin_utils import get_model_name_from_path
from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def split_list(items, num_chunks):
    chunk_size = (len(items) + num_chunks - 1) // num_chunks
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def probe(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        num_task=args.num_task,
        text_tower=args.text_tower,
        routing_config_path=args.routing_config_path,
        stage1_band_schedule_path=args.stage1_band_schedule_path,
    )

    with open(os.path.expanduser(args.question_file), "r", encoding="utf-8") as f:
        questions = json.load(f)
    if args.sample_size > len(questions):
        raise ValueError(f"sample_size={args.sample_size} exceeds dataset size={len(questions)}")
    sampled = random.Random(args.seed).sample(questions, args.sample_size)
    if args.num_chunks > 1:
        chunks = split_list(sampled, args.num_chunks)
        if args.chunk_idx >= len(chunks):
            raise ValueError(
                f"chunk_idx={args.chunk_idx} is out of range for {len(chunks)} chunks"
            )
        sampled = chunks[args.chunk_idx]

    route_records = []
    role_state = None
    last_role_scores = None
    original_route_builder = model._build_progressive_route_plan
    original_score_roles = model._score_roles

    def logged_score_roles(self, *score_args, **score_kwargs):
        nonlocal last_role_scores
        scores = original_score_roles(*score_args, **score_kwargs)
        if getattr(self, "_route_probe_current_sample", None) is not None:
            last_role_scores = scores.detach().float().cpu().tolist()
        return scores

    model._score_roles = types.MethodType(logged_score_roles, model)

    def logged_route_builder(self, active_experts, task_scores, image_summary, text_summary):
        nonlocal last_role_scores
        route_plan = original_route_builder(active_experts, task_scores, image_summary, text_summary)
        sample = getattr(self, "_route_probe_current_sample", None)
        if sample is not None:
            nonlocal role_state
            if role_state is None:
                active_roles = int(self._get_active_role_count())
                role_state = {
                    "active_role_count": active_roles,
                    "task_role_membership": self.task_role_membership[:active_experts, :active_roles]
                    .detach()
                    .float()
                    .cpu()
                    .tolist(),
                }
            late_basis = route_plan["late_basis"].detach().float().cpu().tolist()
            route_records.append(
                {
                    "question_id": sample["question_id"],
                    "image": sample["image"],
                    "selected_expert": int(route_plan["candidate_experts"][0]),
                    "late_basis": late_basis,
                    "task_scores": task_scores.detach().float().cpu().tolist(),
                    "active_experts": int(active_experts),
                    "selected_role": (
                        int(max(range(len(last_role_scores)), key=last_role_scores.__getitem__))
                        if last_role_scores
                        else None
                    ),
                    "role_scores": last_role_scores,
                }
            )
            last_role_scores = None
        return route_plan

    model._build_progressive_route_plan = types.MethodType(logged_route_builder, model)
    model.eval()

    for line in tqdm(sampled, desc="routing probe"):
        model._route_probe_current_sample = line
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        with Image.open(os.path.join(args.image_folder, line["image"])) as image:
            image = image.convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = [KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)]
        with torch.inference_mode():
            model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )
        model._route_probe_current_sample = None

    if len(route_records) != len(sampled):
        raise RuntimeError(
            f"Expected one route record per sample, got {len(route_records)} for {len(sampled)} samples"
        )

    expected_experts = set(args.expected_experts)
    wrong = [
        record
        for record in route_records
        if record["selected_expert"] not in expected_experts
    ]
    histogram = {}
    role_histogram = {}
    for record in route_records:
        key = str(record["selected_expert"])
        histogram[key] = histogram.get(key, 0) + 1
        role_key = str(record["selected_role"])
        role_histogram[role_key] = role_histogram.get(role_key, 0) + 1

    output = {
        "seed": args.seed,
        "sample_size": len(route_records),
        "expected_experts": sorted(expected_experts),
        "selected_histogram": histogram,
        "selected_role_histogram": role_histogram,
        "wrong_count_vs_expected_expert": len(wrong),
        "role_state": role_state,
        "records": route_records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(json.dumps({key: output[key] for key in output if key != "records"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--text-tower", required=True)
    parser.add_argument("--num-task", type=int, required=True)
    parser.add_argument("--routing-config-path", required=True)
    parser.add_argument("--stage1-band-schedule-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--expected-experts", type=int, nargs="+", default=[2])
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    probe(parser.parse_args())
