"""Run VizWiz eval and record average HiDESC routing activations."""

import argparse
import json
import math
import os
import types

import shortuuid
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
    chunk_size = math.ceil(len(items) / num_chunks)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def get_chunk(items, num_chunks, chunk_idx):
    return split_list(items, num_chunks)[chunk_idx]


def _tensor_to_list(tensor):
    if tensor is None:
        return None
    return tensor.detach().float().cpu().tolist()


def _average_per_layer_expert_activation(route_plan, active_experts):
    per_layer = route_plan.get("per_layer", [])
    if not per_layer:
        return [0.0] * active_experts
    stacked = torch.stack([
        weights[:active_experts].detach().float().cpu()
        for weights in per_layer
    ], dim=0)
    return stacked.mean(dim=0).tolist()


def eval_model(args):
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
    if image_processor is None:
        raise RuntimeError(
            f"Failed to initialize image processor for multimodal checkpoint: {model_path}"
        )

    with open(os.path.expanduser(args.question_file), "r", encoding="utf-8") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    route_stats_file = os.path.expanduser(args.route_stats_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    os.makedirs(os.path.dirname(route_stats_file), exist_ok=True)

    route_records = []
    role_state = None
    original_route_builder = model._build_progressive_route_plan

    def logged_route_builder(self, active_experts, task_scores, image_summary, text_summary):
        nonlocal role_state
        route_plan = original_route_builder(active_experts, task_scores, image_summary, text_summary)
        sample = getattr(self, "_route_eval_current_sample", None)
        if sample is not None:
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
            route_records.append(
                {
                    "question_id": sample["question_id"],
                    "image": sample["image"],
                    "selected_expert": int(route_plan["candidate_experts"][0]),
                    "selected_role": int(
                        torch.argmax(route_plan["role_scores"]).item()
                    ) if route_plan.get("role_scores") is not None else None,
                    "role_scores": _tensor_to_list(route_plan.get("role_scores")),
                    "task_scores": _tensor_to_list(task_scores),
                    "candidate_experts": [int(x) for x in route_plan.get("candidate_experts", [])],
                    "early_basis": _tensor_to_list(route_plan["early_basis"]),
                    "middle_basis": _tensor_to_list(route_plan["middle_basis"]),
                    "late_basis": _tensor_to_list(route_plan["late_basis"]),
                    "average_expert_activation": _average_per_layer_expert_activation(
                        route_plan, active_experts
                    ),
                    "active_experts": int(active_experts),
                }
            )
        return route_plan

    model._build_progressive_route_plan = types.MethodType(logged_route_builder, model)
    model.eval()

    with open(answers_file, "w", encoding="utf-8") as ans_file:
        for line in tqdm(questions, desc="vizwiz eval"):
            model._route_eval_current_sample = line
            idx = line["question_id"]
            image_file = line["image"]
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

            with Image.open(os.path.join(args.image_folder, image_file)) as image:
                image = image.convert("RGB")
                image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = [KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                )

            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": cur_prompt,
                        "text": outputs,
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_name,
                        "metadata": {},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            ans_file.flush()
            model._route_eval_current_sample = None

    if len(route_records) != len(questions):
        raise RuntimeError(
            f"Expected {len(questions)} route records, got {len(route_records)}"
        )

    active_experts = max(record["active_experts"] for record in route_records)
    early_sum = [0.0] * active_experts
    middle_sum = [0.0] * active_experts
    late_sum = [0.0] * active_experts
    expert_activation_sum = [0.0] * active_experts
    selected_histogram = {}
    for record in route_records:
        selected_key = str(record["selected_expert"])
        selected_histogram[selected_key] = selected_histogram.get(selected_key, 0) + 1
        for idx in range(active_experts):
            early_sum[idx] += record["early_basis"][idx]
            middle_sum[idx] += record["middle_basis"][idx]
            late_sum[idx] += record["late_basis"][idx]
            expert_activation_sum[idx] += record["average_expert_activation"][idx]
    total = float(len(route_records))
    summary = {
        "sample_count": len(route_records),
        "active_experts": active_experts,
        "selected_histogram": selected_histogram,
        "selected_fraction": {
            key: value / total for key, value in selected_histogram.items()
        },
        "average_early_basis": [value / total for value in early_sum],
        "average_middle_basis": [value / total for value in middle_sum],
        "average_late_basis": [value / total for value in late_sum],
        "average_expert_activation": [value / total for value in expert_activation_sum],
        "role_state": role_state,
        "records": route_records,
    }
    with open(route_stats_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--route-stats-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--text-tower", type=str, required=True)
    parser.add_argument("--num-task", type=int, required=True)
    parser.add_argument("--routing-config-path", type=str, default=None)
    parser.add_argument("--stage1-band-schedule-path", type=str, default=None)
    eval_model(parser.parse_args())
