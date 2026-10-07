"""Record UCIT expert/role routing on small fixed subsets of each dataset."""

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


DATASETS = {
    "ImageNet-R": ("ImageNet-R/test_3000.json", 0),
    "ArxivQA": ("ArxivQA/test_3000.json", 1),
    "VizWiz": ("VizWiz/test_3000.json", 2),
    "IconQA": ("IconQA/test_3000.json", 3),
    "CLEVR-Math": ("CLEVR/test_3000.json", 4),
    "Flickr30k": ("Flickr30k/test_3000.json", 5),
}


def _as_list(tensor):
    return tensor.detach().float().cpu().tolist()


def _split_sample(items, sample_size, seed):
    if sample_size > len(items):
        raise ValueError(f"sample_size={sample_size} exceeds dataset size={len(items)}")
    return random.Random(seed).sample(items, sample_size)


def main(args):
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

    datasets = [name for name in args.datasets if name in DATASETS]
    if not datasets:
        raise ValueError("No supported datasets requested")

    sampled = {}
    for offset, name in enumerate(datasets):
        rel_path, _ = DATASETS[name]
        question_path = os.path.join(args.data_root, rel_path)
        with open(question_path, "r", encoding="utf-8") as handle:
            questions = json.load(handle)
        sampled[name] = _split_sample(questions, args.sample_size, args.seed + offset)

    records = []
    role_state = None
    last_role_scores = None
    current = None
    original_route_builder = model._build_progressive_route_plan
    original_score_roles = model._score_roles

    def logged_score_roles(self, *score_args, **score_kwargs):
        nonlocal last_role_scores
        scores = original_score_roles(*score_args, **score_kwargs)
        if current is not None:
            last_role_scores = _as_list(scores)
        return scores

    def logged_route_builder(self, active_experts, task_scores, image_summary, text_summary):
        nonlocal role_state, last_role_scores
        route_plan = original_route_builder(
            active_experts, task_scores, image_summary, text_summary
        )
        if current is not None:
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
            records.append(
                {
                    "dataset": current["dataset"],
                    "expected_expert": DATASETS[current["dataset"]][1],
                    "question_id": current["question_id"],
                    "image": current["image"],
                    "selected_expert": int(route_plan["candidate_experts"][0]),
                    "selected_role": (
                        int(max(range(len(last_role_scores)), key=last_role_scores.__getitem__))
                        if last_role_scores
                        else None
                    ),
                    "role_scores": last_role_scores,
                    "task_scores": _as_list(task_scores),
                    "early_basis": _as_list(route_plan["early_basis"]),
                    "middle_basis": _as_list(route_plan["middle_basis"]),
                    "late_basis": _as_list(route_plan["late_basis"]),
                    "active_experts": int(active_experts),
                }
            )
            last_role_scores = None
        return route_plan

    model._score_roles = types.MethodType(logged_score_roles, model)
    model._build_progressive_route_plan = types.MethodType(logged_route_builder, model)
    model.eval()

    for dataset in datasets:
        for line in tqdm(sampled[dataset], desc=f"routing {dataset}"):
            current = {"dataset": dataset, **line}
            question = line["text"]
            if model.config.mm_use_im_start_end:
                question = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + question
                )
            else:
                question = DEFAULT_IMAGE_TOKEN + "\n" + question

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()

            image_path = os.path.join(args.image_folder, line["image"])
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensor = image_processor.preprocess(
                    image, return_tensors="pt"
                )["pixel_values"][0]

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
            current = None

    if len(records) != len(datasets) * args.sample_size:
        raise RuntimeError(
            f"Expected {len(datasets) * args.sample_size} route records, got {len(records)}"
        )

    summary = {
        "seed": args.seed,
        "sample_size_per_dataset": args.sample_size,
        "datasets": datasets,
        "checkpoint": model_path,
        "role_state": role_state,
        "by_dataset": {},
        "records": records,
    }
    for dataset in datasets:
        subset = [record for record in records if record["dataset"] == dataset]
        histogram = {}
        role_histogram = {}
        for record in subset:
            expert_key = str(record["selected_expert"])
            role_key = str(record["selected_role"])
            histogram[expert_key] = histogram.get(expert_key, 0) + 1
            role_histogram[role_key] = role_histogram.get(role_key, 0) + 1
        correct = sum(
            record["selected_expert"] == record["expected_expert"] for record in subset
        )
        summary["by_dataset"][dataset] = {
            "expected_expert": DATASETS[dataset][1],
            "sample_count": len(subset),
            "selected_histogram": histogram,
            "selected_role_histogram": role_histogram,
            "expert_hit_count": correct,
            "expert_hit_fraction": correct / len(subset),
        }

    output_path = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--text-tower", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--num-task", type=int, required=True)
    parser.add_argument("--routing-config-path", required=True)
    parser.add_argument("--stage1-band-schedule-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    main(parser.parse_args())
