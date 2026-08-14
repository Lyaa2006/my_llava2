#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from datetime import date
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
HIDESC_ROOT = REPO_ROOT / "LLaVA" / "HiDESC"
if str(HIDESC_ROOT) not in sys.path:
    sys.path.insert(0, str(HIDESC_ROOT))

from llava.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
)
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402


UCIT_TASKS = [
    {
        "task_id": 0,
        "task_name": "ImageNet-R",
        "train_path": "/mnt/lyaa/my_llava/UCIT/ImageNet-R/train.json",
        "test_path": "/mnt/lyaa/my_llava/UCIT/ImageNet-R/test_3000.json",
        "image_folder": "/mnt/lyaa/my_llava/UCIT/datasets",
    },
    {
        "task_id": 1,
        "task_name": "ArxivQA",
        "train_path": "/mnt/lyaa/my_llava/UCIT/ArxivQA/train_4w.json",
        "test_path": "/mnt/lyaa/my_llava/UCIT/ArxivQA/test_3000.json",
        "image_folder": "/mnt/lyaa/my_llava/UCIT/datasets",
    },
    {
        "task_id": 2,
        "task_name": "VizWiz",
        "train_path": "/mnt/lyaa/my_llava/UCIT/VizWiz/train.json",
        "test_path": "/mnt/lyaa/my_llava/UCIT/VizWiz/test_3000.json",
        "image_folder": "/mnt/lyaa/my_llava/UCIT/datasets",
    },
    {
        "task_id": 3,
        "task_name": "IconQA",
        "train_path": "/mnt/lyaa/my_llava/UCIT/IconQA/train.json",
        "test_path": "/mnt/lyaa/my_llava/UCIT/IconQA/test_3000.json",
        "image_folder": "/mnt/lyaa/my_llava/UCIT/datasets",
    },
    {
        "task_id": 4,
        "task_name": "CLEVR-Math",
        "train_path": "/mnt/lyaa/my_llava/UCIT/CLEVR/train_4w.json",
        "test_path": "/mnt/lyaa/my_llava/UCIT/CLEVR/test_3000.json",
        "image_folder": "/mnt/lyaa/my_llava/UCIT/datasets",
    },
    {
        "task_id": 5,
        "task_name": "Flickr30k",
        "train_path": "/mnt/lyaa/my_llava/UCIT/Flickr30k/train_brief_4w.json",
        "test_path": "/mnt/lyaa/my_llava/UCIT/Flickr30k/test_3000.json",
        "image_folder": "/mnt/lyaa/my_llava/UCIT/datasets",
    },
]

DCL_TASKS = [
    {
        "task_id": 0,
        "task_name": "RS",
        "train_path": "/mnt/lyaa/MCITlib/MLLM/domain/RS/train.json",
        "test_path": "/mnt/lyaa/MCITlib/MLLM/domain/RS/test.json",
        "image_folder": "/mnt/lyaa/MCITlib/MLLM/domain/RS",
    },
    {
        "task_id": 1,
        "task_name": "Med",
        "train_path": "/mnt/lyaa/MCITlib/MLLM/domain/Med/train.json",
        "test_path": "/mnt/lyaa/MCITlib/MLLM/domain/Med/test.json",
        "image_folder": "/mnt/lyaa/MCITlib/MLLM/domain/Med",
    },
    {
        "task_id": 2,
        "task_name": "AD",
        "train_path": "/mnt/lyaa/MCITlib/MLLM/domain/AD/train.json",
        "test_path": "/mnt/lyaa/MCITlib/MLLM/domain/AD/test.json",
        "image_folder": "/mnt/lyaa/MCITlib/MLLM/domain/AD",
    },
    {
        "task_id": 3,
        "task_name": "Sci",
        "train_path": "/mnt/lyaa/MCITlib/MLLM/domain/Sci/train.json",
        "test_path": "/mnt/lyaa/MCITlib/MLLM/domain/Sci/test.json",
        "image_folder": "/mnt/lyaa/MCITlib/MLLM/domain/Sci",
    },
    {
        "task_id": 4,
        "task_name": "Fin",
        "train_path": "/mnt/lyaa/MCITlib/MLLM/domain/Fin/train.json",
        "test_path": "/mnt/lyaa/MCITlib/MLLM/domain/Fin/test.json",
        "image_folder": "/mnt/lyaa/MCITlib/MLLM/domain/Fin",
    },
]

BENCHMARK_TASKS = {
    "ucit": UCIT_TASKS,
    "dcl": DCL_TASKS,
}

DEFAULT_MODEL_PATHS = {
    "ucit": (
        "/mnt/lyaa/MCITlib/checkpoints/UCIT/LLaVA/"
        "HiDESC_from_HiDeTask1_g4b8ga2/"
        "Task6_llava_lora_HiDESC_full_gpus0123_20260713_focus04_energy1e3"
    ),
    "dcl": (
        "/mnt/lyaa/MCITlib/runs/MLLM-DCL/HiDe/"
        "hide_dcl_full_20260718_204518/checkpoints/Task5_llava_lora"
    ),
}


def get_tasks(benchmark):
    try:
        return BENCHMARK_TASKS[benchmark]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark: {benchmark}") from exc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARK_TASKS.keys()),
        default="ucit",
    )
    parser.add_argument(
        "--model-path",
        default="",
    )
    parser.add_argument(
        "--model-base",
        default="/mnt/lyaa/my_llava/llava-v1.5-7b",
    )
    parser.add_argument(
        "--text-tower",
        default="/mnt/lyaa/my_llava/clip-vit-large-patch14-336",
    )
    parser.add_argument("--num-task", type=int, default=0)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--prototype-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--prototype-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "analysis_outputs" / "hidesc_spectral_routing"),
    )
    parser.add_argument("--prototype-cache", default="")
    parser.add_argument("--recompute-prototypes", action="store_true")
    parser.add_argument("--routing-config-json", default="")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_routing_config(path):
    if not path:
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Routing override must be a JSON object: {path}")
    return payload


def load_rgb_images(samples, image_folder):
    images = []
    for sample in samples:
        image_path = os.path.join(image_folder, sample["image"])
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images


def build_prompt(question_text, model_config, conv_mode):
    if model_config.mm_use_im_start_end:
        prompt_text = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + question_text
        )
    else:
        prompt_text = DEFAULT_IMAGE_TOKEN + "\n" + question_text
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_input_id_batch(prompts, tokenizer):
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in prompts
    ]
    max_len = max(x.shape[0] for x in input_ids)
    padded = torch.full(
        (len(input_ids), max_len),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    for idx, tensor in enumerate(input_ids):
        padded[idx, : tensor.shape[0]] = tensor
    return padded


def build_clip_text_inputs(model, tokenizer, input_ids_batch):
    input_pad = torch.where(
        input_ids_batch.eq(-200),
        torch.full_like(input_ids_batch, tokenizer.pad_token_id),
        input_ids_batch,
    )
    decoded_inputs = tokenizer.batch_decode(
        input_pad,
        skip_special_tokens=True,
    )
    decoded_hidden_inputs = [
        "\n".join(decoded_input.split("\n")[1:]) for decoded_input in decoded_inputs
    ]
    decoded_clip_inputs = [
        decoded_input.split(" ASSISTANT")[0] for decoded_input in decoded_hidden_inputs
    ]
    return model.clip_tokenizer(
        decoded_clip_inputs,
        padding="longest",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )


def extract_spectral_features(model, image_processor, samples):
    images = load_rgb_images(samples, samples[0]["_image_folder"])
    image_tensor = process_images(images, image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor, dim=0)
    image_tensor = image_tensor.to(
        device=model.device,
        dtype=model.get_vision_tower().dtype,
    )
    with torch.inference_mode():
        (
            clip_image_features,
            _selected_patch_features,
            _final_patch_features,
            projected_patch_features,
        ) = model.get_vision_tower()(image_tensor)
        spectral_features = model._extract_image_spectral_descriptor(
            projected_patch_features.to(model.device)
        )
    return clip_image_features.float(), spectral_features.float()


def compute_task_prototypes(args, model, image_processor, output_dir):
    tasks = get_tasks(args.benchmark)
    prototype_cache = (
        Path(args.prototype_cache)
        if args.prototype_cache
        else Path(output_dir) / f"{args.benchmark}_spectral_prototypes.pt"
    )
    if prototype_cache.exists() and not args.recompute_prototypes:
        payload = torch.load(prototype_cache, map_location="cpu")
        return payload, prototype_cache

    task_names = []
    prototype_list = []
    counts = []
    for task in tasks:
        records = load_json(task["train_path"])
        if args.prototype_limit > 0:
            records = records[: args.prototype_limit]
        for sample in records:
            sample["_image_folder"] = task["image_folder"]
        running_sum = None
        sample_count = 0
        progress = tqdm(
            range(0, len(records), args.prototype_batch_size),
            desc=f"prototype:{task['task_name']}",
        )
        for start in progress:
            batch = records[start : start + args.prototype_batch_size]
            _, spectral_features = extract_spectral_features(
                model,
                image_processor,
                batch,
            )
            if running_sum is None:
                running_sum = torch.zeros(
                    spectral_features.shape[-1],
                    dtype=torch.float32,
                )
            running_sum += spectral_features.cpu().sum(dim=0)
            sample_count += spectral_features.shape[0]
        prototype = running_sum / max(sample_count, 1)
        task_names.append(task["task_name"])
        prototype_list.append(prototype)
        counts.append(sample_count)

    payload = {
        "task_names": task_names,
        "counts": counts,
        "prototypes": torch.stack(prototype_list, dim=0),
        "metadata": {
            "benchmark": args.benchmark,
            "model_path": args.model_path,
            "model_base": args.model_base,
            "text_tower": args.text_tower,
            "spectral_cutoff": model.relation_routing_config["spectral_cutoff"],
            "spectral_low_bins": model.relation_routing_config["spectral_low_bins"],
            "spectral_high_bins": model.relation_routing_config["spectral_high_bins"],
            "prototype_limit": args.prototype_limit,
        },
    }
    prototype_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, prototype_cache)
    return payload, prototype_cache


def apply_spectral_prototypes(model, payload, benchmark=None):
    metadata = payload.get("metadata", {})
    benchmark_name = metadata.get("benchmark", benchmark)
    if benchmark_name is None:
        raise KeyError(
            "Prototype cache metadata is missing benchmark; pass --benchmark or regenerate cache."
        )
    tasks = get_tasks(benchmark_name)
    prototypes = payload["prototypes"]
    counts = payload["counts"]
    if prototypes.shape[0] < len(tasks):
        raise ValueError(
            f"Expected at least {len(tasks)} prototypes, got {prototypes.shape[0]}"
        )
    for task_id in range(len(tasks)):
        model.spectral_image_anchors[task_id].data.copy_(
            prototypes[task_id].to(
                device=model.device,
                dtype=model.spectral_image_anchors[task_id].dtype,
            ).unsqueeze(0)
        )
        model.spectral_image_boundary[task_id].data.copy_(
            torch.tensor(
                [float(counts[task_id])],
                device=model.device,
                dtype=model.spectral_image_boundary[task_id].dtype,
            )
        )


def init_stage_stats(active_experts):
    return {
        stage: {
            "weight_sum": torch.zeros(active_experts, dtype=torch.float64),
            "top1_counts": torch.zeros(active_experts, dtype=torch.long),
            "entropy_sum": 0.0,
            "margin_sum": 0.0,
            "expected_top1": 0,
        }
        for stage in ("early", "middle", "late")
    }


def finalize_stage_stats(stage_stats, sample_count):
    finalized = {}
    for stage, stats in stage_stats.items():
        denom = max(sample_count, 1)
        finalized[stage] = {
            "avg_weight": [round(x, 6) for x in (stats["weight_sum"] / denom).tolist()],
            "top1_rate": [round(x, 6) for x in (stats["top1_counts"].double() / denom).tolist()],
            "entropy_mean": stats["entropy_sum"] / denom,
            "margin_mean": stats["margin_sum"] / denom,
            "expected_expert_top1_rate": stats["expected_top1"] / denom,
        }
    return finalized


def evaluate_task(model, image_processor, tokenizer, task, args):
    records = load_json(task["test_path"])
    if args.eval_limit > 0:
        records = records[: args.eval_limit]
    for sample in records:
        sample["_image_folder"] = task["image_folder"]

    active_experts = max(1, min(int(model.expert_num), len(model.image_anchors)))
    model.ensure_role_bank_initialized(active_experts)
    stage_stats = init_stage_stats(active_experts)
    task_score_sum = torch.zeros(active_experts, dtype=torch.float64)
    task_score_top1_counts = torch.zeros(active_experts, dtype=torch.long)
    task_score_margin_sum = 0.0
    task_score_entropy_sum = 0.0
    task_score_expected_top1 = 0

    progress = tqdm(
        range(0, len(records), args.eval_batch_size),
        desc=f"eval:{task['task_name']}",
    )
    for start in progress:
        batch = records[start : start + args.eval_batch_size]
        prompts = [
            build_prompt(sample["text"], model.config, args.conv_mode)
            for sample in batch
        ]
        input_ids_batch = build_input_id_batch(prompts, tokenizer)
        clip_text_inputs = build_clip_text_inputs(model, tokenizer, input_ids_batch)
        _, spectral_features = extract_spectral_features(
            model,
            image_processor,
            batch,
        )
        with torch.inference_mode():
            text_features = model.get_text_tower()(clip_text_inputs.to(model.device))
            task_scores = model._compute_shared_task_scores(
                image_guide_features=spectral_features.to(model.device),
                text_guide_features=text_features,
                active_experts=active_experts,
            )
        if task_scores.ndim == 1:
            task_scores = task_scores.unsqueeze(0)

        for row_idx in range(task_scores.shape[0]):
            sample_scores = task_scores[row_idx].float()
            image_summary = model._summarize_guide_features(spectral_features[row_idx].to(model.device))
            text_summary = model._summarize_guide_features(text_features[row_idx])
            route_plan = model._build_progressive_route_plan(
                active_experts,
                sample_scores,
                image_summary,
                text_summary,
            )

            sorted_scores = torch.topk(sample_scores, k=min(2, sample_scores.numel())).values
            task_score_sum += sample_scores.detach().cpu().double()
            top_expert = int(torch.argmax(sample_scores).item())
            task_score_top1_counts[top_expert] += 1
            task_score_expected_top1 += int(top_expert == task["task_id"])
            task_score_margin_sum += float(sorted_scores[0].item() - sorted_scores[-1].item())
            task_score_entropy_sum += float(
                -(
                    torch.softmax(sample_scores, dim=0)
                    * torch.log(torch.softmax(sample_scores, dim=0).clamp_min(1e-12))
                )
                .sum()
                .item()
            )

            for stage in ("early", "middle", "late"):
                weights = route_plan[stage].detach().cpu().float()
                stage_stats[stage]["weight_sum"] += weights.double()
                stage_stats[stage]["top1_counts"][int(torch.argmax(weights).item())] += 1
                stage_stats[stage]["expected_top1"] += int(
                    int(torch.argmax(weights).item()) == task["task_id"]
                )
                sorted_weights = torch.topk(weights, k=min(2, weights.numel())).values
                stage_stats[stage]["margin_sum"] += float(
                    sorted_weights[0].item() - sorted_weights[-1].item()
                )
                stage_stats[stage]["entropy_sum"] += float(
                    -(weights * torch.log(weights.clamp_min(1e-12))).sum().item()
                )

    sample_count = len(records)
    return {
        "task_id": task["task_id"],
        "task_name": task["task_name"],
        "sample_count": sample_count,
        "task_score": {
            "avg_score": [round(x, 6) for x in (task_score_sum / max(sample_count, 1)).tolist()],
            "top1_rate": [round(x, 6) for x in (task_score_top1_counts.double() / max(sample_count, 1)).tolist()],
            "expected_expert_top1_rate": task_score_expected_top1 / max(sample_count, 1),
            "margin_mean": task_score_margin_sum / max(sample_count, 1),
            "softmax_entropy_mean": task_score_entropy_sum / max(sample_count, 1),
        },
        "stages": finalize_stage_stats(stage_stats, sample_count),
    }


def summarize_report(task_reports, active_experts):
    summary = {
        "expected_expert_top1_rate": {
            "task_score": 0.0,
            "early": 0.0,
            "middle": 0.0,
            "late": 0.0,
        },
        "mean_entropy": {
            "task_score": 0.0,
            "early": 0.0,
            "middle": 0.0,
            "late": 0.0,
        },
        "mean_margin": {
            "task_score": 0.0,
            "early": 0.0,
            "middle": 0.0,
            "late": 0.0,
        },
        "avg_weight_by_stage": {
            stage: [0.0] * active_experts for stage in ("early", "middle", "late")
        },
    }
    task_count = max(len(task_reports), 1)
    for report in task_reports:
        summary["expected_expert_top1_rate"]["task_score"] += report["task_score"][
            "expected_expert_top1_rate"
        ]
        summary["mean_entropy"]["task_score"] += report["task_score"][
            "softmax_entropy_mean"
        ]
        summary["mean_margin"]["task_score"] += report["task_score"]["margin_mean"]
        for stage in ("early", "middle", "late"):
            summary["expected_expert_top1_rate"][stage] += report["stages"][stage][
                "expected_expert_top1_rate"
            ]
            summary["mean_entropy"][stage] += report["stages"][stage]["entropy_mean"]
            summary["mean_margin"][stage] += report["stages"][stage]["margin_mean"]
            summary["avg_weight_by_stage"][stage] = [
                summary["avg_weight_by_stage"][stage][idx]
                + report["stages"][stage]["avg_weight"][idx]
                for idx in range(active_experts)
            ]
    for group in ("expected_expert_top1_rate", "mean_entropy", "mean_margin"):
        for key in summary[group]:
            summary[group][key] /= task_count
    for stage in summary["avg_weight_by_stage"]:
        summary["avg_weight_by_stage"][stage] = [
            round(value / task_count, 6)
            for value in summary["avg_weight_by_stage"][stage]
        ]
    return summary


def main():
    args = parse_args()
    if not args.model_path:
        args.model_path = DEFAULT_MODEL_PATHS[args.benchmark]
    tasks = get_tasks(args.benchmark)
    if args.num_task <= 0:
        args.num_task = len(tasks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    disable_torch_init()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        Path(args.model_path).name,
        device="cuda",
        num_task=args.num_task,
        text_tower=args.text_tower,
    )
    model.eval()
    routing_override = maybe_load_routing_config(args.routing_config_json)
    if routing_override:
        model.configure_relation_routing(**routing_override)

    prototype_payload, prototype_cache = compute_task_prototypes(
        args,
        model,
        image_processor,
        output_dir,
    )
    apply_spectral_prototypes(model, prototype_payload, benchmark=args.benchmark)

    task_reports = []
    for task in tasks:
        task_reports.append(evaluate_task(model, image_processor, tokenizer, task, args))

    active_experts = max(1, min(int(model.expert_num), len(model.image_anchors)))
    report = {
        "date": str(date.today()),
        "benchmark": args.benchmark,
        "model_path": args.model_path,
        "model_base": args.model_base,
        "text_tower": args.text_tower,
        "routing_config_json": args.routing_config_json,
        "prototype_cache": str(prototype_cache),
        "prototype_limit": args.prototype_limit,
        "eval_limit": args.eval_limit,
        "active_experts": active_experts,
        "routing_config": dict(model.relation_routing_config),
        "prototype_counts": prototype_payload["counts"],
        "tasks": task_reports,
        "summary": summarize_report(task_reports, active_experts),
    }
    output_path = output_dir / f"{args.benchmark}_spectral_routing_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved prototype cache to: {prototype_cache}")
    print(f"Saved routing report to: {output_path}")
    for task_report in task_reports:
        late = task_report["stages"]["late"]
        print(
            f"{task_report['task_name']}: "
            f"late_expected={late['expected_expert_top1_rate']:.4f}, "
            f"late_margin={late['margin_mean']:.4f}, "
            f"late_entropy={late['entropy_mean']:.4f}"
        )
    print("Global summary:")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
