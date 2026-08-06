#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(__file__)
HI_DE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
LOCAL_PEFT_ROOT = os.path.join(HI_DE_ROOT, "HiDe")

os.environ.setdefault("TRITON_CACHE_DIR", os.path.join("/tmp", "triton"))

# Prefer the bundled HiDe/peft package over any environment-installed `peft`.
for path in (SCRIPT_DIR, LOCAL_PEFT_ROOT, HI_DE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import analyze_template_alignment as template_alignment


DEFAULT_OUTPUT_DIR = os.path.join(template_alignment.MCITLIB_ROOT, "docs", "experiment2_l_focus_logic")


def safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def positive_fraction(values: Sequence[float], threshold: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(np.mean(arr > threshold))


def task_logic_summary(
    task_report: Dict[str, object],
    align_drop_threshold: float,
    retrieval_drop_threshold: float,
    margin_drop_threshold: float,
    corr_threshold: float,
) -> Dict[str, object]:
    alignment_drop = float(task_report["alignment_drop_mean"])
    retrieval_drop = float(task_report["retrieval_drop_mean"])
    margin_drop = float(task_report["margin_drop_mean"])
    align_corr = float(task_report["template_alignment_drop_corr"])
    retrieval_corr = float(task_report["template_retrieval_drop_corr"])
    margin_corr = float(task_report["template_margin_drop_corr"])
    template_drift = float(task_report["template_drift_mean"])
    content_drift = float(task_report["content_drift_mean"])

    signals = {
        "alignment_degrades": alignment_drop > align_drop_threshold,
        "retrieval_degrades": retrieval_drop > retrieval_drop_threshold,
        "margin_degrades": margin_drop > margin_drop_threshold,
        "alignment_couples_with_template_drift": align_corr > corr_threshold,
        "retrieval_couples_with_template_drift": retrieval_corr > corr_threshold,
        "margin_couples_with_template_drift": margin_corr > corr_threshold,
        "content_drift_exceeds_template_drift": content_drift > template_drift,
    }

    harmful_channels: List[str] = []
    if signals["alignment_degrades"] and signals["alignment_couples_with_template_drift"]:
        harmful_channels.append("alignment")
    if signals["retrieval_degrades"] and signals["retrieval_couples_with_template_drift"]:
        harmful_channels.append("retrieval")
    if signals["margin_degrades"] and signals["margin_couples_with_template_drift"]:
        harmful_channels.append("margin")

    support_score = sum(int(value) for value in signals.values())
    if len(harmful_channels) >= 2:
        conclusion = "strong_support"
    elif len(harmful_channels) == 1:
        conclusion = "partial_support"
    elif support_score >= 3:
        conclusion = "mixed_signal"
    else:
        conclusion = "weak_or_null"

    return {
        "task_id": int(task_report["task_id"]),
        "task_name": task_report["task_name"],
        "label": task_report["label"],
        "alignment_drop_mean": alignment_drop,
        "retrieval_drop_mean": retrieval_drop,
        "margin_drop_mean": margin_drop,
        "template_drift_mean": template_drift,
        "content_drift_mean": content_drift,
        "template_alignment_drop_corr": align_corr,
        "template_retrieval_drop_corr": retrieval_corr,
        "template_margin_drop_corr": margin_corr,
        "harmful_channels": harmful_channels,
        "support_score": support_score,
        "conclusion": conclusion,
        "signals": signals,
    }


def group_logic_summary(
    group_name: str,
    task_summaries: Sequence[Dict[str, object]],
    align_drop_threshold: float,
    retrieval_drop_threshold: float,
    corr_threshold: float,
) -> Dict[str, object]:
    task_ids = [int(task["task_id"]) for task in task_summaries]
    alignment_drops = [float(task["alignment_drop_mean"]) for task in task_summaries]
    retrieval_drops = [float(task["retrieval_drop_mean"]) for task in task_summaries]
    margin_drops = [float(task["margin_drop_mean"]) for task in task_summaries]
    align_corrs = [float(task["template_alignment_drop_corr"]) for task in task_summaries]
    retrieval_corrs = [float(task["template_retrieval_drop_corr"]) for task in task_summaries]
    margin_corrs = [float(task["template_margin_drop_corr"]) for task in task_summaries]
    support_scores = [float(task["support_score"]) for task in task_summaries]
    harmful_support = [1.0 if task["harmful_channels"] else 0.0 for task in task_summaries]

    return {
        "group_name": group_name,
        "task_ids": task_ids,
        "num_tasks": len(task_ids),
        "alignment_drop_mean": safe_mean(alignment_drops),
        "retrieval_drop_mean": safe_mean(retrieval_drops),
        "margin_drop_mean": safe_mean(margin_drops),
        "template_alignment_drop_corr_mean": safe_mean(align_corrs),
        "template_retrieval_drop_corr_mean": safe_mean(retrieval_corrs),
        "template_margin_drop_corr_mean": safe_mean(margin_corrs),
        "alignment_drop_positive_fraction": positive_fraction(alignment_drops, align_drop_threshold),
        "retrieval_drop_positive_fraction": positive_fraction(retrieval_drops, retrieval_drop_threshold),
        "alignment_corr_positive_fraction": positive_fraction(align_corrs, corr_threshold),
        "retrieval_corr_positive_fraction": positive_fraction(retrieval_corrs, corr_threshold),
        "support_score_mean": safe_mean(support_scores),
        "harmful_template_drift_support_rate": safe_mean(harmful_support),
    }


def build_logic_report(
    task_reports: Sequence[Dict[str, object]],
    late_task_start: int,
    align_drop_threshold: float,
    retrieval_drop_threshold: float,
    margin_drop_threshold: float,
    corr_threshold: float,
) -> Dict[str, object]:
    task_summaries = [
        task_logic_summary(
            task_report,
            align_drop_threshold=align_drop_threshold,
            retrieval_drop_threshold=retrieval_drop_threshold,
            margin_drop_threshold=margin_drop_threshold,
            corr_threshold=corr_threshold,
        )
        for task_report in task_reports
    ]

    early_tasks = [task for task in task_summaries if int(task["task_id"]) < late_task_start]
    late_tasks = [task for task in task_summaries if int(task["task_id"]) >= late_task_start]

    groups = {
        "all": group_logic_summary(
            "all",
            task_summaries,
            align_drop_threshold=align_drop_threshold,
            retrieval_drop_threshold=retrieval_drop_threshold,
            corr_threshold=corr_threshold,
        ),
        "early": group_logic_summary(
            "early",
            early_tasks,
            align_drop_threshold=align_drop_threshold,
            retrieval_drop_threshold=retrieval_drop_threshold,
            corr_threshold=corr_threshold,
        ),
        "late": group_logic_summary(
            "late",
            late_tasks,
            align_drop_threshold=align_drop_threshold,
            retrieval_drop_threshold=retrieval_drop_threshold,
            corr_threshold=corr_threshold,
        ),
    }

    late_support = groups["late"]["harmful_template_drift_support_rate"]
    late_align_corr = groups["late"]["alignment_corr_positive_fraction"]
    late_retrieval_corr = groups["late"]["retrieval_corr_positive_fraction"]

    if late_support >= 0.67 and (late_align_corr >= 0.67 or late_retrieval_corr >= 0.67):
        overall_conclusion = "moderately_consistent_support"
    elif late_support >= 0.34 or groups["all"]["harmful_template_drift_support_rate"] >= 0.5:
        overall_conclusion = "mixed_but_meaningful_support"
    else:
        overall_conclusion = "exploratory_or_weak_support"

    return {
        "principles": [
            "Do not require every task to show degradation; early tasks may sharpen representations.",
            "Treat harmful template drift as a coupling pattern: template drift should align with semantic instability.",
            "Use late-task behavior as stronger evidence than early-task behavior in continual learning.",
            "Do not overclaim causality from Phase A alone; ablation is required for causal support.",
        ],
        "thresholds": {
            "late_task_start": late_task_start,
            "alignment_drop_threshold": align_drop_threshold,
            "retrieval_drop_threshold": retrieval_drop_threshold,
            "margin_drop_threshold": margin_drop_threshold,
            "corr_threshold": corr_threshold,
        },
        "tasks": task_summaries,
        "groups": groups,
        "overall_conclusion": overall_conclusion,
    }


def write_markdown_report(output_path: str, report: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# L_focus Logic-Oriented Experiment Report")
    lines.append("")
    lines.append("## What This Script Tries To Test")
    lines.append("")
    lines.append("This report does not assume that every task must show uniform degradation after training.")
    lines.append("Instead, it checks whether larger template-token drift tends to coincide with weaker cross-template semantic stability,")
    lines.append("which is the specific failure mode that `L_focus` is meant to suppress.")
    lines.append("")
    lines.append("## Analysis Principles")
    lines.append("")
    for principle in report["principles"]:
        lines.append(f"- {principle}")
    lines.append("")
    lines.append("## Group Summary")
    lines.append("")
    for group_name in ("all", "early", "late"):
        group = report["groups"][group_name]
        lines.append(f"### {group_name.capitalize()} Tasks")
        lines.append("")
        lines.append(f"- task ids: {group['task_ids']}")
        lines.append(f"- harmful template-drift support rate: {group['harmful_template_drift_support_rate']:.3f}")
        lines.append(f"- mean alignment drop: {group['alignment_drop_mean']:.4f}")
        lines.append(f"- mean retrieval drop: {group['retrieval_drop_mean']:.4f}")
        lines.append(f"- mean template/alignment corr: {group['template_alignment_drop_corr_mean']:.4f}")
        lines.append(f"- mean template/retrieval corr: {group['template_retrieval_drop_corr_mean']:.4f}")
        lines.append("")
    lines.append("## Per-Task Logic")
    lines.append("")
    for task in report["tasks"]:
        lines.append(f"### {task['label']}")
        lines.append("")
        lines.append(f"- conclusion: `{task['conclusion']}`")
        lines.append(f"- harmful channels: {task['harmful_channels']}")
        lines.append(f"- alignment drop mean: {task['alignment_drop_mean']:.4f}")
        lines.append(f"- retrieval drop mean: {task['retrieval_drop_mean']:.4f}")
        lines.append(f"- margin drop mean: {task['margin_drop_mean']:.4f}")
        lines.append(f"- template/alignment corr: {task['template_alignment_drop_corr']:.4f}")
        lines.append(f"- template/retrieval corr: {task['template_retrieval_drop_corr']:.4f}")
        lines.append(f"- template drift mean: {task['template_drift_mean']:.4f}")
        lines.append(f"- content drift mean: {task['content_drift_mean']:.4f}")
        lines.append("")
    lines.append("## Overall Conclusion")
    lines.append("")
    lines.append(f"- overall conclusion: `{report['overall_conclusion']}`")
    lines.append("- This Phase A report is meant to motivate `L_focus`, not to prove its causal necessity by itself.")
    lines.append("")

    template_alignment.ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the logic-oriented Phase A analysis for L_focus."
    )
    parser.add_argument("--base-model-path", type=str, default=template_alignment.DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--vision-tower-path", type=str, default=template_alignment.DEFAULT_VISION_TOWER_PATH)
    parser.add_argument("--ucit-root", type=str, default=template_alignment.DEFAULT_UCIT_ROOT)
    parser.add_argument("--checkpoint-root", type=str, default=template_alignment.DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--samples-per-task", type=int, default=64)
    parser.add_argument("--description-hidden-layer", type=int, default=-2)
    parser.add_argument("--description-max-tokens", type=int, default=56)
    parser.add_argument("--conv-mode", type=str, default=template_alignment.DEFAULT_CONV_MODE)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--late-task-start", type=int, default=3)
    parser.add_argument("--alignment-drop-threshold", type=float, default=0.0)
    parser.add_argument("--retrieval-drop-threshold", type=float, default=0.0)
    parser.add_argument("--margin-drop-threshold", type=float, default=0.0)
    parser.add_argument("--corr-threshold", type=float, default=0.05)
    args = parser.parse_args()

    template_alignment.ensure_dir(args.output_dir)
    template_alignment.set_seed(args.seed)
    template_alignment.disable_torch_init()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    runtime_device = args.device
    if runtime_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable; falling back to CPU.")
        runtime_device = "cpu"
    model_dtype = torch.float16 if runtime_device.startswith("cuda") else torch.float32

    task_reports: List[Dict[str, object]] = []
    for task_id in args.task_ids:
        task_name, rel_path = template_alignment.TASK_DATA_FILES[task_id]
        data_path = os.path.join(args.ucit_root, rel_path)
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Missing data file: {data_path}")

        samples = template_alignment.load_task_samples(args.ucit_root, task_id, args.samples_per_task, args.seed)
        prev_path, cur_path = template_alignment.task_checkpoint_paths(
            args.checkpoint_root,
            task_id,
            args.base_model_path,
        )
        if not os.path.isdir(cur_path) and not os.path.isfile(cur_path):
            raise FileNotFoundError(f"Missing current checkpoint: {cur_path}")
        if task_id > 1 and not os.path.isdir(prev_path):
            raise FileNotFoundError(f"Missing previous checkpoint: {prev_path}")

        print(f"[Task {task_id}] loading previous model: {prev_path}")
        prev_base = None if task_id == 1 else args.base_model_path
        prev_tokenizer, prev_model, prev_image_processor, prev_mm_use_im_start_end = template_alignment.load_model_bundle(
            prev_path,
            prev_base,
            args.vision_tower_path,
            runtime_device,
            model_dtype,
            task_id - 1 if task_id > 1 else 1,
        )
        before_sets = template_alignment.collect_prompt_state_sets(
            prev_model,
            prev_tokenizer,
            prev_image_processor,
            samples,
            template_alignment.DEFAULT_PROMPT_VARIANTS,
            prev_mm_use_im_start_end,
            args.conv_mode,
            args.description_hidden_layer,
            args.description_max_tokens,
            runtime_device,
            model_dtype,
        )
        del prev_model
        gc.collect()
        if runtime_device.startswith("cuda"):
            torch.cuda.empty_cache()

        print(f"[Task {task_id}] loading current model: {cur_path}")
        cur_tokenizer, cur_model, cur_image_processor, cur_mm_use_im_start_end = template_alignment.load_model_bundle(
            cur_path,
            args.base_model_path,
            args.vision_tower_path,
            runtime_device,
            model_dtype,
            task_id,
        )
        after_sets = template_alignment.collect_prompt_state_sets(
            cur_model,
            cur_tokenizer,
            cur_image_processor,
            samples,
            template_alignment.DEFAULT_PROMPT_VARIANTS,
            cur_mm_use_im_start_end,
            args.conv_mode,
            args.description_hidden_layer,
            args.description_max_tokens,
            runtime_device,
            model_dtype,
        )
        del cur_model
        gc.collect()
        if runtime_device.startswith("cuda"):
            torch.cuda.empty_cache()

        task_report = template_alignment.summarize_task_metrics(
            task_id,
            task_name,
            samples,
            before_sets,
            after_sets,
        )
        task_reports.append(task_report)
        print(
            f"[Task {task_id}] align_drop={task_report['alignment_drop_mean']:.4f} "
            f"retrieval_drop={task_report['retrieval_drop_mean']:.4f} "
            f"template_corr={task_report['template_alignment_drop_corr']:.4f}"
        )

    logic_summary = build_logic_report(
        task_reports,
        late_task_start=args.late_task_start,
        align_drop_threshold=args.alignment_drop_threshold,
        retrieval_drop_threshold=args.retrieval_drop_threshold,
        margin_drop_threshold=args.margin_drop_threshold,
        corr_threshold=args.corr_threshold,
    )

    report = {
        "config": {
            "base_model_path": args.base_model_path,
            "vision_tower_path": args.vision_tower_path,
            "ucit_root": args.ucit_root,
            "checkpoint_root": args.checkpoint_root,
            "samples_per_task": args.samples_per_task,
            "description_hidden_layer": args.description_hidden_layer,
            "description_max_tokens": args.description_max_tokens,
            "prompt_variants": template_alignment.DEFAULT_PROMPT_VARIANTS,
            "seed": args.seed,
        },
        "task_reports": task_reports,
        "logic_summary": logic_summary,
    }

    report_path = os.path.join(args.output_dir, "l_focus_logic_report.json")
    figure_path = os.path.join(args.output_dir, "l_focus_logic_curves.png")
    markdown_path = os.path.join(args.output_dir, "l_focus_logic_summary.md")

    template_alignment.save_report(report_path, report)
    template_alignment.plot_results(figure_path, task_reports)
    write_markdown_report(markdown_path, logic_summary)

    print(f"Saved report to {report_path}")
    print(f"Saved figure to {figure_path}")
    print(f"Saved markdown summary to {markdown_path}")


if __name__ == "__main__":
    main()
