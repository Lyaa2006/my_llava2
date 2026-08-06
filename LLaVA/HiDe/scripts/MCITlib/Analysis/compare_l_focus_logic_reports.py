#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List


def load_report(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_tasks(report: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    return {
        int(task["task_id"]): task
        for task in report["logic_summary"]["tasks"]
    }


def group_delta(focus_off: Dict[str, object], focus_on: Dict[str, object], group_name: str) -> Dict[str, object]:
    off_group = focus_off["logic_summary"]["groups"][group_name]
    on_group = focus_on["logic_summary"]["groups"][group_name]
    return {
        "group_name": group_name,
        "support_rate_delta": float(on_group["harmful_template_drift_support_rate"]) - float(off_group["harmful_template_drift_support_rate"]),
        "alignment_drop_delta": float(off_group["alignment_drop_mean"]) - float(on_group["alignment_drop_mean"]),
        "retrieval_drop_delta": float(off_group["retrieval_drop_mean"]) - float(on_group["retrieval_drop_mean"]),
        "alignment_corr_delta": float(off_group["template_alignment_drop_corr_mean"]) - float(on_group["template_alignment_drop_corr_mean"]),
        "retrieval_corr_delta": float(off_group["template_retrieval_drop_corr_mean"]) - float(on_group["template_retrieval_drop_corr_mean"]),
        "focus_off": off_group,
        "focus_on": on_group,
    }


def task_deltas(focus_off: Dict[str, object], focus_on: Dict[str, object]) -> List[Dict[str, object]]:
    off_tasks = index_tasks(focus_off)
    on_tasks = index_tasks(focus_on)
    shared_ids = sorted(set(off_tasks) & set(on_tasks))
    deltas: List[Dict[str, object]] = []
    for task_id in shared_ids:
        off_task = off_tasks[task_id]
        on_task = on_tasks[task_id]
        deltas.append(
            {
                "task_id": task_id,
                "label": on_task["label"],
                "alignment_drop_delta": float(off_task["alignment_drop_mean"]) - float(on_task["alignment_drop_mean"]),
                "retrieval_drop_delta": float(off_task["retrieval_drop_mean"]) - float(on_task["retrieval_drop_mean"]),
                "alignment_corr_delta": float(off_task["template_alignment_drop_corr"]) - float(on_task["template_alignment_drop_corr"]),
                "retrieval_corr_delta": float(off_task["template_retrieval_drop_corr"]) - float(on_task["template_retrieval_drop_corr"]),
                "focus_off_conclusion": off_task["conclusion"],
                "focus_on_conclusion": on_task["conclusion"],
            }
        )
    return deltas


def write_markdown(path: str, report: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# L_focus Ablation Comparison")
    lines.append("")
    lines.append("Positive deltas mean the `with L_focus` run looks better under the same logic-oriented metric.")
    lines.append("")
    lines.append("## Group Deltas")
    lines.append("")
    for group_name in ("all", "early", "late"):
        group = report["group_deltas"][group_name]
        lines.append(f"### {group_name.capitalize()}")
        lines.append("")
        lines.append(f"- support-rate delta: {group['support_rate_delta']:.4f}")
        lines.append(f"- alignment-drop improvement: {group['alignment_drop_delta']:.4f}")
        lines.append(f"- retrieval-drop improvement: {group['retrieval_drop_delta']:.4f}")
        lines.append(f"- alignment-corr reduction: {group['alignment_corr_delta']:.4f}")
        lines.append(f"- retrieval-corr reduction: {group['retrieval_corr_delta']:.4f}")
        lines.append("")
    lines.append("## Per-Task Deltas")
    lines.append("")
    for task in report["task_deltas"]:
        lines.append(f"### {task['label']}")
        lines.append("")
        lines.append(f"- alignment-drop improvement: {task['alignment_drop_delta']:.4f}")
        lines.append(f"- retrieval-drop improvement: {task['retrieval_drop_delta']:.4f}")
        lines.append(f"- alignment-corr reduction: {task['alignment_corr_delta']:.4f}")
        lines.append(f"- retrieval-corr reduction: {task['retrieval_corr_delta']:.4f}")
        lines.append(f"- focus-off conclusion: `{task['focus_off_conclusion']}`")
        lines.append(f"- focus-on conclusion: `{task['focus_on_conclusion']}`")
        lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare focus-off and focus-on logic reports.")
    parser.add_argument("--focus-off-report", required=True)
    parser.add_argument("--focus-on-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    focus_off = load_report(args.focus_off_report)
    focus_on = load_report(args.focus_on_report)

    comparison = {
        "focus_off_report": args.focus_off_report,
        "focus_on_report": args.focus_on_report,
        "focus_off_overall_conclusion": focus_off["logic_summary"]["overall_conclusion"],
        "focus_on_overall_conclusion": focus_on["logic_summary"]["overall_conclusion"],
        "group_deltas": {
            group_name: group_delta(focus_off, focus_on, group_name)
            for group_name in ("all", "early", "late")
        },
        "task_deltas": task_deltas(focus_off, focus_on),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "l_focus_ablation_comparison.json")
    md_path = os.path.join(args.output_dir, "l_focus_ablation_comparison.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    write_markdown(md_path, comparison)

    print(f"Saved comparison JSON to {json_path}")
    print(f"Saved comparison markdown to {md_path}")


if __name__ == "__main__":
    main()
