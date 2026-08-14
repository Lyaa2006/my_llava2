#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from stage1_protocol_lib import (
    BENCHMARK_SPECS,
    coarse_objective_label,
    detect_answer_format_class,
    detect_prompt_cluster,
    image_cluster_id,
    read_task_config,
    resolve_image_path,
    sample_records,
    split_summary,
    template_id,
    write_json,
    write_jsonl,
)


def load_dataset_records(benchmark: str, tasks: List[str], max_samples_per_task: Optional[int], seed: int) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for task in tasks:
        cfg = read_task_config(benchmark, task)
        raw_samples = json.load(open(cfg["test_path"], "r", encoding="utf-8"))
        sampled = sample_records(raw_samples, max_samples_per_task, seed + len(records))
        for item in sampled:
            prompt_text = str(item.get("text", "")).strip()
            answer_text = "" if item.get("answer") is None else str(item.get("answer", "")).strip()
            prompt_cluster = detect_prompt_cluster(prompt_text)
            answer_format = detect_answer_format_class(prompt_text, answer_text)
            records.append(
                {
                    "sample_id": f"{benchmark}::{task}::{item.get('question_id', len(records))}",
                    "source_dataset": benchmark,
                    "dataset_group": benchmark,
                    "task": task,
                    "question_id": str(item.get("question_id", "")),
                    "image_path": resolve_image_path(str(item["image"]), cfg["image_root"]),
                    "prompt_text": prompt_text,
                    "answer_text": answer_text,
                    "answer_format": answer_format,
                    "prompt_cluster": prompt_cluster,
                    "objective_label": coarse_objective_label(prompt_cluster, answer_format),
                    "probe_family": "native",
                    "probe_variant": "base",
                    "paired_group_id": f"{benchmark}::{task}::{item.get('question_id', len(records))}",
                    "image_group_id": image_cluster_id(resolve_image_path(str(item["image"]), cfg["image_root"])),
                    "template_id": template_id(prompt_text, prompt_cluster),
                    "nuisance_tags": {
                        "prompt_cluster": prompt_cluster,
                        "answer_type": answer_format,
                        "image_cluster": image_cluster_id(resolve_image_path(str(item["image"]), cfg["image_root"])),
                    },
                }
            )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Stage 1 native probe pool.")
    parser.add_argument("--benchmarks", nargs="+", default=["ucit", "acl", "dcl"])
    parser.add_argument("--max-samples-per-task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default=str(Path("configs/probe_configs/stage1_protocol")))
    parser.add_argument("--output-jsonl", type=str, default=None)
    parser.add_argument("--output-summary", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict[str, object]] = []
    for benchmark in args.benchmarks:
        tasks = BENCHMARK_SPECS[benchmark]["tasks"]
        all_records.extend(load_dataset_records(benchmark, tasks, args.max_samples_per_task, args.seed))

    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else output_dir / "native_probe_pool.jsonl"
    output_summary = Path(args.output_summary) if args.output_summary else output_dir / "native_probe_pool.summary.json"
    write_jsonl(output_jsonl, all_records)
    write_json(
        output_summary,
        {
            "num_records": len(all_records),
            "summary": split_summary(all_records),
            "benchmarks": list(args.benchmarks),
            "max_samples_per_task": args.max_samples_per_task,
            "seed": args.seed,
        },
    )
    print(output_jsonl)


if __name__ == "__main__":
    main()
