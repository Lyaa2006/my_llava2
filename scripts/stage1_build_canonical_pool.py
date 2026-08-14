#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from stage1_protocol_lib import balance_by_keys, read_jsonl, split_summary, write_json, write_jsonl


STYLE_VARIANTS = {
    "label_only": "Answer with the option id only.",
    "short_phrase": "Answer using a short phrase.",
    "full_sentence": "Answer in one complete sentence.",
}


OBJECTIVE_VARIANTS = {
    "objective_caption": [
        ("caption", "Describe the image in one sentence."),
        ("general_vqa", "What is happening in the image?"),
        ("recognition", "What object is shown in the image?"),
    ],
    "objective_recognition": [
        ("recognition", "What object is shown in the image?"),
        ("general_vqa", "What is the main thing visible in the image?"),
        ("yes_no", "Is a clearly visible object present in the image?"),
    ],
    "objective_qa": [
        ("general_vqa", "Answer the question using the image."),
        ("yes_no", "Answer yes or no based on the image."),
        ("comparison_reasoning", "Use the image to compare the options and answer carefully."),
    ],
    "objective_reasoning": [
        ("comparison_reasoning", "Reason from the image before answering."),
        ("counting", "How many relevant items are there in the image?"),
        ("mcq_reasoning", "Reason about the options and choose the best one."),
    ],
    "objective_mcq": [
        ("mcq_visual_qa", "Choose the best option based on the image."),
        ("mcq_reasoning", "Reason about the options and choose the best one."),
        ("general_vqa", "Answer based on the image."),
    ],
}


def make_style_variant(record: Dict[str, object], variant_name: str, suffix: str) -> Dict[str, object]:
    clone = dict(record)
    clone["probe_family"] = "style"
    clone["probe_variant"] = variant_name
    clone["objective_label"] = f"style_{variant_name}"
    clone["sample_id"] = f"{record['sample_id']}::style::{variant_name}"
    clone["prompt_text"] = f"{record['prompt_text']}\n{suffix}".strip()
    return clone


def build_reasoning_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for record in records:
        cluster = str(record.get("prompt_cluster"))
        if cluster in {"comparison_reasoning", "counting", "math_reasoning", "mcq_reasoning"}:
            variants = [
                ("comparison_reasoning", "Reason from the image before answering."),
                ("counting", "How many relevant items are there in the image?"),
                ("mcq_reasoning", "Reason about the options and choose the best one."),
            ]
            for variant_cluster, suffix in variants:
                clone = dict(record)
                clone["probe_family"] = "reasoning"
                clone["probe_variant"] = "hard"
                clone["objective_label"] = "reasoning_hard"
                clone["prompt_cluster"] = variant_cluster
                clone["prompt_text"] = f"{record['prompt_text']}\n{suffix}".strip()
                clone["template_id"] = f"{variant_cluster}_{clone['template_id']}"
                output.append(clone)
        elif cluster in {"recognition", "caption", "ocr_reading", "gui_action"}:
            variants = [
                ("recognition", "What object is shown in the image?"),
                ("caption", "Describe the image in one sentence."),
                ("ocr_reading", "Read any visible text in the image."),
            ]
            for variant_cluster, suffix in variants:
                clone = dict(record)
                clone["probe_family"] = "reasoning"
                clone["probe_variant"] = "easy"
                clone["objective_label"] = "reasoning_easy"
                clone["prompt_cluster"] = variant_cluster
                clone["prompt_text"] = f"{record['prompt_text']}\n{suffix}".strip()
                clone["template_id"] = f"{variant_cluster}_{clone['template_id']}"
                output.append(clone)
    return output


def build_objective_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for record in records:
        label = str(record.get("objective_label"))
        if label.startswith("objective_"):
            for variant_cluster, suffix in OBJECTIVE_VARIANTS.get(label, [(str(record.get("prompt_cluster")), "")]):
                clone = dict(record)
                clone["probe_family"] = "objective"
                clone["probe_variant"] = label.replace("objective_", "")
                clone["prompt_cluster"] = variant_cluster
                if suffix:
                    clone["prompt_text"] = f"{record['prompt_text']}\n{suffix}".strip()
                clone["template_id"] = f"{variant_cluster}_{clone['template_id']}"
                output.append(clone)
    return output


def build_style_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for record in records:
        base = dict(record)
        base["probe_family"] = "style"
        base["probe_variant"] = "base"
        output.append(base)
        for variant_name, suffix in STYLE_VARIANTS.items():
            output.append(make_style_variant(record, variant_name, suffix))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Stage 1 canonical probe pool.")
    parser.add_argument("--source-pool", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(Path("configs/probe_configs/stage1_protocol")))
    parser.add_argument("--max-per-objective", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = read_jsonl(Path(args.source_pool))

    canonical_records: List[Dict[str, object]] = []
    canonical_records.extend(build_objective_records(pool))
    canonical_records.extend(build_reasoning_records(pool))
    canonical_records.extend(build_style_records(pool))

    canonical_records = balance_by_keys(canonical_records, ["objective_label", "probe_family", "prompt_cluster"], args.max_per_objective, args.seed)
    write_jsonl(output_dir / "canonical_probe.jsonl", canonical_records)
    write_json(
        output_dir / "canonical_probe.summary.json",
        {
            "num_records": len(canonical_records),
            "summary": split_summary(canonical_records),
            "source_pool": args.source_pool,
            "max_per_objective": args.max_per_objective,
            "seed": args.seed,
        },
    )
    print(output_dir / "canonical_probe.jsonl")


if __name__ == "__main__":
    main()
