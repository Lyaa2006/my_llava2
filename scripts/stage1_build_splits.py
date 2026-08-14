#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from stage1_protocol_lib import ensure_dir, make_manifest_split, read_jsonl, split_summary, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage 1 leave-one splits.")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(Path("configs/probe_configs/stage1_protocol/splits")))
    parser.add_argument("--min-test-size", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    records = read_jsonl(Path(args.input_jsonl))
    summary = split_summary(records)

    template_ids = sorted({str(r.get("template_id")) for r in records if r.get("template_id")})
    dataset_groups = sorted({str(r.get("dataset_group")) for r in records if r.get("dataset_group")})

    split_specs: List[Dict[str, object]] = []
    for template_id in template_ids:
        split = {
            "split_name": f"leave_one_template::{template_id}",
            "split_type": "leave_one_template_out",
            "held_out_key": "template_id",
            "held_out_value": template_id,
            **make_manifest_split(records, "template_id", template_id),
        }
        if len(split["test_ids"]) >= args.min_test_size:
            split_specs.append(split)
    for dataset_group in dataset_groups:
        split = {
            "split_name": f"leave_one_dataset::{dataset_group}",
            "split_type": "leave_one_dataset_out",
            "held_out_key": "dataset_group",
            "held_out_value": dataset_group,
            **make_manifest_split(records, "dataset_group", dataset_group),
        }
        if len(split["test_ids"]) >= args.min_test_size:
            split_specs.append(split)
    for dataset_group in dataset_groups:
        for template_id in template_ids:
            test_ids = [
                str(r["sample_id"])
                for r in records
                if str(r.get("dataset_group")) == dataset_group and str(r.get("template_id")) == template_id
            ]
            if len(test_ids) < args.min_test_size:
                continue
            split_specs.append(
                {
                    "split_name": f"leave_one_dataset_template::{dataset_group}::{template_id}",
                    "split_type": "leave_one_dataset_and_template_out",
                    "held_out_key": "dataset_group/template_id",
                    "held_out_value": f"{dataset_group}::{template_id}",
                    "train_ids": [
                        str(r["sample_id"])
                        for r in records
                        if not (str(r.get("dataset_group")) == dataset_group and str(r.get("template_id")) == template_id)
                    ],
                    "test_ids": test_ids,
                }
            )

    write_json(
        output_dir / "split_manifest.json",
        {
            "input_jsonl": args.input_jsonl,
            "summary": summary,
            "splits": split_specs,
        },
    )
    print(output_dir / "split_manifest.json")


if __name__ == "__main__":
    main()
