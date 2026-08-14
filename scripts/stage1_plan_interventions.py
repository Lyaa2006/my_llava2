#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from stage1_protocol_lib import ensure_dir, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage 1 intervention plans from boundary windows.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer-count", type=int, default=32)
    parser.add_argument("--b1-low", type=int, required=True)
    parser.add_argument("--b1-high", type=int, required=True)
    parser.add_argument("--b2-low", type=int, required=True)
    parser.add_argument("--b2-high", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default=str(Path("configs/probe_configs/stage1_protocol/interventions")))
    return parser


def layer_range(start: int, end: int) -> List[int]:
    if end < start:
        return []
    return list(range(start, end + 1))


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    early = layer_range(1, args.b1_low)
    middle = layer_range(args.b1_high + 1, args.b2_low - 1)
    late = layer_range(args.b2_high, args.layer_count)
    early_core = layer_range(1, max(1, args.b1_low - 1))
    middle_core = layer_range(args.b1_high + 1, max(args.b1_high + 1, args.b2_low - 1))
    late_core = layer_range(min(args.layer_count, args.b2_high + 1), args.layer_count)

    plans = []
    for name, frozen in [
        ("freeze_early", early),
        ("freeze_middle", middle),
        ("freeze_late", late),
        ("train_only_early", early),
        ("train_only_middle", middle),
        ("train_only_late", late),
    ]:
        plans.append(
            {
                "name": name,
                "frozen_layers": frozen if name.startswith("freeze_") else [],
                "trainable_layers": frozen if name.startswith("train_only_") else [],
            }
        )

    payload = {
        "model": args.model,
        "layer_count": args.layer_count,
        "boundary_windows": {
            "b1": [args.b1_low, args.b1_high],
            "b2": [args.b2_low, args.b2_high],
        },
        "regions": {
            "early": early,
            "middle": middle,
            "late": late,
            "early_core": early_core,
            "middle_core": middle_core,
            "late_core": late_core,
        },
        "plans": plans,
        "notes": "This is a manifest only. The trainer must consume frozen_layers/trainable_layers explicitly.",
    }
    write_json(output_dir / f"{args.model}_intervention_plan.json", payload)
    print(output_dir / f"{args.model}_intervention_plan.json")


if __name__ == "__main__":
    main()
