#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Sequence

from stage1_protocol_lib import ensure_dir


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_metric(values: Sequence[float]) -> Dict[str, float]:
    ordered = [float(v) for v in values]
    if not ordered:
        return {}
    return {
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "std": float(statistics.pstdev(ordered)) if len(ordered) > 1 else 0.0,
        "min": float(min(ordered)),
        "max": float(max(ordered)),
    }


def scan_json_reports(root: Path) -> List[Dict[str, object]]:
    reports: List[Dict[str, object]] = []
    boundary_summaries = sorted(root.glob("**/stage1_boundary_summary.json"))
    if boundary_summaries:
        for path in boundary_summaries:
            try:
                payload = load_json(path)
            except Exception:
                continue
            for row in payload.get("reports", []):
                reports.append(
                    {
                        "_path": row.get("report_path", ""),
                        "model": row.get("model"),
                        "dataset": row.get("dataset"),
                        "boundary_windows": {
                            "b1": [row.get("b1"), row.get("b1")],
                            "b2": [row.get("b2"), row.get("b2")],
                        },
                    }
                )
        return reports

    for path in sorted(root.glob("**/*.json")):
        if path.name.endswith(".summary.json"):
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        if isinstance(payload, dict) and ("boundary_windows" in payload or "recommended_boundaries" in payload):
            payload["_path"] = str(path)
            reports.append(payload)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage 1 protocol outputs.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir else input_root
    ensure_dir(output_dir)
    reports = scan_json_reports(input_root)

    rows: List[Dict[str, object]] = []
    for report in reports:
        windows = report.get("boundary_windows") or report.get("recommended_boundaries") or {}
        b1 = windows.get("b1") or [windows.get("b1_low"), windows.get("b1_high")]
        b2 = windows.get("b2") or [windows.get("b2_low"), windows.get("b2_high")]
        rows.append(
            {
                "path": report["_path"],
                "model": report.get("model"),
                "dataset": report.get("dataset"),
                "b1_low": b1[0] if isinstance(b1, list) and len(b1) > 0 else None,
                "b1_high": b1[1] if isinstance(b1, list) and len(b1) > 1 else None,
                "b2_low": b2[0] if isinstance(b2, list) and len(b2) > 0 else None,
                "b2_high": b2[1] if isinstance(b2, list) and len(b2) > 1 else None,
            }
        )

    with open(output_dir / "stage1_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_reports": len(reports),
                "reports": rows,
                "aggregate": {
                    "b1_low": summarize_metric([row["b1_low"] for row in rows if row["b1_low"] is not None]),
                    "b1_high": summarize_metric([row["b1_high"] for row in rows if row["b1_high"] is not None]),
                    "b2_low": summarize_metric([row["b2_low"] for row in rows if row["b2_low"] is not None]),
                    "b2_high": summarize_metric([row["b2_high"] for row in rows if row["b2_high"] is not None]),
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(output_dir / "stage1_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["path"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(output_dir / "stage1_summary.json")


if __name__ == "__main__":
    main()
