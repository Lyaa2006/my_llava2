import argparse
import csv
import json
import os
import re


DATASET_ORDER = ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA"]


def parse_result_value(result_path):
    metric_name = None
    metric_value = None
    with open(result_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = [part.strip() for part in line.split(":", 1)]
            if key == "Accuracy":
                metric_name = "Accuracy"
                metric_value = float(value.rstrip("%"))
                break
            if key == "Average":
                metric_name = "Average"
                metric_value = float(value.rstrip("%"))
                break
    if metric_name is None or metric_value is None:
        raise ValueError(f"Could not parse primary metric from {result_path}")
    return metric_name, metric_value


def collect_native(native_root):
    rows = {}
    for dataset in DATASET_ORDER:
        dataset_dir = os.path.join(native_root, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for stage_name in sorted(os.listdir(dataset_dir)):
            result_path = os.path.join(dataset_dir, stage_name, "Result.text")
            if not os.path.isfile(result_path):
                continue
            match = re.search(r"task(\d+)", stage_name, re.IGNORECASE)
            if match is None:
                continue
            stage_id = int(match.group(1))
            rows[(dataset, stage_id)] = parse_result_value(result_path)
    return rows


def collect_ablation(ablation_root):
    rows = {}
    for dataset in DATASET_ORDER:
        dataset_dir = os.path.join(ablation_root, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for stage_name in sorted(os.listdir(dataset_dir)):
            result_path = os.path.join(dataset_dir, stage_name, "Result.text")
            if not os.path.isfile(result_path):
                continue
            match = re.search(r"task(\d+)", stage_name, re.IGNORECASE)
            if match is None:
                continue
            stage_id = int(match.group(1))
            rows[(dataset, stage_id)] = parse_result_value(result_path)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    native = collect_native(args.native_root)
    ablation = collect_ablation(args.ablation_root)
    keys = sorted(set(native.keys()) & set(ablation.keys()), key=lambda item: (item[1], DATASET_ORDER.index(item[0])))

    rows = []
    for dataset, stage_id in keys:
        native_metric, native_value = native[(dataset, stage_id)]
        ablation_metric, ablation_value = ablation[(dataset, stage_id)]
        if native_metric != ablation_metric:
            raise ValueError(
                f"Metric mismatch for {dataset} task{stage_id}: native={native_metric}, ablation={ablation_metric}"
            )
        rows.append(
            {
                "dataset": dataset,
                "stage_id": stage_id,
                "metric": native_metric,
                "native": round(native_value, 4),
                "band_prior": round(ablation_value, 4),
                "delta": round(ablation_value - native_value, 4),
            }
        )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "stage_id", "metric", "native", "band_prior", "delta"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
