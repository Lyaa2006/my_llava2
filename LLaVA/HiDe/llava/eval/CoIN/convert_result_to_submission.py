import argparse
import json
import os
import re

import pandas as pd


def normalize_question_id(question_id):
    if question_id is None:
        return None
    if isinstance(question_id, int):
        return question_id

    question_id = str(question_id).strip()
    if question_id.startswith("ALL_"):
        question_id = question_id[4:]

    try:
        return int(question_id)
    except ValueError:
        return None


def extract_option(text):
    if text is None:
        return "Z"

    text = str(text).strip().upper()
    match = re.match(r"^([A-Z])(?:[\.\):\s]|$)", text)
    if match:
        return match.group(1)
    return text or "Z"


def load_predictions(result_file):
    predictions = {}
    with open(result_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            question_id = normalize_question_id(entry.get("question_id"))
            if question_id is None:
                continue
            text = str(entry.get("text", "")).strip()
            predictions[question_id] = {"question_id": question_id, "text": text}
    return predictions


def resolve_template_file(template_file, question_file):
    candidates = []
    if template_file:
        candidates.append(template_file)

    default_name = "llava_v1.5_7b_MMT-Bench_ALL_openai_submission.tsv"
    candidates.append(default_name)

    if question_file:
        question_dir = os.path.dirname(os.path.abspath(question_file))
        candidates.append(os.path.join(question_dir, default_name))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def build_dataframe_from_question_file(question_file):
    if not question_file:
        raise FileNotFoundError(
            "No submission template was found and --question-file was not provided."
        )

    with open(question_file, "r") as f:
        questions = json.load(f)

    rows = []
    for item in questions:
        rows.append(
            {
                "index": normalize_question_id(item.get("question_id")),
                "question_id": item.get("question_id"),
                "image": item.get("image"),
                "category": item.get("category"),
                "l2-category": item.get("l2-category"),
                "split": item.get("split"),
                "question": item.get("text"),
                "prediction": "",
                "opt": "",
            }
        )

    return pd.DataFrame(rows)


def update_submission(tsv_data, predictions):
    if "prediction" not in tsv_data.columns:
        tsv_data["prediction"] = ""
    if "opt" not in tsv_data.columns:
        tsv_data["opt"] = ""

    for index, row in tsv_data.iterrows():
        lookup_key = None
        if "index" in tsv_data.columns:
            lookup_key = normalize_question_id(row.get("index"))
        if lookup_key is None and "question_id" in tsv_data.columns:
            lookup_key = normalize_question_id(row.get("question_id"))
        if lookup_key is None:
            lookup_key = index

        matched_entry = predictions.get(lookup_key)
        if matched_entry:
            current_prediction = row.get("prediction", "")
            if pd.isna(current_prediction) or str(current_prediction).strip() == "":
                tsv_data.at[index, "prediction"] = matched_entry["text"]
            tsv_data.at[index, "opt"] = extract_option(matched_entry["text"])
        else:
            current_prediction = row.get("prediction", "")
            if pd.isna(current_prediction) or str(current_prediction).strip() == "":
                tsv_data.at[index, "prediction"] = "0"
            tsv_data.at[index, "opt"] = "Z"

    return tsv_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, default='./LLaVA/results/CoIN_slim_new_0.8/OCRVQA/Finetune/merge.jsonl')
    parser.add_argument('--output_file', type=str, default='./LLaVA/results/CoIN_slim_new_0.8/OCRVQA/Finetune/our_result_for_submission.tsv')
    parser.add_argument('--question-file', type=str, default=None)
    parser.add_argument('--template-file', type=str, default=None)
    args = parser.parse_args()

    predictions = load_predictions(args.result_file)
    template_file = resolve_template_file(args.template_file, args.question_file)

    if template_file:
        tsv_data = pd.read_csv(template_file, sep='\t')
    else:
        tsv_data = build_dataframe_from_question_file(args.question_file)

    tsv_data = update_submission(tsv_data, predictions)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    tsv_data.to_csv(args.output_file, sep='\t', index=False)

    print(f"TSV file written successfully: {args.output_file}")
