#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_SPECS = {
    "ucit": {
        "config_dir": REPO_ROOT / "configs" / "data_configs" / "UCIT",
        "tasks": ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR", "Flickr30k"],
        "config_files": {
            "ImageNet-R": "ImageNet-R.json",
            "ArxivQA": "ArxivQA.json",
            "VizWiz": "VizWiz.json",
            "IconQA": "IconQA.json",
            "CLEVR": "CLEVR-Math.json",
            "Flickr30k": "Flickr30k.json",
        },
        "display_name": "UCIT",
    },
    "acl": {
        "config_dir": REPO_ROOT / "configs" / "data_configs" / "MLLM-ACL",
        "tasks": ["APP", "Math", "OCR", "VP"],
        "config_files": {task: f"{task}.json" for task in ["APP", "Math", "OCR", "VP"]},
        "display_name": "MLLM-ACL",
    },
    "dcl": {
        "config_dir": REPO_ROOT / "configs" / "data_configs" / "MLLM-DCL",
        "tasks": ["AD", "Fin", "Med", "RS", "Sci"],
        "config_files": {task: f"{task}.json" for task in ["AD", "Fin", "Med", "RS", "Sci"]},
        "display_name": "MLLM-DCL",
    },
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "show",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "was",
    "what",
    "which",
    "who",
    "with",
    "would",
    "you",
    "your",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def read_task_config(benchmark: str, task: str) -> Dict[str, str]:
    spec = BENCHMARK_SPECS[benchmark]
    cfg_name = spec["config_files"].get(task, f"{task}.json")
    cfg_path = spec["config_dir"] / cfg_name
    cfg = load_json(cfg_path)
    return {
        "config_path": str(cfg_path),
        "test_path": cfg.get("test_path") or cfg.get("train_path"),
        "image_root": cfg.get("test_folder") or cfg.get("train_folder") or "",
    }


def resolve_image_path(raw_image_path: str, image_root: str) -> str:
    image_path = Path(raw_image_path)
    if image_path.is_absolute():
        return str(image_path)
    return str(Path(image_root) / image_path)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def has_explicit_option_labels(text: str) -> bool:
    return bool(re.findall(r"(?:(?<=\n)|(?<=\s)|^)([A-H]|[0-9]+)[\.\)]\s+\S", text))


def detect_answer_format_class(text: str, answer: str) -> str:
    answer = answer.strip()
    answer_lc = answer.lower()
    if has_explicit_option_labels(text):
        if re.fullmatch(r"[A-Ha-h]|[0-9]+", answer):
            return "mcq_label"
        return "mcq_text"
    if "rationale:" in text.lower() and re.fullmatch(r"[A-Ha-h]|[0-9]+", answer):
        return "mcq_label"
    if answer_lc in {"yes", "no", "yes.", "no.", "true", "false"}:
        return "yes_no"
    compact = answer.replace(",", "")
    if compact and re.fullmatch(r"[-+]?\d+(\.\d+)?", compact):
        return "numeric"
    if len(re.findall(r"\w+", answer)) >= 6 or re.search(r"[.!?]\s*$", answer):
        return "sentence"
    return "short_phrase"


def detect_prompt_cluster(text: str) -> str:
    normalized = text.lower()
    if has_explicit_option_labels(text):
        if re.search(r"\b(why|how|best|most likely|infer|happen)\b", normalized):
            return "mcq_reasoning"
        return "mcq_visual_qa"
    if "rationale:" in normalized:
        return "mcq_reasoning"
    if re.search(r"\bcompare\b|\bdifference\b|\bmore likely\b|\binfer\b|\binference\b", normalized):
        return "comparison_reasoning"
    if re.search(r"\bcaption\b|\bdescribe the image\b|\bbrief caption\b", normalized):
        return "caption"
    if re.search(r"\bwhat is written\b|\bread\b|\btext in the image\b|\bocr\b", normalized):
        return "ocr_reading"
    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", normalized):
        return "counting"
    if re.search(r"\bwhat is the object\b|\bidentify\b|\bwhat is shown\b", normalized):
        return "recognition"
    if re.search(r"\bclick\b|\bbutton\b|\bicon\b|\bgui\b|\bapp\b", normalized):
        return "gui_action"
    if re.search(r"\bcalculate\b|\bsolve\b|\bmath\b|\bequation\b|\bresult\b", normalized):
        return "math_reasoning"
    if re.search(r"\bwould\b|\bis\b|\bare\b|\bdo\b|\bdoes\b|\bcan\b|\bhas\b|\bhave\b", normalized):
        return "yes_no"
    return "general_vqa"


def coarse_objective_label(prompt_cluster: str, answer_format_class: str) -> str:
    if prompt_cluster == "caption":
        return "objective_caption"
    if prompt_cluster in {"recognition", "ocr_reading", "gui_action"}:
        return "objective_recognition"
    if prompt_cluster in {"comparison_reasoning", "counting", "math_reasoning", "mcq_reasoning"}:
        return "objective_reasoning"
    if prompt_cluster.startswith("mcq") or answer_format_class.startswith("mcq"):
        return "objective_mcq"
    if prompt_cluster == "yes_no":
        return "objective_qa"
    return "objective_qa"


def reasoning_difficulty(prompt_cluster: str) -> str:
    if prompt_cluster in {"comparison_reasoning", "counting", "math_reasoning", "mcq_reasoning"}:
        return "hard"
    if prompt_cluster in {"caption", "recognition", "ocr_reading", "gui_action"}:
        return "easy"
    if prompt_cluster in {"mcq_visual_qa", "yes_no", "general_vqa"}:
        return "easy"
    return "hard"


def image_cluster_id(image_path: str, num_buckets: int = 32) -> str:
    image = Path(image_path)
    parent_bits = image.parts[-4:-1] if len(image.parts) >= 4 else image.parts
    key = "/".join(parent_bits) or image.parent.as_posix()
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % max(num_buckets, 1)
    width = max(2, len(str(max(num_buckets - 1, 0))))
    return f"cluster_{bucket:0{width}d}"


def template_skeleton(text: str) -> str:
    tokens = re.findall(r"[A-Za-z]+|[0-9]+|[^\w\s]", normalize_whitespace(text).lower())
    out: List[str] = []
    for tok in tokens:
        if re.fullmatch(r"[0-9]+", tok):
            out.append("<num>")
        elif re.fullmatch(r"[A-Za-z]+", tok):
            out.append(tok if tok in STOPWORDS else "<lex>")
        else:
            out.append(tok)
    return " ".join(out)


def template_id(text: str, prompt_cluster: str) -> str:
    skeleton = template_skeleton(text)
    digest = hashlib.md5(skeleton.encode("utf-8")).hexdigest()[:10]
    return f"tmpl_{prompt_cluster}_{digest}"


def sample_records(records: Sequence[Dict[str, object]], max_count: Optional[int], seed: int) -> List[Dict[str, object]]:
    if max_count is None or max_count >= len(records):
        return list(records)
    rng = random.Random(seed)
    return rng.sample(list(records), max_count)


def balance_by_keys(
    records: Sequence[Dict[str, object]],
    keys: Sequence[str],
    max_per_group: int,
    seed: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[tuple(record.get(k) for k in keys)].append(record)
    balanced: List[Dict[str, object]] = []
    for _, group in sorted(groups.items(), key=lambda item: str(item[0])):
        if len(group) > max_per_group:
            balanced.extend(rng.sample(group, max_per_group))
        else:
            balanced.extend(group)
    return balanced


def make_manifest_split(
    records: Sequence[Dict[str, object]],
    held_out_key: str,
    holdout_value: str,
) -> Dict[str, List[str]]:
    train_ids: List[str] = []
    test_ids: List[str] = []
    for record in records:
        sample_id = str(record["sample_id"])
        if str(record.get(held_out_key)) == holdout_value:
            test_ids.append(sample_id)
        else:
            train_ids.append(sample_id)
    return {"train_ids": train_ids, "test_ids": test_ids}


def split_summary(records: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, int]]:
    prompt_counter = Counter(str(r.get("prompt_cluster")) for r in records)
    dataset_counter = Counter(str(r.get("dataset_group")) for r in records)
    objective_counter = Counter(str(r.get("objective_label")) for r in records)
    return {
        "prompt_cluster_counts": dict(sorted(prompt_counter.items())),
        "dataset_group_counts": dict(sorted(dataset_counter.items())),
        "objective_label_counts": dict(sorted(objective_counter.items())),
    }
