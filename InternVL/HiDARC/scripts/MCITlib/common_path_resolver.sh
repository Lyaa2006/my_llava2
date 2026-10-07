#!/bin/bash

# HiDESC was developed against the legacy dependency stack.  Do not inherit
# an unrelated active conda environment such as MCITlib_copy.
COMMON_SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIDESC_PROJECT_ROOT="${HIDESC_PROJECT_ROOT:-$(realpath "$COMMON_SCRIPT_DIR/../..")}"
HIDESC_ENV_BIN="${HIDESC_ENV_BIN:-/home/lyaa/miniconda3/envs/MCITlib/bin}"
if [ ! -x "$HIDESC_ENV_BIN/python3" ]; then
    echo "Missing HiDESC Python environment: $HIDESC_ENV_BIN/python3" >&2
    return 1 2>/dev/null || exit 1
fi
export PATH="$HIDESC_ENV_BIN:$PATH"
export PYTHONPATH="$HIDESC_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN="${PYTHON_BIN:-$HIDESC_ENV_BIN/python3}"
export DEEPSPEED_BIN="${DEEPSPEED_BIN:-$HIDESC_ENV_BIN/deepspeed}"

if [ "${HIDESC_ENV_CHECKED:-0}" != "1" ]; then
    "$PYTHON_BIN" - <<'PY'
import sys

try:
    import peft
    import transformers
    import deepspeed
except Exception as exc:
    print(f"HiDESC dependency import failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

actual = {
    "peft": getattr(peft, "__version__", "unknown"),
    "transformers": getattr(transformers, "__version__", "unknown"),
    "deepspeed": getattr(deepspeed, "__version__", "unknown"),
}
expected = {"peft": "0.4.0", "transformers": "4.31.0", "deepspeed": "0.9.5"}
bad = [f"{key}={actual[key]} (expected {expected[key]})" for key in expected if actual[key] != expected[key]]
if bad:
    print("Incompatible HiDESC environment: " + "; ".join(bad), file=sys.stderr)
    raise SystemExit(1)
print(
    "HiDESC dependency preflight: "
    + sys.executable
    + " "
    + " ".join(f"{key}={actual[key]}" for key in expected)
)
PY
    export HIDESC_ENV_CHECKED=1
fi

ensure_argument_count() {
    local expected="$1"
    local actual="$2"
    local usage="${3:-}"
    if [ "$actual" -ne "$expected" ]; then
        if [ -n "$usage" ]; then
            echo "Usage: $usage" >&2
        else
            echo "Expected $expected arguments, got $actual" >&2
        fi
        exit 1
    fi
}

ensure_nonempty_value() {
    local value="$1"
    local label="${2:-value}"
    if [ -z "$value" ]; then
        echo "Missing ${label}." >&2
        exit 1
    fi
}

resolve_run_scoped_path() {
    python3 - "$1" "${UCIT_RUN_ID:-}" <<'PY'
import os
import sys

raw_path = sys.argv[1]
run_id = sys.argv[2]
candidates = []

def add(path: str):
    if path and path not in candidates:
        candidates.append(path)

add(raw_path)

if run_id:
    add(f"{raw_path}_{run_id}")

    norm_path = raw_path.replace("\\", "/")
    smoke_marker = "/smoke/"
    if smoke_marker in norm_path:
        prefix, suffix = norm_path.split(smoke_marker, 1)
        add(os.path.join(prefix, run_id, suffix))
        add(os.path.join(prefix, run_id, os.path.basename(raw_path)))
        add(os.path.join(prefix, run_id, f"{os.path.basename(raw_path)}_{run_id}"))

    dirname = os.path.dirname(raw_path)
    basename = os.path.basename(raw_path)
    add(os.path.join(dirname, f"{basename}_{run_id}"))

    parent_dir = os.path.dirname(dirname)
    dirname_name = os.path.basename(dirname)
    if dirname_name == "smoke":
        add(os.path.join(parent_dir, run_id, basename))
        add(os.path.join(parent_dir, run_id, f"{basename}_{run_id}"))

for candidate in candidates:
    if os.path.isdir(candidate):
        print(os.path.abspath(candidate))
        break
else:
    print(raw_path)
PY
}

ensure_existing_file() {
    local path="$1"
    local label="${2:-file}"
    ensure_nonempty_value "$path" "$label"
    if [ ! -f "$path" ]; then
        echo "Missing ${label}: $path" >&2
        exit 1
    fi
}

ensure_existing_dir() {
    local path="$1"
    local label="${2:-directory}"
    ensure_nonempty_value "$path" "$label"
    if [ ! -d "$path" ]; then
        echo "Missing ${label}: $path" >&2
        exit 1
    fi
}

read_optional_json_field() {
    python3 - "$1" "$2" <<'PY'
import json
import sys

cfg_path = sys.argv[1]
key = sys.argv[2]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
value = cfg.get(key, "")
if value is None:
    value = ""
print(value)
PY
}

quarantine_incomplete_cache_dir() {
    local cache_dir="$1"
    local label="${2:-Description cache}"
    if [ ! -d "$cache_dir" ]; then
        return 0
    fi

    local meta_path="$cache_dir/meta.json"
    if [ -f "$meta_path" ]; then
        return 0
    fi

    local shard_count
    shard_count=$(find "$cache_dir" -maxdepth 1 -name '*.pt' | wc -l)
    if [ "$shard_count" -le 0 ]; then
        return 0
    fi

    local quarantine_path="${cache_dir}_stale_$(date +%Y%m%d_%H%M%S)"
    echo "$label has $shard_count shard(s) but no meta.json; moving stale partial cache to: $quarantine_path"
    mv "$cache_dir" "$quarantine_path"
    mkdir -p "$cache_dir"
}
