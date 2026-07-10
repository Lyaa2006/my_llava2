#!/bin/bash

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
