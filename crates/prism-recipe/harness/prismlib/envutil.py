"""Shared env parsing + logging helpers (stdlib only)."""

import json
import os
import sys


def float_env(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def int_env(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def log(msg):
    print("[harness]", msg, flush=True)


def fail(stage, exc):
    """Terminal failure (parent): v1-compatible stage line + EVAL_FAIL."""
    print(json.dumps({"stage": stage, "error": str(exc)[:400]}))
    print("EVAL_FAIL")
    sys.exit(3)
