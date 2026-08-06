#!/usr/bin/env python3
"""PRISM pod harness parent (recipe 1.3.0) — uploaded by prism-lium over SSH.

Multi-file harness package:
  main.py            — this parent orchestrator (network + SSH channel)
  prismlib/          — library modules (dataset, telemetry, train stream,
                       probes, scoring, manifest, subprocess runner, miner entry)
  eval/              — G1–G8 battery registry (modules land in a later step)

Flow: verify the pinned fineweb-edu shard (URL + SHA-256) -> collect pod
manifest -> spawn the miner subprocess (`unshare --net -- python3 -m
prismlib.miner_entry ctx.json` when available; plain subprocess fallback
with a loud warning) -> read its one-line JSON result from the dedicated FD
-> print `METRICS_JSON={...}` (v2) and `EVAL_OK` (or stage line + EVAL_FAIL).

The parent never imports miner code. Miner code runs in the subprocess;
harness-owned code in that same subprocess drives the frozen-val scoring
after train() returns, so the miner never owns the val loop.

ctx contract (JSON file -> subprocess; miner-visible keys):
  dataset_path (deprecated; kept, usage unmonitored — agentic enforces),
  dataset_sha256, seed, train_hours_cap, max_train_steps, max_params,
  val_rows, train_rows, device, seq_len (default 512), batch_size,
  probe_every, probe_time_budget_s — plus in-process objects added by the
  subprocess entry: telemetry (prism_telemetry shim), guard(), tokenizer
  (GPT-2, loaded once by the harness), train_stream (SeededTrainStream
  yielding (input_ids, labels) batches; .tokens_seen is the authoritative
  harness token counter).

METRICS_JSON v2: every v1 key (bpb, tokens_seen, wall_clock_seconds,
gpu_type, notes, val_rows, n_params, recipe, telemetry) plus
metrics_version=2, tokens_seen_source ("train_stream" | "legacy"),
probe_curve, train_metrics, pod_manifest, netns, harness_files_sha256.
"""
import importlib
import json
import os
import time

from prismlib import RECIPE_SEED, RECIPE_VERSION, TOKENIZER, TRAIN_ROWS, VAL_ROWS
from prismlib import dataset
from prismlib import manifest as manifest_mod
from prismlib.envutil import fail, float_env, int_env, log
from prismlib.runner import probe_unshare, run_miner_subprocess

MAX_TRAIN_STEPS = int_env("PRISM_MAX_TRAIN_STEPS", 20000)
TRAIN_HOURS_CAP = float_env("PRISM_TRAIN_HOURS_CAP", 6.0)
# Test-mode knobs (staging/e2e; sim or real Lium): shrink the wall cap and
# the parameter cap so a full lifecycle fits in minutes on tiny models.
_TEST_TRAIN_MINUTES = float_env("PRISM_TEST_TRAIN_MINUTES", 0.0)
if _TEST_TRAIN_MINUTES > 0:
    TRAIN_HOURS_CAP = _TEST_TRAIN_MINUTES / 60.0
MAX_PARAMS = int_env("PRISM_TEST_MAX_PARAMS", int_env("PRISM_MAX_PARAMS", 350000000))
SEQ_LEN = int_env("PRISM_SEQ_LEN", 512)
BATCH_SIZE = int_env("PRISM_TRAIN_BATCH_SIZE", 8)
PROBE_EVERY = int_env("PRISM_PROBE_EVERY", 25)
PROBE_TIME_BUDGET_S = float_env("PRISM_PROBE_TIME_BUDGET_S", 600.0)
BUILD_TIMEOUT_S = float_env("PRISM_BUILD_TIMEOUT_S", 900.0)
SCORE_TIMEOUT_S = float_env("PRISM_SCORE_TIMEOUT_S", 1800.0)
WORKDIR = os.environ.get("PRISM_WORKDIR", "/tmp/prism_eval")


def _eval_battery_status():
    """Discovery-only battery status for the manifest (battery lands in E2)."""
    try:
        batt = importlib.import_module("eval")
        return batt.status_summary()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:200]}


def main():
    t_start = time.time()
    log(f"prism harness parent starting (recipe {RECIPE_VERSION})")
    dataset_url = os.environ["PRISM_DATASET_URL"]
    dataset_sha = os.environ["PRISM_DATASET_SHA256"]
    os.makedirs(WORKDIR, exist_ok=True)

    unshare = probe_unshare()
    netns = bool(unshare["available"])
    if not netns:
        log(
            "WARNING: netns unavailable ("
            + unshare["detail"]
            + "); miner subprocess will run WITHOUT network isolation"
        )

    try:
        parquet = dataset.materialize_dataset(dataset_url, dataset_sha)
    except Exception as exc:  # noqa: BLE001
        fail("dataset", exc)

    import torch

    allow_cpu = os.environ.get("PRISM_ALLOW_CPU") == "1"
    if torch.cuda.is_available():
        device = "cuda"
    elif allow_cpu:
        device = "cpu"
        log("WARNING: PRISM_ALLOW_CPU=1 — CPU device (test mode only)")
    else:
        fail("device", RuntimeError("cuda device required"))
    torch.manual_seed(RECIPE_SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(RECIPE_SEED)

    # Warm the HF cache from the parent (which has network) so the isolated
    # child resolves the tokenizer offline from the same cache.
    try:
        from transformers import GPT2TokenizerFast

        GPT2TokenizerFast.from_pretrained(TOKENIZER)
    except Exception as exc:  # noqa: BLE001
        fail("tokenizer", exc)

    ctx = {
        "dataset_path": parquet,
        "dataset_sha256": dataset_sha,
        "seed": RECIPE_SEED,
        "train_hours_cap": TRAIN_HOURS_CAP,
        "max_train_steps": MAX_TRAIN_STEPS,
        "max_params": MAX_PARAMS,
        "val_rows": VAL_ROWS,
        "train_rows": TRAIN_ROWS,
        "device": device,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "probe_every": PROBE_EVERY,
        "probe_time_budget_s": PROBE_TIME_BUDGET_S,
        "workdir": WORKDIR,
        "arch_path": os.path.join(WORKDIR, "architecture.py"),
        "train_path": os.path.join(WORKDIR, "training.py"),
    }
    ctx_path = os.path.join(WORKDIR, "prism_ctx.json")
    with open(ctx_path, "w", encoding="utf-8") as f:
        json.dump(ctx, f)

    manifest = manifest_mod.collect_manifest(
        netns=netns,
        unshare=unshare,
        eval_battery=_eval_battery_status(),
        started_ts=t_start,
    )

    res = run_miner_subprocess(
        WORKDIR,
        ctx_path,
        train_cap_s=TRAIN_HOURS_CAP * 3600.0,
        build_timeout_s=BUILD_TIMEOUT_S,
        score_timeout_s=SCORE_TIMEOUT_S,
    )
    if res.get("killed_phase"):
        fail(
            res["killed_phase"],
            RuntimeError(f"miner subprocess phase timeout: {res['killed_phase']}"),
        )
    payload = res.get("payload")
    if payload is None:
        tail = " | ".join(res.get("tail", [])[-5:])
        fail(
            "subprocess",
            RuntimeError(f"no result from miner subprocess (rc={res.get('rc')}): {tail[:300]}"),
        )
    if payload.get("status") != "ok":
        fail(
            str(payload.get("stage", "subprocess")),
            RuntimeError(str(payload.get("error", "unknown"))[:400]),
        )

    out = {
        "bpb": payload["bpb"],
        "tokens_seen": payload["tokens_seen"],
        "wall_clock_seconds": payload["wall_clock_seconds"],
        "gpu_type": os.environ.get("PRISM_GPU_TYPE", "unknown"),
        "notes": "recipe-v2 val_ce->bpb",
        "val_rows": payload["val_rows"],
        "n_params": payload["n_params"],
        "recipe": RECIPE_VERSION,
        "telemetry": payload["telemetry"],
        "metrics_version": 2,
        "tokens_seen_source": payload["tokens_seen_source"],
        "probe_curve": payload["probe_curve"],
        "train_metrics": payload.get("train_metrics", {}),
        "pod_manifest": manifest,
        "netns": res["netns"],
        "harness_files_sha256": manifest_mod.harness_files_sha256(),
    }
    print("METRICS_JSON=" + json.dumps(out))
    print("EVAL_OK")
    log(f"eval complete in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
