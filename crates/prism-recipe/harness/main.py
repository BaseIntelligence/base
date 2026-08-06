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

v3 flow (`PRISM_FLOW=v3`, or auto when eval assets / a secret seed are
staged): TRAIN-phase child (`prismlib.train_v3`, saves
`$PRISM_WORKDIR/checkpoint.pt`, no scoring) -> parent gate (process group
hard-killed + survivor check, JIT caches reset, asset tier detection:
`$PRISM_EVAL_ASSETS_DIR` -> "private", else "public_dev") -> fresh
EVAL-phase child (`prismlib.eval_v3`, rebuilds the model from the miner's
architecture.py + checkpoint, runs the G1–G8 battery + v1 bpb; the secret
generator seed arrives via env only) -> METRICS_JSON v2 + `flow`/
`eval_tier`/`gate`/`battery`/`items`. The v1 path is byte-identical in
behavior.

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
EVAL_TIMEOUT_S = float_env("PRISM_EVAL_TIMEOUT_S", 3 * 3600.0)
WORKDIR = os.environ.get("PRISM_WORKDIR", "/tmp/prism_eval")


def _eval_battery_status():
    """Discovery-only battery status for the manifest."""
    try:
        batt = importlib.import_module("eval")
        return batt.status_summary()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:200]}


def _detect_flow():
    """v1 (legacy single invocation) vs v3 (two-phase train/eval).

    Explicit `PRISM_FLOW=v1|v3` wins. Otherwise the flow stays
    v1-compatible until the operator stages private assets or a secret
    seed — the battery then runs in the v3 child with the public dev
    family when assets are absent (`eval_tier: "public_dev"`).
    """
    f = os.environ.get("PRISM_FLOW", "").strip().lower()
    if f in ("v1", "v3"):
        return f
    if os.environ.get("PRISM_EVAL_ASSETS_DIR") or os.environ.get("PRISM_EVAL_SECRET_SEED"):
        return "v3"
    return "v1"


def _cheatguard():
    """Optional anti-cheat hooks (landed by a parallel change); missing
    module is a silent no-op per the E2 contract."""
    try:
        from prismlib import cheatguard

        return cheatguard
    except Exception:  # noqa: BLE001
        return None


def _cheatguard_call(name, *args):
    cg = _cheatguard()
    if cg is None:
        return None
    fn = getattr(cg, name, None)
    if not callable(fn):
        return None
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001
        log(f"cheatguard.{name} error (ignored): {exc}")
        return None


def _jit_caches_reset():
    """Post-train gate step: drop any parent-side torch/JIT caches."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            import torch._dynamo

            torch._dynamo.reset()
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


def _run_v3(ctx, ctx_path, manifest, t_start, netns):
    """Two-phase v3 flow: train child -> gate (dead miners, JIT reset,
    asset tier) -> fresh eval child (battery + v1 bpb) -> METRICS_JSON."""
    from prismlib import v3flow

    train_res = v3flow.run_phase(
        "prismlib.train_v3",
        WORKDIR,
        ctx_path,
        budgets={
            "build": BUILD_TIMEOUT_S,
            "train": TRAIN_HOURS_CAP * 3600.0 + 120.0,
            "checkpoint": SCORE_TIMEOUT_S,
        },
    )
    if train_res.get("killed_phase"):
        fail(
            train_res["killed_phase"],
            RuntimeError(f"v3 train phase timeout: {train_res['killed_phase']}"),
        )
    tpayload = train_res.get("payload")
    if tpayload is None or tpayload.get("status") != "ok":
        tail = " | ".join(train_res.get("tail", [])[-5:])
        fail(
            str((tpayload or {}).get("stage", "train")),
            RuntimeError(str((tpayload or {}).get("error", f"no v3 train payload: {tail[:300]}"))[:400]),
        )

    # Gate: miner process group must be dead before assets/eval.
    _jit_caches_reset()
    gate = {
        "survivors_after_train": train_res.get("survivors"),
        "train_netns": train_res["netns"],
    }
    if train_res.get("survivors"):
        log("WARNING: train-phase survivor processes detected (recorded in gate)")

    assets_dir = os.environ.get("PRISM_EVAL_ASSETS_DIR", "").strip() or None
    if assets_dir and not os.path.isdir(assets_dir):
        log(f"WARNING: PRISM_EVAL_ASSETS_DIR={assets_dir} is not a directory — public_dev tier")
        assets_dir = None
    eval_tier = "private" if assets_dir else "public_dev"

    eval_ctx = dict(ctx)
    eval_ctx.update(
        {
            "eval_assets_dir": assets_dir,
            "eval_tier": eval_tier,
            "probe_curve": tpayload.get("probe_curve") or [],
            "telemetry_series": (tpayload.get("telemetry") or {}).get("loss_series") or [],
            "tokens_seen": tpayload.get("tokens_seen", 0),
            "n_params": tpayload.get("n_params", 0),
            "train_wall_s": tpayload.get("wall_clock_seconds", 0.0),
        }
    )
    eval_ctx_path = os.path.join(WORKDIR, "prism_eval_ctx.json")
    with open(eval_ctx_path, "w", encoding="utf-8") as f:
        json.dump(eval_ctx, f)

    _cheatguard_call("pre_eval", eval_ctx)
    eval_res = v3flow.run_phase(
        "prismlib.eval_v3",
        WORKDIR,
        eval_ctx_path,
        budgets={"eval": EVAL_TIMEOUT_S, "build": BUILD_TIMEOUT_S, "battery": EVAL_TIMEOUT_S, "score": SCORE_TIMEOUT_S},
    )
    # Secret seed was passed through the child env; drop it from ours now.
    os.environ.pop("PRISM_EVAL_SECRET_SEED", None)
    if eval_res.get("killed_phase"):
        fail(
            eval_res["killed_phase"],
            RuntimeError(f"v3 eval phase timeout: {eval_res['killed_phase']}"),
        )
    epayload = eval_res.get("payload")
    if epayload is None or epayload.get("status") != "ok":
        tail = " | ".join(eval_res.get("tail", [])[-5:])
        fail(
            str((epayload or {}).get("stage", "eval")),
            RuntimeError(str((epayload or {}).get("error", f"no v3 eval payload: {tail[:300]}"))[:400]),
        )
    gate["survivors_after_eval"] = eval_res.get("survivors")
    gate["eval_netns"] = eval_res["netns"]

    out = {
        "bpb": epayload["bpb"],
        "tokens_seen": tpayload["tokens_seen"],
        "wall_clock_seconds": tpayload["wall_clock_seconds"],
        "gpu_type": os.environ.get("PRISM_GPU_TYPE", "unknown"),
        "notes": "recipe-v2 val_ce->bpb (v3 eval phase)",
        "val_rows": epayload["val_rows"],
        "n_params": epayload["n_params"],
        "recipe": RECIPE_VERSION,
        "telemetry": tpayload["telemetry"],
        "metrics_version": 2,
        "tokens_seen_source": tpayload["tokens_seen_source"],
        "probe_curve": tpayload["probe_curve"],
        "train_metrics": tpayload.get("train_metrics", {}),
        "pod_manifest": manifest,
        "netns": train_res["netns"],
        "harness_files_sha256": manifest_mod.harness_files_sha256(),
        "flow": "v3",
        "eval_tier": eval_tier,
        "gate": gate,
        "battery": epayload.get("battery", {}),
        "items": epayload.get("items", {}),
    }
    _cheatguard_call("post_eval", out)
    print("METRICS_JSON=" + json.dumps(out))
    print("EVAL_OK")
    log(f"v3 eval complete in {time.time()-t_start:.0f}s")


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

    _cheatguard_call("pre_train", ctx)

    if _detect_flow() == "v3":
        _run_v3(ctx, ctx_path, manifest, t_start, netns)
        return

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
