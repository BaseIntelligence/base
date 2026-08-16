#!/usr/bin/env python3
"""PRISM pod harness parent (recipe 2.0.0) — uploaded by prism-lium over SSH.

Multi-file harness package:
  main.py            — this parent orchestrator (network + SSH channel)
  prismlib/          — library modules (dataset, telemetry, train stream,
                       probes, scoring, manifest, subprocess runner, miner
                       entry, AutoModel staging adapter)
  eval/              — G1–G8 battery registry (modules land in a later step)

Flow: verify the pinned fineweb-edu shard (URL + SHA-256) -> (recipe 2.0)
stage AutoModel pin + applied miner tree when env/workdir requests it ->
collect pod manifest -> spawn the miner subprocess (`unshare --net --
python3 -m prismlib.miner_entry ctx.json` when available; plain subprocess
fallback with a loud warning) -> read its one-line JSON result from the
dedicated FD -> print `METRICS_JSON={...}` (v2) and `EVAL_OK` (or stage
line + EVAL_FAIL). Also writes `/tmp/prism_eval/metrics.json` so Lium
harvest can recover blobs larger than a fixed log-tail window.

AutoModel fixture/sim (`PRISM_AUTOMODEL_FIXTURE=1`) exercises staging +
entry gates only — never treat fixture markers as proof of a real train.

v3 flow (`PRISM_FLOW=v3`, or auto when eval assets / a secret seed are
staged): TRAIN-phase child (`prismlib.train_v3`, saves
`$PRISM_WORKDIR/checkpoint.pt`, no scoring) -> parent gate (process group
hard-killed + survivor check, JIT caches reset). With
`$PRISM_EVAL_ASSETS_DIR` set (staged pack → default `eval_tier=public`)
the gate then prints the bare `PHASE_TRAIN_DONE` marker line (the Lium
client stages the assets tar + `SECRET_SEED` + `.ready` on it) and holds
for the `.ready` sentinel — fail-closed, never a silent downgrade to the
embedded `public_dev` fixtures — before spawning a fresh EVAL-phase child
(`prismlib.eval_v3`, rebuilds the model from the miner's architecture.py +
checkpoint, runs the G1–G8 battery + v1 bpb; the generator seed is read
from the staged hex file and handed to the eval child via env only).
Without the assets env the tier is `public_dev` (same battery, published
dev seed / tiny fixtures). -> METRICS_JSON v2 + `flow`/
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
  (the SUBMITTED tokenizer, resolved and validated once per phase by
  prismlib.tokenizer: `tokenizer/` files, a `build_tokenizer(ctx)` hook in
  architecture.py, else the pinned fallback), vocab_size (of that
  tokenizer — size embeddings from it, not from a hardcoded constant),
  train_stream (SeededTrainStream yielding (input_ids, labels) batches;
  .tokens_seen is the authoritative harness token counter).

METRICS_JSON v2: every v1 key (bpb, tokens_seen, wall_clock_seconds,
gpu_type, notes, val_rows, n_params, recipe, telemetry) plus
metrics_version=2, tokens_seen_source ("train_stream" | "legacy"),
probe_curve, train_metrics, pod_manifest, netns, harness_files_sha256,
tokenizer (resolved spec: source, vocab_size, fingerprint) and
bits_per_byte (tokenizer-neutral unit beside the per-token `bpb`).
"""
import importlib
import importlib.util  # `import importlib` alone does not bind `importlib.util`
import json
import os
import time

from prismlib import RECIPE_SEED, RECIPE_VERSION
from prismlib import TRAIN_ROWS as _RECIPE_TRAIN_ROWS
from prismlib import VAL_ROWS as _RECIPE_VAL_ROWS
from prismlib import dataset
from prismlib import deps as deps_mod
from prismlib import manifest as manifest_mod
from prismlib import tokenizer as tok_contract
from prismlib.envutil import fail, float_env, int_env, log
from prismlib.runner import probe_unshare, run_miner_subprocess

MAX_TRAIN_STEPS = int_env("PRISM_MAX_TRAIN_STEPS", 20000)
TRAIN_HOURS_CAP = float_env("PRISM_TRAIN_HOURS_CAP", 6.0)
# Test-mode knobs (staging/e2e; sim or real Lium): shrink the wall cap and
# the parameter cap so a full lifecycle fits in minutes on tiny models.
_TEST_TRAIN_MINUTES = float_env("PRISM_TEST_TRAIN_MINUTES", 0.0)
if _TEST_TRAIN_MINUTES > 0:
    TRAIN_HOURS_CAP = _TEST_TRAIN_MINUTES / 60.0
MAX_PARAMS = int_env("PRISM_TEST_MAX_PARAMS", int_env("PRISM_MAX_PARAMS", 1000000000))
# Test-mode row overrides (staging/e2e only): shrink the train slice and the
# frozen val cut so small procedural fixtures satisfy the harness contract.
# Production is always the prismlib constants (2048 train / 256 val).
VAL_ROWS = int_env("PRISM_TEST_VAL_ROWS", _RECIPE_VAL_ROWS)
TRAIN_ROWS = int_env("PRISM_TEST_TRAIN_ROWS", _RECIPE_TRAIN_ROWS)
SEQ_LEN = int_env("PRISM_SEQ_LEN", 512)
BATCH_SIZE = int_env("PRISM_TRAIN_BATCH_SIZE", 8)
PROBE_EVERY = int_env("PRISM_PROBE_EVERY", 25)
PROBE_TIME_BUDGET_S = float_env("PRISM_PROBE_TIME_BUDGET_S", 600.0)
BUILD_TIMEOUT_S = float_env("PRISM_BUILD_TIMEOUT_S", 900.0)
SCORE_TIMEOUT_S = float_env("PRISM_SCORE_TIMEOUT_S", 1800.0)
EVAL_TIMEOUT_S = float_env("PRISM_EVAL_TIMEOUT_S", 3 * 3600.0)
WORKDIR = os.environ.get("PRISM_WORKDIR", "/tmp/prism_eval")
# Sidecar for Lium harvest: v3 METRICS_JSON lines can exceed the historical
# 32 KiB log-tail window; master greps this file (or the full log line).
METRICS_SIDECAR = os.path.join(WORKDIR, "metrics.json")
# Two-phase private-tier staging (E5): with `$PRISM_EVAL_ASSETS_DIR` set the
# parent announces train-done by printing this exact bare line (the Lium
# client matches it as a raw line — a `log()` prefix would not match) and
# then holds the post-train gate on `$PRISM_EVAL_ASSETS_DIR/.ready`.
TRAIN_DONE_MARKER = os.environ.get("PRISM_TRAIN_DONE_MARKER", "PHASE_TRAIN_DONE")
ASSETS_WAIT_S = float_env("PRISM_ASSETS_WAIT_S", 900.0)
ENTRY_MARKER = ".prism_entry"


def _resolve_miner_paths(workdir):
    """Resolve architecture/training paths for two-script or source-tree layout.

    Full trees land under `submission/` (see prism_tree::POD_SUBMISSION_DIR)
    with a harness-owned `.prism_entry` naming the training entry module.
    Legacy two-script uploads keep `architecture.py` / `training.py` at the
    workdir root.

    Recipe 2.0 AutoModel: when env/workdir requests the patch layout,
    `prismlib.automodel.stage_and_prepare` materialises the applied tree and
    writes harness-owned seams at the workdir root (returned here).
    """
    from prismlib import automodel as am

    if am.layout_requested(workdir):
        try:
            prep = am.stage_and_prepare(workdir)
        except Exception as exc:  # noqa: BLE001
            fail("automodel_stage", exc)
        return prep["arch_path"], prep["train_path"], prep.get("automodel")

    sub = os.path.join(workdir, "submission")
    if os.path.isdir(sub):
        entry = "training.py"
        marker = os.path.join(sub, ENTRY_MARKER)
        if os.path.isfile(marker):
            with open(marker, "r", encoding="utf-8") as f:
                entry = f.read().strip() or entry
        arch = os.path.join(sub, "architecture.py")
        train = os.path.join(sub, entry)
        if not os.path.isfile(arch):
            fail("contract", FileNotFoundError(f"missing {arch}"))
        if not os.path.isfile(train):
            fail("contract", FileNotFoundError(f"missing entry {train}"))
        return arch, train, None
    return (
        os.path.join(workdir, "architecture.py"),
        os.path.join(workdir, "training.py"),
        None,
    )


def _eval_battery_status():
    """Discovery-only battery status for the manifest."""
    try:
        batt = importlib.import_module("eval")
        return batt.status_summary()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:200]}


def _detect_flow():
    """v1 (legacy single invocation) vs v3 (two-phase train/eval + G1–G8).

    Explicit `PRISM_FLOW=v1|v3` wins. Default is **v3** so scored runs always
    execute the public battery (full pack when `PRISM_EVAL_ASSETS_DIR` is
    staged; otherwise `eval_tier=public_dev` fixtures — never silent BPB-only).
    Set `PRISM_FLOW=v1` only for legacy single-shot compatibility.
    """
    f = os.environ.get("PRISM_FLOW", "").strip().lower()
    if f in ("v1", "v3"):
        return f
    return "v3"


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


def _emit_metrics(out):
    """Print `METRICS_JSON=` and write `metrics.json` sidecar for harvest.

    Battery blobs often exceed the historical 32 KiB harness.log tail window;
    the Lium client prefers this sidecar (else greps the full log line).
    """
    blob = json.dumps(out)
    try:
        os.makedirs(WORKDIR, exist_ok=True)
        with open(METRICS_SIDECAR, "w", encoding="utf-8") as f:
            f.write(blob)
    except OSError:
        pass
    print("METRICS_JSON=" + blob)


def _cap_exceeded_out(payload, manifest, t_start):
    """Miner-attributable parameter-cap breach: machine-readable terminal.

    Emits a minimal METRICS_JSON v2 (bpb sentinel 0 — gated off client-side
    by the flag) plus the exact `CAP_EXCEEDED` line the Lium client matches;
    the orchestrator maps this to a terminal Score(0), never a measured
    score. Deliberately no EVAL_OK: the run is not a measurement.
    """
    out = {
        "bpb": 0.0,
        "tokens_seen": 0,
        "wall_clock_seconds": time.time() - t_start,
        "gpu_type": os.environ.get("PRISM_GPU_TYPE", "unknown"),
        "notes": "parameter cap exceeded",
        "val_rows": 0,
        "n_params": payload.get("n_params"),
        "recipe": RECIPE_VERSION,
        "metrics_version": 2,
        "cap_exceeded": True,
        "pod_manifest": manifest,
        "harness_files_sha256": manifest_mod.harness_files_sha256(),
    }
    _emit_metrics(out)
    print("CAP_EXCEEDED")
    log(f"parameter cap exceeded: {payload.get('error', '')[:200]}")


def _await_eval_assets(assets_dir):
    """Hold the post-train gate until the master stages the eval-assets pack.

    Emits the bare train-done marker (exact raw line — the Lium client
    matches it to trigger staging over a second SSH channel), then polls for
    the `.ready` sentinel. Fail-closed: no `.ready` within `ASSETS_WAIT_S`
    is terminal, never a silent downgrade to embedded `public_dev` fixtures.
    Returns the staged `SECRET_SEED` (128-bit hex file) as a decimal string
    for the eval child's `PRISM_EVAL_SECRET_SEED` env — file-only delivery
    keeps the train-phase child from ever inheriting it.
    """
    print(TRAIN_DONE_MARKER, flush=True)
    ready = os.path.join(assets_dir, ".ready")
    deadline = time.monotonic() + ASSETS_WAIT_S
    while not os.path.exists(ready):
        if time.monotonic() > deadline:
            fail(
                "assets",
                RuntimeError(
                    f"eval assets .ready not staged within {ASSETS_WAIT_S:.0f}s ({assets_dir})"
                ),
            )
        time.sleep(1.0)
    log(f"eval assets staged (.ready) — pack ready in {assets_dir}")
    try:
        with open(os.path.join(assets_dir, "SECRET_SEED"), "r", encoding="utf-8") as f:
            hex_seed = f.read().strip()
    except OSError as exc:
        fail("assets", RuntimeError(f"staged assets missing SECRET_SEED: {exc}"))
    try:
        return str(int(hex_seed, 16))
    except ValueError:
        fail("assets", RuntimeError("staged SECRET_SEED is not a hex string"))


def _run_v3(ctx, ctx_path, manifest, t_start, netns):
    """Two-phase v3 flow: train child -> gate (dead miners, JIT reset,
    asset staging) -> fresh eval child (battery + v1 bpb) -> METRICS_JSON."""
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
    if tpayload and tpayload.get("cap_exceeded"):
        _cap_exceeded_out(tpayload, manifest, t_start)
        return
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
    staged_seed = None
    if assets_dir:
        # Staged pack (default eval_tier=public from HF held-out assets):
        # train-phase process group is dead; announce train-done so the
        # master stages the assets + SECRET_SEED, then hold on `.ready`
        # (fail-closed — never a silent downgrade to public_dev fixtures).
        staged_seed = _await_eval_assets(assets_dir)
    from eval import common as eval_common

    eval_tier = eval_common.resolve_eval_tier(assets_dir)

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
    # Staged seed rides the eval child's env only (decimal string; the
    # hex file stays in the assets dir for cheatguard). An operator-preset
    # `PRISM_EVAL_SECRET_SEED` (local runs) is inherited instead.
    eval_extra = {"PRISM_EVAL_SECRET_SEED": staged_seed} if staged_seed else None
    eval_res = v3flow.run_phase(
        "prismlib.eval_v3",
        WORKDIR,
        eval_ctx_path,
        budgets={"eval": EVAL_TIMEOUT_S, "build": BUILD_TIMEOUT_S, "battery": EVAL_TIMEOUT_S, "score": SCORE_TIMEOUT_S},
        extra_env=eval_extra,
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
        "bits_per_byte": epayload.get("bits_per_byte"),
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
        "tokenizer": epayload.get("tokenizer") or tpayload.get("tokenizer") or {},
        "flow": "v3",
        "eval_tier": eval_tier,
        "gate": gate,
        "battery": epayload.get("battery", {}),
        "items": epayload.get("items", {}),
        "inference_traces": epayload.get("inference_traces") or {},
    }
    _cheatguard_call("post_eval", out)
    _emit_metrics(out)
    print("EVAL_OK")
    log(f"v3 eval complete in {time.time()-t_start:.0f}s")


def _run_automodel_fixture_stub(t_start):
    """CI/Sim AutoModel fixture: stage + entry only (no FineWeb / CUDA / EVAL_OK)."""
    from prismlib import automodel as am

    log(
        "automodel fixture/stub path — staging + entry gates only "
        "(not proof of a real NVIDIA AutoModel train)"
    )
    try:
        prep = am.stage_and_prepare(WORKDIR)
        am.invoke_entry(prep["automodel"], ctx=None)
    except Exception as exc:  # noqa: BLE001
        fail("automodel_fixture", exc)
    manifest = prep["automodel"]
    unshare = probe_unshare()
    _emit_metrics(
        {
            "bpb": 0.0,
            "tokens_seen": 0,
            "wall_clock_seconds": time.time() - t_start,
            "gpu_type": os.environ.get("PRISM_GPU_TYPE", "sim-fixture"),
            "notes": "automodel fixture stub — not real train proof",
            "val_rows": 0,
            "n_params": 0,
            "recipe": RECIPE_VERSION,
            "metrics_version": 2,
            "automodel_stub": True,
            "real_train": False,
            "netns": bool(unshare.get("available")),
            "automodel": {
                "base": manifest.get("base"),
                "entry": manifest.get("entry"),
                "mode": manifest.get("mode"),
            },
            "harness_files_sha256": manifest_mod.harness_files_sha256(),
        }
    )
    print("AUTOMODEL_FIXTURE_OK")
    # Deliberately no EVAL_OK — fixture markers are not scored proof.


def main():
    t_start = time.time()
    log(f"prism harness parent starting (recipe {RECIPE_VERSION})")
    os.makedirs(WORKDIR, exist_ok=True)

    from prismlib import automodel as am

    # Fixture/host-sim short-circuit before FineWeb/CUDA (like DESIGN_FORCE_SIM).
    if am.fixture_requested() and am.layout_requested(WORKDIR):
        _run_automodel_fixture_stub(t_start)
        return
    if am.fixture_requested() and not (
        os.environ.get(am.ENV_APPLIED_DIR) or os.environ.get(am.ENV_PIN_DIR)
    ):
        # Explicit fixture flag alone is enough for local/CI staging smoke.
        _run_automodel_fixture_stub(t_start)
        return

    dataset_url = os.environ["PRISM_DATASET_URL"]
    dataset_sha = os.environ["PRISM_DATASET_SHA256"]

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

    arch_path, train_path, automodel_manifest = _resolve_miner_paths(WORKDIR)
    if automodel_manifest and automodel_manifest.get("stub"):
        # Applied fixture tree without early ENV_FIXTURE — still not scored.
        _run_automodel_fixture_stub(t_start)
        return

    # Warm the HF cache from the parent (which has network) so the isolated
    # child can resolve the PINNED FALLBACK tokenizer offline from the same
    # cache. A submission that declares its own tokenizer (files in the
    # source tree or a `build_tokenizer(ctx)` hook — probed by string scan,
    # the parent never imports miner code) needs no network at all, so a
    # warm failure is only fatal when the fallback is the only option.
    declares_tok = tok_contract.declared(WORKDIR, arch_path, train_path)
    try:
        tok_contract.warm_default_cache()
    except Exception as exc:  # noqa: BLE001
        if not declares_tok:
            fail("tokenizer", exc)
        log(f"WARNING: default tokenizer warm failed ({exc}); submission declares its own")

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
        # Multi-GPU contract: the pod rents 4×RTX 5090 by default
        # (PRISM_POD_GPU_COUNT). Miners may use every visible GPU for
        # build/train (e.g. torch.distributed / FSDP over env:// on
        # 127.0.0.1 — the harness brings `lo` up inside the train netns).
        # The eval battery stays pinned to GPU 0 so G7 timings stay
        # comparable across submissions.
        "gpu_count": torch.cuda.device_count() if device == "cuda" else 0,
        # Transformer Engine ships in the CUDA13 pod image; miners opt into
        # TE fp8/fp4 (NVFP4) autocast themselves. The harness never forces a
        # dtype on miner build/train code.
        "te_available": importlib.util.find_spec("transformer_engine") is not None,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "probe_every": PROBE_EVERY,
        "probe_time_budget_s": PROBE_TIME_BUDGET_S,
        "workdir": WORKDIR,
        "arch_path": arch_path,
        "train_path": train_path,
        "layout": "automodel" if automodel_manifest else "legacy",
        "automodel": automodel_manifest,
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

    # Miner dependency install phase (recipe-v10): install a shipped
    # requirements.txt / pyproject.toml while the parent still has network,
    # BEFORE the netns-isolated children. A failure is miner-fixable
    # (`install_deps` class) — resubmit at will, no eligibility burned.
    try:
        installed = deps_mod.install_miner_deps(
            WORKDIR, int_env("PRISM_INSTALL_TIMEOUT_SECS", deps_mod.DEFAULT_INSTALL_TIMEOUT_S), log
        )
        if installed:
            log(f"miner deps installed ({installed}); continuing to train/eval")
    except Exception as exc:  # noqa: BLE001 — miner-attributable, routed to install_deps
        print("DEPS_INSTALL_FAIL")
        fail("install_deps", exc)

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
    if payload.get("cap_exceeded"):
        _cap_exceeded_out(payload, manifest, t_start)
        return
    if payload.get("status") != "ok":
        fail(
            str(payload.get("stage", "subprocess")),
            RuntimeError(str(payload.get("error", "unknown"))[:400]),
        )

    out = {
        "bpb": payload["bpb"],
        "bits_per_byte": payload.get("bits_per_byte"),
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
        "tokenizer": payload.get("tokenizer", {}),
    }
    _emit_metrics(out)
    print("EVAL_OK")
    log(f"eval complete in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
