"""v3 TRAIN-phase child entry (harness-owned; spawned by prismlib.v3flow).

Usage: `python3 -m prismlib.train_v3 <ctx.json>` (usually wrapped in
`unshare --net --`; no eval assets exist on disk yet — the operator
stages them only after this process group is dead).

Mirrors `prismlib.miner_entry` for build/train (telemetry shim, submitted
tokenizer, G6 probes, seeded train stream, per-phase markers) but, per the
v3 architecture, does **not** score: it ends by saving the trained
`state_dict` + build metadata (including the resolved tokenizer spec, which
`eval_v3` re-checks) to `$PRISM_WORKDIR/checkpoint.pt`
(sharded into `checkpoint.shardNN.pt` + `checkpoint.index.json` when a
single file would exceed ~1.5 GiB) and reporting train telemetry. The
v1 bpb scoring happens in the EVAL-phase child (`eval_v3`) — the trained
weights never cross to the parent process.
"""

import importlib.util
import json
import os
import sys
import time
import traceback

from . import RECIPE_SEED, TRAIN_ROWS, VAL_ROWS, tokenizer as tok_contract
from . import flops as flops_mod
from .dataset import load_texts
from .flops import BudgetExhausted
from .probes import ProbeRunner, select_probe_texts
from .stream import SeededTrainStream
from .telemetry import FinishEvaluation, build_telemetry_module

_HARNESS_CTX_KEYS = ("arch_path", "train_path", "workdir")
_SHARD_BYTES = 1_500_000_000


def _log(msg):
    print(f"[train_v3] {msg}", flush=True)


def _phase(name):
    print(f"PRISM_PHASE={name}", flush=True)


def _result_fd():
    try:
        return int(os.environ.get("PRISM_RESULT_FD", "3"))
    except ValueError:
        return 3


def _emit(payload):
    line = json.dumps(payload, separators=(",", ":"), default=str)
    fd = os.dup(_result_fd())
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _CapExceeded(Exception):
    pass


class _ParamCapExceeded(Exception):
    """Miner-attributable parameter-cap breach (n_params > max_params).

    Carries the measured count so the fail payload can flag it
    machine-readably (`cap_exceeded`) for the parent's terminal path."""

    def __init__(self, n_params, max_params):
        super().__init__(f"model exceeds parameter cap: {n_params} > {max_params}")
        self.n_params = n_params


def _save_checkpoint(model, workdir, meta):
    """state_dict (CPU tensors) + build metadata; sharded when large."""
    import torch

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    total = sum(v.numel() * v.element_size() for v in sd.values())
    if total <= _SHARD_BYTES:
        path = os.path.join(workdir, "checkpoint.pt")
        torch.save({"state_dict": sd, "meta": meta}, path)
        return {"path": path, "bytes": total, "shards": []}

    shards, cur, cur_bytes, idx = [], {}, 0, 0
    for k, v in sd.items():
        b = v.numel() * v.element_size()
        if cur and cur_bytes + b > _SHARD_BYTES:
            shards.append(cur)
            cur, cur_bytes = {}, 0
        cur[k] = v
        cur_bytes += b
    if cur:
        shards.append(cur)
    names = []
    for i, shard in enumerate(shards):
        name = f"checkpoint.shard{i:02d}.pt"
        torch.save(shard, os.path.join(workdir, name))
        names.append(name)
    index = {"shards": names, "meta": meta, "bytes": total}
    with open(os.path.join(workdir, "checkpoint.index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    return {"path": os.path.join(workdir, "checkpoint.index.json"), "bytes": total, "shards": names}


def _attest_flops(model, stream, cfg, seq_len, flops_cap):
    """Probe FLOPs/token, cross-check it analytically, arm the stream cap.

    Never fatal. A probe that cannot run (no `FlopCounterMode`, an
    architecture the harness-driven fwd/bwd cannot drive) leaves the FLOPs
    cap disarmed and records `flops_probe_error` — the wall-clock bound
    still contains the run, and the failure is visible rather than silently
    treated as a zero-cost model.
    """
    out = {"flops_per_token": 0.0, "cv": 0.0, "unstable": False}
    try:
        probe = flops_mod.probe_flops_per_token(
            model,
            stream,
            # None ⇒ prismlib.flops resolves the probe secret itself. The
            # eval secret is staged only AFTER train, so on a real pod this
            # is a fresh urandom draw the miner cannot predict.
            cfg.get("flops_probe_secret"),
            n=int(cfg.get("flops_probe_samples", flops_mod.FLOPS_PROBE_SAMPLES)),
            log=_log,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"flops probe unavailable ({exc}); FLOPs cap disarmed, wall cap stands")
        out["error"] = str(exc)[:200]
        return out
    out.update(probe)
    # The secret never leaves the train child: it is unpredictable-in-advance
    # rather than cryptographically hidden, and echoing it into the emitted
    # payload would publish the index draw for the next run to imitate.
    out.pop("secret", None)
    try:
        out["cross_check"] = flops_mod.cross_check(
            probe["flops_per_token"],
            model,
            seq_len,
            gap_max=float(cfg.get("flops_analytic_gap_max", flops_mod.FLOPS_ANALYTIC_GAP_MAX)),
        )
        cc = out["cross_check"]
        _log(
            f"flops cross-check: analytic={cc['analytic_flops_per_token']:.4g} "
            f"ratio={cc['analytic_ratio']:.3f} gap={cc['analytic_gap']:.3f} "
            f"mismatch={cc['mismatch']}"
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"flops cross-check failed (ignored): {exc}")
    if flops_cap > 0.0 and probe["flops_per_token"] > 0.0:
        stream.set_flops_per_token(probe["flops_per_token"])
        budget_tokens = flops_cap / probe["flops_per_token"]
        _log(f"budget: {flops_cap:.3g} FLOPs ≈ {budget_tokens/1e9:.2f}B tokens")
    return out


def _diag_metrics(stream, attest, wall_s):
    """`org.diag.*` telemetry: observed-only, absent from every anchor set.

    These are deliberately NOT in an anchor set. A metric in the anchor set
    but missing from metrics.json is a hard completeness failure, so new
    keys must be emitted first and declared later. Emitting them now is what
    lets v3 anchors be calibrated on measured distributions.
    """
    rep = stream.budget_report()
    n_gpu = flops_mod.n_gpus_visible()
    attested = rep["flops_attested"]
    ok, ceiling = flops_mod.physically_possible(attested, wall_s, n_gpu=n_gpu)
    out = {
        "org.diag.flops_attested": attested,
        "org.diag.flops_per_token_probe": rep["flops_per_token"],
        "org.diag.flops_probe_cv": float(attest.get("cv", 0.0)),
        "org.diag.flops_probe_unstable": 1.0 if attest.get("unstable") else 0.0,
        "org.diag.flops_probe_samples": float(attest.get("n_samples", 0)),
        # A reduced-batch probe is a valid per-token measurement but a
        # DIFFERENT measurement condition, so it is visible, not implicit.
        "org.diag.flops_probe_rows": float(attest.get("probe_rows", 0)),
        "org.diag.flops_probe_rows_reduced": 1.0 if attest.get("probe_rows_reduced") else 0.0,
        "org.diag.spend_fraction": rep["spend_fraction"],
        "org.diag.binding_cap": rep["binding_cap"],
        "org.diag.mfu_achieved": flops_mod.mfu(attested, wall_s, n_gpu=n_gpu),
        "org.diag.n_gpu_attested": float(n_gpu),
        "org.diag.flops_physically_possible": 1.0 if ok else 0.0,
        "org.diag.flops_physical_ceiling": ceiling,
        "org.diag.tokenizer_bytes_per_token": rep["bytes_per_token"],
        "org.diag.bytes_seen": float(rep["bytes_seen"]),
    }
    if attest.get("error"):
        out["org.diag.flops_probe_error"] = 1.0
    cc = attest.get("cross_check")
    if cc:
        out["org.diag.flops_analytic_ratio"] = cc["analytic_ratio"]
        out["org.diag.flops_analytic_gap"] = cc["analytic_gap"]
        out["org.diag.flops_analytic_mismatch"] = 1.0 if cc["mismatch"] else 0.0
        bd = cc["breakdown"]
        out["org.diag.n_params_body"] = float(bd["n_params_body"])
        out["org.diag.n_params_embed"] = float(bd["n_params_embed"])
        out["org.diag.effective_flops_per_token_ratio"] = bd["r_eff"]
    return out


def _run(cfg, st):
    import torch

    device = str(cfg.get("device", "cuda"))
    if device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda device required")
        device = "cuda"
    # Seed from ctx, not the constant: the parent may be running the
    # operator seed-variance sweep (PRISM_SEED_OVERRIDE), and a child that
    # re-seeded from the lattice constant would silently defeat it — every
    # "different seed" run would train identically.
    train_seed = int(cfg.get("seed", RECIPE_SEED))
    torch.manual_seed(train_seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(train_seed)

    telemetry_mod, state = build_telemetry_module(log=_log)
    sys.modules["prism_telemetry"] = telemetry_mod

    st["stage"] = "contract"
    arch = _load_mod("miner_architecture", cfg["arch_path"])
    if not hasattr(arch, "build_model"):
        raise RuntimeError("build_model missing")
    train_mod = _load_mod("miner_training", cfg["train_path"])
    if not hasattr(train_mod, "train"):
        raise RuntimeError("train missing")

    # Submitted tokenizer; the spec goes into the checkpoint meta so the
    # EVAL child can prove it rebuilt the very same tokenizer.
    st["stage"] = "tokenizer"
    tok, tok_spec = tok_contract.resolve(
        cfg, device, arch_mod=arch, train_mod=train_mod, telemetry=telemetry_mod, log=_log
    )
    _log(
        f"tokenizer: source={tok_spec['source']} vocab={tok_spec['vocab_size']} "
        f"fp={tok_spec['fingerprint'][:16]}"
    )

    st["stage"] = "dataset"
    texts = load_texts(cfg["dataset_path"])
    train_rows = int(cfg.get("train_rows", TRAIN_ROWS))
    val_rows = int(cfg.get("val_rows", VAL_ROWS))
    train_texts = texts[:train_rows] + texts[train_rows + val_rows :]
    probe_texts = select_probe_texts(texts, train_rows)

    seq_len = int(cfg.get("seq_len", 512))
    batch_size = int(cfg.get("batch_size", 8))
    train_hours_cap = float(cfg.get("train_hours_cap", 5.0))
    flops_cap = float(cfg.get("train_flops_cap", 0.0) or 0.0)
    stream = SeededTrainStream(
        train_texts,
        tok,
        device,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=int(cfg.get("seed", RECIPE_SEED)),
        flops_cap=flops_cap,
        wall_cap_s=train_hours_cap * 3600.0,
        steps_cap=int(cfg.get("max_train_steps", 20000)),
    )

    ctx = {k: v for k, v in cfg.items() if k not in _HARNESS_CTX_KEYS}
    ctx["device"] = device
    ctx["telemetry"] = telemetry_mod
    ctx["tokenizer"] = tok
    ctx["vocab_size"] = tok_spec["vocab_size"]
    ctx["train_stream"] = stream

    st["stage"] = "build"
    t0 = time.time()
    model = arch.build_model(ctx)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("build_model must return nn.Module")
    n_params = sum(p.numel() for p in model.parameters())
    _log(f"model params: {n_params/1e6:.1f}M")
    max_params = int(cfg.get("max_params", 1000000000))
    if n_params > max_params:
        # Product hard cap: fail before CUDA / train (machine-readable).
        raise _ParamCapExceeded(n_params, max_params)
    model = model.to(device)

    # ------------------------------------------------- FLOPs attestation
    # Established BEFORE training so the budget is enforced from the first
    # batch. Harness-driven forward+backward under FlopCounterMode on
    # batches at secret stream indices; the miner reports nothing.
    st["stage"] = "flops_probe"
    stream.wall_cap_s = train_hours_cap * 3600.0
    stream._t0 = t0  # noqa: SLF001 — same module family; wall clock is harness-owned
    attest = _attest_flops(model, stream, cfg, seq_len, flops_cap)

    # The dual cap is enforced inside the stream (`next_batch` refuses to
    # yield past a cap). `guard` stays for backward compatibility with
    # submissions that call it, and now reports the same verdict rather
    # than a second, independent clock.
    def guard():
        stream._check_budget()  # noqa: SLF001 — harness-owned enforcement point

    ctx["guard"] = guard
    ctx["train_flops_cap"] = float(flops_cap)
    ctx["flops_per_token_probe"] = float(attest["flops_per_token"])
    ctx["train_hours_cap"] = train_hours_cap

    probes = ProbeRunner(
        model=model,
        stream=stream,
        tok=tok,
        texts=probe_texts,
        device=device,
        seq_len=seq_len,
        every=int(cfg.get("probe_every", 25)),
        time_budget_s=float(cfg.get("probe_time_budget_s", 600.0)),
        log=_log,
    )
    state["probe_hook"] = probes.maybe_probe

    st["stage"] = "train"
    _phase("train")
    finish_reason = "train_returned"
    try:
        metrics = train_mod.train(model, ctx)
    except FinishEvaluation:
        metrics = {}
        finish_reason = "finish_evaluation"
    except BudgetExhausted as exc:
        # Reaching your own budget is the EXPECTED outcome, so it routes to
        # the same graceful path as finish_evaluation(): checkpoint, then
        # eval. The predecessor (_CapExceeded) was caught by nothing and
        # failed the whole run, which made spending the full budget a way to
        # score zero.
        metrics = {}
        finish_reason = f"budget_{exc.cap}"
        _log(f"budget exhausted ({exc.cap}): spent={exc.spent} limit={exc.limit}")
    if not isinstance(metrics, dict):
        raise TypeError("train must return dict")
    train_s = time.time() - t0
    _log(f"train done in {train_s:.0f}s ({finish_reason})")

    st["stage"] = "checkpoint"
    _phase("checkpoint")
    meta = {
        "n_params": int(n_params),
        "seq_len": seq_len,
        "batch_size": batch_size,
        "device": device,
        "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finish_reason": finish_reason,
        "torch": torch.__version__,
        "tokenizer": tok_spec,
    }
    ckpt = _save_checkpoint(model, cfg["workdir"], meta)
    _log(f"checkpoint saved: {ckpt['path']} ({ckpt['bytes']/1e6:.0f} MB)")

    tokens_seen = int(stream.tokens_seen)
    tokens_seen_source = "train_stream"
    if tokens_seen <= 0:
        tokens_seen = int(cfg.get("train_rows", TRAIN_ROWS))
        tokens_seen_source = "legacy"

    budget = stream.budget_report()
    diag = _diag_metrics(stream, attest, train_s)
    _log(
        f"budget: attested={budget['flops_attested']:.4g} FLOPs "
        f"({budget['spend_fraction']*100:.1f}% of cap) bound_by={budget['binding_cap']} "
        f"mfu={diag['org.diag.mfu_achieved']*100:.1f}%"
    )

    _emit(
        {
            "status": "ok",
            "flow": "v3",
            "phase": "train",
            "n_params": int(n_params),
            "tokens_seen": tokens_seen,
            "tokens_seen_source": tokens_seen_source,
            "wall_clock_seconds": train_s,
            "finish_reason": finish_reason,
            "budget": budget,
            "diag_metrics": diag,
            "tokenizer": tok_spec,
            "checkpoint": ckpt,
            "telemetry": {
                "finish_reason": finish_reason,
                "report_count": state["reports"],
                "loss_series": state["series"],
            },
            "probe_curve": state["probe_curve"],
            "train_metrics": {
                str(k)[:64]: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))
            },
            "seq_len": seq_len,
            "batch_size": batch_size,
        }
    )
    return 0


def main():
    st = {"stage": "ctx"}
    try:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            cfg = json.load(f)
        _phase("build")
        return _run(cfg, st)
    except FinishEvaluation:
        _emit(
            {
                "status": "fail",
                "stage": st["stage"],
                "error": "finish_evaluation raised outside train()",
            }
        )
        return 3
    except _ParamCapExceeded as exc:
        _emit(
            {
                "status": "fail",
                "stage": st["stage"],
                "error": str(exc)[:400],
                "cap_exceeded": True,
                "n_params": int(exc.n_params),
            }
        )
        return 3
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        try:
            _emit({"status": "fail", "stage": st["stage"], "error": str(exc)[:400]})
        except Exception:  # noqa: BLE001
            pass
        return 3


if __name__ == "__main__":
    sys.exit(main())
