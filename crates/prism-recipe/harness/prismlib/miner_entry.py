"""Miner subprocess entrypoint (harness-owned; spawned by prismlib.runner).

Usage: `python3 -m prismlib.miner_entry <ctx.json>` (usually wrapped in
`unshare --net --` by the parent, so this process is born without network).

Reads the harness ctx JSON file, imports the miner's `architecture.py` /
`training.py` (the only place miner code is ever imported — never in the
parent), resolves the submitted tokenizer (`prismlib.tokenizer`), runs
build/train under the telemetry shim + G6 probes, then scores the frozen val
cut with the harness-owned loop and writes exactly one JSON result line to fd
`$PRISM_RESULT_FD` (default 3).

stdout/stderr are streamed to the parent harness log; phase transitions are
announced with `PRISM_PHASE=<build|train|score>` marker lines that drive
the parent's per-phase timeout budgets.
"""

import importlib.util
import json
import os
import sys
import time
import traceback

from . import RECIPE_SEED, TRAIN_ROWS, VAL_ROWS, tokenizer as tok_contract
from .dataset import load_texts
from .params import ParamRangeError as _ParamCapExceeded
from .params import enforce_param_range
from .probes import ProbeRunner, select_probe_texts
from .scoring import val_ce_bpb
from .stream import SeededTrainStream
from .telemetry import FinishEvaluation, build_telemetry_module, ingest_ddp_sidecar

_HARNESS_CTX_KEYS = ("arch_path", "train_path", "workdir")


def _log(msg):
    print(f"[miner_entry] {msg}", flush=True)


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


def _sanitize_train_metrics(metrics):
    out = {}
    if not isinstance(metrics, dict):
        return out
    for k, v in metrics.items():
        if len(out) >= 64:
            break
        key = str(k)[:64]
        if isinstance(v, bool):
            out[key] = v
        elif isinstance(v, (int, float)):
            out[key] = v
        elif isinstance(v, str):
            out[key] = v[:200]
    return out


class _CapExceeded(Exception):
    pass


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(cfg, st):
    import torch

    device = str(cfg.get("device", "cuda"))
    if device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda device required")
        device = "cuda"
    torch.manual_seed(RECIPE_SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(RECIPE_SEED)

    # Miner-facing telemetry shim: registered before miner code loads so a
    # top-level `import prism_telemetry` in training.py resolves.
    telemetry_mod, state = build_telemetry_module(log=_log)
    sys.modules["prism_telemetry"] = telemetry_mod

    st["stage"] = "contract"
    arch = _load_mod("miner_architecture", cfg["arch_path"])
    if not hasattr(arch, "build_model"):
        raise RuntimeError("build_model missing")
    train_mod = _load_mod("miner_training", cfg["train_path"])
    if not hasattr(train_mod, "train"):
        raise RuntimeError("train missing")

    # Submitted tokenizer (files / build_tokenizer hook / pinned default);
    # miner code is already imported, so the hook can run here.
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
    val_texts = texts[train_rows : train_rows + val_rows]
    train_texts = texts[:train_rows] + texts[train_rows + val_rows :]
    probe_texts = select_probe_texts(texts, train_rows)

    seq_len = int(cfg.get("seq_len", 512))
    batch_size = int(cfg.get("batch_size", 8))
    stream = SeededTrainStream(
        train_texts,
        tok,
        device,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=int(cfg.get("seed", RECIPE_SEED)),
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
    _log(f"model params: {n_params/1e6:.1f}M n_params={int(n_params)}")
    max_params = int(cfg.get("max_params", 1000000000))
    min_params = int(cfg.get("min_params", 0) or 0)
    enforce_param_range(n_params, min_params, max_params)
    model = model.to(device)

    train_hours_cap = float(cfg.get("train_hours_cap", 4.0))

    def guard():
        if time.time() - t0 > train_hours_cap * 3600.0:
            raise _CapExceeded("train time cap exceeded")

    ctx["guard"] = guard

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
        # Miner-signalled stop: fall through and score the model as-is.
        metrics = {}
        finish_reason = "finish_evaluation"
    if not isinstance(metrics, dict):
        raise TypeError("train must return dict")
    train_s = time.time() - t0
    _log(f"train done in {train_s:.0f}s ({finish_reason})")
    n_side = ingest_ddp_sidecar(state, cfg.get("workdir"))
    if n_side:
        _log(f"ingested {n_side} DDP telemetry reports")

    st["stage"] = "score"
    _phase("score")
    ce, bpb, val_tokens, bits_per_byte = val_ce_bpb(model, tok, val_texts, device)

    tokens_seen = int(stream.tokens_seen)
    tokens_seen_source = "train_stream"
    if tokens_seen <= 0:
        # Miner bypassed the harness train stream: keep the legacy counter
        # behavior so METRICS_JSON stays internally consistent.
        tokens_seen = int(cfg.get("train_rows", TRAIN_ROWS))
        tokens_seen_source = "legacy"

    _emit(
        {
            "status": "ok",
            "bpb": bpb,
            "ce": ce,
            "bits_per_byte": bits_per_byte,
            "val_rows": val_rows,
            "val_tokens": val_tokens,
            "n_params": int(n_params),
            "tokens_seen": tokens_seen,
            "tokens_seen_source": tokens_seen_source,
            "wall_clock_seconds": train_s,
            "finish_reason": finish_reason,
            "tokenizer": tok_spec,
            "telemetry": {
                "finish_reason": finish_reason,
                "report_count": state["reports"],
                "loss_series": state["series"],
            },
            "probe_curve": state["probe_curve"],
            "train_metrics": _sanitize_train_metrics(metrics),
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
                "floor_missed": bool(exc.under),
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
