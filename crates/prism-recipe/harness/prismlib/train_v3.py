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
from .dataset import load_texts
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
    stream = SeededTrainStream(
        train_texts,
        tok,
        device,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=int(cfg.get("seed", RECIPE_SEED)),
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

    train_hours_cap = float(cfg.get("train_hours_cap", 6.0))

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
        metrics = {}
        finish_reason = "finish_evaluation"
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
