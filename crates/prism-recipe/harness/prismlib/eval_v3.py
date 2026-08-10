"""v3 EVAL-phase child entry (harness-owned; spawned by prismlib.v3flow).

Usage: `python3 -m prismlib.eval_v3 <eval_ctx.json>` — a FRESH subprocess
(netns on), started only after the train-phase process group is dead and
private eval assets have been staged by the operator (or not —
public_dev tier).

Reimports the miner's `architecture.py` (miner code only ever runs in
the isolated child), re-resolves the submitted tokenizer through
`prismlib.tokenizer` and asserts it matches the spec recorded in the
checkpoint (fail-closed — a tokenizer that does not reconstruct identically
is never scored), rebuilds the model, loads `$PRISM_WORKDIR/
checkpoint.pt` (or the sharded index), then runs the v1 frozen-val bpb
scoring plus the full G1–G8 battery via `eval.run_battery`. The secret
generator seed arrives via env `PRISM_EVAL_SECRET_SEED` only — never via
the ctx JSON on disk — and is unset in this process immediately after
reading. Results go to fd `$PRISM_RESULT_FD` as one JSON line.
"""

import importlib.util
import json
import os
import sys
import traceback

from . import RECIPE_SEED, TRAIN_ROWS, VAL_ROWS, tokenizer as tok_contract
from .dataset import load_texts
from .scoring import val_ce_bpb
from .stream import SeededTrainStream

_HARNESS_CTX_KEYS = ("arch_path", "train_path", "workdir")


def _log(msg):
    print(f"[eval_v3] {msg}", flush=True)


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


def _load_checkpoint(model, workdir, device):
    import torch

    single = os.path.join(workdir, "checkpoint.pt")
    index = os.path.join(workdir, "checkpoint.index.json")
    if os.path.isfile(single):
        blob = torch.load(single, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
        return blob.get("meta", {})
    if os.path.isfile(index):
        with open(index, "r", encoding="utf-8") as f:
            idx = json.load(f)
        sd = {}
        for name in idx["shards"]:
            sd.update(torch.load(os.path.join(workdir, name), map_location="cpu", weights_only=True))
        model.load_state_dict(sd)
        return idx.get("meta", {})
    raise RuntimeError("no checkpoint found (checkpoint.pt / checkpoint.index.json)")


def _run(cfg, st):
    import torch

    # Secret seed: env only, never the ctx file; unset immediately.
    secret_raw = os.environ.pop("PRISM_EVAL_SECRET_SEED", None)
    secret_seed = None
    if secret_raw is not None:
        try:
            secret_seed = int(secret_raw)
        except ValueError:
            secret_seed = None

    device = str(cfg.get("device", "cuda"))
    if device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda device required")
        device = "cuda"
    torch.manual_seed(RECIPE_SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(RECIPE_SEED)

    st["stage"] = "contract"
    arch = _load_mod("miner_architecture", cfg["arch_path"])
    if not hasattr(arch, "build_model"):
        raise RuntimeError("build_model missing")

    # Same resolution path as the TRAIN child (files / build_tokenizer hook /
    # pinned default) — this fresh subprocess must reconstruct exactly the
    # tokenizer the checkpoint was trained with; verified below against the
    # spec the TRAIN child stored in the checkpoint meta.
    st["stage"] = "tokenizer"
    tok, tok_spec = tok_contract.resolve(cfg, device, arch_mod=arch, log=_log)
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

    st["stage"] = "rebuild"
    build_ctx = {k: v for k, v in cfg.items() if k not in _HARNESS_CTX_KEYS}
    build_ctx["device"] = device
    build_ctx["tokenizer"] = tok
    build_ctx["vocab_size"] = tok_spec["vocab_size"]
    model = arch.build_model(build_ctx)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("build_model must return nn.Module")
    meta = _load_checkpoint(model, cfg["workdir"], device)
    tok_check = tok_contract.assert_matches(tok_spec, (meta or {}).get("tokenizer"))
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    st["stage"] = "battery"
    _phase("eval")
    import eval as battery_pkg  # the eval/ package (harness root on PYTHONPATH)

    eval_ctx = {
        "tokenizer": tok,
        "device": device,
        "seed": int(cfg.get("seed", RECIPE_SEED)),
        "seq_len": int(cfg.get("seq_len", 512)),
        "val_texts": val_texts,
        "eval_assets_dir": cfg.get("eval_assets_dir"),
        "eval_tier": cfg.get("eval_tier", "public_dev"),
        "eval_secret_seed": secret_seed,
        "probe_curve": cfg.get("probe_curve") or [],
        "telemetry_series": cfg.get("telemetry_series") or [],
        "tokens_seen": int(cfg.get("tokens_seen", 0)),
        "n_params": n_params,
        "build_model": arch.build_model,
        "build_ctx": build_ctx,
        "micro_stream": SeededTrainStream(
            train_texts,
            tok,
            device,
            seq_len=int(cfg.get("seq_len", 512)),
            batch_size=max(1, int(cfg.get("batch_size", 8)) // 2),
            seed=int(cfg.get("seed", RECIPE_SEED)) + 7919,
        ),
    }
    from eval.common import ItemRecorder

    eval_ctx["items"] = ItemRecorder()
    battery = battery_pkg.run_battery(model, eval_ctx)
    # Composite contract view: flat canonical org.* metrics + mirror pairs
    # beside the nested per-group debug view (eval/rollup.py).
    from eval import rollup as battery_rollup

    battery = battery_rollup.rollup_battery(battery, eval_ctx, model=model)

    st["stage"] = "score"
    ce, bpb, val_tokens, bits_per_byte = val_ce_bpb(model, tok, val_texts, device)

    _emit(
        {
            "status": "ok",
            "flow": "v3",
            "phase": "eval",
            "bpb": bpb,
            "ce": ce,
            "bits_per_byte": bits_per_byte,
            "val_rows": val_rows,
            "val_tokens": val_tokens,
            "n_params": int(n_params),
            "checkpoint_meta": meta,
            "tokenizer": dict(tok_spec, cross_phase=tok_check),
            "eval_tier": cfg.get("eval_tier", "public_dev"),
            "battery": battery,
            "items": eval_ctx["items"].dump(),
            "inference_traces": eval_ctx["items"].dump_traces(),
        }
    )
    # Optional sidecar for operators who harvest artifact dirs (METRICS_JSON
    # remains the authoritative store copy on master).
    side = os.environ.get("PRISM_INFERENCE_TRACES_PATH")
    if side:
        try:
            with open(side, "w", encoding="utf-8") as f:
                json.dump(eval_ctx["items"].dump_traces(), f, ensure_ascii=False)
        except OSError:
            pass
    return 0


def main():
    st = {"stage": "ctx"}
    try:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            cfg = json.load(f)
        _phase("eval")
        return _run(cfg, st)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        try:
            _emit({"status": "fail", "stage": st["stage"], "error": str(exc)[:400]})
        except Exception:  # noqa: BLE001
            pass
        return 3


if __name__ == "__main__":
    sys.exit(main())
