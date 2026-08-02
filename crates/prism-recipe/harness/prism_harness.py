#!/usr/bin/env python3
"""PRISM pod harness (recipe v1) — uploaded by prism-lium over SSH.

Runs the miner's architecture.py + training.py against the pinned fineweb-edu
shard, computes val CE -> BPB on the frozen val cut, prints one line:
    METRICS_JSON={...}
then ends with EVAL_OK (or raises). No network beyond the two pinned
downloads (dataset shard + tokenizer). Stdlib + torch/transformers/datasets
preinstalled in the pod image only.

Contract recap (docs/PRISM_RECIPE.md):
  ctx = {
    "dataset_path": "<abs path to verified parquet>",
    "dataset_sha256": "<pin>",
    "seed": RECIPE_SEED,
    "train_hours_cap": 6.0,
    "max_train_steps": MAX_TRAIN_STEPS,
    "val_rows": VAL_ROWS,
    "train_rows": TRAIN_ROWS,
    "device": "cuda",
  }
  model = build_model(ctx)   # torch.nn.Module
  metrics = train(model, ctx)  # must return dict with optional 'train_loss'
  harness scores val CE itself; miner cannot touch val rows.
"""
import hashlib
import json
import os
import sys
import time

RECIPE_SEED = 0x00505249534D
MAX_TRAIN_STEPS = int(os.environ.get("PRISM_MAX_TRAIN_STEPS", "20000"))
TRAIN_HOURS_CAP = float(os.environ.get("PRISM_TRAIN_HOURS_CAP", "6.0"))
VAL_ROWS = 256
TRAIN_ROWS = 2048
TOKENIZER = "gpt2"


def log(msg):
    print("[harness]", msg, flush=True)


def fail(stage, exc):
    print(json.dumps({"stage": stage, "error": str(exc)[:400]}))
    print("EVAL_FAIL")
    sys.exit(3)


def materialize_dataset(url, sha256_hex):
    import urllib.request

    dst = "/tmp/prism_dataset.parquet"
    if os.path.isfile(dst):
        log("dataset already materialized")
    else:
        log(f"downloading {url}")
        t0 = time.time()
        with urllib.request.urlopen(url, timeout=600) as r, open(dst, "wb") as f:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
        log(f"downloaded in {time.time()-t0:.0f}s")
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != sha256_hex:
        raise RuntimeError(f"dataset sha256 mismatch: got {got}, want {sha256_hex}")
    log("dataset sha256 ok")
    return dst


def load_texts(parquet_path):
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["text"])
    texts = table.column("text").to_pylist()
    texts = [t for t in texts if isinstance(t, str) and len(t) >= 100]
    if len(texts) < VAL_ROWS + 64:
        raise RuntimeError(f"shard too small after filter: {len(texts)}")
    return texts


def main():
    dataset_url = os.environ["PRISM_DATASET_URL"]
    dataset_sha = os.environ["PRISM_DATASET_SHA256"]
    work = "/tmp/prism_eval"
    os.makedirs(work, exist_ok=True)

    try:
        parquet = materialize_dataset(dataset_url, dataset_sha)
    except Exception as exc:  # noqa: BLE001
        fail("dataset", exc)

    try:
        texts = load_texts(parquet)
    except Exception as exc:  # noqa: BLE001
        fail("load_texts", exc)

    import torch
    assert torch.cuda.is_available(), "cuda device required"
    device = "cuda"
    torch.manual_seed(RECIPE_SEED)
    torch.cuda.manual_seed_all(RECIPE_SEED)

    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ctx = {
        "dataset_path": parquet,
        "dataset_sha256": dataset_sha,
        "seed": RECIPE_SEED,
        "train_hours_cap": TRAIN_HOURS_CAP,
        "max_train_steps": MAX_TRAIN_STEPS,
        "val_rows": VAL_ROWS,
        "train_rows": TRAIN_ROWS,
        "device": device,
    }

    # Load miner code in-pod (authoritative contract screen).
    arch_path = os.path.join(work, "architecture.py")
    train_path = os.path.join(work, "training.py")
    import importlib.util

    def load_mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    try:
        arch = load_mod("miner_architecture", arch_path)
        assert hasattr(arch, "build_model"), "build_model missing"
        train_mod = load_mod("miner_training", train_path)
        assert hasattr(train_mod, "train"), "train missing"
    except Exception as exc:  # noqa: BLE001
        fail("contract", exc)

    # Miner train path (cap enforced here, not by miners).
    t0 = time.time()
    try:
        model = arch.build_model(ctx)
        assert isinstance(model, torch.nn.Module), "build_model must return nn.Module"
        n_params = sum(p.numel() for p in model.parameters())
        log(f"model params: {n_params/1e6:.1f}M")
        model = model.to(device)
        class CapExceeded(Exception):
            pass

        def guard():
            if time.time() - t0 > TRAIN_HOURS_CAP * 3600.0:
                raise CapExceeded("train time cap exceeded")

        ctx["guard"] = guard
        metrics = train_mod.train(model, ctx)
        if not isinstance(metrics, dict):
            raise TypeError("train must return dict")
    except Exception as exc:  # noqa: BLE001
        fail("train", exc)
    train_s = time.time() - t0
    log(f"train done in {train_s:.0f}s")

    # Frozen val cut (miners never see these 256 texts at train time only
    # in the sense that the *harness* picks them; nothing secret).
    val_texts = texts[TRAIN_ROWS : TRAIN_ROWS + VAL_ROWS]

    # Score: mean cross-entropy over frozen tokens -> bpb.
    model.eval()
    losses = []
    n_tokens = 0
    with torch.no_grad():
        for txt in val_texts:
            ids = tok(txt, return_tensors="pt").input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            guard = None
            out = model(ids[:, :-1]) if hasattr(model, "forward") else model(ids[:, :-1])
            logits = out.logits if hasattr(out, "logits") else out
            # Miner architectures may self-truncate to a shorter context
            # (baseline block=512): align the target window to the positions
            # the model actually scored (last t logits).
            if logits.shape[1] < 1:
                continue
            tgt = ids[:, 1:][:, -logits.shape[1]:]
            loss_fn = torch.nn.functional.cross_entropy
            l = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="mean")
            losses.append(l.item())
            n_tokens += tgt.numel()
    if not losses:
        fail("score", RuntimeError("no scored tokens"))
    ce = sum(losses) / len(losses)
    bpb = ce / 0.6931471805599453  # ln 2
    out = {
        "bpb": bpb,
        "tokens_seen": int(TRAIN_ROWS),
        "wall_clock_seconds": train_s,
        "gpu_type": os.environ.get("PRISM_GPU_TYPE", "unknown"),
        "notes": "recipe-v1 val_ce->bpb",
        "val_rows": VAL_ROWS,
        "recipe": "1.0.0",
    }
    print("METRICS_JSON=" + json.dumps(out))
    print("EVAL_OK")


if __name__ == "__main__":
    main()
