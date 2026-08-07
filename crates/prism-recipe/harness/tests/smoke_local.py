#!/usr/bin/env python3
"""Local end-to-end smoke for the PRISM harness package (CPU-friendly).

Generates a tiny parquet fixture itself, writes a stub miner
architecture.py + training.py, then runs `main.py` as a subprocess with:
  PRISM_TEST_TRAIN_MINUTES=2  PRISM_TEST_MAX_PARAMS=2000000
  PRISM_ALLOW_CPU=1           PRISM_PROBE_EVERY=2
  PRISM_SEQ_LEN=64            PRISM_TRAIN_BATCH_SIZE=2

Asserts the METRICS_JSON v2 contract: v1 keys present, metrics_version==2,
tokens_seen_source=="train_stream", probe_curve non-empty, pod_manifest +
netns + harness_files_sha256 recorded.

Self-skips (exit 2) when torch/transformers are not importable — e.g. CI
boxes without the pod image deps. Requires pyarrow for the fixture.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]

STUB_ARCH = '''
import torch
import torch.nn as nn


class Tiny(nn.Module):
    def __init__(self, vocab=50257, d=16, block=64):
        super().__init__()
        self.block = block
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block, d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids):
        b, t = ids.shape
        t = min(t, self.block)
        ids = ids[:, -t:]
        pos = torch.arange(t, device=ids.device)
        x = self.emb(ids) + self.pos(pos)[None, :, :]
        return self.head(x)


def build_model(ctx):
    torch.manual_seed(int(ctx.get("seed", 0)))
    return Tiny()
'''

STUB_TRAIN = '''
import torch

try:
    import prism_telemetry
except ImportError:
    prism_telemetry = None


def train(model, ctx):
    torch.manual_seed(int(ctx["seed"]))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    stream = ctx["train_stream"]
    model.train()
    steps = 0
    last = 0.0
    for input_ids, labels in stream:
        out = model(input_ids)
        logits = out.logits if hasattr(out, "logits") else out
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.item())
        steps += 1
        if prism_telemetry is not None:
            prism_telemetry.report(loss=last, step=steps)
        if steps >= 8:
            break
    return {"train_loss": last, "train_steps": steps}
'''

STEPS = 8
SEQ_LEN = 64
BATCH_SIZE = 2
# Test-mode row contract: the 400-row fixture must cover train+val+probes.
TEST_TRAIN_ROWS = 256
TEST_VAL_ROWS = 64


def _make_fixture(path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = hashlib.sha256(b"prism-smoke").digest()
    seed = int.from_bytes(rng[:8], "big")
    import random

    r = random.Random(seed)
    words = (
        "the quick brown fox jumps over a lazy dog while researchers train "
        "small language models on pinned datasets and measure bits per byte "
        "with a frozen validation cut and deterministic seeds"
    ).split()
    texts = []
    for _ in range(400):
        n = r.randint(30, 90)
        texts.append(" ".join(r.choice(words) for _ in range(n)))
    pq.write_table(pa.table({"text": texts}), path)


def main():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        print(f"SMOKE SKIP: {exc} (install torch+transformers to run the full smoke)")
        return 2
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        print(f"SMOKE SKIP: {exc} (pyarrow needed for the fixture)")
        return 2

    with tempfile.TemporaryDirectory(prefix="prism_smoke_") as tmp:
        tmp = Path(tmp)
        fixture = tmp / "fixture.parquet"
        _make_fixture(fixture)
        sha = hashlib.sha256(fixture.read_bytes()).hexdigest()

        work = tmp / "work"
        work.mkdir()
        (work / "architecture.py").write_text(STUB_ARCH)
        (work / "training.py").write_text(STUB_TRAIN)

        env = dict(os.environ)
        env.update(
            {
                "PRISM_DATASET_URL": f"file://{fixture}",
                "PRISM_DATASET_SHA256": sha,
                "PRISM_DATASET_PATH": str(tmp / "dataset.parquet"),
                "PRISM_WORKDIR": str(work),
                "PRISM_TEST_TRAIN_MINUTES": "2",
                "PRISM_TEST_MAX_PARAMS": "2000000",
                # The 400-row fixture cannot cover the production 2048+256
                # slice; test-mode row overrides shrink the contract cut.
                "PRISM_TEST_TRAIN_ROWS": str(TEST_TRAIN_ROWS),
                "PRISM_TEST_VAL_ROWS": str(TEST_VAL_ROWS),
                "PRISM_ALLOW_CPU": "1",
                "PRISM_PROBE_EVERY": "2",
                "PRISM_SEQ_LEN": str(SEQ_LEN),
                "PRISM_TRAIN_BATCH_SIZE": str(BATCH_SIZE),
                "PRISM_GPU_TYPE": "smoke-cpu",
            }
        )
        proc = subprocess.run(
            [sys.executable, "main.py"],
            cwd=HARNESS_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            print(f"SMOKE FAIL: main.py rc={proc.returncode}")
            return 1

        line = next(
            (l for l in proc.stdout.splitlines() if l.startswith("METRICS_JSON=")),
            None,
        )
        assert line is not None, "no METRICS_JSON line"
        m = json.loads(line[len("METRICS_JSON=") :])
        assert "EVAL_OK" in proc.stdout

        # v1 keys preserved.
        for key in (
            "bpb",
            "tokens_seen",
            "wall_clock_seconds",
            "gpu_type",
            "notes",
            "val_rows",
            "n_params",
            "recipe",
            "telemetry",
        ):
            assert key in m, f"missing v1 key {key}"
        assert m["recipe"] == "1.4.0"
        assert m["val_rows"] == TEST_VAL_ROWS

        # v2 keys.
        assert m["metrics_version"] == 2
        assert m["tokens_seen"] == STEPS * BATCH_SIZE * SEQ_LEN, m["tokens_seen"]
        assert m["tokens_seen_source"] == "train_stream"
        assert isinstance(m["netns"], bool)
        assert len(m["harness_files_sha256"]) == 64
        curve = m["probe_curve"]
        assert len(curve) >= 3, f"probe_curve too short: {curve}"
        for pt in curve:
            assert set(pt) == {"step", "tokens_seen", "wall_s", "probe_loss"}
            assert pt["probe_loss"] > 0
        assert curve[-1]["tokens_seen"] == STEPS * BATCH_SIZE * SEQ_LEN
        pm = m["pod_manifest"]
        assert pm["netns"] == m["netns"]
        assert "unshare" in pm and "available" in pm["unshare"]
        assert pm["python"]
        assert m["telemetry"]["report_count"] == STEPS
        assert m["telemetry"]["finish_reason"] == "train_returned"
        assert m["train_metrics"]["train_steps"] == STEPS

        print("SMOKE OK: METRICS_JSON v2 verified")
        print(json.dumps({k: m[k] for k in ("bpb", "tokens_seen", "tokens_seen_source", "netns", "metrics_version")}, indent=2))
        return 0


if __name__ == "__main__":
    sys.exit(main())
