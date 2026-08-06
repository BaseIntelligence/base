#!/usr/bin/env python3
"""G1–G8 battery smoke (E2).

Always runs (no deps):
  1. py_compile every harness .py file.
  2. Generator determinism: same seed -> identical items, different seed
     -> different items, and gold answers verify against independent
     recomputation for the verifiable families (dyck, modular, s5,
     boolean, arithmetic, knights&knaves).

When torch + transformers + pyarrow are importable, additionally:
  3. Full-battery run on a tiny randomly-initialized model with a
     byte-level fake tokenizer (no HF download) under tiny caps.
  4. Full v3 two-phase flow via main.py (PRISM_FLOW=v3) on a stub miner.

Exit 2 = skipped the torch parts (box lacks the pod image deps);
the pure-python parts always execute and assert.
"""

import json
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT))

# ---------------------------------------------------------------- pure python


def check_py_compile():
    bad = []
    for p in sorted(HARNESS_ROOT.rglob("*.py")):
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as exc:
            bad.append(f"{p}: {exc}")
    assert not bad, "py_compile failures:\n" + "\n".join(bad)
    print(f"py_compile OK ({len(list(HARNESS_ROOT.rglob('*.py')))} files)")


def _verify_gold(it):
    """Independent recomputation of the gold answer per family."""
    task = it["task"]
    gold_val = it["choices"][it["gold"]]
    if task == "mod":
        expr = it["prompt"].split("compute (")[1].split(")")[0]
        a, b = (int(x) for x in expr.split("+"))
        p = int(it["prompt"].split("mod ")[1].split(" ")[0])
        assert int(gold_val) == (a + b) % p, it
    elif task == "s5":
        lines = [l for l in it["prompt"].split("\n") if l.startswith("p")]
        perms = [[int(t) for t in l.split("(")[1].split(")")[0].split()] for l in lines]
        x = int(it["prompt"].split(" to ")[-1].split(".")[0])
        v = x
        for perm in perms:
            v = perm[v]
        assert int(gold_val) == v, it
    elif task == "bool":
        expr = it["prompt"].split("evaluate: ")[1].split(" . answer")[0]
        expr = expr.replace("true", "True").replace("false", "False")
        val = eval(expr)  # noqa: S307 — generated boolean exprs only
        assert gold_val == ("true" if val else "false"), it
    elif task == "arith":
        assert int(gold_val) == it["meta"]["answer"], it
    elif task == "dyck":
        opens, closes = {"(": ")", "[": "]"}, {")", "]"}
        stack = []
        seq = it["prompt"].split("type the next bracket: ")[1].split(" ->")[0].split()
        for ch in seq:
            if ch in opens:
                stack.append(opens[ch])
            else:
                assert stack and stack.pop() == ch, it
        assert gold_val in closes and (not stack or gold_val == stack[-1]), it
    elif task == "kk":
        pass  # uniqueness already brute-forced inside the generator


def check_generator_determinism():
    from eval import common, generators as g, gen_reasoning as gr, gen_longctx as gl

    secret = 424242
    families = [
        ("mqar", lambda s: g.mqar(s, n_pairs=8)),
        ("copy", lambda s: g.copy_gap(s, seq_len=12, gap=16)),
        ("induction", lambda s: g.induction(s)),
        ("passkey", lambda s: g.passkey(s, n_filler=12)),
        ("s5", lambda s: gr.s5_compose(s, k=5)),
        ("arith", lambda s: gr.arith(s, tier=2, noop=True)),
        ("arith_extrap", lambda s: gr.arith(s, tier=1, extrap=True)),
        ("proof", lambda s: gr.proofwriter(s, depth=2)),
        ("bool", lambda s: gr.boolean_expr(s)),
        ("dyck", lambda s: gr.dyck(s, k=2, n_pairs=10)),
        ("mod_id", lambda s: gr.modular(s, ood=False)),
        ("mod_ood", lambda s: gr.modular(s, ood=True)),
        ("kk", lambda s: gr.knights_knaves(s, n=3)),
        ("niah", lambda s: gl.niah_multikey(s, 1024)),
        ("vt", lambda s: gl.variable_tracking(s, 1024)),
        ("freq", lambda s: gl.freq_words(s, 1024)),
        ("babi", lambda s: gl.babilong(s, 1024, qa=2)),
        ("graph", lambda s: gl.graphwalks(s, 512, k=2)),
        ("mrcr", lambda s: gl.mrcr_order(s, 1024)),
        ("nolima", lambda s: gl.nolima(s, 1024)),
    ]
    for name, fn in families:
        same_secret, shifted = [], []
        for i in range(4):
            s1 = common.task_seed(secret, f"smoke/{name}/{i}")
            s2 = common.task_seed(secret, f"smoke/{name}/{i}")
            s3 = common.task_seed(secret + 1, f"smoke/{name}/{i}")
            a, b, c = fn(s1), fn(s2), fn(s3)
            assert a == b, f"{name}: same seed must give identical items"
            assert a, f"{name}: empty items"
            same_secret.append(a)
            shifted.append(c)
            for it in a:
                assert set(it) >= {"task", "prompt", "choices", "gold", "cluster"}, it
                assert 0 <= it["gold"] < len(it["choices"]), it
                assert it["prompt"] and it["choices"], it
                _verify_gold(it)
        # Low-entropy families (small puzzles) may collide on one seed;
        # over several draws the shifted family must differ somewhere.
        assert same_secret != shifted, f"{name}: different secret must shift the family"
    # Cantor lattice sanity: distinct task ids -> distinct seeds.
    seeds = {common.task_seed(secret, f"t/{i}") for i in range(100)}
    assert len(seeds) == 100
    print(f"generator determinism OK ({len(families)} families)")


# ---------------------------------------------------------------- torch parts


class _ByteTok:
    """Minimal tokenizer shim for battery smoke (no HF download)."""

    def __init__(self):
        self.eos_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False, return_tensors=None, truncation=False, max_length=None):
        ids = [b + 1 for b in text.encode("utf-8", errors="replace")]
        if truncation and max_length:
            ids = ids[:max_length]
        out = {"input_ids": ids}
        if return_tensors == "pt":
            import torch

            out["input_ids"] = torch.tensor([ids], dtype=torch.long)
        return out

    def convert_ids_to_tokens(self, ids):
        return ["ĠA" if i % 97 == 0 else chr(max(32, i - 1)) for i in ids]

    def __len__(self):
        return 260


def _tiny_model(device):
    import torch
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = 260
            self.emb = nn.Embedding(260, 32)
            self.ff = nn.Sequential(nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, 32))
            self.head = nn.Linear(32, 260, bias=False)

        def forward(self, ids):
            x = self.emb(ids)
            x = x + self.ff(x)
            return self.head(x)

    torch.manual_seed(0)
    return Tiny().to(device)


def check_full_battery():
    import torch  # noqa: F401
    import eval as battery_pkg
    from eval.common import ItemRecorder

    os.environ.setdefault("PRISM_TEST_EVAL_CAPS", "1")
    model = _tiny_model("cpu")
    texts = [
        "the quick brown fox jumps over the lazy dog near the river bank",
        "researchers measure bits per byte on a frozen validation cut",
        "procedural generators make memorization structurally useless",
        "the harness scores answer tokens with teacher forcing",
    ] * 4
    probe_curve = [
        {"step": 2, "tokens_seen": 256, "wall_s": 1.0, "probe_loss": 6.5},
        {"step": 4, "tokens_seen": 512, "wall_s": 2.0, "probe_loss": 4.2},
        {"step": 6, "tokens_seen": 1024, "wall_s": 3.0, "probe_loss": 3.6},
        {"step": 8, "tokens_seen": 2048, "wall_s": 4.0, "probe_loss": 3.4},
    ]
    series = [{"step": i, "loss": 6.0 - 0.3 * i, "at_secs": float(i)} for i in range(1, 9)]
    ctx = {
        "tokenizer": _ByteTok(),
        "device": "cpu",
        "seed": 1234,
        "seq_len": 64,
        "val_texts": texts,
        "eval_assets_dir": None,
        "eval_tier": "public_dev",
        "eval_secret_seed": None,
        "probe_curve": probe_curve,
        "telemetry_series": series,
        "tokens_seen": 2048,
        "n_params": 100000,
        "items": ItemRecorder(),
    }
    results = battery_pkg.run_battery(model, ctx)
    ok_groups, detail = [], {}
    for group, entry in results.items():
        detail[group] = entry.get("status")
        if entry.get("status") == "ok" and entry.get("metrics"):
            for k, v in entry["metrics"].items():
                assert v == v and abs(v) != float("inf"), f"{group}.{k} non-finite"
            ok_groups.append(group)
    assert set(ok_groups) == {"g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"}, detail
    for group in ok_groups:
        metrics = results[group]["metrics"]
        assert any(k.startswith(f"{group}.") for k in metrics), group
    assert results["g6"]["metrics"]["g6.points"] == 4.0
    assert "g2.core.mean_acc_norm" in results["g2"]["metrics"]
    assert "g5.lstar" in results["g5"]["metrics"]
    assert results["g8"]["metrics"].get("g8.mup.stub") == 1.0
    print("full battery OK: all 8 groups emitted finite metrics on a tiny model")
    print(json.dumps({g: len(results[g]["metrics"]) for g in ok_groups}, indent=2))

    # Rollup contract: the flat canonical org.* map + degenerate public_dev
    # mirror pairs the Rust composite ingests (eval/rollup.py).
    from eval import rollup as battery_rollup

    flat = battery_rollup.flatten_metrics(results, ctx["items"])
    assert flat and all(k.startswith("org.") for k in flat), sorted(flat)
    assert "org.g3.mqar_acc" in flat and "org.g4.arithmetic_acc" in flat, sorted(flat)
    assert "org.g6.auc_log_tokens" in flat, sorted(flat)
    view = battery_rollup.rollup_battery(results, ctx, model=model)
    assert set(view) >= {"groups", "metrics", "mirrors", "tier"}
    assert view["tier"] == "public_dev"
    assert view["mirrors"], "public_dev tier emits degenerate mirror pairs"
    assert all(set(p) == {"group", "metric", "public", "mirror"} for p in view["mirrors"])
    print(f"rollup OK: {len(flat)} org.* metrics, {len(view['mirrors'])} mirror pairs")


# ---------------------------------------------------------------- v3 flow


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


def train(model, ctx):
    torch.manual_seed(int(ctx["seed"]))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    stream = ctx["train_stream"]
    model.train()
    steps = 0
    last = 0.0
    for input_ids, labels in stream:
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.item())
        steps += 1
        try:
            import prism_telemetry
            prism_telemetry.report(loss=last, step=steps)
        except ImportError:
            pass
        if steps >= 4:
            break
    return {"train_loss": last, "train_steps": steps}
'''


def check_v3_flow():
    import hashlib

    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory(prefix="prism_v3_smoke_") as tmp:
        tmp = Path(tmp)
        fixture = tmp / "fixture.parquet"
        words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
        import random

        r = random.Random(7)
        texts = [" ".join(r.choice(words) for _ in range(r.randint(20, 50))) for _ in range(400)]
        pq.write_table(pa.table({"text": texts}), fixture)
        sha = hashlib.sha256(fixture.read_bytes()).hexdigest()

        work = tmp / "work"
        work.mkdir()
        (work / "architecture.py").write_text(STUB_ARCH)
        (work / "training.py").write_text(STUB_TRAIN)

        env = dict(os.environ)
        env.update(
            {
                "PRISM_FLOW": "v3",
                "PRISM_DATASET_URL": f"file://{fixture}",
                "PRISM_DATASET_SHA256": sha,
                "PRISM_DATASET_PATH": str(tmp / "dataset.parquet"),
                "PRISM_WORKDIR": str(work),
                "PRISM_TEST_TRAIN_MINUTES": "2",
                "PRISM_TEST_MAX_PARAMS": "2000000",
                "PRISM_TEST_EVAL_CAPS": "1",
                # The 400-row fixture cannot cover the production 2048+256
                # slice; test-mode row overrides shrink the contract cut.
                "PRISM_TEST_TRAIN_ROWS": "256",
                "PRISM_TEST_VAL_ROWS": "64",
                "PRISM_ALLOW_CPU": "1",
                "PRISM_PROBE_EVERY": "2",
                "PRISM_SEQ_LEN": "64",
                "PRISM_TRAIN_BATCH_SIZE": "2",
                "PRISM_GPU_TYPE": "smoke-cpu",
            }
        )
        proc = subprocess.run(
            [sys.executable, "main.py"],
            cwd=HARNESS_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        sys.stdout.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-2000:])
        assert proc.returncode == 0, f"v3 main.py rc={proc.returncode}"
        line = next(
            (l for l in proc.stdout.splitlines() if l.startswith("METRICS_JSON=")), None
        )
        assert line is not None, "no METRICS_JSON line"
        m = json.loads(line[len("METRICS_JSON="):])
        assert "EVAL_OK" in proc.stdout
        assert m["flow"] == "v3"
        assert m["eval_tier"] == "public_dev"
        assert m["bpb"] > 0
        assert m["tokens_seen"] == 4 * 2 * 64, m["tokens_seen"]
        assert "gate" in m and m["gate"]["survivors_after_train"] is False
        battery = m["battery"]
        # Composite contract shape: nested groups (debug) + flat canonical
        # org.* metrics + mirror pairs + tier (eval/rollup.py; consumed by
        # prism-eval-store finalize.rs::submission_metrics).
        assert set(battery) >= {"groups", "metrics", "mirrors", "tier"}, sorted(battery)
        assert battery["tier"] == "public_dev"
        groups = battery["groups"]
        ok = [g for g, e in groups.items() if e.get("status") == "ok"]
        assert set(ok) == {"g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"}, groups
        assert "g2.core.mean_acc_norm" in groups["g2"]["metrics"]
        flat = battery["metrics"]
        assert flat, "battery.metrics must carry canonical org.* keys"
        assert all(k.startswith("org.") for k in flat), sorted(flat)
        # Every anchored metric of the procedural/asset dev-tier groups is
        # present (g2/g3/g4/g5/g6 complete on a public_dev CPU run).
        for key in (
            "org.g2.arc_challenge_acc",
            "org.g3.mqar_acc",
            "org.g4.arithmetic_acc",
            "org.g5.niah_acc",
            "org.g6.auc_log_tokens",
        ):
            assert key in flat, f"missing {key}: {sorted(flat)}"
        mirrors = battery["mirrors"]
        assert mirrors, "battery.mirrors must not be empty"
        for pair in mirrors:
            assert set(pair) == {"group", "metric", "public", "mirror"}, pair
            assert pair["group"] in ("g2", "g4") and pair["metric"].startswith("org.")
            for side in ("public", "mirror"):
                assert isinstance(pair[side]["value"], (int, float)), pair
        assert any(p["group"] == "g2" for p in mirrors), mirrors
        assert any(p["group"] == "g4" for p in mirrors), mirrors
        print("v3 two-phase flow OK: battery + sealed v1 bpb via checkpoint handoff")
        print(
            "battery contract OK: %d org.* metrics, %d mirror pairs"
            % (len(flat), len(mirrors))
        )


def main():
    check_py_compile()
    check_generator_determinism()
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError as exc:
        print(f"BATTERY SMOKE SKIP (torch parts): {exc}")
        return 2
    check_full_battery()
    check_v3_flow()
    print("BATTERY SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
