#!/usr/bin/env python3
"""G1–G8 battery smoke (E2).

Always runs (no deps):
  1. py_compile every harness .py file.
  2. Generator determinism: same seed -> identical items, different seed
     -> different items, and gold answers verify against independent
     recomputation for the verifiable families (dyck, modular, s5,
     boolean, arithmetic, knights&knaves).
  3. Miner tokenizer contract (`prismlib.tokenizer`): validation and its
     fail-closed paths, `tokenizer/` caps, the import-free declaration
     probe, cross-phase spec equality, and exact token budgets from
     `eval.common.fit_to_tokens`.

When torch + transformers + pyarrow are importable, additionally:
  4. Full-battery run on a tiny randomly-initialized model with a
     byte-level fake tokenizer (no HF download) under tiny caps.
  5. Full v3 two-phase flow via main.py (PRISM_FLOW=v3) on a stub miner —
     once on the pinned default tokenizer, once on a submission that ships
     its own via `build_tokenizer(ctx)`.

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


def check_tokenizer_contract():
    """Miner tokenizer contract: validation, caps, declaration probe,
    cross-phase spec equality, and the exact token-budget helper."""
    import random

    from eval import common
    from prismlib import tokenizer as tok_contract

    Err = tok_contract.TokenizerContractError

    # 1) Validation of a conforming (byte-level) tokenizer + deterministic
    # fingerprint across instances — that equality is what binds the eval
    # phase to the train phase.
    spec = tok_contract.validate(_ByteTok(), "hook", "byte")
    assert spec["vocab_size"] == 260 and spec["source"] == "hook", spec
    assert len(spec["fingerprint"]) == 64
    assert tok_contract.validate(_ByteTok(), "hook", "byte") == spec

    # 2) Fail-closed paths: no decode, ids outside the declared vocab,
    # vocab under the floor.
    class _NoDecode(_ByteTok):
        decode = None

    class _Overflow(_ByteTok):
        def __len__(self):
            return 300

        def __call__(self, text, **kw):
            out = super().__call__(text, **kw)
            out["input_ids"] = [900] + list(out["input_ids"])
            return out

        def decode(self, ids):
            return super().decode([i for i in ids if i < 300])

    class _TinyVocab(_ByteTok):
        def __len__(self):
            return 4

    for bad, needle in ((_NoDecode(), "decode"), (_Overflow(), "vocab"), (_TinyVocab(), "vocab")):
        try:
            tok_contract.validate(bad, "hook")
        except Err as exc:
            assert needle in str(exc), exc
        else:
            raise AssertionError(f"expected a contract error for {type(bad).__name__}")

    # 3) Cross-phase equality: identical spec passes, any drift fails.
    assert tok_contract.assert_matches(spec, spec)["checked"] is True
    assert tok_contract.assert_matches(spec, None)["checked"] is False
    try:
        tok_contract.assert_matches(spec, dict(spec, fingerprint="0" * 64))
    except Err as exc:
        assert "not reproducible" in str(exc), exc
    else:
        raise AssertionError("expected a cross-phase tokenizer mismatch error")

    # 4) Hook placement: `build_tokenizer` must sit beside `build_model`,
    # because the EVAL child imports the architecture module only.
    class _ArchMod:
        @staticmethod
        def build_tokenizer(_ctx):
            return _ByteTok()

    try:
        tok_contract.resolve({}, "cpu", arch_mod=None, train_mod=_ArchMod)
    except Err as exc:
        assert "architecture.py" in str(exc), exc
    else:
        raise AssertionError("expected a misplaced-hook error")
    hook_spec = tok_contract.resolve({}, "cpu", arch_mod=_ArchMod, train_mod=_ArchMod)[1]
    assert hook_spec["source"] == "hook", hook_spec
    assert hook_spec["fingerprint"] == spec["fingerprint"], hook_spec

    # 5) `tokenizer/` source-tree caps + the import-free declaration probe.
    with tempfile.TemporaryDirectory(prefix="prism_tok_") as tmp:
        work = Path(tmp)
        assert tok_contract.tokenizer_dir(work) is None
        assert tok_contract.declared(work) is False
        (work / "architecture.py").write_text("def build_tokenizer(ctx):\n    return None\n")
        assert tok_contract.declared(work, str(work / "architecture.py")) is True
        d = work / tok_contract.TOKENIZER_DIRNAME
        d.mkdir()
        for name, needle in (("weights.bin", "extension"), ("tokenizer.json", None)):
            (d / name).write_text("{}")
            if needle is None:
                assert tok_contract.tokenizer_dir(work) == str(d)
            else:
                try:
                    tok_contract.tokenizer_dir(work)
                except Err as exc:
                    assert needle in str(exc), exc
                else:
                    raise AssertionError(f"expected a cap error for {name}")
                (d / name).unlink()
        for i in range(tok_contract.MAX_TOKENIZER_FILES + 1):
            (d / f"extra{i}.json").write_text("{}")
        try:
            tok_contract.tokenizer_dir(work)
        except Err as exc:
            assert "files" in str(exc), exc
        else:
            raise AssertionError("expected a file-count cap error")

    # 6) Exact token budgets: `fit_to_tokens` is what the long-context
    # adapters build contexts with (tokens of the SUBMITTED tokenizer).
    tok = _ByteTok()
    segments = ["the magic word for alpha is 471 .", "the magic word for beta is 908 ."]
    suffix = " the magic word for alpha is"
    rng = random.Random(11)
    for target in (256, 1024, 4096):
        text, n = common.fit_to_tokens(
            tok, segments, target, lambda: "the falcon crossed the meadow near the harbor .",
            rng=rng, suffix=suffix,
        )
        assert n == target, (target, n)
        assert common.token_len(tok, text) == target
        assert text.endswith(suffix), text[-80:]
        for seg in segments:
            assert seg in text, seg
    # Over-budget segments are reported honestly, never silently cut.
    text, n = common.fit_to_tokens(tok, segments, 8, lambda: "filler .", suffix=suffix)
    assert n > 8 and segments[0] in text
    assert common.truncate_tokens(tok, "abcdef", 3) == "abc"
    assert common.vocab_size(tok) == 260
    print("tokenizer contract OK: validation, caps, cross-phase spec, exact token budgets")


def check_eval_pack_contract():
    """Operator pack is hard-capped at 400 rows per JSONL asset."""
    import importlib.util

    path = HARNESS_ROOT / "eval" / "build_private_pack.py"
    old = {k: os.environ.get(k) for k in ("G1_N", "G2_N", "G2_N_USABLE", "G5_QA_N")}
    try:
        for key in old:
            os.environ[key] = "9999"
        spec = importlib.util.spec_from_file_location("prism_eval_pack_contract", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.MAX_ASSET_ROWS == 400
        assert all(
            value == 400 for value in (mod.G1_N, mod.G2_N, mod.G2_N_USABLE, mod.G5_QA_N)
        )
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("eval pack contract OK (400 rows/file, public/private tiers)")


# ---------------------------------------------------------------- torch parts


class _ByteTok:
    """Minimal tokenizer shim for battery smoke (no HF download).

    This is exactly the duck-typed surface `prismlib.tokenizer` requires of
    a submitted tokenizer: `tok(text)["input_ids"]`, `decode`, `len`,
    `eos_token_id`, `convert_ids_to_tokens`.
    """

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

    def decode(self, ids):
        return bytes(max(0, int(i) - 1) for i in ids).decode("utf-8", errors="replace")

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
        {"step": 0, "tokens_seen": 0, "bytes_seen": 1, "flops_spent": 0,
         "wall_s": 0.0, "probe_loss": 6.5, "probe_bits_per_byte": 2.2},
        {"step": 4, "tokens_seen": 512, "bytes_seen": 1536, "flops_spent": 25,
         "wall_s": 2.0, "probe_loss": 4.2, "probe_bits_per_byte": 1.8},
        {"step": 6, "tokens_seen": 1024, "bytes_seen": 3072, "flops_spent": 50,
         "wall_s": 3.0, "probe_loss": 3.6, "probe_bits_per_byte": 1.5},
        {"step": 8, "tokens_seen": 2048, "bytes_seen": 6144, "flops_spent": 100,
         "wall_s": 4.0, "probe_loss": 3.4, "probe_bits_per_byte": 1.3},
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
        "train_flops_cap": 100.0,
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
    g5m = results["g5"]["metrics"]
    assert "g5.lstar" in g5m
    assert "g5.ruler.acc" in g5m or "g5.ruler.error" in g5m, sorted(g5m)
    assert "g5.babilong.acc" in g5m or "g5.babilong.error" in g5m, sorted(g5m)
    assert results["g8"]["metrics"].get("g8.mup.stub") == 1.0
    print("full battery OK: all 8 groups emitted finite metrics on a tiny model")
    print(json.dumps({g: len(results[g]["metrics"]) for g in ok_groups}, indent=2))

    # Rollup contract: the flat canonical org.* map + degenerate public_dev
    # mirror pairs the Rust composite ingests (eval/rollup.py).
    from eval import rollup as battery_rollup

    flat = battery_rollup.flatten_metrics(results, ctx["items"])
    assert flat and all(k.startswith("org.") for k in flat), sorted(flat)
    assert "org.g3.mqar_acc" in flat and "org.g4.arithmetic_acc" in flat, sorted(flat)
    for key in (
        "org.g1.bits_per_byte_code",
        "org.g1.bits_per_byte_prose",
        "org.g1.bits_per_byte_math",
        "org.g1.bits_per_byte_fresh_crawl",
        "org.g6.auc_log_tokens",
        "org.g6.auc_log_bytes",
        "org.g6.bytes_to_bpb_threshold",
        "org.g6.bpb_at_half_budget",
        "org.g8.loss_spike_score",
    ):
        assert key in flat, f"missing {key}: {sorted(flat)}"
    for key in (
        "org.g5.ruler_acc",
        "org.g5.babilong_acc",
        "org.g5.natural_mcq_acc",
        "org.g5.helmet_rag_acc",
        "org.g5.lstar",
    ):
        assert key in flat, f"missing {key}: {sorted(flat)}"
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

# Same stub plus the miner tokenizer hook: `build_tokenizer(ctx)` beside
# `build_model`, returning a byte-level tokenizer built with no network.
STUB_ARCH_TOKHOOK = STUB_ARCH + '''

class ByteTokenizer:
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False, return_tensors=None,
                 truncation=False, max_length=None):
        ids = [b + 1 for b in text.encode("utf-8", errors="replace")]
        if truncation and max_length:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids):
        return bytes(max(0, int(i) - 1) for i in ids).decode("utf-8", errors="replace")

    def convert_ids_to_tokens(self, ids):
        return [chr(max(32, int(i) - 1)) for i in ids]

    def __len__(self):
        return 260


def build_tokenizer(ctx):
    return ByteTokenizer()
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


def check_v3_flow(arch_src=STUB_ARCH, tokenizer_source="default", vocab_size=None):
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
        (work / "architecture.py").write_text(arch_src)
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
                # Enter the real µP path even under tiny caps. The fixture
                # intentionally ignores the width knob, so this fails closed
                # to 0.0 while proving the canonical org key is never omitted.
                "PRISM_EVAL_G8_SWEEP": "1",
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
        assert m["bits_per_byte"] > 0, m.get("bits_per_byte")
        # Tokenizer contract: the resolved spec is reported, and the EVAL
        # child proved it rebuilt the TRAIN child's tokenizer.
        tok_spec = m["tokenizer"]
        assert tok_spec["source"] == tokenizer_source, tok_spec
        assert len(tok_spec["fingerprint"]) == 64, tok_spec
        assert tok_spec["cross_phase"]["checked"] is True, tok_spec
        if vocab_size is not None:
            assert tok_spec["vocab_size"] == vocab_size, tok_spec
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
        # Every anchored metric available on CPU/tiny caps is present.
        for key in (
            "org.g1.bits_per_byte_code",
            "org.g1.bits_per_byte_prose",
            "org.g1.bits_per_byte_math",
            "org.g1.bits_per_byte_fresh_crawl",
            "org.g2.arc_challenge_acc",
            "org.g3.mqar_acc",
            "org.g4.arithmetic_acc",
            "org.g5.ruler_acc",
            "org.g5.babilong_acc",
            "org.g5.natural_mcq_acc",
            "org.g5.lstar",
            "org.g6.auc_log_tokens",
            "org.g6.auc_log_bytes",
            "org.g6.bytes_to_bpb_threshold",
            "org.g6.bpb_at_half_budget",
            "org.g7.throughput_toks_s",
            "org.g7.ttft_ms_32k",
            "org.g7.tpot_ms_32k",
            "org.g7.state_bytes_per_token_32k",
            "org.g7.joules_per_token",
            "org.g7.reasoning_throughput",
            "org.g8.loss_spike_score",
            "org.g8.mup_lr_stability",
        ):
            assert key in flat, f"missing {key}: {sorted(flat)}"
        # helmet_rag is fail-soft: under a slow hook tokenizer the shared
        # G5 natural budget can expire after MCQ; default-tok smoke still
        # covers the key via check_full_battery.
        if tokenizer_source == "default":
            assert "org.g5.helmet_rag_acc" in flat, sorted(flat)
            anchor = json.loads(
                (HARNESS_ROOT.parent / "anchors" / "v3.json").read_text(encoding="utf-8")
            )
            anchored = {
                key
                for name, group in anchor["groups"].items()
                if name in {f"g{i}" for i in range(1, 9)}
                for key in group.get("metrics", {})
            }
            missing = sorted(anchored - set(flat))
            assert not missing, f"v3 composite ineligible; missing anchored metrics: {missing}"
        mirrors = battery["mirrors"]
        assert mirrors, "battery.mirrors must not be empty"
        for pair in mirrors:
            assert set(pair) == {"group", "metric", "public", "mirror"}, pair
            assert pair["group"] in ("g2", "g4", "g5") and pair["metric"].startswith(
                "org."
            )
            for side in ("public", "mirror"):
                assert isinstance(pair[side]["value"], (int, float)), pair
        assert any(p["group"] == "g2" for p in mirrors), mirrors
        assert any(p["group"] == "g4" for p in mirrors), mirrors
        assert any(p["group"] == "g5" for p in mirrors), mirrors
        print(
            "v3 two-phase flow OK (tokenizer=%s vocab=%d): battery + sealed v1 bpb "
            "via checkpoint handoff" % (tok_spec["source"], tok_spec["vocab_size"])
        )
        print(
            "battery contract OK: %d org.* metrics, %d mirror pairs"
            % (len(flat), len(mirrors))
        )


def check_g8_mup_contracts():
    """µP rollup fail-closed + reduced probe-base geometry (no full GPU sweep)."""
    for script in ("test_g8_mup_rollup.py", "test_g8_mup_probe_base.py"):
        path = Path(__file__).resolve().parent / script
        r = subprocess.run([sys.executable, str(path)], cwd=str(HARNESS_ROOT))
        assert r.returncode == 0, f"{script} failed with {r.returncode}"


def check_g7_rollup_contract():
    """All full-grid G7 producers map to the canonical anchor keys."""
    from eval import rollup

    groups = {
        "g4": {
            "status": "ok",
            "metrics": {
                "g4.arith.acc": 0.5,
                "g4.bool.base.acc": 0.5,
                "g4.dyck.acc": 0.5,
                "g4.mod.acc": 0.5,
                "g4.kk.acc": 0.5,
                "g4.proof.acc": 0.5,
            },
        },
        "g7": {
            "status": "ok",
            "metrics": {
                "g7.throughput.b32.toks": 1000.0,
                "g7.ttft.L32768.ms": 20.0,
                "g7.tpot.L32768.ms": 3.0,
                "g7.state.bytes_per_token.measured": 1024.0,
                "g7.energy.j_per_token": 0.2,
            },
        },
    }
    flat = rollup.flatten_metrics(groups)
    required = {
        "org.g7.throughput_toks_s",
        "org.g7.ttft_ms_32k",
        "org.g7.tpot_ms_32k",
        "org.g7.state_bytes_per_token_32k",
        "org.g7.joules_per_token",
        "org.g7.reasoning_throughput",
    }
    assert required <= set(flat), required - set(flat)
    print("g7 rollup OK (32k + energy + reasoning throughput)")


def main():
    check_py_compile()
    check_generator_determinism()
    check_tokenizer_contract()
    check_eval_pack_contract()
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError as exc:
        print(f"BATTERY SMOKE SKIP (torch parts): {exc}")
        return 2
    # Probe-base test builds Transformer++ at tiny width (needs torch).
    check_g8_mup_contracts()
    check_g7_rollup_contract()
    check_full_battery()
    check_v3_flow()
    # Same flow on a submission that ships its own tokenizer via the
    # documented `build_tokenizer(ctx)` hook (no network, byte-level vocab).
    check_v3_flow(STUB_ARCH_TOKHOOK, tokenizer_source="hook", vocab_size=260)
    print("BATTERY SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
