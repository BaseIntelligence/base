#!/usr/bin/env python3
"""G5 natural-document slice smoke (`eval/natural_docs.py`).

Always runs (stdlib only — the scorer is monkeypatched, so no torch):

 1. `py_compile` + the shipped `public_dev` fixtures load and their gold
    answers are the hand-checked ones.
 2. Prompt construction under a byte-level fake tokenizer: the slice token
    budget is respected, over-length contexts keep both ends, and the
    tokenizer-without-`decode` fallback still fits the budget.
 3. Anti-leak: a built MCQ prompt never contains any choice text or the
    gold letter/index, and the gold index is uniform over a large pool
    because the choice order is redrawn from the secret seed.
 4. Chance level: three prompt-blind degenerate scorers (first, last and
    prompt-hash) each land at 1/k on a length-balanced 4-way pool.
 5. Determinism: the same secret seed reproduces the run exactly; a
    different secret seed redraws the item set and the choice order.
 6. Mirror pairs match `eval/rollup.py`'s `{group, metric, public, mirror}`
    contract and read the disjoint `public_dev/` pool when it is staged.

With torch importable, additionally scores the real length-normalized
logprob path on a tiny random model.

Exit 2 = skipped the torch part; the stdlib parts always execute and assert.
"""

import hashlib
import json
import os
import py_compile
import random
import sys
import tempfile
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT))

from eval import common, natural_docs as nat  # noqa: E402

FIXTURES = HARNESS_ROOT / "eval" / "public_dev" / "g5" / "natural"
SECRET = 987_654_321

# Hand-checked gold answers for the shipped fixtures: the gold text must be
# the one a human reading the fixture context picks.
EXPECTED_MCQ_GOLD = {
    "fixture-mcq-001": "Winter barley, which gained 14 percent over the control plots.",
    "fixture-mcq-002": "Both recommend delaying the migration until the audit log is append-only.",
    "fixture-mcq-003": "Nadia owns it, after Tomas hands it over.",
    "fixture-mcq-004": "Westhaven.",
    "fixture-mcq-005": "It returns an empty window and records a skipped counter.",
    "fixture-mcq-006": "copper-3",
}
EXPECTED_RAG_GOLD = {
    "fixture-rag-nq-001": "the Vell",
    "fixture-rag-nq-002": "1834",
    "fixture-rag-tqa-001": "tin",
    "fixture-rag-tqa-002": "the Marl bridge",
    "fixture-rag-hqa-001": "the Carrow hills",
    "fixture-rag-pop-001": "cartographer",
}


class ByteTok:
    """Byte-level fake tokenizer: one id per UTF-8 byte, decode included."""

    def __init__(self, decodable=True):
        self.decodable = decodable
        self.eos_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [b + 1 for b in text.encode("utf-8", errors="replace")]
        out = {"input_ids": ids}
        if return_tensors == "pt":
            import torch

            out["input_ids"] = torch.tensor([ids], dtype=torch.long)
        return out

    def __getattr__(self, name):
        # A tokenizer built without `decode` must still work (char fallback).
        if name == "decode" and self.__dict__.get("decodable"):
            return self._decode
        raise AttributeError(name)

    def _decode(self, ids):
        return bytes(max(0, i - 1) for i in ids).decode("utf-8", errors="replace")

    def __len__(self):
        return 260


class SeamTok(ByteTok):
    """Byte tokenizer that re-encodes a decoded cut *longer* than the slice
    it came from, the way a real BPE does when merges break at the seam.

    Every encode adds a 2% surcharge, so the naive head+tail cut always
    overshoots and `_fit_middle`'s correction loop has to converge.
    """

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = super().__call__(text)["input_ids"]
        return {"input_ids": ids + [0] * (1 + len(ids) // 50)}

    def _decode(self, ids):
        return bytes(i - 1 for i in ids if i > 0).decode("utf-8", errors="replace")


def fixture_ctx(assets_dir=None, tok=None):
    return {
        "tokenizer": tok or ByteTok(),
        "device": "cpu",
        "eval_assets_dir": assets_dir,
        "eval_tier": "private" if assets_dir else "public_dev",
        "eval_secret_seed": SECRET,
        "items": common.ItemRecorder(),
    }


# ------------------------------------------------------------- 1. fixtures


def check_compile_and_fixtures():
    for path in sorted((HARNESS_ROOT / "eval").rglob("*.py")):
        py_compile.compile(str(path), doraise=True)

    mcq = common.load_jsonl(str(FIXTURES / "natural_mcq.jsonl"))
    rag = common.load_jsonl(str(FIXTURES / "helmet_rag.jsonl"))
    demos = common.load_jsonl(str(FIXTURES / "helmet_rag.demos.jsonl"))
    assert len(mcq) == len(EXPECTED_MCQ_GOLD), len(mcq)
    assert len(rag) == len(EXPECTED_RAG_GOLD), len(rag)
    assert demos, "demo pool must not be empty"

    for row in mcq:
        assert row["slice"] == "natural_mcq" and row["meta"]["fixture"] is True, row["id"]
        assert len(row["choices"]) == 4 and 0 <= row["gold"] < 4, row["id"]
        assert row["choices"][row["gold"]] == EXPECTED_MCQ_GOLD[row["id"]], row["id"]
        assert row["context"] and row["question"] and row["cluster"], row["id"]
    for row in rag:
        assert row["slice"] == "helmet_rag", row["id"]
        assert row["answers"][0] == EXPECTED_RAG_GOLD[row["id"]], row["id"]
        assert row["passages"] and all(p["title"] and p["text"] for p in row["passages"])
    clusters = {d["cluster"] for d in demos}
    assert clusters >= {r["cluster"] for r in rag}, clusters
    for row in demos:
        assert row["answer"] and row["passages"], row["id"]
    print(f"fixtures OK: {len(mcq)} MCQ + {len(rag)} RAG rows, gold hand-checked")


# ------------------------------------------------- 2. prompts and truncation


def check_prompt_budget():
    long_ctx = "The archive states that the seal is intact. " * 4000
    row = {
        "id": "synthetic-long",
        "cluster": "test",
        "question": "Is the seal intact?",
        "choices": ["yes", "no", "partly", "unknown"],
        "gold": 0,
        "context": long_ctx,
    }
    # Tokenizer with `decode`, without it (char fallback), and one whose
    # re-encode overshoots the cut (BPE seam) — all must land in budget.
    for label, tok in (
        ("decodable", ByteTok()),
        ("no-decode", ByteTok(decodable=False)),
        ("bpe-seam", SeamTok()),
    ):
        prompt, choices, gold, n_tokens = nat._mcq_prompt(tok, row, SECRET)
        assert n_tokens <= nat.MAX_TOKENS, (label, n_tokens)
        assert choices[gold].strip() == "yes"
        # Middle-dropped, both ends kept.
        assert prompt.startswith(long_ctx[:64]), label
        assert "Question: Is the seal intact?\nAnswer:" in prompt, label
        assert len(prompt) < len(long_ctx), label

    # A short context is passed through untouched.
    short = dict(row, id="synthetic-short", context="The seal is intact.")
    prompt, _, _, n_tokens = nat._mcq_prompt(ByteTok(), short, SECRET)
    assert prompt.startswith("The seal is intact.\n\nQuestion:"), prompt[:80]
    assert n_tokens < 200, n_tokens

    tok = ByteTok()
    rag_row = common.load_jsonl(str(FIXTURES / "helmet_rag.jsonl"))[0]
    demo_pool = common.load_jsonl(str(FIXTURES / "helmet_rag.demos.jsonl"))
    demos = [d for d in demo_pool if d["cluster"] == rag_row["cluster"]][: nat._SHOTS]
    prompt, n_tokens = nat._rag_prompt(tok, rag_row, demos)
    assert n_tokens <= nat.MAX_TOKENS, n_tokens
    assert prompt.startswith("Use the given documents"), prompt[:40]
    assert prompt.endswith(f"Question: {rag_row['question']}\nAnswer:"), prompt[-120:]
    # HELMET's frozen non-chat rendering: no chat turns, `Document (Title: ...)`.
    assert "Document (Title: " in prompt
    assert "User:" not in prompt and "Assistant:" not in prompt
    # One `Answer:` in the instruction's format hint, one per demo, one for
    # the item — and the item's own is the empty completion slot.
    assert prompt.count("\nAnswer:") == 2 + nat._SHOTS, prompt.count("\nAnswer:")
    for answer in rag_row["answers"]:
        assert f"Answer: {answer}" not in prompt, answer
    print(f"prompt budget OK: MCQ + RAG fit {nat.MAX_TOKENS} tokens, ends preserved")


# ---------------------------------------------------------- 3+4. leak/chance


def synthetic_pool(n):
    """4-way MCQ pool with gold at a fixed upstream index.

    Every choice is the same length, so no surface heuristic can separate
    them: whatever a degenerate model does must average to 1/4 unless the
    adapter leaks the gold position.
    """
    rng = random.Random(11)
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"synthetic-{i:04d}",
                "cluster": f"c{i % 5}",
                "question": "What did the auditor find?",
                "choices": [f"finding {j} of record {i:04d}" for j in range(4)],
                "gold": 0,
                "context": " ".join(rng.choice(["alpha", "beta", "gamma"]) for _ in range(50)),
            }
        )
    return rows


def check_no_gold_leak_and_chance_level():
    tok = ByteTok()
    rows = synthetic_pool(400)
    golds = []
    for row in rows:
        prompt, choices, gold, _ = nat._mcq_prompt(tok, row, SECRET)
        golds.append(gold)
        for choice in choices:
            assert choice.strip() not in prompt, row["id"]
        assert "Answer: " not in prompt, row["id"]
        assert not any(f"answer is {ltr}" in prompt.lower() for ltr in "abcd"), row["id"]
        assert len({len(c) for c in choices}) == 1, "test pool must be length-balanced"
    share = [golds.count(k) / len(golds) for k in range(4)]
    assert all(0.18 < s < 0.32 for s in share), share

    # Degenerate scorers that see only the prompt (never gold or position)
    # must all sit at chance. Any positional leak breaks the first two; a
    # prompt-borne leak breaks the third.
    def prompt_hash(prompt, choices):
        digest = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        return digest % len(choices)

    for name, pick in (
        ("always-first", lambda prompt, choices: 0),
        ("always-last", lambda prompt, choices: len(choices) - 1),
        ("prompt-hash", prompt_hash),
    ):
        acc = run_with_scorer(rows, pick)
        assert 0.18 < acc < 0.32, f"{name}: acc {acc:.3f} is not chance level"
        print(f"chance level OK: {name} scores {acc:.3f} on a 4-way pool")


def run_with_scorer(rows, pick):
    """Score `rows` through `natural_docs` with a prompt-blind scorer."""
    real = common.score_choices
    common.score_choices = lambda model, tok, device, prompt, choices, gold: (
        1.0 if pick(prompt, choices) == gold else 0.0,
        1.0,
    )
    try:
        ctx = fixture_ctx()
        out = {}
        series = nat._mcq_series(None, ctx, rows, SECRET, common.Budget(120.0), out, "g5.natural_mcq")
        assert out["g5.natural_mcq.n"] == float(len(rows)), out
        return series["value"]
    finally:
        common.score_choices = real


# ------------------------------------------------------- 5+6. run / mirrors


def synthetic_rag_pool(n):
    """RAG pool where the answer is the last document's last word iff `i` is
    even, so `fake_greedy` below lands at a known EM of ~0.5."""
    rows, demos = [], []
    for i in range(n):
        cluster = ("nq", "triviaqa", "hotpotqa", "popqa")[i % 4]
        answer = f"marker{i:04d}" if i % 2 == 0 else f"unseen{i:04d}"
        rows.append(
            {
                "id": f"rag-{i:04d}",
                "slice": nat.RAG,
                "cluster": cluster,
                "question": f"Which marker did survey {i:04d} record?",
                "answers": [answer],
                "passages": [
                    {"title": "Unrelated", "text": "The ledger was sealed in spring."},
                    {"title": f"Survey {i:04d}", "text": f"The survey logged marker{i:04d}"},
                ],
            }
        )
        demos.append(
            {
                "id": f"demo-{i:04d}",
                "cluster": cluster,
                "question": f"Which marker did drill {i:04d} record?",
                "answer": f"marker{i:04d}",
                "passages": [{"title": f"Drill {i:04d}", "text": f"The drill logged marker{i:04d}"}],
            }
        )
    return rows, demos


def fake_greedy(model, tok, device, prompt):
    """Torch-free stand-in for `_greedy_line`: copies the last word of the
    scored item's last document, the way a retrieval-following LM would."""
    body = prompt.rsplit("Document (Title: ", 1)[-1]
    return body.split("\n")[0].split()[-1]


def staged_assets(tmp):
    """Operator-style assets dir: private pools + disjoint mirror pools."""
    root = Path(tmp) / "eval-assets" / nat.PACK_DIR
    (root / "public_dev").mkdir(parents=True)
    mcq = synthetic_pool(120)
    rag, demos = synthetic_rag_pool(48)
    chunks = {
        "natural_mcq.jsonl": mcq[:60],
        "public_dev/natural_mcq.jsonl": mcq[60:],
        "helmet_rag.jsonl": rag[:24],
        "public_dev/helmet_rag.jsonl": rag[24:],
        "helmet_rag.demos.jsonl": demos[:24],
        "public_dev/helmet_rag.demos.jsonl": demos[24:],
    }
    for rel, chunk in chunks.items():
        (root / rel).write_text("".join(json.dumps(r) + "\n" for r in chunk), encoding="utf-8")
    private = {r["id"] for r in mcq[:60]} | {r["id"] for r in rag[:24]}
    mirror = {r["id"] for r in mcq[60:]} | {r["id"] for r in rag[24:]}
    return str(Path(tmp) / "eval-assets"), private, mirror


def check_run_and_mirrors():
    real_score, real_greedy = common.score_choices, nat._greedy_line
    common.score_choices = lambda model, tok, device, prompt, choices, gold: (
        1.0 if choices[gold].strip().startswith("the archive") else 0.0,
        2.5,
    )
    nat._greedy_line = fake_greedy
    try:
        with tempfile.TemporaryDirectory(prefix="prism_nat_") as tmp:
            assets, private_ids, mirror_ids = staged_assets(tmp)
            assert not private_ids & mirror_ids, "pack sides must be disjoint"

            os.environ["PRISM_EVAL_NATURAL_ITEMS"] = "16"
            first = nat.run(None, fixture_ctx(assets))
            again = nat.run(None, fixture_ctx(assets))
            assert first == again, "same secret seed must reproduce the run"
            assert first["g5.natural.staged"] == 1.0, first
            assert first["g5.natural.pool_rows.natural_mcq"] == 60.0, first
            assert first["g5.natural.pool_rows.helmet_rag"] == 24.0, first
            assert first["g5.natural_mcq.n"] == 16.0, first
            assert first["g5.helmet_rag.n"] == 8.0, first
            assert "g5.natural_mcq.acc" in first, sorted(first)
            # `L<bucket>` keys are what rollup's existing G5 branch averages.
            assert any(k.startswith("g5.natural_mcq.L") for k in first), sorted(first)
            # The stand-in copies the document, so EM tracks the planted half.
            assert 0.2 < first["g5.helmet_rag.acc"] < 0.8, first

            # A different operator secret must redraw the item set, the choice
            # order and the demos — asserted on the draws themselves, since
            # two different draws can still average to the same accuracy.
            pool = common.load_jsonl(str(Path(assets) / nat.PACK_DIR / "natural_mcq.jsonl"))
            drawn = [
                [r["id"] for r in nat._sample(pool, 16, s, f"g5/natural/{nat.MCQ}/items")]
                for s in (SECRET, SECRET + 1)
            ]
            assert drawn[0] != drawn[1], "a new secret seed must redraw the item set"
            assert drawn[0] == [
                r["id"] for r in nat._sample(pool, 16, SECRET, f"g5/natural/{nat.MCQ}/items")
            ], "the same secret seed must redraw the same item set"
            orders = [
                [nat._mcq_prompt(ByteTok(), r, s)[1] for r in pool[:8]] for s in (SECRET, SECRET + 1)
            ]
            assert orders[0] != orders[1], "a new secret seed must reshuffle choices"

            pairs = nat.mirror_pairs(None, fixture_ctx(assets))
            assert {p["metric"] for p in pairs} == {
                "org.g5.natural_mcq_acc",
                "org.g5.helmet_rag_acc",
            }, pairs
            for pair in pairs:
                assert set(pair) == {"group", "metric", "public", "mirror"}, pair
                assert pair["group"] == "g5"
                for side in ("public", "mirror"):
                    assert isinstance(pair[side]["value"], float), pair
                    assert pair[side]["clusters"], pair
            print(f"run + mirrors OK: {len(first)} metrics, {len(pairs)} mirror pair(s)")

            # Public-dev tier: fixtures only, honestly reported as unstaged.
            del os.environ["PRISM_EVAL_NATURAL_ITEMS"]
            os.environ["PRISM_TEST_EVAL_CAPS"] = "1"
            dev = nat.run(None, fixture_ctx())
            assert dev["g5.natural.staged"] == 0.0, dev
            assert dev["g5.natural.pool_rows.natural_mcq"] == 6.0, dev
            assert dev["g5.natural.pool_rows.helmet_rag"] == 6.0, dev
            degenerate = nat.mirror_pairs(None, fixture_ctx())
            for pair in degenerate:
                assert pair["public"]["value"] == pair["mirror"]["value"], pair
            print(f"public_dev tier OK: {len(dev)} metrics, gap-0 mirrors on fixtures")
    finally:
        common.score_choices = real_score
        nat._greedy_line = real_greedy
        os.environ.pop("PRISM_TEST_EVAL_CAPS", None)
        os.environ.pop("PRISM_EVAL_NATURAL_ITEMS", None)


# ------------------------------------------------------------- torch scoring


def check_real_scorer():
    import torch
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(260, 24)
            self.head = nn.Linear(24, 260, bias=False)

        def forward(self, ids):
            return self.head(self.emb(ids[:, -512:]))

    torch.manual_seed(0)
    model = Tiny().eval()
    os.environ["PRISM_TEST_EVAL_CAPS"] = "1"
    try:
        out = nat.run(model, fixture_ctx())
    finally:
        os.environ.pop("PRISM_TEST_EVAL_CAPS", None)
    for key, value in out.items():
        assert value == value and abs(value) != float("inf"), (key, value)
    for key in ("g5.natural_mcq.acc", "g5.helmet_rag.acc", "g5.helmet_rag.rank_acc"):
        assert 0.0 <= out[key] <= 1.0, (key, out)
    # Both RAG paths ran for real: closed-set ranking and greedy decode + EM.
    assert out["g5.helmet_rag.rank_n"] > 0 and out["g5.helmet_rag.n"] > 0, out
    print(f"real logprob scorer OK on a tiny model: {len(out)} finite metrics")


def main():
    check_compile_and_fixtures()
    check_prompt_budget()
    check_no_gold_leak_and_chance_level()
    check_run_and_mirrors()
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        print(f"NATURAL SMOKE SKIP (torch part): {exc}")
        return 2
    check_real_scorer()
    print("NATURAL SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
