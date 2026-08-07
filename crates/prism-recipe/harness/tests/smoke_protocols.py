#!/usr/bin/env python3
"""G5 community-protocol smoke: vendored RULER + BABILong generators.

Always runs (no deps):
  1. Determinism — same lattice seed gives identical instances, a shifted
     secret shifts them.
  2. Gold correctness — every gold answer is recomputed independently from
     the rendered prompt (needle bindings, variable chains, bAbI replay) and
     also has to pass the official upstream metric functions.
  3. Chance level — degenerate scorers (constant slot, longest string,
     coin flip) land at 1/n_choices, so the choice sets carry no free signal.
  4. Length fidelity — the byte-level fake tokenizer of `smoke_battery`
     fits every probe to its exact token target with the evidence intact.

With torch importable, additionally runs both adapters end to end on the
tiny randomly-initialized model under tiny caps.

Exit 2 = skipped the torch part (box lacks the pod image deps); the
pure-python parts always execute and assert.
"""

import random
import re
import statistics
import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT))
sys.path.insert(0, str(HARNESS_ROOT / "tests"))

from eval import common, g5_babilong as gb, g5_ruler as grl, toklen  # noqa: E402
from eval import vendor_babilong as vb, vendor_ruler as vr  # noqa: E402
from smoke_battery import _ByteTok  # noqa: E402

SECRET = 424242


class ByteTok(_ByteTok):
    """`_ByteTok` plus the `decode` half of the tokenizer contract."""

    def decode(self, ids, skip_special_tokens=True):
        return bytes(max(0, min(255, int(i) - 1)) for i in ids).decode(
            "utf-8", errors="replace"
        )


# ---------------------------------------------------------------- determinism


def _families():
    rows = _synthetic_qa_rows()
    return [
        ("niah_mk", lambda s: vr.niah(s, 24, num_needle_k=4)),
        ("niah_mq", lambda s: vr.niah(s, 24, num_needle_k=4, num_needle_q=4)),
        ("niah_mv", lambda s: vr.niah(s, 24, num_needle_v=4)),
        ("vt", lambda s: vr.variable_tracking(s, 24)),
        ("qa_synth", lambda s: vr.qa(s, 6)),
        ("qa_pack", lambda s: vr.qa(s, 6, rows=rows)),
        ("babi_qa1", lambda s: vb.story(s, 1)),
        ("babi_qa2", lambda s: vb.story(s, 2)),
        ("babi_qa3", lambda s: vb.story(s, 3)),
        ("babi_qa4", lambda s: vb.story(s, 4)),
        ("babi_qa5", lambda s: vb.story(s, 5)),
        ("filler", lambda s: vb.synthetic_filler(s, 4)),
        ("sampler", lambda s: vb.sample_sentences(s, vb.synthetic_filler(11, 8), 12)),
    ]


def _synthetic_qa_rows():
    return vr._synthetic_docs(random.Random(7), 12)


def _canon(value):
    """Comparable view of an instance (drops the lazy filler callables)."""
    if isinstance(value, dict):
        return {k: _canon(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    return value


def check_determinism():
    for name, fn in _families():
        same, shifted = [], []
        for i in range(6):
            s1 = common.task_seed(SECRET, f"smoke/{name}/{i}")
            s2 = common.task_seed(SECRET, f"smoke/{name}/{i}")
            s3 = common.task_seed(SECRET + 1, f"smoke/{name}/{i}")
            a, b, c = _canon(fn(s1)), _canon(fn(s2)), _canon(fn(s3))
            assert a == b, f"{name}: same seed must give identical instances"
            assert a, f"{name}: empty instance"
            same.append(a)
            shifted.append(c)
        assert same != shifted, f"{name}: a different secret must shift the family"
    # Whole-prompt determinism: the same seed and tokenizer must rebuild the
    # exact prompt, filler stream and fitting included.
    tok = ByteTok()
    docs = vb.synthetic_filler(3, 16)
    for probe in grl.PROBES:
        prompts = set()
        for secret in (SECRET, SECRET, SECRET + 1):
            seed = common.task_seed(secret, f"det/{probe}")
            shot = grl._shot(tok, grl._maker(probe, seed + 1, None), 2048, seed + 1)
            prompts.add(grl._build(tok, grl._maker(probe, seed, None), 2048, shot, seed)[1])
        assert len(prompts) == 2, f"ruler {probe}: prompt not seed-deterministic"
    for qa in gb.TASKS:
        prompts = set()
        for secret in (SECRET, SECRET, SECRET + 1):
            seed = common.task_seed(secret, f"det/qa{qa}")
            prompts.add(gb._item(tok, seed, qa, 2048, docs)["prompt"])
        assert len(prompts) == 2, f"babilong qa{qa}: prompt not seed-deterministic"
    seeds = {common.task_seed(SECRET, f"g5/ruler/niah_mk/4096/{i}") for i in range(64)}
    assert len(seeds) == 64
    print(f"determinism OK ({len(_families())} families + fitted prompts)")


# ---------------------------------------------------------------- gold: ruler


def _needle_bindings(prompt):
    """key -> {values} as stated by the needle sentences in the prompt."""
    out = {}
    pat = re.compile(
        r"One of the special magic numbers? for ([\w-]+) is: (\d+)\.",
    )
    for key, value in pat.findall(prompt):
        out.setdefault(key, set()).add(value)
    return out


def _resolve_chains(prompt):
    """var -> value by replaying the `VAR X = ...` statements in order."""
    values = {}
    for line in prompt.splitlines():
        m = re.match(r"VAR ([A-Z]+) = (\d+)\s*$", line)
        if m:
            values[m.group(1)] = m.group(2)
            continue
        m = re.match(r"VAR ([A-Z]+) = VAR ([A-Z]+)\s*$", line)
        if m and m.group(2) in values:
            values[m.group(1)] = values[m.group(2)]
    return values


_QUESTION = re.compile(
    r"What (?:are all the|is the) special magic numbers? "
    r"for (.+?) mentioned in the provided text\?"
)


def _query_keys(prompt):
    m = _QUESTION.search(prompt)
    assert m, prompt[-400:]
    return [k.strip() for k in m.group(1).replace(", and ", ", ").split(", ")]


def check_ruler_gold():
    tok = ByteTok()
    checked = 0
    for probe in grl.PROBES:
        for i in range(4):
            seed = common.task_seed(SECRET, f"gold/{probe}/{i}")
            rows = _synthetic_qa_rows() if probe == "qa" else None
            make = grl._maker(probe, seed, rows)
            inst, prompt, _n, intact = grl._build(tok, make, 2048, "")
            assert intact, f"{probe}: evidence truncated at 2048"
            answers = [s["answer"] for s in inst["slots"]]
            assert answers == list(inst["meta"]["answers"]), inst["meta"]
            for slot in inst["slots"]:
                assert slot["choices"][slot["gold"]] == slot["answer"], slot
                assert len(set(slot["choices"])) == len(slot["choices"]), slot

            if probe.startswith("niah"):
                bindings = _needle_bindings(prompt)
                for key in _query_keys(prompt):
                    assert key in bindings, (key, sorted(bindings)[:4])
                stated = {v for key in _query_keys(prompt) for v in bindings[key]}
                assert set(answers) <= stated, (answers, sorted(stated))
                # The distractor needles must not also state a gold value,
                # or "read any needle" would score.
                others = [
                    v
                    for key, vals in bindings.items()
                    if key not in _query_keys(prompt)
                    for v in vals
                ]
                assert not (set(answers) & set(others)), answers
            elif probe == "vt":
                values = _resolve_chains(prompt)
                query = re.search(r"assigned the value (\d+)", prompt).group(1)
                for var in answers:
                    assert values.get(var) == query, (var, values.get(var), query)
                assert sorted(v for v, x in values.items() if x == query) == sorted(
                    answers
                ), (answers, values)
            else:
                assert answers[0] in prompt
                assert prompt.count("Document 1:") == 1
                # Monotonic numbering; the last filler document may have been
                # trimmed mid-header to land on the token target.
                numbers = [int(m) for m in re.findall(r"^Document (\d+):", prompt, re.M)]
                assert numbers == sorted(numbers) == list(range(1, len(numbers) + 1))

            # Upstream metrics accept the generated gold; a wrong answer fails.
            refs = [[a] for a in answers]
            preds = [", ".join(answers)] * len(answers)
            assert vr.string_match_all(preds, refs) == 100.0
            assert vr.string_match_part(preds, refs) == 100.0
            assert vr.string_match_all(["nothing at all"] * len(answers), refs) == 0.0
            checked += 1
    print(f"ruler gold OK ({checked} instances hand-verified against upstream metrics)")


# ---------------------------------------------------------------- gold: babi

_FACT = re.compile(r"^(\w+) (.+?) the (\w+)( there)?\.$")


def _replay(facts):
    """bAbI world model: actor rooms, object rooms, per-object room history."""
    where, held, at = {}, {}, {}
    history, gave = {o: [] for o in vb.OBJECTS}, []

    def _seen(obj, room):
        if room is not None and history[obj][-1:] != [room]:
            history[obj].append(room)

    for fact in facts:
        m = _FACT.match(fact.strip())
        if m is None:
            m2 = re.match(r"^(\w+) (\w+) the (\w+) to (\w+)\.$", fact.strip())
            assert m2, fact
            actor, _verb, obj, other = m2.groups()
            held[obj] = other
            gave.append((obj, actor, other))
            _seen(obj, where.get(other))
            continue
        actor, verb, noun, _there = m.groups()
        if noun in vb.LOCATIONS:
            where[actor] = noun
            for obj, holder in held.items():
                if holder == actor:
                    _seen(obj, noun)
        elif verb in vb.TAKE_VERBS:
            held[noun] = actor
            at.pop(noun, None)
            _seen(noun, where.get(actor))
        else:
            assert verb in vb.DROP_VERBS, fact
            held.pop(noun, None)
            at[noun] = where[actor]
            _seen(noun, where[actor])
    # qa5 hands objects to actors who may never have entered a room; their
    # position is simply unknown (qa5 never asks for it).
    place = {}
    for obj in vb.OBJECTS:
        room = where.get(held[obj]) if obj in held else at.get(obj)
        if room is not None:
            place[obj] = room
    return where, place, history, gave


def _verify_story(st):
    facts, question, answer = st["facts"], st["question"], st["answer"]
    assert answer in st["choices"], st
    if st["qa"] == 4:
        rel = [
            re.match(r"The (\w+) is (\w+) of the (\w+)\.", f).groups() for f in facts
        ]
        m = re.match(r"What is (\w+) of the (\w+)\?", question)
        if m:
            direction, anchor = m.groups()
            hits = [s for s, d, x in rel if d == direction and x == anchor]
        else:
            m = re.match(r"What is the (\w+) (\w+) of\?", question)
            anchor, direction = m.groups()
            hits = [s for s, d, x in rel if x == anchor and d == vb.OPPOSITE[direction]]
        assert hits == [answer], (question, rel, answer)
        return
    where, place, history, gave = _replay(facts)
    if st["qa"] == 1:
        actor = re.match(r"Where is (\w+)\?", question).group(1)
        assert where[actor] == answer, (question, where, answer)
    elif st["qa"] == 2:
        obj = re.match(r"Where is the (\w+)\?", question).group(1)
        assert place[obj] == answer, (question, place, answer)
    elif st["qa"] == 3:
        obj, now = re.match(r"Where was the (\w+) before the (\w+)\?", question).groups()
        assert history[obj][-1] == now, (question, history[obj])
        assert history[obj][-2] == answer, (question, history[obj], answer)
    elif st["qa"] == 5:
        m = re.match(r"Who gave the (\w+)( to \w+)?\?", question)
        if m:
            hits = {g for o, g, _r in gave if o == m.group(1)}
        elif question.startswith("Who received"):
            obj = re.match(r"Who received the (\w+)\?", question).group(1)
            hits = {r for o, _g, r in gave if o == obj}
        elif question.startswith("Who did"):
            giver, obj = re.match(r"Who did (\w+) give the (\w+) to\?", question).groups()
            hits = {r for o, g, r in gave if (o, g) == (obj, giver)}
        else:
            giver, recv = re.match(r"What did (\w+) give to (\w+)\?", question).groups()
            hits = {o for o, g, r in gave if (g, r) == (giver, recv)}
        assert hits == {answer}, (question, gave, answer)


def check_babilong_gold():
    checked = 0
    for qa in vb.TASK_IDS:
        for i in range(24):
            st = vb.story(common.task_seed(SECRET, f"gold/babi/{qa}/{i}"), qa)
            assert st is not None, f"qa{qa}: story grammar came up dry"
            _verify_story(st)
            assert vb.compare_answers(st["answer"], f"{st['answer']}. blah")
            assert not vb.compare_answers(st["answer"], "<context>")
            checked += 1
    # Facts keep their story order and stay intact once the fitter has
    # scattered background prose between them.
    tok = ByteTok()
    docs = vb.synthetic_filler(5, 16)
    for qa in vb.TASK_IDS:
        item = gb._item(tok, common.task_seed(SECRET, f"order/{qa}"), qa, 3072, docs)
        st = vb.story(common.task_seed(SECRET, f"order/{qa}"), qa)
        pos = 0
        for fact in st["facts"]:  # forward scan: bAbI stories repeat sentences
            found = item["prompt"].find(fact, pos)
            assert found >= 0, (qa, fact, pos)
            pos = found + len(fact)
    print(f"babilong gold OK ({checked} stories replayed independently)")


# ---------------------------------------------------------------- chance level


def _slot_sets():
    """(n_choices, gold_index, choices) over freshly generated instances."""
    out = []
    for probe in grl.PROBES:
        rows = _synthetic_qa_rows() if probe == "qa" else None
        for i in range(24):
            seed = common.task_seed(SECRET, f"chance/{probe}/{i}")
            inst = grl._maker(probe, seed, rows)(16)
            out.extend((s["choices"], s["gold"]) for s in inst["slots"])
    for qa in vb.TASK_IDS:
        for i in range(24):
            st = vb.story(common.task_seed(SECRET, f"chance/babi/{qa}/{i}"), qa)
            if st is None:
                continue
            choices = list(st["choices"])
            random.Random(common.cantor(i, 6151)).shuffle(choices)
            out.append((choices, choices.index(st["answer"])))
    return out


def check_chance_level():
    slots = _slot_sets()
    assert len(slots) > 300, len(slots)
    rng = random.Random(0)
    policies = {
        "first": lambda cs: 0,
        "last": lambda cs: len(cs) - 1,
        "longest": lambda cs: max(range(len(cs)), key=lambda j: (len(cs[j]), cs[j])),
        "coin": lambda cs: rng.randrange(len(cs)),
    }
    chance = statistics.fmean(1.0 / len(cs) for cs, _g in slots)
    for name, pick in policies.items():
        acc = statistics.fmean(1.0 if pick(cs) == g else 0.0 for cs, g in slots)
        assert acc <= chance + 0.06, f"{name}: {acc:.3f} beats chance {chance:.3f}"
    # No fixed-position bias either: a flat-logprob model riding the argmax
    # tie-break must not find the gold sitting in the same slot every time.
    for n_choices in (4, 6):
        golds = [g for cs, g in slots if len(cs) == n_choices]
        freq = [golds.count(j) / len(golds) for j in range(n_choices)]
        assert max(freq) <= 1.0 / n_choices + 0.06, (n_choices, freq)
    print(f"chance level OK ({len(slots)} slots, chance={chance:.3f})")


# ---------------------------------------------------------------- length fit


def check_length_fidelity():
    tok = ByteTok()
    worst, spread = 0.0, []
    for probe in grl.PROBES:
        for length in grl.GRID_TINY:
            seed = common.task_seed(SECRET, f"fit/{probe}/{length}")
            shot = grl._shot(
                tok, grl._maker(probe, seed + 1, None), grl._shot_budget(length), seed + 1
            )
            inst, prompt, n_prompt, intact = grl._build(
                tok, grl._maker(probe, seed, None), length, shot, seed
            )
            assert intact, f"ruler {probe}@{length}: evidence truncated"
            assert toklen.n_tokens(tok, prompt) == n_prompt
            for slot in inst["slots"]:
                assert slot["answer"] in prompt or probe == "qa"
            got = toklen.depths(prompt, inst["evidence"])
            assert len(got) == len(inst["evidence"]) and all(0.0 <= d <= 1.0 for d in got)
            spread.extend(got)
            worst = max(worst, abs(n_prompt - length) / length)
    docs, from_pack = gb._filler_docs({"eval_assets_dir": None}, 3)
    assert not from_pack and docs
    for qa in gb.TASKS:
        for length in gb.GRID_TINY:
            seed = common.task_seed(SECRET, f"fit/babi{qa}/{length}")
            item = gb._item(tok, seed, qa, length, docs)
            assert item is not None, f"babilong qa{qa}@{length}: no item"
            assert item["intact"], f"babilong qa{qa}@{length}: fact truncated"
            assert item["choices"][item["gold"]].strip() == item["answer"]
            assert len(item["depths"]) == item["n_facts"], item["depths"]
            spread.extend(item["depths"])
            worst = max(worst, abs(item["n_prompt"] - length) / length)
    assert worst < 0.01, f"length error {worst:.4f} too large"
    # Evidence must reach deep into the context, not hug the prompt head.
    assert min(spread) < 0.25 and max(spread) > 0.7, (min(spread), max(spread))
    print(
        f"length fidelity OK (byte tokenizer, worst |err| = {worst:.4f}, "
        f"depth {min(spread):.2f}–{max(spread):.2f})"
    )


# ---------------------------------------------------------------- adapters


def check_adapters():
    import os

    from eval.common import ItemRecorder
    from smoke_battery import _tiny_model

    os.environ.setdefault("PRISM_TEST_EVAL_CAPS", "1")
    ctx = {
        "tokenizer": ByteTok(),
        "device": "cpu",
        "eval_assets_dir": None,
        "eval_tier": "public_dev",
        "eval_secret_seed": None,
        "items": ItemRecorder(),
    }
    model = _tiny_model("cpu")
    for name, mod, families in (
        ("ruler", grl, grl.PROBES),
        ("babilong", gb, [f"qa{q}" for q in gb.TASKS]),
    ):
        out = mod.run(model, ctx)
        assert out, name
        for key, value in out.items():
            assert key.startswith(f"g5.{name}."), key
            assert isinstance(value, float) and value == value, (key, value)
        assert 0.0 <= out[f"g5.{name}.acc"] <= 1.0
        assert out[f"g5.{name}.evidence_miss"] == 0.0, out
        assert 0.2 < out[f"g5.{name}.mean_depth"] < 0.8, out
        for family in families:
            assert f"g5.{name}.{family}.acc" in out, sorted(out)
        for length in mod.GRID_TINY:
            assert f"g5.{name}.L{length}.acc" in out, sorted(out)
            assert out[f"g5.{name}.L{length}.len_err"] < 0.02, out
            for family in families:
                assert f"g5.{name}.{family}.L{length}.acc" in out, sorted(out)
        print(f"{name} adapter OK ({len(out)} metrics, acc={out[f'g5.{name}.acc']:.3f})")

    clusters = {c["cluster"] for v in ctx["items"].dump().values() for c in v}
    assert len(clusters) == len(grl.PROBES) * len(grl.GRID_TINY) + len(gb.TASKS) * len(
        gb.GRID_TINY
    ), sorted(clusters)
    # Grid + probe selection stay parameters (Phase C adds the 64k tier).
    out = grl.run(
        model, ctx, grid=(1024,), probes=("niah_mk", "vt"), n_items=1,
        probe_grid={"vt": (1536,)},
    )
    assert "g5.ruler.niah_mk.L1024.acc" in out, sorted(out)
    assert "g5.ruler.vt.L1536.acc" in out, sorted(out)
    out = gb.run(model, ctx, grid=(1024,), tasks=(1, 2), n_items=1, task_grid={2: (1536,)})
    assert "g5.babilong.qa1.L1024.acc" in out, sorted(out)
    assert "g5.babilong.qa2.L1536.acc" in out, sorted(out)
    print("grid override OK (Phase C can pass grid / probes / probe_grid / task_grid)")


def main():
    check_determinism()
    check_ruler_gold()
    check_babilong_gold()
    check_chance_level()
    check_length_fidelity()
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        print(f"PROTOCOL SMOKE SKIP (torch part): {exc}")
        return 2
    check_adapters()
    print("PROTOCOL SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
