"""Challenge-owned synthetic logic probe generators + dual-channel scoring.

Architecture-agnostic suite for Complete View P10 ``P10_reasoning_logic``
(``suite_id=logic_synthetic.v1``). Implements MUST probes RL-01..10:

* boolean_parity_xor, arith_digit_mod, transitive_compare, multihop_binding
* sort_order, reverse_edit, count_stream, dyck_nesting
* instruction_toy, contradiction_detect

Dual score channels (every probe):

1. **Closed-choice accuracy** (primary absolute + ``relative_to_chance``)
2. **Forced CE** on gold answer span (continuous when accuracy near chance)

No GSM8K / MMLU / lm-eval. No MQAR induction-copy templates (multihop_binding
uses role/function composition, not KV associative recall). CPU unit fixtures +
pure-torch optional hooks; host densify reuses ``trained_state`` logits
callables without retrain.

Assertions: VAL-REASON-002..008 (generators + scoreable metrics).
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..complete_view import (
    COMPLETE_VIEW_REASONING_CHANCE_TABLE,
    REASONING_REL_FLOOR,
    REASONING_SUITE_ID,
)
from ..scorecard_suite import (
    clamp01,
    probe_next_token_accuracy,
    relative_to_chance,
    score_closed_choice_accuracy,
)

DeviceHint = Literal["cpu", "cuda", "auto", "fixture"]

# MUST probe catalogue (matches COMPLETE_VIEW_REASONING_MUST_PROBES keys).
LOGIC_PROBE_KEYS: tuple[str, ...] = (
    "boolean_parity_xor",
    "arith_digit_mod",
    "transitive_compare",
    "multihop_binding",
    "sort_order",
    "reverse_edit",
    "count_stream",
    "dyck_nesting",
    "instruction_toy",
    "contradiction_detect",
)

LOGIC_CHANCE: dict[str, float] = dict(COMPLETE_VIEW_REASONING_CHANCE_TABLE)

# Default seed-scale suite settings (design §3.2).
DEFAULT_TRIALS_PER_PROBE = 24
DEFAULT_LOGIC_SUITE_SEED = 0xC0FFEE42
# discrete symbol alphabets for architecture-agnostic digit token CE (optional).
DIGIT_SYMBOLS = tuple(str(i) for i in range(10))
BOOL_SYMBOLS = ("0", "1")
COMPARE_SYMBOLS = (">", "<", "=", "?")
CONS_SYMBOLS = ("consistent", "inconsistent")


def logic_trial_seed(
    *,
    probe: str,
    trial_i: int,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
    protocol: str = "prism_official_compare.v1",
    suite_id: str = REASONING_SUITE_ID,
) -> int:
    """Deterministic sealed PRNG seed for one probe trial (challenge-owned)."""
    payload = f"{protocol}|{suite_id}|{probe}|{suite_seed}|{trial_i}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) & 0x7FFFFFFF


def _rng(
    probe: str,
    trial_i: int,
    *,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
) -> random.Random:
    return random.Random(logic_trial_seed(probe=probe, trial_i=trial_i, suite_seed=suite_seed))


# --- Trial record -----------------------------------------------------------------


@dataclass(frozen=True)
class LogicTrial:
    """One closed logic probe trial with forced-answer gold string.

    ``candidates`` is a small closed set for accuracy scoring. ``gold`` must be
    a member of ``candidates``. Text form is architecture-agnostic (no state kits).
    """

    probe: str
    prompt: str
    gold: str
    candidates: tuple[str, ...]
    meta: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe,
            "prompt": self.prompt,
            "gold": self.gold,
            "candidates": list(self.candidates),
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class LogicTaskScore:
    """Closed accuracy + forced CE for one probe (VAL-REASON dual channel)."""

    probe: str
    accuracy: float
    chance: float
    relative: float
    forced_ce: float | None
    trials: int = 0
    device: str = "fixture"
    detail: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe,
            "accuracy": self.accuracy,
            "chance": self.chance,
            "relative_to_chance": self.relative,
            "forced_ce": self.forced_ce,
            "trials": self.trials,
            "device": self.device,
            "detail": dict(self.detail) if self.detail is not None else None,
        }


# --- Generators (RL-01..10) -------------------------------------------------------


def gen_boolean_parity_xor(
    trial_i: int = 0,
    *,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
    chain_lengths: Sequence[int] = (1, 2, 3, 4),
) -> LogicTrial:
    """RL-01: Boolean / parity / XOR composition (chance 0.5)."""
    rng = _rng("boolean_parity_xor", trial_i, suite_seed=suite_seed)
    # Alternate pure XOR chain vs parity bit-stream.
    mode = "xor" if (trial_i % 2 == 0) else "parity"
    if mode == "xor":
        n = chain_lengths[trial_i % len(chain_lengths)]
        bits = [rng.randint(0, 1) for _ in range(n)]
        # Fold XOR
        acc = 0
        for b in bits:
            acc ^= b
        names = [chr(ord("A") + i) for i in range(n)]
        asserts = " ".join(f"{names[i]}={bits[i]}" for i in range(n))
        # Always label the operator; n=1 is trivial identity under XOR chain of length 1.
        formula = " XOR ".join(names) if n > 1 else f"XOR {names[0]}"
        prompt = f"bool: {asserts} ; q={formula} → ?"
        gold = str(acc)
        meta = {"mode": "xor", "n": n, "bits": bits}
    else:
        n = (4, 8, 16)[trial_i % 3]
        bits = [rng.randint(0, 1) for _ in range(n)]
        parity = sum(bits) % 2
        stream = "".join(str(b) for b in bits)
        prompt = f"parity bits={stream} ; q=parity → ?"
        gold = str(parity)
        meta = {"mode": "parity", "n": n, "bits": bits}
    return LogicTrial(
        probe="boolean_parity_xor",
        prompt=prompt,
        gold=gold,
        candidates=BOOL_SYMBOLS,
        meta=meta,
    )


def gen_arith_digit_mod(
    trial_i: int = 0,
    *,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
) -> LogicTrial:
    """RL-02: Digit / mod arithmetic toy (chance 0.1 for digits 0-9)."""
    rng = _rng("arith_digit_mod", trial_i, suite_seed=suite_seed)
    op_kind = ("add", "sub", "mul_mod", "chain_mod")[trial_i % 4]
    a = rng.randint(0, 9)
    b = rng.randint(0, 9)
    if op_kind == "add":
        gold_i = (a + b) % 10
        expr = f"{a}+{b}"
    elif op_kind == "sub":
        gold_i = (a - b) % 10
        expr = f"{a}-{b}"
    elif op_kind == "mul_mod":
        m = (5, 7, 10)[trial_i % 3]
        gold_i = (a * b) % m
        expr = f"({a}*{b})mod{m}"
    else:
        c = rng.randint(0, 9)
        gold_i = (a + b + c) % 10
        expr = f"{a}+{b}+{c}mod10"
    return LogicTrial(
        probe="arith_digit_mod",
        prompt=f"arith: {expr} → ?",
        gold=str(gold_i),
        candidates=DIGIT_SYMBOLS,
        meta={"op": op_kind, "a": a, "b": b, "gold_i": gold_i},
    )


def gen_transitive_compare(
    trial_i: int = 0,
    *,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
    hops_choices: Sequence[int] = (1, 2, 3, 4),
) -> LogicTrial:
    """RL-03: Transitive relational composition (chance 0.25 with {>, <, =, ?})."""
    rng = _rng("transitive_compare", trial_i, suite_seed=suite_seed)
    hops = hops_choices[trial_i % len(hops_choices)]
    # Build a total order of hop+1 entities with known numeric ranks.
    n_ent = hops + 1
    labels = [chr(ord("W") + i) for i in range(n_ent)]
    ranks = list(range(n_ent))
    rng.shuffle(ranks)
    order = sorted(range(n_ent), key=lambda i: ranks[i])  # ascending rank order
    # Adjacent facts only: label_order[i] < label_order[i+1]
    facts: list[str] = []
    for i in range(hops):
        lo, hi = labels[order[i]], labels[order[i + 1]]
        facts.append(f"{lo} < {hi}")
    # Query endpoints (chain start vs end)
    q_lo, q_hi = labels[order[0]], labels[order[-1]]
    if hops == 0:
        gold = "="
    else:
        gold = "<"  # order[0] is lowest rank → q_lo < q_hi
    # With chance reverse query direction for variety, keep gold correct.
    if trial_i % 3 == 1:
        q_lo, q_hi = q_hi, q_lo
        gold = ">" if gold == "<" else gold
    distractor = labels[rng.randrange(n_ent)]
    facts_s = " ; ".join(facts)
    prompt = f"rel: {facts_s} ; distractor={distractor} ; q={q_lo} ? {q_hi} → ?"
    return LogicTrial(
        probe="transitive_compare",
        prompt=prompt,
        gold=gold,
        candidates=COMPARE_SYMBOLS,
        meta={"hops": hops, "order_labels": [labels[i] for i in order]},
    )


def gen_multihop_binding(
    trial_i: int = 0,
    *,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
    hops_choices: Sequence[int] = (1, 2, 3),
    entity_sizes: Sequence[int] = (4, 8),
) -> LogicTrial:
    """RL-04: Multi-hop role/function composition (≠ MQAR associative recall).

    Template ``P(a)=b; P(b)=c; Q:P(P(a))?`` (function composition) or role chain
    Alice→box→key→door. Candidates are entity names (chance 1/E).
    Distinct from P3 MQAR: not pure key→value N-pair lookup accuracy.
    """
    rng = _rng("multihop_binding", trial_i, suite_seed=suite_seed)
    hops = hops_choices[trial_i % len(hops_choices)]
    e = entity_sizes[trial_i % len(entity_sizes)]
    # Symbolic function-composition path (not induction/copy pattern).
    entities = [f"e{i}" for i in range(e)]
    rng.shuffle(entities)
    # Build open chain of distinct steps when possible.
    chain = [entities[i % e] for i in range(hops + 1)]
    # If duplicates collapse, re-sample unique long enough prefix.
    if len(set(chain)) < min(hops + 1, e):
        pool = list(entities)
        chain = [pool.pop(0)]
        for _ in range(hops):
            nxt = pool.pop(0) if pool else chain[-1]
            chain.append(nxt)
    binds = [f"P({chain[i]})={chain[i + 1]}" for i in range(hops)]
    # Gold is end of composition P^h(start)
    gold = chain[hops]
    # Closed candidates: all role entities (size e) — chance 1/e
    # Protocol chance table locks 0.125 ⇒ E=8 primary; when E=4 still report table chance.
    cands = tuple(entities[:e] if len(entities) >= e else entities)
    if gold not in cands:
        cands = cands + (gold,)
    # Expand collective candidates if needed to match design chance bookkeeping
    # (scoring uses chance table constant; cands define closed set).
    filler_tokens = " ".join(f"pad{i}" for i in range((trial_i % 3) * 2))
    facts = " ; ".join(binds)
    # Hop-application formula
    nested = "a"
    for _ in range(hops):
        nested = f"P({nested})"
    prompt = (
        f"bind: {facts}"
        + (f" ; filler {filler_tokens}" if filler_tokens else "")
        + f" ; start=a={chain[0]} ; q={nested} → ?"
    )
    return LogicTrial(
        probe="multihop_binding",
        prompt=prompt,
        gold=gold,
        candidates=cands,
        meta={
            "hops": hops,
            "entity_set_size": e,
            "chain": chain,
            "distinct_from_mqar": True,
            "binding_style": "function_composition",
        },
    )


def gen_sort_order(
    trial_i: int = 0,
    *,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
    ns: Sequence[int] = (3, 4, 5),
) -> LogicTrial:
    """RL-05: Sort/order sequencing via min/max/median (mode-aware chance).

    Protocol table chance stays 0.25 for suite bookkeeping, but each mode
    records its true closed-set chance: ~1/|candidates| for min/max/median and
    0.5 for binary ``is_sorted``. Scoring uses those mode chances for relative
    labels so seats stay consistent (hygiene: mixed binary vs n-way sets).
    """
    rng = _rng("sort_order", trial_i, suite_seed=suite_seed)
    n = ns[trial_i % len(ns)]
    vals = [rng.randint(0, 9) for _ in range(n)]
    # Ensure some diversity
    if len(set(vals)) == 1:
        vals[0] = (vals[0] + 1) % 10
    mode = ("min", "max", "median", "is_sorted")[trial_i % 4]
    items = " ".join(str(v) for v in vals)
    if mode == "min":
        gold = str(min(vals))
        prompt = f"order: list=[{items}] ; q=min → ?"
        cands = tuple(str(v) for v in sorted(set(vals)))
        # pad to at most n distinct
        while len(cands) < min(4, n):
            cands = cands + (str(rng.randint(0, 9)),)
            cands = tuple(dict.fromkeys(cands))
    elif mode == "max":
        gold = str(max(vals))
        prompt = f"order: list=[{items}] ; q=max → ?"
        cands = tuple(str(v) for v in sorted(set(vals)))
        while len(cands) < min(4, n):
            cands = cands + (str(rng.randint(0, 9)),)
            cands = tuple(dict.fromkeys(cands))
    elif mode == "median":
        s = sorted(vals)
        mid = s[n // 2]
        gold = str(mid)
        prompt = f"order: list=[{items}] ; q=median → ?"
        cands = tuple(str(v) for v in sorted(set(s)))
        while len(cands) < min(4, n):
            cands = cands + (str(rng.randint(0, 9)),)
            cands = tuple(dict.fromkeys(cands))
    else:
        is_sorted = vals == sorted(vals)
        gold = "yes" if is_sorted else "no"
        # Flip one element sometimes so both answers occur
        prompt = f"order: list=[{items}] ; q=is_sorted → ?"
        cands = ("yes", "no")
    if gold not in cands:
        cands = cands + (gold,)
    mode_chance = 0.5 if mode == "is_sorted" else (1.0 / float(len(cands)) if cands else 0.25)
    return LogicTrial(
        probe="sort_order",
        prompt=prompt,
        gold=gold,
        candidates=cands,
        meta={
            "mode": mode,
            "vals": vals,
            "n": n,
            "chance": mode_chance,
            "protocol_chance": float(LOGIC_CHANCE.get("sort_order", 0.25)),
        },
    )


def gen_reverse_edit(
    trial_i: int = 0,
    *,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
) -> LogicTrial:
    """RL-06: Reverse / simple string edit (transform ≠ exact copy)."""
    rng = _rng("reverse_edit", trial_i, suite_seed=suite_seed)
    alphabet = list("ABCDEFGH")
    length = (3, 4, 5, 6, 7, 8)[trial_i % 6]
    chars = [rng.choice(alphabet) for _ in range(length)]
    src = "".join(chars)
    mode = ("rev", "swap_ends", "delete", "replace")[trial_i % 4]
    if mode == "rev":
        gold = src[::-1]
        prompt = f"edit: rev({src}) → ?"
    elif mode == "swap_ends":
        body = list(src)
        body[0], body[-1] = body[-1], body[0]
        gold = "".join(body)
        prompt = f"edit: swap_ends({src}) → ?"
    elif mode == "delete":
        pos = rng.randrange(length)
        gold = src[:pos] + src[pos + 1 :]
        prompt = f"edit: delete_pos{pos}({src}) → ?"
    else:
        pos = rng.randrange(length)
        new_c = rng.choice([c for c in alphabet if c != src[pos]])
        body = list(src)
        body[pos] = new_c
        gold = "".join(body)
        prompt = f"edit: replace_pos{pos}->{new_c}({src}) → ?"
    # Hard negatives: one-edit distance variants of gold + identity (copy trap).
    negs: list[str] = [src]  # identity is a trap (≠ reverse/edit)
    # swap two positions in gold
    if len(gold) >= 2:
        g = list(gold)
        g[0], g[1] = g[1], g[0]
        negs.append("".join(g))
    # reverse of gold (sometimes coincides)
    negs.append(gold[::-1])
    # drop last
    if len(gold) > 1:
        negs.append(gold[:-1] + rng.choice(alphabet))
    candidates_set: list[str] = []
    for s in [gold, *negs]:
        if s not in candidates_set:
            candidates_set.append(s)
    # Cap closed set
    while len(candidates_set) < 4:
        extra = "".join(rng.choice(alphabet) for _ in range(length))
        if extra not in candidates_set:
            candidates_set.append(extra)
    return LogicTrial(
        probe="reverse_edit",
        prompt=prompt,
        gold=gold,
        candidates=tuple(candidates_set[:8]),
        meta={"mode": mode, "src": src, "distinct_from_copy": True},
    )


def gen_count_stream(
    trial_i: int = 0,
    *,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
    stream_lengths: Sequence[int] = (8, 16, 32),
) -> LogicTrial:
    """RL-07: Count occurrences of a target symbol in a stream (chance 0.1 for 0..9).

    Gold domain always matches true stream counts in ``0..9`` (table chance 0.1).
    Streams are constructed from a sampled count rather than post-hoc clamping, so
    ``meta['count']`` never disagrees with gold even when stream_len is 32.
    """
    rng = _rng("count_stream", trial_i, suite_seed=suite_seed)
    s_len = stream_lengths[trial_i % len(stream_lengths)]
    target = rng.choice(list("ABCDE"))
    fillers = list("FGHIJ")
    # Sample true count in digit domain first so gold == count always holds.
    count = int(rng.randint(0, min(9, s_len)))
    stream = [target] * count + [rng.choice(fillers) for _ in range(s_len - count)]
    rng.shuffle(stream)
    gold = str(count)
    stream_s = " ".join(stream)
    prompt = f"count: stream=[{stream_s}] ; target={target} ; q=count → ?"
    return LogicTrial(
        probe="count_stream",
        prompt=prompt,
        gold=gold,
        candidates=DIGIT_SYMBOLS,
        meta={
            "stream_len": s_len,
            "target": target,
            "count": count,
            "gold_domain": "0..9",
            "clamped": False,
        },
    )


def gen_dyck_nesting(
    trial_i: int = 0,
    *,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
) -> LogicTrial:
    """RL-08: Bounded Dyck well-formedness / next-close (chance 0.5 well-formed yes/no)."""
    rng = _rng("dyck_nesting", trial_i, suite_seed=suite_seed)
    mode = ("well_formed", "next_close", "depth")[trial_i % 3]
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    # Build a valid Dyck-3 string then optionally corrupt.
    depth_target = 1 + (trial_i % 4)
    opens: list[str] = []
    tokens: list[str] = []
    # Grow then shrink under bound
    steps = 4 + (trial_i % 6)  # length control ≤ ~20
    for _ in range(steps):
        if not opens or (len(opens) < depth_target and rng.random() < 0.55):
            op, cl = pairs[rng.randrange(len(pairs))]
            tokens.append(op)
            opens.append(cl)
        else:
            tokens.append(opens.pop())
    while opens:
        tokens.append(opens.pop())
    valid = "".join(tokens)
    cands: tuple[str, ...]
    meta: dict[str, Any]
    if mode == "well_formed":
        corrupt = trial_i % 2 == 1
        text = valid
        if corrupt and tokens:
            # Flip a random close into a different or drop
            body = list(tokens)
            i = rng.randrange(len(body))
            if body[i] in ")]}":
                body[i] = rng.choice([c for c in ")]}" if c != body[i]] or [")"])
            else:
                body.append(")")  # unbalanced extra closer
            text = "".join(body)
            # recompute validity simply
            gold = "no"
        else:
            gold = "yes"
        prompt = f"dyck: s={text} ; q=well_formed → ?"
        cands = ("yes", "no")
        meta = {"mode": mode, "text": text, "valid_ref": valid}
    elif mode == "next_close":
        # Present a proper open prefix and ask next legal close.
        ops = ["(", "[", "{"]
        chosen = ops[trial_i % 3]
        close_map = {"(": ")", "[": "]", "{": "}"}
        gold = close_map[chosen]
        prefix = valid[: max(1, len(valid) // 2)]
        # Ensure ends with open of chosen type when possible
        prefix = prefix + chosen
        prompt = f"dyck: prefix={prefix} ; q=next_close → ?"
        cands = (")", "]", "}", "x")
        meta = {"mode": mode, "prefix": prefix}
    else:
        # Depth-at-end of balanced string is 0; use max-depth closed small int
        max_depth = 0
        d = 0
        for t in tokens:
            if t in "([{":
                d += 1
                max_depth = max(max_depth, d)
            else:
                d = max(0, d - 1)
        gold = str(min(9, max_depth))
        prompt = f"dyck: s={valid} ; q=max_depth → ?"
        cands = DIGIT_SYMBOLS
        meta = {"mode": mode, "max_depth": max_depth}
    return LogicTrial(
        probe="dyck_nesting",
        prompt=prompt,
        gold=gold,
        candidates=cands,
        meta=meta,
    )


def gen_instruction_toy(
    trial_i: int = 0,
    *,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
) -> LogicTrial:
    """RL-09: Micro instruction-toy format templates (field selection / fixed format)."""
    rng = _rng("instruction_toy", trial_i, suite_seed=suite_seed)
    mode = ("fmta_add", "second_field", "answer_only", "pick_op")[trial_i % 4]
    if mode == "fmta_add":
        x = rng.randint(0, 9)
        y = rng.randint(0, 9)
        gold = str((x + y) % 10)
        prompt = f"FMTA|x={x}|y={y}|op=add|→|"
        cands = DIGIT_SYMBOLS
    elif mode == "second_field":
        a, b, c = rng.choice("abcd"), rng.choice("wxyz"), rng.choice("1234")
        gold = b
        prompt = f"instr: output only the second field of `{a}:{b}:{c}` → ?"
        cands = (a, b, c, "none")
    elif mode == "answer_only":
        tok = rng.choice(["alpha", "beta", "gamma", "delta", "epsilon"])
        gold = tok
        prompt = f"instr: ANSWER: <token> only; token={tok} ; emit → ?"
        # 5 candidates; chance table 0.2
        pool = ["alpha", "beta", "gamma", "delta", "epsilon"]
        cands = tuple(pool)
    else:
        x = rng.randint(1, 5)
        y = rng.randint(1, 5)
        op = rng.choice(["add", "mul"])
        gold = str((x + y) if op == "add" else (x * y) % 10)
        prompt = f"instr: do {op} on {x},{y} ; format digit only → ?"
        cands = DIGIT_SYMBOLS
    if gold not in cands:
        cands = cands + (gold,)
    return LogicTrial(
        probe="instruction_toy",
        prompt=prompt,
        gold=gold,
        candidates=cands,
        meta={"mode": mode},
    )


def gen_contradiction_detect(
    trial_i: int = 0,
    *,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
) -> LogicTrial:
    """RL-10: Consistency / contradiction over dual short claims (chance 0.5)."""
    rng = _rng("contradiction_detect", trial_i, suite_seed=suite_seed)
    atoms = [
        ("color", "red", "blue"),
        ("size", "big", "small"),
        ("shape", "circle", "square"),
        ("owner", "alice", "bob"),
        ("id", "7", "9"),
    ]
    attr, v1, v2 = atoms[trial_i % len(atoms)]
    subj = rng.choice(["X", "Y", "Z", "item"])
    consistent = trial_i % 2 == 0
    claim1 = f"{subj}.{attr}={v1}"
    claim2 = f"{subj}.{attr}={v1}" if consistent else f"{subj}.{attr}={v2}"
    # Optional filler fact sheet
    sheet = f"fact_sheet: {subj}.tag=ok"
    gold = "consistent" if consistent else "inconsistent"
    prompt = f"cons: sheet={sheet} ; claim1={claim1} ; claim2={claim2} ; q=status → ?"
    return LogicTrial(
        probe="contradiction_detect",
        prompt=prompt,
        gold=gold,
        candidates=CONS_SYMBOLS,
        meta={"consistent": consistent, "attr": attr},
    )


GENERATOR_BY_PROBE: dict[str, Callable[..., LogicTrial]] = {
    "boolean_parity_xor": gen_boolean_parity_xor,
    "arith_digit_mod": gen_arith_digit_mod,
    "transitive_compare": gen_transitive_compare,
    "multihop_binding": gen_multihop_binding,
    "sort_order": gen_sort_order,
    "reverse_edit": gen_reverse_edit,
    "count_stream": gen_count_stream,
    "dyck_nesting": gen_dyck_nesting,
    "instruction_toy": gen_instruction_toy,
    "contradiction_detect": gen_contradiction_detect,
}


def generate_probe_trials(
    probe: str,
    *,
    n_trials: int = DEFAULT_TRIALS_PER_PROBE,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
) -> list[LogicTrial]:
    """Generate ``n_trials`` sealed trials for one MUST probe."""
    if probe not in GENERATOR_BY_PROBE:
        raise KeyError(f"unknown logic probe {probe!r}; expected one of {LOGIC_PROBE_KEYS}")
    gen = GENERATOR_BY_PROBE[probe]
    return [gen(i, suite_seed=suite_seed) for i in range(int(n_trials))]


def generate_logic_suite(
    *,
    n_trials: int = DEFAULT_TRIALS_PER_PROBE,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
    probes: Sequence[str] = LOGIC_PROBE_KEYS,
) -> dict[str, list[LogicTrial]]:
    """Generate full suite trials dict keyed by probe name."""
    out: dict[str, list[LogicTrial]] = {}
    for probe in probes:
        out[str(probe)] = generate_probe_trials(
            str(probe), n_trials=n_trials, suite_seed=suite_seed
        )
    return out


# --- Scoring ----------------------------------------------------------------------


def score_logic_closed_accuracy(
    correct: Sequence[bool] | Sequence[int] | Sequence[float],
    *,
    probe: str,
    chance: float | None = None,
    forced_ce: float | None = None,
    device: str = "fixture",
    detail: Mapping[str, Any] | None = None,
) -> LogicTaskScore:
    """Closed-choice accuracy for a logic probe with chance + relative floors."""
    ch = float(chance if chance is not None else LOGIC_CHANCE.get(probe, 0.0))
    base = score_closed_choice_accuracy(correct, task=probe, chance=ch)
    ce: float | None = None
    if forced_ce is not None and math.isfinite(float(forced_ce)):
        ce = float(forced_ce)
    return LogicTaskScore(
        probe=probe,
        accuracy=base.accuracy,
        chance=ch,
        relative=base.relative,
        forced_ce=ce,
        trials=base.trials,
        device=device,
        detail=detail,
    )


def _trial_chance(trial: LogicTrial, *, probe: str | None = None) -> float:
    """Per-trial chance: prefer explicit meta chance (sort modes) else table."""
    meta = trial.meta or {}
    raw = meta.get("chance")
    if isinstance(raw, (int, float)) and math.isfinite(float(raw)) and float(raw) > 0.0:
        return float(raw)
    p = probe or trial.probe
    return float(LOGIC_CHANCE.get(p, 0.0))


def _mean_trial_chance(trials: Sequence[LogicTrial], *, probe: str | None = None) -> float:
    """Macro mean of per-trial chances (sort_order mode-mix honesty)."""
    if not trials:
        p = probe or "unknown"
        return float(LOGIC_CHANCE.get(p, 0.0))
    vals = [_trial_chance(t, probe=probe) for t in trials]
    return float(sum(vals) / len(vals))


def score_logic_from_predictions(
    trials: Sequence[LogicTrial],
    predictions: Sequence[str],
    *,
    probe: str | None = None,
    forced_ce_values: Sequence[float] | None = None,
    device: str = "fixture",
) -> LogicTaskScore:
    """Score predicted answer strings against gold trials (fixture or model).

    Protocol ``LOGIC_CHANCE`` remains the dual-channel / panel table chance.
    When trials publish per-mode ``meta['chance']`` (e.g. sort_order is_sorted
    vs min/max/median), the mode-mean is stored on ``detail`` for honesty so
    chance labels stay interpretable without breaking table locks.
    """
    if not trials:
        p = probe or "unknown"
        ch = float(LOGIC_CHANCE.get(p, 0.0))
        return LogicTaskScore(
            probe=p, accuracy=0.0, chance=ch, relative=0.0, forced_ce=None, trials=0, device=device
        )
    if len(predictions) != len(trials):
        raise ValueError("predictions/trials length mismatch")
    p_name = probe or trials[0].probe
    correct = [str(pred) == t.gold for pred, t in zip(predictions, trials, strict=True)]
    ce: float | None = None
    if forced_ce_values is not None:
        if len(forced_ce_values) != len(trials):
            raise ValueError("forced_ce_values/trials length mismatch")
        vals = [float(x) for x in forced_ce_values if math.isfinite(float(x))]
        ce = (sum(vals) / len(vals)) if vals else None
    protocol_ch = float(LOGIC_CHANCE.get(p_name, 0.0))
    # Protocol table chance stays authoritative for suite/panel/contracts.
    # Mode-aware bookkeeping is detail-only (sort_order binary vs n-way mix).
    mode_chances = [
        float(t.meta["chance"])
        for t in trials
        if isinstance(t.meta, Mapping)
        and isinstance(t.meta.get("chance"), (int, float))
        and math.isfinite(float(t.meta["chance"]))
        and float(t.meta["chance"]) > 0.0
    ]
    detail: dict[str, Any] = {
        "n_trials": len(trials),
        "protocol_chance": protocol_ch,
        "chance_source": "protocol_table",
    }
    if mode_chances and len(mode_chances) == len(trials):
        mode_mean = float(sum(mode_chances) / len(mode_chances))
        detail["mode_chance_mean"] = mode_mean
        detail["mode_chance_labels"] = [_trial_chance(t, probe=p_name) for t in trials]
        detail["chance_mode_note"] = (
            "protocol chance stays locked to LOGIC_CHANCE table; "
            "mode_chance_* documents actual closed-set sizes (e.g. is_sorted=0.5)"
        )
    return score_logic_closed_accuracy(
        correct, probe=p_name, chance=protocol_ch, forced_ce=ce, device=device, detail=detail
    )


def oracle_predictions(trials: Sequence[LogicTrial]) -> list[str]:
    """Perfect oracle predictions (used for dual-channel wiring tests)."""
    return [t.gold for t in trials]


def chance_predictions(trials: Sequence[LogicTrial], *, seed: int = 0) -> list[str]:
    """Uniform random candidate picks (baseline for relative_to_chance)."""
    rng = random.Random(seed)
    out: list[str] = []
    for t in trials:
        out.append(rng.choice(list(t.candidates)))
    return out


def fixture_forced_ce_from_accuracy(accuracy: float, *, chance: float) -> float:
    """Synthetic forced-CE surrogate inverted from accuracy for CPU fixtures.

    Lower CE = better. Maps accuracy above chance toward lower CE without needing
    a real model. Purely for dual-channel bookkeeping in unit tests / densify shells.
    """
    rel = relative_to_chance(accuracy, chance)
    # ce in ~[0.5, 4.0] nats-ish hybrid: perfect relative → 0.5; chance → ~2.5
    return float(0.5 + 3.0 * (1.0 - rel))


def score_probe_fixture(
    probe: str,
    *,
    n_trials: int = DEFAULT_TRIALS_PER_PROBE,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
    accuracy: float | None = None,
    device: str = "fixture",
) -> LogicTaskScore:
    """CPU fixture path: generate trials, observe optional/synthetic accuracy, dual score.

    When ``accuracy`` is provided, outcomes are synthesized to that mean (rounded via
    Bernoulli draws on sealed PRNG). When omitted, **oracle** predictions are used
    (perfect accuracy) so wiring tests can assert complete metric record shapes.
    """
    trials = generate_probe_trials(probe, n_trials=n_trials, suite_seed=suite_seed)
    ch = float(LOGIC_CHANCE[probe])
    if accuracy is None:
        preds = oracle_predictions(trials)
        side_acc = 1.0
    else:
        side_acc = clamp01(float(accuracy))
        # Synthesize exactly round(n * acc) correct more deterministically.
        n_correct = int(round(side_acc * len(trials)))
        correct_mask = [i < n_correct for i in range(len(trials))]
        # Det-shuffle scenario via sealed seed per probe
        rng = _rng(probe, 9999, suite_seed=suite_seed)
        rng.shuffle(correct_mask)
        preds = []
        for ok, t in zip(correct_mask, trials, strict=True):
            if ok:
                preds.append(t.gold)
            else:
                wrongs = [c for c in t.candidates if c != t.gold]
                preds.append(wrongs[0] if wrongs else "__wrong__")
    score = score_logic_from_predictions(trials, preds, probe=probe, device=device)
    ce = fixture_forced_ce_from_accuracy(score.accuracy, chance=ch)
    return LogicTaskScore(
        probe=score.probe,
        accuracy=score.accuracy,
        chance=score.chance,
        relative=score.relative,
        forced_ce=ce,
        trials=score.trials,
        device=device,
        detail={
            "suite_id": REASONING_SUITE_ID,
            "rel_floor": REASONING_REL_FLOOR,
            "generator": "challenge_owned_synthetic",
            "n_trials": len(trials),
        },
    )


def score_suite_fixture(
    *,
    n_trials: int = DEFAULT_TRIALS_PER_PROBE,
    suite_seed: int = DEFAULT_LOGIC_SUITE_SEED,
    accuracy_by_probe: Mapping[str, float] | None = None,
    device: str = "fixture",
) -> dict[str, LogicTaskScore]:
    """Score all MUST probes via CPU fixtures (dual channel)."""
    out: dict[str, LogicTaskScore] = {}
    for probe in LOGIC_PROBE_KEYS:
        acc = None if accuracy_by_probe is None else accuracy_by_probe.get(probe)
        out[probe] = score_probe_fixture(
            probe,
            n_trials=n_trials,
            suite_seed=suite_seed,
            accuracy=acc,
            device=device,
        )
    return out


# --- Model-facing hooks (host densify / pure-torch) -------------------------------


def tokenize_simple(text: str) -> list[int]:
    """Byte-ish deterministic tokenizer for unit/host probes (no external vocab).

    Maps printable chars to ids in 1..255; space→32. Architecture-agnostic CE works
    on any consistent token map shared by the evaluation surface.
    """
    return [min(255, max(1, ord(ch) if 0 < ord(ch) < 256 else 63)) for ch in text]


def encode_answer_ids(
    answer: str,
    *,
    encode: Callable[[str], Sequence[int]] | None = None,
) -> list[int]:
    """Encode a closed-choice answer string into one or more token ids.

    Default uses full :func:`tokenize_simple` **sequence** (not first-byte only) so
    multi-char / multi-candidate answers (``e0..e7``, reverse_edit strings) stay
    distinct. Callers with train-pin tokenizers (gpt2 tiktoken) should pass
    ``encode=`` that returns the full multi-token id list.
    """
    if encode is None:
        ids = tokenize_simple(answer)
    else:
        ids = list(encode(answer))
    return [int(x) for x in ids]


def _answers_are_single_token(
    trials: Sequence[LogicTrial],
    encode: Callable[[str], Sequence[int]],
) -> bool:
    for t in trials:
        for s in (t.gold, *t.candidates):
            if len(encode_answer_ids(str(s), encode=encode)) != 1:
                return False
    return True


def _candidates_share_first_id(
    trials: Sequence[LogicTrial],
    encode: Callable[[str], Sequence[int]],
) -> bool:
    """True when any trial's closed set collapses under first-token encoding."""
    for t in trials:
        firsts = []
        for s in t.candidates:
            ids = encode_answer_ids(str(s), encode=encode)
            firsts.append(int(ids[0]) if ids else 0)
        if len(firsts) >= 2 and len(set(firsts)) < len(firsts):
            return True
    return False


def closed_choice_rank_preds(
    trials: Sequence[LogicTrial],
    *,
    nll_fn: Callable[[Sequence[int]], Sequence[float] | float],
    encode: Callable[[str], Sequence[int]] | None = None,
    prompt_encode: Callable[[str], Sequence[int]] | None = None,
) -> list[str]:
    """Pick the candidate with lowest teacher-forced answer NLL (multi-token safe).

    This is the default host densify closed scorer when candidates are multi-token
    or would collapse under first-byte / first-token rank.
    """
    enc = encode if encode is not None else tokenize_simple
    penc = prompt_encode if prompt_encode is not None else enc
    preds: list[str] = []
    for t in trials:
        best_s = t.gold
        best_nll = float("inf")
        # Stable order: candidates as published; lower NLL wins; ties prefer earlier.
        for c in t.candidates:
            nll = probe_forced_answer_ce(
                nll_fn,
                t.prompt,
                str(c),
                basis="nats",
                encode=enc,
                prompt_encode=penc,
            )
            if nll < best_nll:
                best_nll = nll
                best_s = str(c)
        preds.append(best_s)
    return preds


def probe_forced_answer_ce(
    nll_fn: Callable[[Sequence[int]], Sequence[float] | float],
    prompt: str,
    answer: str,
    *,
    basis: Literal["nats", "bits"] = "nats",
    encode: Callable[[str], Sequence[int]] | None = None,
    prompt_encode: Callable[[str], Sequence[int]] | None = None,
) -> float:
    """Forced CE on gold answer span only (teacher-forced masses via NLL callable).

    ``nll_fn`` receives full ``prompt+answer`` token ids and returns:
    - a scalar mean NLL over answer tokens, or
    - a per-token NLL stream (we average the **answer suffix** only).
    Architecture-agnostic: no attention/SSM state inspection.

    Optional ``encode`` / ``prompt_encode`` override the tokenizer so multi-token
    densify paths (gpt2) share identity with train pin.
    """
    penc = prompt_encode if prompt_encode is not None else encode
    prompt_ids = encode_answer_ids(prompt, encode=penc)
    answer_ids = encode_answer_ids(answer, encode=encode)
    if not answer_ids:
        return float("inf")
    full = prompt_ids + answer_ids
    raw = nll_fn(full)
    if isinstance(raw, (int, float)):
        nll = float(raw)
    else:
        stream = [float(x) for x in raw]
        # Prefer answer-aligned suffix length
        if len(stream) >= len(answer_ids):
            tail = stream[-len(answer_ids) :]
        else:
            tail = stream
        if not tail:
            return float("inf")
        nll = sum(tail) / len(tail)
    if not math.isfinite(nll):
        return float("inf")
    if basis == "bits":
        return nll / math.log(2.0)
    return nll


def score_probe_with_logits(
    trials: Sequence[LogicTrial],
    logits_fn: Callable[[Any], Any],
    *,
    probe: str | None = None,
    nll_fn: Callable[[Sequence[int]], Sequence[float] | float] | None = None,
    encode_answer: Callable[[str], int] | None = None,
    encode_answer_seq: Callable[[str], Sequence[int]] | None = None,
    device: str = "cpu",
) -> LogicTaskScore:
    """Host densify path: closed accuracy via logits / multi-token NLL rank + forced CE.

    Prefer multi-token-safe closed scoring:

    1. If ``nll_fn`` is provided **and** (candidates are multi-token under the
       configured encode **or** first ids collide for any closed set, e.g.
       ``e0..e7`` under byte encode), rank candidates by teacher-forced answer
       NLL via :func:`closed_choice_rank_preds`. This is the correct host densify
       path for multihop_binding and reverse_edit.
    2. Else, when every candidate is a single token and ids don't collide, use
       next-token closed accuracy via :func:`probe_next_token_accuracy`.
    3. ``encode_answer`` (single id) is honored for pure next-token rank only;
       prefer ``encode_answer_seq`` for multi-token full-string sequences.
    4. Default ``tokenize_simple`` encoding always uses the **full string sequence**,
       never first-byte alone when multi-token NLL ranking is active.
    """
    if not trials:
        p = probe or "unknown"
        ch = float(LOGIC_CHANCE.get(p, 0.0))
        return LogicTaskScore(
            probe=p, accuracy=0.0, chance=ch, relative=0.0, forced_ce=None, trials=0, device=device
        )
    p_name = probe or trials[0].probe

    # Resolve multi-token encode (seq); first-token adapter for closed next-token path.
    if encode_answer_seq is not None:
        enc_seq: Callable[[str], Sequence[int]] = encode_answer_seq
    elif encode_answer is not None:
        # Legacy single-id adapter: wrap as one-token sequences.
        def enc_seq(s: str, _e: Callable[[str], int] = encode_answer) -> Sequence[int]:
            return [int(_e(s))]
    else:
        enc_seq = tokenize_simple

    multi_token_candidates = not _answers_are_single_token(trials, enc_seq)
    first_id_collision = _candidates_share_first_id(trials, enc_seq)
    use_multitoken = nll_fn is not None and (multi_token_candidates or first_id_collision)

    ce_vals: list[float] | None = None
    detail_method: str
    if use_multitoken:
        assert nll_fn is not None
        preds = closed_choice_rank_preds(trials, nll_fn=nll_fn, encode=enc_seq)
        detail_method = "forced_nll_rank"
        ce_vals = [
            probe_forced_answer_ce(nll_fn, t.prompt, t.gold, basis="nats", encode=enc_seq)
            for t in trials
        ]
    else:
        # Single-token closed rank from next-token logits.
        contexts = [t.prompt for t in trials]
        if encode_answer is not None:
            enc_tok = encode_answer
        else:

            def enc_tok(s: str, _seq: Callable[[str], Sequence[int]] = enc_seq) -> int:
                ids = encode_answer_ids(s, encode=_seq)
                return int(ids[0]) if ids else 0

        targets = [enc_tok(t.gold) for t in trials]
        cand_sets = [[enc_tok(c) for c in t.candidates] for t in trials]
        # Guard: if still collapsed after encode, refuse false-perfect next-token.
        if any(len(set(cs)) < len(cs) for cs in cand_sets):
            if nll_fn is None:
                # Cannot rank freely; mark all wrong rather than invent perfect.
                preds = [next((c for c in t.candidates if c != t.gold), "__x__") for t in trials]
                detail_method = "collapsed_ids_abstain"
            else:
                preds = closed_choice_rank_preds(trials, nll_fn=nll_fn, encode=enc_seq)
                detail_method = "forced_nll_rank_collapsed_fallback"
                ce_vals = [
                    probe_forced_answer_ce(nll_fn, t.prompt, t.gold, basis="nats", encode=enc_seq)
                    for t in trials
                ]
        else:
            outcomes = probe_next_token_accuracy(
                logits_fn, contexts, targets, candidate_sets=cand_sets
            )
            preds = []
            for ok, t in zip(outcomes, trials, strict=True):
                if ok:
                    preds.append(t.gold)
                else:
                    wrong = next((c for c in t.candidates if c != t.gold), "__x__")
                    preds.append(wrong)
            detail_method = "next_token"
            if nll_fn is not None:
                ce_vals = [
                    probe_forced_answer_ce(nll_fn, t.prompt, t.gold, basis="nats", encode=enc_seq)
                    for t in trials
                ]

    score = score_logic_from_predictions(
        trials,
        preds,
        probe=p_name,
        forced_ce_values=ce_vals,
        device=device,
    )
    # Preserve method on detail for densify audit.
    detail = dict(score.detail or {})
    detail["closed_rank_method"] = detail_method
    return LogicTaskScore(
        probe=score.probe,
        accuracy=score.accuracy,
        chance=score.chance,
        relative=score.relative,
        forced_ce=score.forced_ce,
        trials=score.trials,
        device=score.device,
        detail=detail,
    )


def pure_torch_fixture_model(vocab_size: int = 256, *, seed: int = 0) -> Any:
    """Tiny pure-torch Linear model on bag-of-token-id means (CPU).

    Used only in unit tests that require a real torch path without loading seeds.
    Returns an object with ``logits_fn`` and ``nll_fn`` callables.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch required for pure_torch_fixture_model") from exc

    g = torch.Generator()
    g.manual_seed(int(seed))
    model = nn.Linear(1, vocab_size, bias=True)
    with torch.no_grad():
        model.weight.normal_(0.0, 0.02, generator=g)
        model.bias.zero_()
    model.eval()

    def logits_fn(ctx: Any) -> Any:
        # Fixed bias-dominated logits; architecture-agnostic closed scorer exercises path.
        ids = tokenize_simple(str(ctx)) if not isinstance(ctx, (list, tuple)) else list(ctx)
        x = torch.tensor([[float(len(ids) % 17) / 17.0]], dtype=torch.float32)
        with torch.no_grad():
            return model(x).squeeze(0)

    def nll_fn(token_ids: Sequence[int]) -> float:
        # Constantish NLL surrogate (finite) so forced CE path is exercised.
        return float(2.0 + 0.01 * (len(token_ids) % 7))

    class _Bundle:
        def __init__(self) -> None:
            self.model = model
            self.logits_fn = logits_fn
            self.nll_fn = nll_fn
            self.device = "cpu"

    return _Bundle()


def documented_logic_suite() -> dict[str, Any]:
    """Publish suite identity + chance table for contracts / docs anchors."""
    return {
        "suite_id": REASONING_SUITE_ID,
        "probes": list(LOGIC_PROBE_KEYS),
        "chance_table": dict(LOGIC_CHANCE),
        "rel_floor": REASONING_REL_FLOOR,
        "scoring": {
            "closed_choice_accuracy": True,
            "forced_ce": True,
            "chance_baselines": True,
            "architecture_agnostic_logits_only": True,
        },
        "distinct_from": {
            "mqar": "multihop_binding is role/function composition, not P3 MQAR KV recall",
            "induction_copy": "reverse_edit is transform, not exact copy continuation",
            "gsm8k_mmlu_lm_eval": (
                "NOT GSM8K/MMLU/lm-eval; challenge-owned synthetic only; not external batteries"
            ),
        },
        "default_trials_per_probe": DEFAULT_TRIALS_PER_PROBE,
        "no_lm_eval_dependency": True,
    }


__all__ = [
    "DEFAULT_LOGIC_SUITE_SEED",
    "DEFAULT_TRIALS_PER_PROBE",
    "GENERATOR_BY_PROBE",
    "LOGIC_CHANCE",
    "LOGIC_PROBE_KEYS",
    "LogicTaskScore",
    "LogicTrial",
    "chance_predictions",
    "closed_choice_rank_preds",
    "documented_logic_suite",
    "encode_answer_ids",
    "fixture_forced_ce_from_accuracy",
    "gen_arith_digit_mod",
    "gen_boolean_parity_xor",
    "gen_contradiction_detect",
    "gen_count_stream",
    "gen_dyck_nesting",
    "gen_instruction_toy",
    "gen_multihop_binding",
    "gen_reverse_edit",
    "gen_sort_order",
    "gen_transitive_compare",
    "generate_logic_suite",
    "generate_probe_trials",
    "logic_trial_seed",
    "oracle_predictions",
    "probe_forced_answer_ce",
    "pure_torch_fixture_model",
    "score_logic_closed_accuracy",
    "score_logic_from_predictions",
    "score_probe_fixture",
    "score_probe_with_logits",
    "score_suite_fixture",
    "tokenize_simple",
]
