"""Shared helpers for the G1–G8 battery modules (E2).

Pure stdlib at import time (torch is imported lazily inside the scoring
helpers) so the battery registry can discover modules on any box,
including CPU-only dev machines.

Seeding lattice: every procedural item derives from

    seed = cantor(cantor(RECIPE_SEED, secret_seed), int(sha256(task_id)[:8]))

where `secret_seed` is delivered via env `PRISM_EVAL_SECRET_SEED` by the
operator *after* the train subprocess is dead (private tier), or falls
back to the published public-dev seed (public_dev tier). Gold answers
are always computed by the harness from the seed — never read from
miner-visible files.

Item format (produced by generators, consumed by battery modules):
    {"task": str, "prompt": str, "choices": [str], "gold": int,
     "cluster": str, "meta": {...}}
The universal scorer is likelihood-based (`acc_norm`: max sum-logprob of
the continuation normalized per character), which keeps every metric
smooth at 100M–350M scale (Schaeffer 2304.15004 design rule). The
per-item side channel (ItemRecorder) stores {cluster, value} records —
cluster = template/variant id, the unit of randomization for the
clustered bootstrap in the Rust composite.

Every length in this battery is measured in tokens of the **submitted**
tokenizer (`prismlib.tokenizer`), reached through [`tokenizer_of`] /
[`encode`] / [`token_len`]; [`fit_to_tokens`] is the one place that builds
a context of an exact token budget. Nothing here may assume GPT-2 or
approximate token counts from word counts.
"""

import hashlib
import json
import math
import os
import time

from prismlib import LN2, RECIPE_SEED
from prismlib import tokenizer as tok_contract
from prismlib.envutil import float_env, int_env, log

PUBLIC_DEV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public_dev")
PUBLIC_DEV_SEED = 2654435761  # published dev seed; mirrors public_dev/seeds.json
ITEM_CAP_PER_METRIC = 2000
ITEM_CAP_GLOBAL = 30000


# ---------------------------------------------------------------- seeds / env


def cantor(a, b):
    """Cantor pairing π(a,b) — deterministic seed combiner."""
    a, b = int(a), int(b)
    s = a + b
    return (s * (s + 1)) // 2 + b


def task_seed(secret_seed, task_id):
    """seed = cantor(RECIPE_SEED, secret, task_id) per the v3 lattice."""
    tid = int.from_bytes(hashlib.sha256(str(task_id).encode()).digest()[:8], "big")
    return cantor(cantor(RECIPE_SEED, int(secret_seed)), tid)


# torch seeding APIs take a signed int64; the raw Cantor lattice value is a
# ~190-bit integer and overflows (`Overflow when unpacking long long`).
TORCH_SEED_MODULUS = (1 << 63) - 1


def torch_seed(secret_seed, task_id):
    """Lattice seed reduced into torch's int64 range.

    Python's `random` keeps the full Cantor value (the determinism
    contract); torch's `manual_seed` / `torch.Generator` get the same
    lattice point reduced mod 2^63-1 — deterministic across platforms for
    the same (secret, task) pair.
    """
    return task_seed(secret_seed, task_id) % TORCH_SEED_MODULUS


def resolve_secret_seed(ctx):
    """Private seed from ctx (injected from env by eval_v3) else public dev."""
    s = ctx.get("eval_secret_seed")
    if s is None:
        return PUBLIC_DEV_SEED
    try:
        return int(s)
    except (TypeError, ValueError):
        return PUBLIC_DEV_SEED


def eval_tier(ctx):
    if ctx.get("eval_tier"):
        return str(ctx["eval_tier"])
    return "private" if ctx.get("eval_assets_dir") else "public_dev"


def tiny_caps():
    """Test-mode tiny caps (grids 512–2k, fewer items, sweeps stubbed)."""
    if int_env("PRISM_TEST_EVAL_CAPS", 0) == 1:
        return True
    if float_env("PRISM_TEST_TRAIN_MINUTES", 0.0) > 0:
        return True
    return os.environ.get("PRISM_TEST_MAX_PARAMS") is not None


def group_budget_s(group, default):
    return float_env(f"PRISM_EVAL_{group.upper()}_BUDGET_S", default)


class Budget:
    """Per-group wall-clock budget: check between items, never mid-item."""

    def __init__(self, seconds):
        self.seconds = float(seconds)
        self.t0 = time.time()
        self.expired = False

    def ok(self):
        if self.expired:
            return False
        if time.time() - self.t0 > self.seconds:
            self.expired = True
            return False
        return True


class ItemRecorder:
    """Per-item {cluster, value} side channel (≤2000/metric, global cap)."""

    def __init__(self):
        self.items = {}
        self.total = 0

    def add(self, metric, cluster, value):
        if self.total >= ITEM_CAP_GLOBAL:
            return
        recs = self.items.setdefault(str(metric), [])
        if len(recs) >= ITEM_CAP_PER_METRIC:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        recs.append({"cluster": str(cluster)[:128], "value": v})
        self.total += 1

    def dump(self):
        return self.items


def record(ctx, metric, cluster, value):
    rec = ctx.get("items")
    if rec is not None and hasattr(rec, "add"):
        rec.add(metric, cluster, value)


# ---------------------------------------------------------------- assets


def assets_path(ctx, rel):
    """Resolve an eval asset: private mirror first, public-dev fallback.

    Private tier: `$PRISM_EVAL_ASSETS_DIR/<rel>`. Public dev tier (or
    missing private file): `eval/public_dev/<rel>`. Returns None when the
    file exists in neither place.
    """
    base = ctx.get("eval_assets_dir")
    if base:
        p = os.path.join(str(base), rel)
        if os.path.isfile(p):
            return p
    p = os.path.join(PUBLIC_DEV_DIR, rel)
    return p if os.path.isfile(p) else None


def load_jsonl(path, cap=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
            if cap is not None and len(rows) >= cap:
                break
    return rows


# ---------------------------------------------------------- tokenizer contract


def tokenizer_of(ctx):
    """The submitted tokenizer from the eval ctx (fail-closed).

    The harness resolves and validates it once per phase
    (`prismlib.tokenizer`); battery modules never load a tokenizer
    themselves and never assume GPT-2.
    """
    tok = ctx.get("tokenizer")
    if tok is None:
        raise RuntimeError(
            "ctx['tokenizer'] missing — the harness injects the submitted tokenizer"
        )
    return tok


def encode(tok, text):
    """Token ids of `text` without special tokens (the battery length unit)."""
    return tok_contract.encode(tok, text)


def decode(tok, ids):
    """Text of `ids` — required by the tokenizer contract."""
    return tok_contract.decode(tok, ids)


def token_len(tok, text):
    """Length of `text` in tokens of the submitted tokenizer."""
    return len(encode(tok, text))


def vocab_size(tok, model=None):
    """Vocab size: the model's own declaration first, the tokenizer's next."""
    for attr in ("vocab_size", "n_vocab"):
        v = getattr(model, attr, None) or getattr(getattr(model, "config", None), attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return tok_contract.vocab_size(tok)


def truncate_tokens(tok, text, max_tokens):
    """`text` cut to at most `max_tokens` tokens (decode of the id prefix)."""
    ids = encode(tok, text)
    if len(ids) <= max_tokens:
        return text
    return decode(tok, ids[: max(0, int(max_tokens))])


def utf8_bytes(text):
    """UTF-8 byte length — the denominator of the tokenizer-neutral unit."""
    return len(text.encode("utf-8", errors="replace"))


def fit_to_tokens(
    tok, segments, target_tokens, filler, rng=None, suffix="", joiner=" ", max_rounds=64
):
    """Build a context of `target_tokens` tokens of the SUBMITTED tokenizer.

    `segments` are the load-bearing sentences (needles, facts, edges): kept
    verbatim, never truncated. `filler()` returns one distractor sentence
    and is called as often as needed; fillers are spliced at random
    positions when `rng` is given (Context-Rot / BABILong rule: distractors
    share the task grammar and the needle depth is randomized) and appended
    otherwise. `suffix` (the query / answer cue) always stays at the tail
    and counts against the budget.

    Growth is estimated in batches from the measured tokens-per-filler, so
    a 32k context costs a handful of encodes rather than one per sentence.
    Overshoot is absorbed by truncating or dropping **trailing filler
    only**, so the returned text is exactly `target_tokens` tokens whenever
    the tokenizer decode is lossless (byte-level BPE) and the segments plus
    suffix fit at all.

    Returns `(text, n_tokens)` — `n_tokens` is the measured length, so
    callers can assert instead of trusting the target.
    """
    target = max(1, int(target_tokens))
    parts = [[str(s), False] for s in segments]

    def text_of():
        return joiner.join(p for p, _ in parts) + suffix

    n = token_len(tok, text_of())
    per_filler, rounds = None, 0
    while n < target and rounds < max_rounds:
        deficit = target - n
        # 0.9 biases toward undershoot: the trim path below only ever eats
        # trailing filler, so a small final overshoot is cheaper than a big one.
        batch = 1 if per_filler is None else max(1, min(4096, int(0.9 * deficit / per_filler)))
        for _ in range(batch):
            pos = rng.randrange(len(parts) + 1) if rng is not None else len(parts)
            parts.insert(pos, [str(filler()), True])
        prev, n = n, token_len(tok, text_of())
        if n > prev:
            per_filler = (n - prev) / float(batch)
        rounds += 1

    guard = len(parts) + max_rounds
    while n > target and guard > 0:
        guard -= 1
        idx = next((i for i in range(len(parts) - 1, -1, -1) if parts[i][1]), None)
        if idx is None:
            break  # segments + suffix alone exceed the budget: report honestly
        ids = encode(tok, parts[idx][0])
        over = n - target
        if len(ids) >= over:
            # Shrink in place (possibly to ""), so the joiner count — and
            # therefore the token delta — stays exactly `over`.
            parts[idx][0] = decode(tok, ids[: len(ids) - over])
        else:
            del parts[idx]
        prev, n = n, token_len(tok, text_of())
        if n == prev:
            break  # decode is not lossless here; stop instead of spinning
    return text_of(), n


# ---------------------------------------------------------------- scoring


def _logits_of(out):
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def continuation_logprob(model, tok, device, prompt, continuation):
    """(sum_logprob, n_tokens_scored) of `continuation` given `prompt`.

    Teacher-forced; the continuation window is aligned to the tail of the
    positions the model actually scored, so self-truncating architectures
    (short internal context) are scored on what they saw — the answer
    tokens sit at the tail by construction.
    """
    import torch

    p_ids = encode(tok, prompt)
    joint = encode(tok, prompt + continuation)
    cont = joint[len(p_ids) :]
    if not cont:
        cont = encode(tok, continuation)
        joint = p_ids + cont
    if len(joint) < 2:
        return 0.0, 0
    ids = torch.tensor([joint], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = _logits_of(model(ids[:, :-1]))
    tgt_full = ids[:, 1:]
    n_scored = logits.shape[1]
    k = min(len(cont), n_scored)
    if k <= 0:
        return 0.0, 0
    tgt = tgt_full[:, -k:]
    lp = torch.log_softmax(logits[:, -k:, :].float(), dim=-1)
    token_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return float(token_lp.sum().item()), k


def score_choices(model, tok, device, prompt, choices, gold):
    """OLMES-style acc_norm: argmax over choices of sum_logprob / chars.

    Returns (acc01, gold_answer_nll_per_token).
    """
    best, best_score, gold_nll = -1, -float("inf"), 0.0
    for i, choice in enumerate(choices):
        lp, ntok = continuation_logprob(model, tok, device, prompt, choice)
        norm = lp / max(1, len(choice))
        if norm > best_score:
            best, best_score = i, norm
        if i == gold:
            gold_nll = -lp / max(1, ntok)
    return (1.0 if best == gold else 0.0), gold_nll


def is_key_token_id(tok, tok_id, _cache={}):
    """Harness-defined key-token heuristic (LongPPL selectivity without a
    reference scorer): numeric tokens and sentence-initial capitalized
    word pieces. Deterministic, fixed across submissions."""
    if tok_id in _cache:
        return _cache[tok_id]
    try:
        s = tok.convert_ids_to_tokens([int(tok_id)])[0]
    except Exception:  # noqa: BLE001
        s = ""
    key = any(ch.isdigit() for ch in s) or (
        s.startswith("Ġ") and len(s) > 1 and s[1].isupper()
    )
    _cache[tok_id] = key
    return key


def doc_loss_stats(model, tok, text, device, position_edges=(128, 256, 384, 512)):
    """One teacher-forced pass over a doc -> mean CE, key-token CE,
    per-position-bucket CE (self-truncation aligned, v1 semantics) and the
    tokenizer-neutral `bits_per_byte` (total bits over the UTF-8 bytes of
    the scored region — the unit that stays comparable across submitted
    tokenizers, where CE per token does not)."""
    import torch

    ids = tok_contract.encode_tensor(tok, text, device)
    if ids.shape[1] < 2:
        return None
    with torch.no_grad():
        logits = _logits_of(model(ids[:, :-1]))
    if logits.shape[1] < 1:
        return None
    avail = max(1, ids.shape[1] - 1)
    tgt = ids[:, 1:][:, -logits.shape[1] :]
    losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        tgt.reshape(-1),
        reduction="none",
    )
    flat_tgt = tgt.reshape(-1)
    n = losses.numel()
    mean_ce = float(losses.mean().item())
    key_mask = torch.tensor(
        [is_key_token_id(tok, t) for t in flat_tgt.tolist()],
        dtype=torch.bool,
    )
    key_ce = float(losses[key_mask].mean().item()) if key_mask.any() else None
    buckets = {}
    pos = torch.arange(n)
    lo = 0
    for hi in position_edges:
        m = (pos >= lo) & (pos < hi)
        if m.any():
            buckets[f"p{lo}_{hi}"] = float(losses[m].mean().item())
        lo = hi
    m = pos >= lo
    if m.any():
        buckets[f"p{lo}_end"] = float(losses[m].mean().item())
    # Bytes of the scored region: the model may self-truncate, so the doc's
    # UTF-8 length is prorated by the fraction of targets actually scored.
    scored_bytes = utf8_bytes(text) * (n / float(avail))
    return {
        "ce": mean_ce,
        "key_ce": key_ce,
        "buckets": buckets,
        "n_tokens": n,
        "bits_per_byte": float(losses.sum().item()) / LN2 / max(1.0, scored_bytes),
    }


def mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else None


def emit(out, key, value):
    """Store only finite floats — METRICS_JSON must stay clean."""
    if value is None:
        return
    try:
        v = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(v):
        out[str(key)] = v


def bpb(ce):
    """CE (nats/token) -> bits/token. Historical `g1.bpb.*` /
    `org.g1.bpb_*` unit; comparable only within one tokenizer. The
    tokenizer-neutral number is `doc_loss_stats(...)["bits_per_byte"]`."""
    return ce / LN2 if ce is not None else None
