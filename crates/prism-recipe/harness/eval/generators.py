"""Seeded procedural generators — core utilities + G3 recall family.

Every generator is pure stdlib (no torch, no tokenizer) and every gold
answer is computed offline from the seed: the same call always yields
the same items, a different secret seed yields a shifted family. The
private family is the same code with operator-held seeds; the public dev
family uses the published seed in `public_dev/seeds.json` (see
public_dev/README.md for the regeneration ceremony).

Item: {"task", "prompt", "choices", "gold", "cluster", "meta"}.
Clusters name the template/variant — the unit of randomization for the
clustered bootstrap in the composite.
"""

import random
import string

_CONS = "bcdfghjklmnpqrstvwxz"
_VOW = "aeiou"


def word(rng, n_syll=2):
    """Synthetic vocab word (Zoology/MQAR-style; never real English)."""
    out = []
    for _ in range(n_syll):
        out.append(rng.choice(_CONS))
        out.append(rng.choice(_VOW))
        if rng.random() < 0.5:
            out.append(rng.choice(_CONS))
    return "".join(out)


def lexicon(rng, n, n_syll=2):
    seen, out = set(), []
    while len(out) < n:
        w = word(rng, n_syll)
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def digits(rng, n):
    return "".join(rng.choice(string.digits) for _ in range(n))


def _item(task, prompt, choices, gold, cluster, **meta):
    return {
        "task": task,
        "prompt": prompt,
        "choices": [str(c) for c in choices],
        "gold": int(gold),
        "cluster": cluster,
        "meta": meta,
    }


def _numeric_choices(rng, gold, n=4, spread=None):
    """Gold + seeded unique numeric distractors (as strings)."""
    spread = spread or max(2, abs(gold) // 4 + 1)
    out = {int(gold)}
    while len(out) < n:
        out.add(int(gold) + rng.randint(1, spread) * rng.choice((-1, 1)))
    out = sorted(out)
    rng.shuffle(out)
    return [str(v) for v in out], out.index(int(gold))


# ------------------------------------------------------------ G3: recall


def mqar(seed, n_pairs=8, n_queries=4):
    """Multi-query associative recall (Zoology 2312.04927): store N random
    key->value pairs, query keys at the end. The attention-vs-SSM
    discriminator; choices = the stored values."""
    rng = random.Random(int(seed))
    keys = lexicon(rng, n_pairs)
    vals = [digits(rng, 2) for _ in range(n_pairs)]
    pairs = " ".join(f"{k} {v}" for k, v in zip(keys, vals))
    items = []
    for q in rng.sample(range(n_pairs), min(n_queries, n_pairs)):
        prompt = f"{pairs} . query: {keys[q]} ->"
        items.append(
            _item("mqar", prompt, vals, q, f"mqar/n{n_pairs}", n_pairs=n_pairs)
        )
    return items


def copy_gap(seed, seq_len=16, gap=0, probes=3):
    """Repeat-after-me copying (2402.01032) with a filler gap between the
    two occurrences; gap sweep exposes fixed-state capacity limits."""
    rng = random.Random(int(seed))
    seq = lexicon(rng, seq_len, n_syll=3)
    filler = lexicon(rng, max(1, gap))
    base = " ".join(seq) + " | " + " ".join(filler) + " | "
    items = []
    idxs = sorted({0, 1, seq_len // 2, seq_len - 1})[:probes]
    for j in idxs:
        prompt = base + " ".join(seq[:j]) + (" " if j else "")
        others = [w for w in seq if w != seq[j]]
        rng.shuffle(others)
        choices = [seq[j]] + others[:3]
        rng.shuffle(choices)
        items.append(
            _item(
                "copy",
                prompt,
                choices,
                choices.index(seq[j]),
                f"copy/len{seq_len}/gap{gap}",
                pos=j,
                gap=gap,
            )
        )
    return items


def induction(seed, n_reps=2, gap=8):
    """Induction-head probe (2209.11895): [A][B] ... [A] -> [B] over
    random tokens, distance between occurrences swept."""
    rng = random.Random(int(seed))
    a, b = lexicon(rng, 2, n_syll=3)
    filler = lexicon(rng, gap)
    parts = []
    for _ in range(n_reps):
        parts.append(f"{a} {b}")
        parts.append(" ".join(lexicon(rng, max(1, gap // 2))))
    prompt = " ".join(parts) + f" {a}"
    others = lexicon(rng, 3, n_syll=3)
    choices = [b] + others
    rng.shuffle(choices)
    return [
        _item(
            "induction",
            prompt,
            choices,
            choices.index(b),
            f"induction/gap{gap}/r{n_reps}",
        )
    ]


def passkey(seed, n_filler=40, depth=None):
    """Passkey retrieval (2306.14000): one random digit needle in
    same-grammar filler at a randomized depth."""
    rng = random.Random(int(seed))
    key = digits(rng, 5)
    nouns = lexicon(rng, 6)
    fillers = [
        f"the {rng.choice(nouns)} {rng.choice(('sat', 'lay', 'grew', 'fell'))} "
        f"near the {rng.choice(nouns)} ."
        for _ in range(n_filler)
    ]
    d = depth if depth is not None else rng.random()
    pos = min(n_filler, max(0, int(d * n_filler)))
    fillers.insert(pos, f"the pass key is {key} .")
    prompt = " ".join(fillers) + " the pass key is"
    choices = [key] + [digits(rng, 5) for _ in range(3)]
    rng.shuffle(choices)
    return [
        _item(
            "passkey",
            prompt,
            choices,
            choices.index(key),
            f"passkey/f{n_filler}",
            depth=d,
        )
    ]
