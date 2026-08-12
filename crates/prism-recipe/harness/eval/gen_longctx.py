"""G5 long-context generators (E2) — pure stdlib, answers from seed.

RULER-like NIAH variants + variable tracking + frequent-words,
BABILong-like qa1–qa5 with same-grammar distractors, GraphWalks,
MRCR-style same-distribution ordering, NoLiMa-style latent needles.
Filler sentences share the task grammar (Context-Rot/BABILong design
rule: the task must be recall/reasoning, not distribution-shift
detection).

**Length unit — pending G5 rework.** These generators still convert their
nominal token targets to word budgets with WORDS_PER_TOKEN ≈ 0.75, an
assumption about GPT-2 on this kind of synthetic text. It survives only
because the current grid is self-normalized (L*), and it is *not* the
tokenizer contract: with a miner-chosen tokenizer the honest measurement is
`eval.common.fit_to_tokens(tok, segments, target_tokens, filler, ...)`,
which pads/truncates to an exact token count of the submitted tokenizer.
The G5 length-grid conversion to that helper (RULER/BABILong adapters +
the 4k–32k grid) is the long-context rework's job; every NEW generator must
take the tokenizer and use `fit_to_tokens` instead of word budgets.
"""

import random

from .generators import _item, digits, lexicon

# Legacy word-budget approximation (GPT-2-shaped). See the module docstring:
# new/reworked length grids must measure with `common.fit_to_tokens`.
WORDS_PER_TOKEN = 0.75

_NOUNS = [
    "falcon", "river", "lantern", "meadow", "anchor", "willow", "compass",
    "harbor", "cinder", "maple", "quartz", "saddle", "thicket", "vessel",
]
_VERBS = ["crossed", "entered", "left", "approached", "circled", "reached"]
_PLACES = [
    "garden", "kitchen", "office", "attic", "cellar", "market", "harbor",
    "chapel", "studio", "barn",
]
_ACTORS = [
    "mary", "john", "susan", "peter", "lucy", "oscar", "nina", "carl",
]
_ITEMS = ["milk", "key", "letter", "candle", "rope", "mirror", "bell"]


def _filler(rng, nouns):
    return (
        f"the {rng.choice(nouns)} {rng.choice(_VERBS)} the "
        f"{rng.choice(nouns)} near the {rng.choice(_PLACES)} ."
    )


def _pad(rng, sentences, target_words, nouns):
    """Splice distractor sentences at random positions (never truncate
    mid-content — distractor-aware padding)."""
    out = list(sentences)
    words = sum(len(s.split()) for s in out)
    while words < target_words:
        out.insert(rng.randrange(len(out) + 1), _filler(rng, nouns))
        words = sum(len(s.split()) for s in out)
    return out


def _words_for(length_tokens):
    return max(8, int(length_tokens * WORDS_PER_TOKEN))


# ------------------------------------------------------------ RULER-like


def niah_multikey(seed, length, n_keys=4):
    """Multi-key NIAH (RULER 2404.06654): several key->value needles plus
    same-grammar filler; query one key at a random depth."""
    rng = random.Random(int(seed))
    keys = lexicon(rng, n_keys)
    vals = [digits(rng, 3) for _ in range(n_keys)]
    nouns = lexicon(rng, 6)
    needles = [f"the magic word for {k} is {v} ." for k, v in zip(keys, vals)]
    body = _pad(rng, needles, _words_for(length), nouns)
    q = rng.randrange(n_keys)
    prompt = " ".join(body) + f" the magic word for {keys[q]} is"
    return [
        _item("niah", prompt, vals, q, f"niah/k{n_keys}", length=length)
    ]


def variable_tracking(seed, length, chains=2, hops=4):
    """RULER VT: X1=V, X2=X1, … chains scattered in filler; query the
    last variable of one chain (minimal multi-hop coreference)."""
    rng = random.Random(int(seed))
    value = lexicon(rng, 1)[0]
    all_vars, chain_last = [], []
    stmts = []
    for c in range(chains):
        names = [f"x{c}_{h}" for h in range(hops + 1)]
        stmts.append(f"{names[0]} = {value} .")
        for h in range(hops):
            stmts.append(f"{names[h+1]} = {names[h]} .")
        all_vars.extend(names)
        chain_last.append(names[-1])
    nouns = lexicon(rng, 6)
    body = _pad(rng, stmts, _words_for(length), nouns)
    c = rng.randrange(chains)
    prompt = " ".join(body) + f" the last variable that equals {value} is"
    return [
        _item(
            "vt",
            prompt,
            all_vars,
            all_vars.index(chain_last[c]),
            f"vt/c{chains}h{hops}",
            length=length,
        )
    ]


def freq_words(seed, length, n_words=6):
    """Common/frequent-words extraction (RULER aggregation): the stream
    itself is the data; gold = the most frequent candidate word."""
    rng = random.Random(int(seed))
    words = lexicon(rng, n_words)
    counts = sorted((rng.randint(2, 12) for _ in range(n_words)), reverse=True)
    counts[0] = counts[1] + rng.randint(2, 5)  # unique argmax
    stream = []
    for w, c in zip(words, counts):
        stream.extend([w] * c)
    rng.shuffle(stream)
    nouns = lexicon(rng, 4)
    fillers = [_filler(rng, nouns) for _ in range(max(2, _words_for(length) // 12))]
    # Interleave the word stream with filler so it spans the document.
    doc, si = [], 0
    for f in fillers:
        doc.append(f)
        chunk = stream[si : si + 12]
        si += len(chunk)
        if chunk:
            doc.append(" ".join(chunk) + " .")
    doc.append(" ".join(stream[si:]) + " .")
    prompt = " ".join(doc) + " question: which word appears most often? the most frequent word is"
    return [
        _item("freq", prompt, words, 0, f"freq/w{n_words}", length=length)
    ]


# ------------------------------------------------------------ BABILong-like


def babilong(seed, length, qa=1):
    """bAbI-style fact chaining scattered in same-grammar distractors
    (2406.10149 leak-proof design; distractors share the task grammar so
    the probe is reasoning, not shift detection)."""
    rng = random.Random(int(seed))
    actor, other = rng.sample(_ACTORS, 2)
    place_a, place_b = rng.sample(_PLACES, 2)
    thing = rng.choice(_ITEMS)
    stmts, answer, qtext = [], None, ""
    if qa == 1:  # single supporting fact
        stmts = [f"{actor} moved to the {place_a} .", f"{other} went to the {place_b} ."]
        qtext = f"where is {actor}? {actor} is in the"
        answer = place_a
    elif qa == 2:  # two supporting facts
        stmts = [
            f"{actor} moved to the {place_a} .",
            f"{actor} picked up the {thing} .",
            f"{other} went to the {place_b} .",
        ]
        qtext = f"where is the {thing}? the {thing} is in the"
        answer = place_a
    elif qa == 3:  # three supporting facts
        stmts = [
            f"{other} went to the {place_b} .",
            f"{other} picked up the {thing} .",
            f"{other} moved to the {place_a} .",
        ]
        qtext = f"where is the {thing}? the {thing} is in the"
        answer = place_a
    elif qa == 4:  # simple deduction over is-a facts
        animal, cls = rng.choice([("cats", "animals"), ("dogs", "animals")])
        pet = rng.choice(["lily", "rex", "milo"])
        stmts = [f"{animal} are {cls} .", f"{pet} is a {animal[:-1]} .", f"{other} went to the {place_b} ."]
        qtext = f"what is {pet}? {pet} is a"
        answer = cls[:-1]
    else:  # qa5: three-argument relation
        stmts = [
            f"{actor} gave the {thing} to {other} .",
            f"{other} went to the {place_b} .",
        ]
        qtext = f"who did {actor} give the {thing} to? {actor} gave the {thing} to"
        answer = other
    choices = sorted({answer, place_a, place_b, actor, other} - {answer}) + [answer]
    rng.shuffle(choices)
    nouns = lexicon(rng, 6)
    rng.shuffle(stmts)
    body = _pad(rng, stmts, _words_for(length), nouns)
    prompt = " ".join(body) + f" {qtext}"
    return [
        _item("babi", prompt, choices, choices.index(answer), f"babi/qa{qa}", length=length)
    ]


# ------------------------------------------------------------ GraphWalks


def graphwalks(seed, length, n_nodes=10, k=2):
    """GraphWalks (openai/graphwalks, Feb 2026 prompt rules): exact-k-step
    walk on a path with unique out-edges + a unique-parent query. Node
    names are synthetic words; filler pads to length."""
    rng = random.Random(int(seed))
    nodes = lexicon(rng, n_nodes)
    start = rng.randrange(n_nodes - k)
    path = [nodes[(start + i) % n_nodes] for i in range(k + 1)]
    edges = [(path[i], path[i + 1]) for i in range(k)]  # out-degree 1 on path
    extra_src = rng.sample([n for n in nodes if n not in path[:-1]], min(3, n_nodes - k))
    for s in extra_src:
        t = rng.choice([n for n in nodes if n != s and (s, n) not in edges])
        edges.append((s, t))
    rng.shuffle(edges)
    edge_txt = [f"{a} -> {b}" for a, b in edges]
    nouns = lexicon(rng, 6)
    body = _pad(rng, edge_txt, _words_for(length), nouns)
    prompt = (
        "edges: " + " , ".join(body)
        + f" . start at {path[0]} and follow -> for exactly {k} steps. you end at"
    )
    choices = list(nodes)
    rng.shuffle(choices)
    return [
        _item(
            "graph",
            prompt,
            choices,
            choices.index(path[-1]),
            f"graph/n{n_nodes}k{k}",
            length=length,
        )
    ]


# ------------------------------------------------------------ MRCR-style ordering


def mrcr_order(seed, length, n_texts=5, dup_topic=2):
    """MRCR/Michelangelo (2409.12640): several same-distribution texts,
    two about the same topic; return the opening word of the dup_topic-th
    text about that topic (order discrimination, hash-free)."""
    rng = random.Random(int(seed))
    topics = lexicon(rng, 3)
    texts = []
    for i in range(n_texts):
        topic = topics[0] if i in (1, 3) else rng.choice(topics[1:])
        opener = lexicon(rng, 1, n_syll=3)[0]
        line = " ".join(lexicon(rng, 10, n_syll=2))
        texts.append((topic, opener, f"a short note about {topic} : {opener} {line} ."))
    rng.shuffle(texts)
    t_idx = [i for i, (t, _, _) in enumerate(texts) if t == topics[0]]
    gold_pos_in_doc = t_idx[dup_topic - 1] if len(t_idx) >= dup_topic else t_idx[-1]
    gold_opener = texts[gold_pos_in_doc][1]
    nouns = lexicon(rng, 6)
    body = _pad(rng, [tx[2] for tx in texts], _words_for(length), nouns)
    ordinal = {1: "first", 2: "second", 3: "third"}[dup_topic]
    prompt = (
        " ".join(body)
        + f" the opening word of the {ordinal} note about {topics[0]} is"
    )
    choices = sorted({tx[1] for tx in texts})
    return [
        _item(
            "mrcr",
            prompt,
            choices,
            choices.index(gold_opener),
            f"mrcr/t{n_texts}d{dup_topic}",
            length=length,
        )
    ]


# ------------------------------------------------------------ NoLiMa-style latent needles


def nolima(seed, length):
    """NoLiMa (2502.05167): the question shares no content words with the
    needle ('spouse'/'works at' for 'married to'/'employed by') — kills
    the literal-match shortcut."""
    rng = random.Random(int(seed))
    a, b = rng.sample(_ACTORS, 2)
    firm, firm2 = lexicon(rng, 2)
    stmts = [
        f"{a} is married to {b} .",
        f"{b} is employed by {firm} .",
        f"{rng.choice(_ACTORS)} is employed by {firm2} .",
    ]
    nouns = lexicon(rng, 6)
    rng.shuffle(stmts)
    body = _pad(rng, stmts, _words_for(length), nouns)
    prompt = " ".join(body) + f" question: where does the spouse of {a} work? the spouse of {a} works at"
    choices = [firm, firm2] + lexicon(rng, 2)
    rng.shuffle(choices)
    return [
        _item("nolima", prompt, choices, choices.index(firm), "nolima/spouse", length=length)
    ]
