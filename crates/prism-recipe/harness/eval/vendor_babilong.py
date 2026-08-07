"""Vendored BABILong protocol + seeded bAbI qa1–qa5 stories (pure stdlib).

Upstream provenance is machine-readable in `UPSTREAM` below and narrated in
`eval/VENDOR.md`. Kept faithful to upstream BABILong: the background-sentence
sampler (random start offset in a random document, sentence-length filter),
the `<context>` framing and the official `compare_answers` metric. Fact
placement (facts scattered through the background, story order preserved) is
done by the shared fitter, `eval.toklen`. Kept faithful to bAbI v1.2: the qa1–qa5
vocabularies (actors, locations, objects), the movement / take / drop /
transfer verb sets, the question phrasings and the answer keys, all read off
`tasks_1-20_v1-2/en-10k`.

Deviations (all required by the harness contract, none silent):

- Stories are **generated** from the lattice seed rather than read from the
  published bAbI 10k files: a fixed instance set is memorizable across runs,
  which is the whole point of the private-seed tier. Only one transfer per
  object is emitted in qa5 and the qa4 relation set is kept collision-free,
  so every gold answer is unique by construction.
- Filler comes from a private pinned corpus (`FILLER_ASSET`) instead of raw
  public PG-19; with no pack staged a seeded synthetic prose pool is used
  and the adapter reports which source it scored.
- `random.Random(seed)` replaces numpy `default_rng`; a regex sentence
  splitter replaces `nltk.PunktSentenceTokenizer` (no NLTK data offline);
  the sampler's minimum-length filter is in words, not tokens, so the
  vendored code stays tokenizer-agnostic (the adapter owns token budgets
  through `eval.toklen`).
- `DEFAULT_PROMPTS` (upstream instruction / post-prompt scaffolding) is
  deliberately **not** vendored: it is instruction-tuned prompting, which
  this battery excludes. The adapter builds a base-LM few-shot completion.
"""

import random
import re

UPSTREAM = {
    "name": "BABILong",
    "repo": "https://github.com/booydar/babilong",
    "commit": "7a6efee29f5cac03c3c410e6799c80fd2ffe3610",
    "commit_date": "2026-06-01",
    "license": "Apache-2.0",
    "paper": "arXiv:2406.10149",
    "files": ("babilong/babilong_utils.py", "babilong/prompts.py"),
    "tasks": (1, 2, 3, 4, 5),
    "task_source": {
        "name": "bAbI tasks v1.2 (qa1-qa5 grammar)",
        "repo": "https://github.com/facebookarchive/bAbI-tasks",
        "license": "BSD-3-Clause",
        "paper": "arXiv:1502.05698",
    },
    "filler_source": {
        "asset": "g5/babilong_filler.jsonl",
        "manifest": "g5/babilong_filler.manifest.json",
        "note": "operator-pinned private corpus; upstream uses public PG-19",
    },
}

# Private filler pack: one JSON object per line, `{"text": "..."}`. The
# sibling manifest records {source, license, sha256, n_docs, pinned_at}.
FILLER_ASSET = "g5/babilong_filler.jsonl"
FILLER_MANIFEST = "g5/babilong_filler.manifest.json"

# babilong/prompts.py (verbatim).
USER_TEMPLATE = "<context>\n{context}\n</context>\n\nQuestion: {question}"
# Base-LM completion variant: upstream's context/question framing plus the
# bare answer prefix that upstream's own `<example>` blocks use. Upstream's
# `instruction` / `post_prompt` scaffolding is instruction tuning and is
# deliberately excluded from this battery.
BASE_TEMPLATE = USER_TEMPLATE + "\nAnswer:"

# ---------------------------------------------------------------- bAbI v1.2

ACTORS = ("Daniel", "John", "Mary", "Sandra")
ACTORS_QA5 = ("Bill", "Fred", "Jeff", "Mary")
LOCATIONS = ("bathroom", "bedroom", "garden", "hallway", "kitchen", "office")
OBJECTS = ("apple", "football", "milk")
MOVE_VERBS = ("journeyed to", "moved to", "travelled to", "went to")
RETURN_VERB = "went back to"
TAKE_VERBS = ("got", "grabbed", "picked up", "took")
DROP_VERBS = ("discarded", "dropped", "left", "put down")
TRANSFER_VERBS = ("gave", "handed", "passed")
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}

TASK_IDS = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------- metric


def compare_answers(target, output):
    """Upstream `babilong.babilong_utils.compare_answers`."""
    target = target.lower()
    output = output.lower()
    output = output.split(".")[0]
    output = output.split("<context>")[0]
    output = output.split("<example>")[0]
    return target in output


# ---------------------------------------------------------------- stories


def _move(rng, actor, dest, seen):
    verb = RETURN_VERB if dest in seen else rng.choice(MOVE_VERBS)
    return f"{actor} {verb} the {dest}."


def _qa1(rng, n_facts):
    where, seen, facts = {}, {a: set() for a in ACTORS}, []
    for _ in range(n_facts):
        actor = rng.choice(ACTORS)
        dest = rng.choice([p for p in LOCATIONS if p != where.get(actor)])
        facts.append(_move(rng, actor, dest, seen[actor]))
        seen[actor].add(dest)
        where[actor] = dest
    actor = rng.choice(sorted(where))
    return facts, f"Where is {actor}?", where[actor], list(LOCATIONS)


def _object_story(rng, n_facts):
    """Shared qa2/qa3 statement pool with full object location history."""
    where, seen = {}, {a: set() for a in ACTORS}
    held, at, history = {}, {}, {o: [] for o in OBJECTS}
    facts = []
    for _ in range(n_facts):
        actor = rng.choice(ACTORS)
        action = rng.random()
        carried = [o for o, h in held.items() if h == actor]
        loose = [o for o in OBJECTS if o not in held and at.get(o) == where.get(actor)]
        if action < 0.45 or actor not in where:
            dest = rng.choice([p for p in LOCATIONS if p != where.get(actor)])
            facts.append(_move(rng, actor, dest, seen[actor]))
            seen[actor].add(dest)
            where[actor] = dest
            for obj in carried:
                if history[obj][-1:] != [dest]:
                    history[obj].append(dest)
        elif action < 0.75 and loose:
            obj = rng.choice(loose)
            facts.append(f"{actor} {rng.choice(TAKE_VERBS)} the {obj} there.")
            held[obj] = actor
            at.pop(obj, None)
        elif carried:
            obj = rng.choice(carried)
            verb = rng.choice(DROP_VERBS)
            tail = " there." if rng.random() < 0.2 else "."
            facts.append(f"{actor} {verb} the {obj}{tail}")
            held.pop(obj, None)
            at[obj] = where[actor]
        else:
            free = [o for o in OBJECTS if o not in held]
            if not free:  # everything is carried: just move somebody
                dest = rng.choice([p for p in LOCATIONS if p != where.get(actor)])
                facts.append(_move(rng, actor, dest, seen[actor]))
                seen[actor].add(dest)
                where[actor] = dest
                for obj in carried:
                    if history[obj][-1:] != [dest]:
                        history[obj].append(dest)
                continue
            obj = rng.choice(free)
            if obj not in at:
                at[obj] = where[actor]
            dest = at[obj]
            if dest != where[actor]:
                facts.append(_move(rng, actor, dest, seen[actor]))
                seen[actor].add(dest)
                where[actor] = dest
            facts.append(f"{actor} {rng.choice(TAKE_VERBS)} the {obj} there.")
            held[obj] = actor
            if history[obj][-1:] != [dest]:
                history[obj].append(dest)
    place = {}
    for obj in OBJECTS:
        if obj in held:
            place[obj] = where[held[obj]]
        elif obj in at:
            place[obj] = at[obj]
    return facts, place, history


def _qa2(rng, n_facts):
    facts, place, _hist = _object_story(rng, n_facts)
    tracked = sorted(place)
    if not tracked:
        return None
    obj = rng.choice(tracked)
    return facts, f"Where is the {obj}?", place[obj], list(LOCATIONS)


def _qa3(rng, n_facts):
    facts, place, history = _object_story(rng, n_facts)
    tracked = [o for o in sorted(place) if len(history[o]) >= 2 and history[o][-1] == place[o]]
    if not tracked:
        return None
    obj = rng.choice(tracked)
    now, before = history[obj][-1], history[obj][-2]
    if before == now:
        return None
    return (
        facts,
        f"Where was the {obj} before the {now}?",
        before,
        list(LOCATIONS),
    )


def _qa4(rng, _n_facts):
    """Two `The X is <dir> of the Y.` facts; ask either direction of the pair.

    `X is d of Y` is equivalent to `Y is OPPOSITE[d] of X`, which is the two
    question shapes bAbI emits (`What is north of the kitchen?` /
    `What is the garden east of?`).
    """
    a, b, c = rng.sample(LOCATIONS, 3)
    axis = rng.choice((("north", "south"), ("east", "west")))
    d1, d2 = rng.choice(axis), rng.choice(axis)
    anchor2 = b if d1 != d2 else a
    rel = ((a, d1, b), (c, d2, anchor2))  # (subject, direction, anchor)
    facts = [f"The {s} is {d} of the {x}." for s, d, x in rel]
    subj, direction, anchor = rel[rng.randrange(len(rel))]
    if len([s for s, d, x in rel if d == direction and x == anchor]) != 1:
        return None
    if rng.random() < 0.5:
        return facts, f"What is {direction} of the {anchor}?", subj, list(LOCATIONS)
    return (
        facts,
        f"What is the {anchor} {OPPOSITE[direction]} of?",
        subj,
        list(LOCATIONS),
    )


def _qa5(rng, n_facts):
    where, seen, held = {}, {a: set() for a in ACTORS_QA5}, {}
    facts, transfers = [], []
    for _ in range(n_facts):
        actor = rng.choice(ACTORS_QA5)
        carried = [o for o, h in held.items() if h == actor]
        moved = [o for o, _g, _r in transfers]
        if actor not in where or rng.random() < 0.4:
            dest = rng.choice([p for p in LOCATIONS if p != where.get(actor)])
            facts.append(_move(rng, actor, dest, seen[actor]))
            seen[actor].add(dest)
            where[actor] = dest
        elif carried and rng.random() < 0.6:
            obj = rng.choice(carried)
            if obj in moved:
                continue
            other = rng.choice([p for p in ACTORS_QA5 if p != actor])
            facts.append(f"{actor} {rng.choice(TRANSFER_VERBS)} the {obj} to {other}.")
            held[obj] = other
            transfers.append((obj, actor, other))
        else:
            free = [o for o in OBJECTS if o not in held and o not in moved]
            if not free:
                continue
            obj = rng.choice(free)
            facts.append(f"{actor} {rng.choice(TAKE_VERBS)} the {obj} there.")
            held[obj] = actor
    if not transfers:
        return None
    obj, giver, receiver = transfers[rng.randrange(len(transfers))]
    forms = [
        (f"Who gave the {obj}?", giver, list(ACTORS_QA5)),
        (f"Who gave the {obj} to {receiver}?", giver, list(ACTORS_QA5)),
        (f"Who received the {obj}?", receiver, list(ACTORS_QA5)),
        (f"Who did {giver} give the {obj} to?", receiver, list(ACTORS_QA5)),
    ]
    # `What did X give to Y?` is only unambiguous when that pair traded once.
    if sum(1 for _o, g, r in transfers if (g, r) == (giver, receiver)) == 1:
        forms.append((f"What did {giver} give to {receiver}?", obj, list(OBJECTS)))
    question, answer, choices = forms[rng.randrange(len(forms))]
    return facts, question, answer, choices


_BUILDERS = {1: _qa1, 2: _qa2, 3: _qa3, 4: _qa4, 5: _qa5}
_N_FACTS = {1: 8, 2: 12, 3: 16, 4: 2, 5: 12}


def story(seed, qa, n_facts=None):
    """One single-question bAbI qa`N` story (facts + question + gold).

    Retries with a fresh draw when a draw yields no unambiguous question
    (e.g. no object ever changed rooms for qa3); returns None if the task
    grammar cannot produce one, which the adapter treats as a skipped item.
    """
    if qa not in _BUILDERS:
        raise NotImplementedError(f"bAbI qa{qa} is not vendored")
    n = int(n_facts or _N_FACTS[qa])
    for attempt in range(24):
        rng = random.Random(int(seed) + attempt)
        built = _BUILDERS[qa](rng, n)
        if built is None:
            continue
        facts, question, answer, choices = built
        if answer not in choices:
            continue
        return {
            "qa": int(qa),
            "facts": list(facts),
            "question": question,
            "answer": answer,
            "choices": list(choices),
            "cluster": f"babilong/qa{qa}",
        }
    return None


# ---------------------------------------------------------------- filler

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

_SUBJ = (
    "the old shepherd", "her brother", "the young clerk", "my uncle",
    "the widow", "the schoolmaster", "a passing stranger", "the captain",
    "the girl in grey", "the innkeeper",
)
_PRED = (
    "had never once spoken of the matter", "walked slowly along the quay",
    "kept a small brass key on a chain", "remembered the winter of the flood",
    "wrote three letters and burned two", "watched the lamps come on below",
    "counted the money twice before speaking", "laughed and would not explain",
    "left the door open behind him", "said nothing for a long while",
)
_CIRC = (
    "on the morning of the fair", "before the bells had finished",
    "in the shadow of the tall hedge", "as the tide began to turn",
    "while the others were at supper", "without so much as a glance",
    "for reasons he never gave", "in a voice pitched low",
)


def synthetic_filler(seed, n_docs=64):
    """Seeded prose stand-in for the private filler pack."""
    rng = random.Random(int(seed))
    docs = []
    for _ in range(n_docs):
        sents = []
        for _ in range(rng.randint(12, 24)):
            sents.append(
                f"{rng.choice(_SUBJ).capitalize()} {rng.choice(_PRED)} "
                f"{rng.choice(_CIRC)}."
            )
        docs.append(" ".join(sents))
    return docs


def sample_sentences(seed, docs, n_sentences, min_words=6, max_words=60):
    """Upstream `SentenceSampler` with `shuffle=True`, in word units.

    Picks a random document, starts at a random offset inside it, keeps
    sentences whose length passes the filter, and repeats until `n_sentences`
    are collected.
    """
    rng = random.Random(int(seed))
    docs = [d for d in docs if d]
    if not docs:
        return []
    out = []
    guard = 0
    while len(out) < n_sentences and guard < 4 * n_sentences + 64:
        guard += 1
        text = docs[rng.randrange(len(docs))]
        text = text[rng.randrange(len(text)) :]
        sents = _SENT_SPLIT.split(text)
        if len(sents) > 2:
            sents = sents[1:-1]
        for sent in sents:
            sent = sent.strip()
            n_words = len(sent.split())
            if n_words < min_words or n_words > max_words:
                continue
            out.append(sent)
            if len(out) >= n_sentences:
                break
    return out


def sentences(text):
    """Sentence split (regex stand-in for upstream's Punkt tokenizer)."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
