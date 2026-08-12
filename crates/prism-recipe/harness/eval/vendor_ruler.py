"""Vendored RULER synthetic generators — upstream parity, pure stdlib.

Upstream provenance is machine-readable in `UPSTREAM` below and narrated in
`eval/VENDOR.md`. Kept verbatim from upstream: the `TASKS` template /
answer-prefix table, the needle sentence format, the `noise` haystack
string, the `num_needle_k = max(num_needle_k, num_needle_q)` clamp, the
`num_needle_q * num_needle_v == 1` singular rewrite, the needle-haystack
splice, the variable-assignment chain grammar, the `Document {i}:` QA
prompt, and the official `string_match_*` metrics.

Deviations (all required by the harness contract, none silent):

- Seeding: `random.Random(seed)` per call instead of upstream's global
  `random.seed` + numpy, so one generator call is a pure function of its
  lattice seed (`eval.common.task_seed`).
- Haystack: upstream's `essay` type (`json/PaulGrahamEssays.json`, ~1 MB of
  public text) is not vendored. Every probe here uses an upstream haystack
  type that needs no corpus — `needle` (upstream `niah_multikey_2/3`) or
  `noise` (upstream `vt`) — so the haystack itself regenerates from the
  private secret seed and adds no payload.
- Key vocabulary: upstream draws `adjective-noun` keys from `wonderwords`;
  no PyPI at eval time, so keys come from the harness's seeded synthetic
  lexicon (`eval.generators.lexicon`).
- Length fitting: upstream binary-searches a haystack size against a fixed
  tokenizer and truncates. Here the generator emits the spliced haystack
  plus a `filler()` of the same sentence grammar and `eval.toklen` grows or
  trims filler only, so a needle can never be cut to hit the target.
- Scoring: upstream generates and applies `string_match_*`. Base LMs are
  scored by log-probability over in-distribution candidate answers instead,
  so each instance also carries `slots` (per-answer candidate sets). The
  templates the model sees are unchanged.
"""

import random
import string

from .generators import lexicon

UPSTREAM = {
    "name": "RULER",
    "repo": "https://github.com/NVIDIA/RULER",
    "commit": "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
    "commit_date": "2026-07-22",
    "license": "Apache-2.0",
    "paper": "arXiv:2404.06654",
    "files": (
        "scripts/data/synthetic/constants.py",
        "scripts/data/synthetic/niah.py",
        "scripts/data/synthetic/variable_tracking.py",
        "scripts/data/synthetic/qa.py",
        "scripts/eval/synthetic/constants.py",
        "scripts/synthetic.yaml",
    ),
}

# ---------------------------------------------------------------- verbatim

# scripts/data/synthetic/constants.py
TASKS = {
    "niah": {
        "tokens_to_generate": 128,
        "template": """Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?""",
        "answer_prefix": """ The special magic {type_needle_v} for {query} mentioned in the provided text are""",
    },
    "variable_tracking": {
        "tokens_to_generate": 30,
        "template": """Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above.""",
        "answer_prefix": """ Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assigned the value {query}, they are: """,
    },
    "qa": {
        "tokens_to_generate": 32,
        "template": """Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query}""",
        "answer_prefix": """ Answer:""",
    },
}

# scripts/data/synthetic/niah.py
NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
NOISE_HAYSTACK = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)

# scripts/data/synthetic/qa.py
DOCUMENT_PROMPT = "Document {i}:\n{document}"

# Upstream QA asset (private tier): one JSON object per line shaped like
# upstream `read_squad` output — {"question": str, "answers": [str],
# "context": str}. The document pool is every row's `context`.
QA_ASSET = "g5/ruler_qa.jsonl"


# scripts/eval/synthetic/constants.py — the official metrics, used by the
# module smoke to prove generated golds pass upstream scoring.
def string_match_all(preds, refs):
    score = (
        sum(
            sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / len(ref)
            for pred, ref in zip(preds, refs)
        )
        / len(preds)
        * 100
    )
    return round(score, 2)


def string_match_part(preds, refs):
    score = (
        sum(
            max(1.0 if r.lower() in pred.lower() else 0.0 for r in ref)
            for pred, ref in zip(preds, refs)
        )
        / len(preds)
        * 100
    )
    return round(score, 2)


# ---------------------------------------------------------------- helpers


def _rand_number(rng, num_digits=7):
    lo = 10 ** (num_digits - 1)
    return str(rng.randint(lo, 10**num_digits - 1))


def _rand_word(rng):
    return "-".join(lexicon(rng, 2, n_syll=2))


def _rand_uuid(rng):
    hexs = "".join(rng.choice(string.hexdigits[:16].lower()) for _ in range(32))
    return f"{hexs[:8]}-{hexs[8:12]}-4{hexs[13:16]}-{hexs[16:20]}-{hexs[20:]}"


def _rand(rng, kind):
    if kind == "numbers":
        return _rand_number(rng)
    if kind == "words":
        return _rand_word(rng)
    if kind == "uuids":
        return _rand_uuid(rng)
    raise NotImplementedError(f"{kind} is not implemented.")


def _split_template(task, singular=False, **fields):
    """Upstream `template + answer_prefix`, split around `{context}`.

    The halves are what the adapter wraps a fitted context in, so the
    rendered prompt is byte-identical to upstream's `template.format(...)`.
    `singular` is upstream's rewrite for the single-needle single-query case;
    like upstream it runs on the raw template, before the fields are
    substituted, so a key that happens to contain `are` is not rewritten.
    """
    cfg = TASKS[task]
    template = cfg["template"] + cfg["answer_prefix"]
    if singular:
        template = (
            template.replace("Some", "A")
            .replace("are all", "is")
            .replace("are", "is")
            .replace("answers", "answer")
        )
    head, tail = template.split("{context}")
    return head.format(**fields), tail.format(**fields)


def _slot(rng, gold, others, n_choices=4):
    """One scored answer slot: gold + in-distribution distractors.

    Choices are shuffled so a model with flat log-probabilities lands at
    chance instead of riding the argmax tie-break onto a fixed gold slot.
    """
    pool = [o for o in dict.fromkeys(others) if o != gold]
    choices = [gold] + pool[: max(1, n_choices - 1)]
    rng.shuffle(choices)
    return {"answer": gold, "choices": choices, "gold": choices.index(gold)}


def _instance(
    task, head, tail, segments, filler, slots, cluster, sep, evidence, splice, **meta
):
    """One instance in the shape `eval.toklen.fit_prompt` consumes.

    `segments` are the load-bearing context sentences, evidence spliced in by
    the upstream rule; `filler()` yields one more expendable sentence of the
    same grammar, so the fitter reaches any length target without touching
    the evidence. `splice` asks the fitter to interleave that filler at
    random positions (upstream's randomized needle depth); probes whose
    sentences are numbered set it False and take tail filler instead.
    """
    return {
        "task": task,
        "head": head,
        "tail": tail,
        "segments": segments,
        "filler": filler,
        "sep": sep,
        "slots": slots,
        "cluster": cluster,
        # Context spans the answer depends on; the adapter asserts they
        # survived the fitting.
        "evidence": evidence,
        "splice": splice,
        "meta": meta,
    }


# ---------------------------------------------------------------- niah


def niah(
    seed,
    num_haystack,
    num_needle_k=1,
    num_needle_v=1,
    num_needle_q=1,
    type_needle_k="words",
    type_needle_v="numbers",
):
    """Needle-in-a-haystack over the upstream `needle` haystack type.

    `num_needle_*` follow upstream `synthetic.yaml`: multikey = 4 keys /
    1 query, multivalue = 1 key / 4 values, multiquery = 4 keys / 4 queries.
    """
    rng = random.Random(int(seed))
    num_needle_k = max(num_needle_k, num_needle_q)  # upstream clamp
    keys, values, needles = [], [], []
    for _ in range(num_needle_k):
        keys.append(_rand(rng, type_needle_k))
        value = []
        for _ in range(num_needle_v):
            value.append(_rand(rng, type_needle_v))
            needles.append(
                NEEDLE.format(
                    type_needle_v=type_needle_v, key=keys[-1], value=value[-1]
                )
            )
        values.append(value)
    rng.shuffle(needles)

    num_haystack = max(len(needles), int(num_haystack))
    hay_vals = []

    def _hay():
        v = _rand(rng, type_needle_v)
        hay_vals.append(v)
        return NEEDLE.format(
            type_needle_v=type_needle_v, key=_rand(rng, type_needle_k), value=v
        )

    sentences = [_hay() for _ in range(num_haystack)]
    indexes = sorted(rng.sample(range(num_haystack), len(needles)), reverse=True)
    for index, element in zip(indexes, needles):
        sentences.insert(index, element)

    indices = rng.sample(range(num_needle_k), num_needle_q)
    queries = [keys[i] for i in indices]
    answers = [a for i in indices for a in values[i]]
    query = (
        ", ".join(queries[:-1]) + ", and " + queries[-1]
        if len(queries) > 1
        else queries[0]
    )

    singular = num_needle_q * num_needle_v == 1  # upstream singular rewrite
    tnv = type_needle_v[:-1] if singular else type_needle_v
    head, tail = _split_template("niah", singular=singular, type_needle_v=tnv, query=query)

    if num_needle_q > 1:
        # Multi-query: the model must bind each queried key to its value, so
        # every answer competes against the other queried keys' values.
        slots = [_slot(rng, a, answers) for a in answers]
    else:
        others = [v for group in values for v in group if v not in answers] + hay_vals
        slots = [_slot(rng, a, others) for a in answers]

    return _instance(
        "niah",
        head,
        tail,
        sentences,
        _hay,
        slots,
        f"niah/k{num_needle_k}v{num_needle_v}q{num_needle_q}",
        "\n",
        [
            NEEDLE.format(type_needle_v=type_needle_v, key=keys[i], value=v)
            for i in indices
            for v in values[i]
        ],
        True,
        answers=answers,
        num_haystack=num_haystack,
    )


# ---------------------------------------------------------------- variable tracking


def variable_tracking(seed, num_noises, num_chains=2, num_hops=4):
    """RULER VT over the upstream `noise` haystack.

    Upstream `synthetic.yaml` pins `num_chains: 1`; a second chain is used
    here so the non-queried chain supplies in-distribution negatives for
    log-probability scoring (with one chain every variable in the text is a
    gold answer and any scorer is trivially perfect).
    """
    rng = random.Random(int(seed))
    k = 5
    n_vars = (num_hops + 1) * num_chains
    vars_all = ["".join(rng.choices(string.ascii_uppercase, k=k)) for _ in range(n_vars)]
    while len(set(vars_all)) < n_vars:
        vars_all.append("".join(rng.choices(string.ascii_uppercase, k=k)))

    vars_ret, chains_ret = [], []
    for i in range(0, len(vars_all), num_hops + 1):
        this_vars = vars_all[i : i + num_hops + 1]
        if len(this_vars) < num_hops + 1:
            break
        vars_ret.append(this_vars)
        chain = [f"VAR {this_vars[0]} = {rng.randint(10000, 99999)}"]
        for j in range(num_hops):
            chain.append(f"VAR {this_vars[j + 1]} = VAR {this_vars[j]} ")
        chains_ret.append(chain)

    value = chains_ret[0][0].split("=")[-1].strip()
    sentences = [NOISE_HAYSTACK] * max(len(vars_all), int(num_noises))
    for chain in chains_ret:
        positions = sorted(rng.sample(range(len(sentences)), len(chain)))
        for insert_pi, j in zip(positions, range(len(chain))):
            sentences.insert(insert_pi + j, chain[j])

    others = [v for group in vars_ret[1:] for v in group]
    slots = [_slot(rng, v, others, n_choices=1 + len(others)) for v in vars_ret[0]]
    head, tail = _split_template(
        "variable_tracking", query=value, num_v=num_hops + 1
    )
    return _instance(
        "variable_tracking",
        head,
        tail,
        sentences,
        lambda: NOISE_HAYSTACK,
        slots,
        f"vt/c{num_chains}h{num_hops}",
        "\n",
        list(chains_ret[0]),
        True,
        answers=list(vars_ret[0]),
        num_haystack=len(sentences),
    )


# ---------------------------------------------------------------- qa


def _synthetic_docs(rng, n):
    """Seeded stand-in for the private QA pack: one fact per document."""
    rows = []
    for _ in range(n):
        entity = " ".join(w.capitalize() for w in lexicon(rng, 2, n_syll=2))
        attr = rng.choice(("founding year", "catalogue number", "registry code"))
        answer = _rand_number(rng, 4)
        filler = " ".join(
            f"The {w} district records were revised in a later survey."
            for w in lexicon(rng, 3, n_syll=2)
        )
        rows.append(
            {
                "question": f"What is the {attr} of {entity}?",
                "answers": [answer],
                "context": f"{entity}. The {attr} of {entity} is {answer}. {filler}",
            }
        )
    return rows


def qa(seed, num_docs, rows=None):
    """RULER QA: the gold document hidden among distractor documents.

    `rows` is the private QA pack (`QA_ASSET`, upstream squad shape); with
    no pack staged a seeded synthetic pool is used and the adapter reports
    which source it scored.
    """
    rng = random.Random(int(seed))
    num_docs = max(4, int(num_docs))
    synthetic = rows is None
    pool = list(rows) if rows else _synthetic_docs(rng, num_docs + 4)
    target = pool[rng.randrange(len(pool))]
    docs = [target["context"]]
    spare = [r["context"] for r in pool if r is not target]
    rng.shuffle(spare)
    while len(docs) < num_docs and spare:
        docs.append(spare.pop())
    body_docs = list(docs)
    rng.shuffle(body_docs)
    body = [
        DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(body_docs)
    ]

    # Filler documents keep the numbering monotonic, so a longer context is
    # still a well-formed upstream document list.
    spare = [r["context"] for r in pool if r is not target] or [target["context"]]
    counter = [len(body)]

    def _filler():
        counter[0] += 1
        return DOCUMENT_PROMPT.format(
            i=counter[0], document=spare[counter[0] % len(spare)]
        )

    gold = str(target["answers"][0])
    others = [str(r["answers"][0]) for r in pool if r is not target]
    head, tail = _split_template("qa", query=target["question"])
    return _instance(
        "qa",
        head,
        tail,
        body,
        _filler,
        [_slot(rng, gold, others)],
        "qa/synthetic" if synthetic else "qa/pack",
        "\n\n",
        [str(target["context"])],
        False,  # `Document {i}:` numbering has to stay monotonic
        answers=[gold],
        num_haystack=len(body),
        synthetic=synthetic,
    )


def render(instance, context):
    """Upstream template rendered around a `context` (smoke / debug view)."""
    return instance["head"] + context + instance["tail"]
