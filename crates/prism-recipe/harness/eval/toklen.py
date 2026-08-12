"""Token-exact prompt assembly for the G5 community-protocol adapters.

The single seam between the vendored RULER / BABILong generators
(`g5_ruler`, `g5_babilong`) and the modular tokenizer contract: every
length target is expressed in tokens of the **submitted** tokenizer
(`ctx["tokenizer"]`) and every encode / fit goes through
`common.tokenizer_of` / `common.encode` / `common.token_len` /
`common.fit_to_tokens`.

Contract (same as `common.fit_to_tokens`): `segments` are load-bearing
(needles, chain statements, bAbI facts, the gold document) and are never
truncated or reordered; slack is absorbed by `filler()` sentences —
spliced at random positions when an `rng` is given, so the evidence sits
at a random depth, appended at the tail otherwise — and overshoot by
trimming **trailing filler only**. `head` (upstream template opening,
few-shot shots) and `tail` (the question plus the upstream answer prefix)
wrap the fitted context verbatim, so the prompt is byte-identical to
upstream's `template.format(...)` and the reported length covers the
whole prompt — RULER's `max_seq_length` semantics.
"""

from . import common


def tokenizer_of(ctx):
    """The submitted tokenizer, fail-closed (never a hardcoded default)."""
    return common.tokenizer_of(ctx)


def encode(tok, text):
    """Token ids of `text` without special tokens."""
    return common.encode(tok, text)


def n_tokens(tok, text):
    """Length of `text` in tokens of the submitted tokenizer."""
    return common.token_len(tok, text)


def fit(tok, segments, n_target, filler, rng=None, suffix="", sep=" "):
    """Context of ~`n_target` tokens: `segments` verbatim plus filler."""
    text, n = common.fit_to_tokens(
        tok, segments, n_target, filler, rng=rng, suffix=suffix, joiner=sep
    )
    return text, int(n)


def fit_prompt(tok, head, segments, tail, n_target, filler, sep=" ", rng=None):
    """`head` + fitted context + `tail`, as close to `n_target` as it fits.

    Returns `(prompt, n_prompt_tokens)` with the length **measured** on the
    final prompt, so callers report length fidelity instead of trusting the
    target (concatenating `head` can shift the count by a token when the
    tokenizer merges across the boundary).
    """
    body, _n = fit(
        tok,
        segments,
        max(8, int(n_target) - n_tokens(tok, head)),
        filler,
        rng=rng,
        suffix=tail,
        sep=sep,
    )
    prompt = head + body
    return prompt, n_tokens(tok, prompt)


def depths(prompt, evidence):
    """Where the answer-bearing spans sit in the prompt, in [0, 1].

    Measured after fitting, in character space. The scan runs forward so
    repeated spans (bAbI stories reuse sentences) are located at their own
    occurrence, and restarts for spans the context holds out of evidence
    order (RULER shuffles its needles). A mean near 0.5 is the evidence that
    the protocol really spreads its evidence through the context instead of
    parking it at the head.
    """
    n = max(1, len(prompt))
    out, pos = [], 0
    for span in evidence:
        span = str(span)
        found = prompt.find(span, pos)
        if found < 0:
            found = prompt.find(span)
        if found < 0:
            continue
        out.append(found / n)
        pos = found + len(span)
    return out
