"""G6 — sample efficiency from the E1 probe curve (research/11 §4).

Consumes `ctx["probe_curve"]` — [{step, tokens_seen, wall_s, probe_loss}]
recorded by the harness probe hook during train — and scores the curve
as the object of interest, not just the endpoint: area under
loss-vs-log10(tokens) (mean loss over log tokens; lower is better) plus
tokens-to-threshold at three pre-registered CE levels {4.0, 3.5, 3.0}
with right-censoring flags. No model use; pure telemetry math.

Both scored G6 quantities are **lower-better** (see `anchors/v2.json`,
where `cap < reference` encodes the direction):

- `g6.auc.log_tokens` is a mean cross-entropy per decade of tokens, so a
  *smaller* area is a better learning curve. The historical v0/v1 anchor
  annotated it "higher-better" over [0.5, 0.95], which no plausible CE
  can express — v2 re-anchors it to the quantity this module actually
  computes. It is CE per **token** of the submitted tokenizer, so it is
  not tokenizer-neutral the way `org.g1.bits_per_byte_*` is; a bits/byte
  form needs byte counts on the probe curve (a v3 recipe change).
- `g6.tokens_to_ce*` is tokens spent to reach a CE level.

**Censoring is fail-closed.** When the curve never reaches a level, the
run did not demonstrate that level at any token count, so the honest
lower-better value is "unbounded", not the small `tokens_seen` it
happened to stop at. Emitting the raw endpoint would make *training
less* score better (a censored 1e8 normalizes to 1.0 under
`reference 2e9 / cap 5e8`), which is directly exploitable. The scored
key therefore carries [`CENSORED_TOKENS`] — a sentinel far above any
plausible reference, so the metric normalizes to the 0.0 floor — while
the raw endpoint stays visible under `g6.tokens_to_ce*.observed`.

This mirrors the fail-closed convention already used for
`org.g8.mup_lr_stability`: a real measurement that failed emits the
worst value rather than being omitted, so the group stays complete and
the composite's completeness gate is not the thing that fires. Omitting
instead would make the whole submission ineligible via `missing_metric`
(the group does *not* fall back to its other metric), which is a much
blunter outcome for a model that simply trained too little.
"""

import math

from . import common

_LEVELS = (4.0, 3.5, 3.0)

# Fail-closed sentinel for a right-censored tokens-to-threshold curve
# (~5e5x the v2 `reference` of 2e9): any `efficiency_log_ratio` anchor with
# `cap < reference` normalizes this to the 0.0 floor. Kept finite so
# METRICS_JSON stays clean JSON (`common.emit` drops non-finite values).
CENSORED_TOKENS = 1e15


def _tokens_to(points, level):
    """First interpolated tokens_seen where probe_loss <= level."""
    prev = None
    for pt in points:
        loss, tok = pt["probe_loss"], pt["tokens_seen"]
        if loss <= level:
            if prev is None or prev[0] == loss:
                return float(tok), False
            # Linear interpolation in loss between the two bracketing points.
            frac = (prev[0] - level) / (prev[0] - loss)
            return prev[1] + frac * (tok - prev[1]), False
        prev = (loss, tok)
    return float(points[-1]["tokens_seen"]), True


def run(model, ctx):
    out = {}
    curve = [
        pt for pt in (ctx.get("probe_curve") or [])
        if isinstance(pt, dict)
        and math.isfinite(float(pt.get("probe_loss", float("nan"))))
        and int(pt.get("tokens_seen", 0)) > 0
    ]
    curve.sort(key=lambda pt: pt["tokens_seen"])
    out["g6.points"] = float(len(curve))
    if len(curve) < 2:
        out["g6.stub"] = 1.0
        return out

    # AUC over log10(tokens): trapezoid integral normalized by the log
    # span == mean probe loss per decade of tokens (curve kind, lower better).
    xs = [math.log10(pt["tokens_seen"]) for pt in curve]
    ys = [float(pt["probe_loss"]) for pt in curve]
    auc = sum(
        0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i]) for i in range(len(xs) - 1)
    )
    span = xs[-1] - xs[0]
    common.emit(out, "g6.auc.log_tokens", auc / span if span > 0 else None)
    common.emit(out, "g6.probe.final_loss", ys[-1])
    common.emit(out, "g6.probe.best_loss", min(ys))

    for level in _LEVELS:
        tok, censored = _tokens_to(curve, level)
        tag = str(level)
        # Fail-closed: a censored curve never reached this level, so the
        # scored key gets the sentinel (normalizes to 0.0) and the raw
        # endpoint is preserved as an observed-only sibling.
        scored = CENSORED_TOKENS if censored else tok
        common.emit(out, f"g6.tokens_to_ce{tag}", scored)
        out[f"g6.tokens_to_ce{tag}.censored"] = 1.0 if censored else 0.0
        if censored:
            common.emit(out, f"g6.tokens_to_ce{tag}.observed", tok)
        # Bootstrap side channel sees the same scored value, so a censored
        # curve cannot resample its way back to a good score.
        common.record(ctx, "g6.tokens_to", f"ce{tag}", scored)
    return out
