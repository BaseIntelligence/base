"""Battery rollup — the composite-scoring contract view of a G1–G8 run.

`eval.run_battery` returns a NESTED per-group view (`{group: {status,
module, metrics}}`) which stays in METRICS_JSON v2 for debugging. The Rust
ingestion path (`prism-eval-store` `finalize.rs::submission_metrics`)
consumes a different, additive shape — the blob *is* the contract:

  battery = {
    "groups":  {<nested per-group view, unchanged>},
    "metrics": {"org.<group>.<name>": float | {"value": f, "clusters": {id: f}}},
    "mirrors": [{"group", "metric", "public": series, "mirror": series}],
    "tier":    "public_dev" | "private",
  }

This module is the single place where internal per-group metric names
reconcile with the canonical `org.*` keys of the pre-registered anchor set
(`crates/prism-recipe/anchors/v0.json` — the Rust anchor definitions are
the authority). A metric that was never measured is simply ABSENT (the
composite then records a `missing_metric` gate failure) — nothing is
fabricated, except fail-closed floors that *are* measurements (e.g.
`org.g8.mup_lr_stability = 0.0` when the µP sweep ran and diverged).

Cluster values come from the battery's per-item side channel
(`ItemRecorder`): cluster = template/variant id, the unit of
randomization for the clustered bootstrap in the Rust composite.

Mirror pairs (composite step 2.5, groups g2/g4/g5): the same metric scored
on the PUBLIC anchor family (published dev seed / public_dev jsonl
anchors) vs the PRIVATE mirror family (operator secret seed / staged
private mirrors). In the public_dev tier the two seeds coincide and no
private assets exist, so each pair is degenerate (gap 0 — the run is its
own mirror); in the private tier both families are scored for real.
G5 natural slices append pairs via `natural_docs.mirror_pairs`.
"""

import math

from prismlib import LN2
from prismlib.envutil import log

from . import common
from . import gen_reasoning as gr
from . import g2_downstream as g2_mod
from . import natural_docs

# ---------------------------------------------------------------- org.* map

# G2 task id -> canonical anchor task name.
_G2_TASK_ORG = {
    "lambada": "lambada",
    "hellaswag": "hellaswag",
    "piqa": "piqa",
    "arc_easy": "arc_easy",
    "arc_challenge": "arc_challenge",
    "winogrande": "winogrande",
    "boolq": "boolq",
    "openbookqa": "obqa",
}

# G5 scored-path rollup (recipe ≥ 1.4.0): protocol + natural adapters.
# Internal key -> canonical org.* anchor name. Procedural gen_longctx
# families are no longer mapped into the composite.
_G5_DIRECT = {
    "org.g5.ruler_acc": "g5.ruler.acc",
    "org.g5.babilong_acc": "g5.babilong.acc",
    "org.g5.natural_mcq_acc": "g5.natural_mcq.acc",
    "org.g5.helmet_rag_acc": "g5.helmet_rag.acc",
    "org.g5.lstar": "g5.lstar",
}

# Direct renames: org key -> (battery group, internal metric key).
_DIRECT = {
    # G1 — intrinsic fit (tokenizer-neutral bits/byte; per-token g1.bpb.*
    # siblings remain in the group view for debugging only).
    "org.g1.bits_per_byte_code": ("g1", "g1.bits_per_byte.domain.code"),
    "org.g1.bits_per_byte_prose": ("g1", "g1.bits_per_byte.domain.prose"),
    "org.g1.bits_per_byte_math": ("g1", "g1.bits_per_byte.domain.math"),
    "org.g1.bits_per_byte_fresh_crawl": ("g1", "g1.bits_per_byte.fresh"),
    "org.g1.bits_per_byte_key_token": ("g1", "g1.bits_per_byte.key_token"),
    # G3 — recall probes (family means / single-variant accuracies).
    "org.g3.mqar_acc": ("g3", "g3.mqar.acc"),
    "org.g3.copying_acc": ("g3", "g3.copy.acc"),
    "org.g3.induction_acc": ("g3", "g3.induction.base.acc"),
    "org.g3.passkey_acc": ("g3", "g3.passkey.base.acc"),
    # G4 — reasoning (family means / single-variant accuracies).
    "org.g4.arithmetic_acc": ("g4", "g4.arith.acc"),
    "org.g4.boolean_logic_acc": ("g4", "g4.bool.base.acc"),
    "org.g4.dyck_acc": ("g4", "g4.dyck.acc"),
    "org.g4.modular_acc": ("g4", "g4.mod.acc"),
    "org.g4.knights_knaves_acc": ("g4", "g4.kk.acc"),
    "org.g4.proofwriter_acc": ("g4", "g4.proof.acc"),
    # G6 — sample efficiency from the train probe curve.
    "org.g6.auc_log_tokens": ("g6", "g6.auc.log_tokens"),
    "org.g6.tokens_to_threshold": ("g6", "g6.tokens_to_ce4.0"),
    # G7 — inference efficiency (32k grid points + state/energy cards are
    # CUDA/full-caps only; absent otherwise, never fabricated).
    "org.g7.throughput_toks_s": ("g7", "g7.throughput.b32.toks"),
    "org.g7.ttft_ms_32k": ("g7", "g7.ttft.L32768.ms"),
    "org.g7.tpot_ms_32k": ("g7", "g7.tpot.L32768.ms"),
    "org.g7.state_bytes_per_token_32k": ("g7", "g7.state.bytes_per_token.measured"),
    "org.g7.joules_per_token": ("g7", "g7.energy.j_per_token"),
}

# ItemRecorder metric name -> (org key, required cluster prefix, value
# transform). Cluster values are per-cluster means of the recorded items.
_ITEM_CLUSTERS = {
    "org.g1.bits_per_byte_code": ("g1.domain.code.bits_per_byte", None, None),
    "org.g1.bits_per_byte_prose": ("g1.domain.prose.bits_per_byte", None, None),
    "org.g1.bits_per_byte_math": ("g1.domain.math.bits_per_byte", None, None),
    "org.g1.bits_per_byte_fresh_crawl": ("g1.fresh.bits_per_byte", None, None),
    # Key-token bits/byte comes from the frozen val cut (`g1.val.*`), so its
    # bootstrap units are the same per-doc clusters as the domains above.
    "org.g1.bits_per_byte_key_token": ("g1.val.key_bits_per_byte", None, None),
    "org.g3.mqar_acc": ("g3.item.acc", "mqar/", None),
    "org.g3.copying_acc": ("g3.item.acc", "copy/", None),
    "org.g3.induction_acc": ("g3.item.acc", "induction/", None),
    "org.g3.passkey_acc": ("g3.item.acc", "passkey/", None),
    "org.g4.arithmetic_acc": ("g4.item.acc", "arith/", None),
    "org.g4.boolean_logic_acc": ("g4.item.acc", "bool/", None),
    "org.g4.dyck_acc": ("g4.item.acc", "dyck/", None),
    "org.g4.modular_acc": ("g4.item.acc", "mod/", None),
    "org.g4.knights_knaves_acc": ("g4.item.acc", "kk/", None),
    "org.g4.proofwriter_acc": ("g4.item.acc", "proof/", None),
    "org.g6.tokens_to_threshold": ("g6.tokens_to", "ce4.0", None),
    "org.g7.throughput_toks_s": ("g7.throughput", "b32", None),
    "org.g7.ttft_ms_32k": ("g7.ttft", "L32768", None),
}


def _group_metrics(battery_groups, group):
    entry = battery_groups.get(group) or {}
    if entry.get("status") != "ok":
        return {}
    return entry.get("metrics") or {}


def _cluster_means(items, metric, prefix, transform):
    """{cluster_id: mean recorded value} for one ItemRecorder metric."""
    recs = (items or {}).get(metric) or []
    buckets = {}
    for rec in recs:
        cluster = str(rec.get("cluster", ""))
        if prefix is not None and not cluster.startswith(prefix):
            continue
        v = rec.get("value")
        if not isinstance(v, (int, float)):
            continue
        if transform == "bpb":
            v = v / LN2
        buckets.setdefault(cluster, []).append(float(v))
    return {c: sum(vs) / len(vs) for c, vs in buckets.items() if vs}


def _series(value, clusters):
    """Rust `series_from` shape: bare number or {value, clusters}."""
    if clusters:
        return {"value": float(value), "clusters": clusters}
    return float(value)


def flatten_metrics(battery_groups, items=None):
    """Nested per-group battery -> flat canonical `org.*` metric map.

    Only measured values appear; absent internal metrics stay absent (the
    composite's completeness gate records the gap).
    """
    items_dump = items.dump() if hasattr(items, "dump") else (items or {})
    out = {}

    def put(org_key, value):
        if value is None:
            return
        spec = _ITEM_CLUSTERS.get(org_key)
        clusters = {}
        if spec is not None:
            clusters = _cluster_means(items_dump, spec[0], spec[1], spec[2])
        out[org_key] = _series(value, clusters)

    for org_key, (group, internal) in _DIRECT.items():
        put(org_key, _group_metrics(battery_groups, group).get(internal))

    g2 = _group_metrics(battery_groups, "g2")
    for task, org_task in _G2_TASK_ORG.items():
        org_key = f"org.g2.{org_task}_acc"
        clusters = _cluster_means(items_dump, f"g2.{task}.acc", None, None)
        v = g2.get(f"g2.{task}.acc_norm")
        if v is not None:
            out[org_key] = _series(v, clusters)
    # Strict LAMBADA (anchors v2): canonical greedy last-word exact match.
    # The 4-way MC key above stays for anchor sets v0/v1; this one has real
    # headroom (the MC form saturates ~0.95+ because random-word distractors
    # cannot compete with a context-determined gold word).
    v = g2.get("g2.lambada_strict.acc")
    if v is not None:
        clusters = _cluster_means(items_dump, "g2.lambada_strict.acc", None, None)
        out["org.g2.lambada_strict_acc"] = _series(v, clusters)

    g5 = _group_metrics(battery_groups, "g5")
    for org_key, internal in _G5_DIRECT.items():
        value = g5.get(internal)
        if value is None:
            continue
        if org_key == "org.g5.ruler_acc":
            clusters = _cluster_means(items_dump, "g5.ruler.item.acc", None, None)
        elif org_key == "org.g5.babilong_acc":
            clusters = _cluster_means(items_dump, "g5.babilong.item.acc", None, None)
        elif org_key == "org.g5.natural_mcq_acc":
            clusters = _cluster_means(items_dump, "g5.item.acc", "natural_mcq@", None)
        elif org_key == "org.g5.helmet_rag_acc":
            clusters = _cluster_means(items_dump, "g5.item.acc", "helmet_rag@", None)
        else:
            clusters = {}
        out[org_key] = _series(value, clusters)

    g8 = _group_metrics(battery_groups, "g8")
    # Bounded stability score: divergence-free fraction with a spike-rate
    # penalty divisor (anchor note: "fraction of seeds without divergence
    # minus spike penalty, pre-bounded [0,1]"). Emitted only when at least
    # one divergence fact was measured.
    nan_fracs = [
        g8.get("g8.divergence.series_nan_frac"),
        g8.get("g8.divergence.probe_nan_frac"),
    ]
    nan_fracs = [x for x in nan_fracs if x is not None]
    if nan_fracs:
        spikes = g8.get("g8.spikes.per_1k_steps") or 0.0
        score = (1.0 - max(nan_fracs)) / (1.0 + max(0.0, spikes))
        out["org.g8.loss_spike_score"] = max(0.0, min(1.0, score))
    # µP LR-transfer stability (anchors/v0.json org.g8.mup_lr_stability):
    # - g8.mup.stability present → use it (success path or fail-closed 0.0)
    # - else legacy: stub==0 + lr_ratio_log2_abs → 1/(1+|ratio|)
    # Tiny-caps skips omit the key (sweep never entered). A diverged/
    # failed sweep MUST emit 0.0 so the G8 composite stays complete.
    if "g8.mup.stability" in g8:
        stab = g8.get("g8.mup.stability")
        if isinstance(stab, (int, float)) and math.isfinite(stab):
            out["org.g8.mup_lr_stability"] = max(0.0, min(1.0, float(stab)))
    elif g8.get("g8.mup.stub") == 0.0:
        ratio = g8.get("g8.mup.lr_ratio_log2_abs")
        if ratio is not None:
            out["org.g8.mup_lr_stability"] = 1.0 / (1.0 + max(0.0, ratio))
    # v2.1 (anchors ≥ v1): local scaling-slope probe from the µP width
    # sweep. Same fail-closed contract as mup_lr_stability — present after
    # any real sweep (0.0 on failure), absent on tiny-caps skips. Ignored
    # by the composite under anchor set v0 (unknown keys are inert).
    slope = g8.get("g8.mup.scaling_slope")
    if isinstance(slope, (int, float)) and math.isfinite(slope):
        out["org.g8.mup_scaling_slope"] = max(0.0, float(slope))

    # v2.1 (anchors ≥ v1): compute-normalized reasoning — the accuracy ×
    # decode-throughput product. A model that "thinks" via extra depth or
    # loops pays its inference cost inside the same key that credits its
    # reasoning gain, so adaptive-compute architectures compete fairly
    # (raw G7 alone structurally penalizes them). Only measured values
    # combine; absent inputs keep the key absent (never fabricated).
    reasoning = _reasoning_throughput(out)
    if reasoning is not None:
        out["org.g7.reasoning_throughput"] = reasoning
    return out


_G4_ORG_KEYS = (
    "org.g4.arithmetic_acc",
    "org.g4.boolean_logic_acc",
    "org.g4.dyck_acc",
    "org.g4.modular_acc",
    "org.g4.knights_knaves_acc",
    "org.g4.proofwriter_acc",
)


def _raw_value(series):
    """Bare number or {value, clusters} → float | None."""
    if isinstance(series, dict):
        series = series.get("value")
    if isinstance(series, (int, float)) and math.isfinite(series):
        return float(series)
    return None


def _reasoning_throughput(out):
    """acc(G4 mean) × decode toks/s — None unless both sides measured."""
    tput = _raw_value(out.get("org.g7.throughput_toks_s"))
    if tput is None or tput <= 0.0:
        return None
    accs = [v for v in (_raw_value(out.get(k)) for k in _G4_ORG_KEYS) if v is not None]
    if not accs:
        return None
    return (sum(accs) / len(accs)) * tput


# ---------------------------------------------------------------- mirrors

# Mirrored G4 families: (org key, generator, kwargs) — one representative
# variant per anchored family, scored on both seed families.
_G4_MIRROR = (
    ("org.g4.arithmetic_acc", gr.arith, {"tier": 2}),
    ("org.g4.boolean_logic_acc", gr.boolean_expr, {"depth": 2}),
    ("org.g4.dyck_acc", gr.dyck, {"k": 2, "n_pairs": 10}),
    ("org.g4.modular_acc", gr.modular, {"ood": False}),
    ("org.g4.knights_knaves_acc", gr.knights_knaves, {"n": 3}),
    ("org.g4.proofwriter_acc", gr.proofwriter, {"depth": 1}),
)


def _score_family(model, ctx, gen, kwargs, seed, n_items, budget):
    """Mean acc + per-item clusters over n seeded draws of one family."""
    accs, clusters = [], {}
    for i in range(n_items):
        if not budget.ok():
            break
        items = gen(common.task_seed(seed, f"mirror/{gen.__name__}/{i}"), **kwargs)
        item_accs = []
        for it in items:
            try:
                acc, _nll = common.score_choices(
                    model, ctx["tokenizer"], ctx["device"], it["prompt"], it["choices"], it["gold"]
                )
            except Exception:  # noqa: BLE001 — one bad item never kills a pair
                continue
            item_accs.append(acc)
        if item_accs:
            v = sum(item_accs) / len(item_accs)
            accs.append(v)
            clusters[f"{gen.__name__}#{i}"] = v
    if not accs:
        return None
    return {"value": sum(accs) / len(accs), "clusters": clusters}


def _score_rows(model, ctx, rows, tag, budget):
    """Mean acc + per-row clusters over frozen G2 anchor rows."""
    accs, clusters = [], {}
    for i, row in enumerate(rows):
        if not budget.ok():
            break
        prompt, choices, gold = row.get("prompt"), row.get("choices"), row.get("gold")
        if not prompt or not choices or gold is None:
            continue
        try:
            acc, _nll = common.score_choices(model, ctx["tokenizer"], ctx["device"], prompt, choices, int(gold))
        except Exception:  # noqa: BLE001
            continue
        accs.append(acc)
        clusters[f"{tag}#{i}"] = acc
    if not accs:
        return None
    return {"value": sum(accs) / len(accs), "clusters": clusters}


def build_mirrors(model, ctx):
    """Public-vs-private mirror pairs for the contamination gap (step 2.5).

    G4 (procedural): public side = the published dev seed family, mirror
    side = the resolved secret seed family (identical in the public_dev
    tier by construction). G2 (assets): public side = the shipped
    public_dev anchors, mirror side = the staged private mirrors when
    present, else the same public anchors (degenerate gap 0). G5 natural
    slices: the same public_dev / private pack orientation via
    `natural_docs.mirror_pairs`. Bounded by `PRISM_EVAL_MIRROR_BUDGET_S`
    (default 600 s); on expiry the pairs collected so far are returned.
    """
    budget = common.Budget(
        common.float_env("PRISM_EVAL_MIRROR_BUDGET_S", common.group_budget_s("mirror"))
    )
    n_items = common.eval_n_items(default_full=4, default_tiny=2)
    secret = common.resolve_secret_seed(ctx)
    pairs = []

    for org_key, gen, kwargs in _G4_MIRROR:
        if not budget.ok():
            break
        public = _score_family(model, ctx, gen, kwargs, common.PUBLIC_DEV_SEED, n_items, budget)
        mirror = _score_family(model, ctx, gen, kwargs, secret, n_items, budget)
        if public is not None and mirror is not None:
            pairs.append({"group": "g4", "metric": org_key, "public": public, "mirror": mirror})

    cap = common.eval_asset_cap(4, 2, env_key="PRISM_EVAL_MIRROR_G2_CAP")
    for task in g2_mod.TASKS:
        if not budget.ok():
            break
        public_path = common.assets_path({**ctx, "eval_assets_dir": None}, f"g2/{task}.jsonl")
        mirror_path = (
            common.assets_path(ctx, f"g2/{task}.jsonl") if ctx.get("eval_assets_dir") else None
        )
        if public_path is None:
            continue
        public = _score_rows(model, ctx, common.load_jsonl(public_path, cap=cap), task, budget)
        if mirror_path is not None and mirror_path != public_path:
            mirror = _score_rows(model, ctx, common.load_jsonl(mirror_path, cap=cap), task, budget)
        else:
            # Public dev tier: no private mirror exists — the run is its
            # own mirror (gap 0 by construction, honestly labelled).
            mirror = dict(public) if public is not None else None
            if mirror is not None:
                mirror["clusters"] = dict(public["clusters"])
        if public is not None and mirror is not None:
            org_key = f"org.g2.{_G2_TASK_ORG[task]}_acc"
            pairs.append({"group": "g2", "metric": org_key, "public": public, "mirror": mirror})

    if budget.ok():
        pairs.extend(natural_docs.mirror_pairs(model, ctx, budget=budget))
    return pairs


def _pair_is_inert(pair):
    """True when a mirror pair cannot express a contamination gap.

    Inert means the two sides are the *same measurement*, so the gap is
    identically 0 no matter how contaminated the submission is. That is
    what `build_mirrors` / `natural_docs.mirror_pairs` produce in the
    `public_dev` tier ("the run is its own mirror"), and it is a measured
    fact about the pair rather than a guess about the tier.
    """
    public, mirror = pair.get("public"), pair.get("mirror")
    if not isinstance(public, dict) or not isinstance(mirror, dict):
        return True
    if public.get("value") != mirror.get("value"):
        return False
    return (public.get("clusters") or {}) == (mirror.get("clusters") or {})


def mirror_report(pairs, ctx):
    """Loud status of the contamination (mirror-gap) defence.

    **Why this exists.** The mirror-gap penalty is the designed
    contamination detector, and in the `public_dev` tier it is inert *by
    construction*: `build_mirrors` sets `mirror = dict(public)` when no
    private asset differs, so `gap ≡ 0` and the penalty deducts nothing.
    That was honestly labelled in a comment but nothing surfaced it in the
    output, so a scored run in `public_dev` looked contamination-checked
    when it was not. This makes the state explicit and machine-readable:

      - `contamination_checked: false` ⇒ **no contamination evidence was
        produced by this run.** A zero mirror penalty is then the absence
        of a measurement, NOT evidence of a clean submission.
      - `inert_pairs` / `live_pairs` count the pairs that can and cannot
        move, so a partially-staged pack is visible rather than averaged
        away.

    Operators must not read "mirror penalty = 0" as "no contamination".
    """
    pairs = list(pairs or [])
    inert = [p for p in pairs if _pair_is_inert(p)]
    live = [p for p in pairs if not _pair_is_inert(p)]
    tier = common.eval_tier(ctx)
    if not pairs:
        reason = "no mirror pairs were built (no model, or every side missing)"
    elif not live:
        reason = (
            f"every mirror pair is degenerate in tier '{tier}': public and "
            "private sides are the same measurement, so the gap is 0 by "
            "construction and the contamination penalty is INERT"
        )
    elif inert:
        reason = (
            f"{len(inert)} of {len(pairs)} mirror pairs are degenerate in "
            f"tier '{tier}'; only {len(live)} can express a gap"
        )
    else:
        reason = f"all {len(pairs)} mirror pairs are live in tier '{tier}'"
    return {
        "tier": tier,
        "pairs": len(pairs),
        "inert_pairs": len(inert),
        "live_pairs": len(live),
        # The single flag downstream consumers should branch on.
        "contamination_checked": bool(live),
        "inert": not live,
        "reason": reason,
        "inert_metrics": sorted({str(p.get("metric")) for p in inert}),
    }


def budget_report(battery_groups):
    """Loud truncation report for the operator.

    Any `*.partial` flag a group emits means that group hit its
    wall-clock ceiling and scored FEWER items than the protocol asks for,
    which makes its metric not comparable across submissions. Group views
    already carried these flags, but nothing surfaced them at the battery
    level, so budget truncation was effectively silent. This aggregates
    them next to the budget actually in force.
    """
    partial = sorted(
        group
        for group, entry in (battery_groups or {}).items()
        if any(str(k).endswith(".partial") for k in ((entry or {}).get("metrics") or {}))
    )
    return {
        "battery_budget_s": common.battery_budget_s(),
        "group_budgets_s": {
            g: common.group_budget_s(g) for g in sorted(common.budget_shares())
        },
        "truncated": bool(partial),
        "partial_groups": partial,
    }


def rollup_battery(battery_groups, ctx, model=None):
    """The METRICS_JSON v2 `battery` object: nested groups (unchanged,
    debug) + flat canonical org.* metrics + mirror pairs + tier label +
    the time-budget / truncation report + the loud mirror-defence status.

    `mirror_defence.contamination_checked` is the flag that says whether
    this run produced any contamination evidence at all; see
    `mirror_report`.
    """
    mirrors = build_mirrors(model, ctx) if model is not None else []
    report = mirror_report(mirrors, ctx)
    if report["inert"]:
        # Loud on the operator's console too, not only in the JSON blob.
        log(
            "WARNING mirror defence INERT: "
            f"{report['reason']} — a zero contamination penalty in this run "
            "is the ABSENCE of a check, not a clean result"
        )
    return {
        "groups": battery_groups,
        "metrics": flatten_metrics(battery_groups, ctx.get("items")),
        "mirrors": mirrors,
        "tier": common.eval_tier(ctx),
        "budget": budget_report(battery_groups),
        "mirror_defence": report,
    }
