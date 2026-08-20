"""G1 — intrinsic distributional fit (research/04 §6.1, research/05 §4).

Unit note (modular tokenizer): the historical `g1.bpb.*` keys are CE / ln 2,
i.e. bits per **token** of the submitted tokenizer — comparable within one
tokenizer only. Each of them ships a `g1.bits_per_byte.*` sibling: total
bits over the UTF-8 bytes actually scored, the **tokenizer-neutral** unit
that survives miners choosing different vocabularies. The scored `org.g1.*`
anchor keys read the per-byte siblings (`org.g1.bits_per_byte_*`); per-token
`g1.bpb.*` remain in the battery group view for debugging only.

bpb on the harness frozen val cut (v1 semantics), bpb on multi-domain
held-out assets + a fresh FineWeb dump stream (staged `public` pack —
held-out, not secret; built from public HF, never the FineWeb-Edu train
pin), a harness-defined key-token-weighted variant (numeric +
capitalized-initial tokens; the LongPPL selectivity idea with zero
evaluator circularity), and a per-position loss decomposition
(truncation-cheat signature).

Assets (JSONL, `{"text": ...}` rows):
  $PRISM_EVAL_ASSETS_DIR/g1/domains/<name>.jsonl  (staged public HF pack)
  $PRISM_EVAL_ASSETS_DIR/g1/fresh.jsonl           (held-out FineWeb dump)
  eval/public_dev/g1/domains/<name>.jsonl         (tiny public_dev anchors)
With no assets at all, only the val-cut + key-token + position metrics
are emitted (never an error). Canonical scored domains (prose / math /
fresh-crawl) still emit: alias from news/crawl when those slices exist,
otherwise the chance floor (`_CENSORED_BITS_PER_BYTE`) so a pack that
only staged code+news cannot silently omit `org.g1.bits_per_byte_*`.
"""

import os

from . import common

_CENSORED_BITS_PER_BYTE = 3.6

_POS_EDGES = (128, 256, 384, 512)

# Scored org.g1 keys require these internal names. Extra pack slices (news,
# crawl, wiki) are debug-scored and then aliased into the canonical slots.
_CANONICAL_DOMAINS = ("code", "prose", "math")
_DOMAIN_ALIASES = {
    "prose": ("prose", "news", "wiki"),
    "math": ("math", "stem"),
    "code": ("code",),
}
_FRESH_RELS = (
    "g1/fresh.jsonl",
    "g1/domains/fresh.jsonl",
    "g1/domains/crawl.jsonl",
    "g1/domains/news.jsonl",
    "g1/domains/prose.jsonl",
)


def _score_texts(model, tok, texts, device, budget, ctx, tag, out, prefix):
    ces, key_ces, bpbytes, key_bpbytes = [], [], [], []
    bucket_vals = {}
    for i, txt in enumerate(texts):
        if not budget.ok():
            out[f"{prefix}.partial"] = 1.0
            break
        try:
            stats = common.doc_loss_stats(model, tok, txt, device, _POS_EDGES)
        except Exception:  # noqa: BLE001 — one bad doc never kills G1
            continue
        if stats is None:
            continue
        # Per-DOC cluster id (`<tag>#<i>`): the bootstrap resamples clusters
        # with replacement, so a constant tag would collapse the whole
        # metric to one cluster and contribute exactly zero variance. The
        # document is the unit of randomization here, matching the per-row
        # convention `rollup.build_mirrors` already uses.
        cluster = f"{tag}#{i}"
        ces.append(stats["ce"])
        bpbytes.append(stats["bits_per_byte"])
        common.record(ctx, f"{prefix}.doc_ce", cluster, stats["ce"])
        common.record(ctx, f"{prefix}.bits_per_byte", cluster, stats["bits_per_byte"])
        # G1 stays summary-first: short prompt excerpt only (cheap completeness).
        common.record_trace(
            ctx,
            {
                "kind": "loss",
                "group": "g1",
                "task": prefix,
                "metric": f"{prefix}.bits_per_byte",
                "cluster": cluster,
                "prompt_excerpt": txt,
                "value": stats["bits_per_byte"],
                "meta": {"ce": stats["ce"], "n_tokens": stats.get("n_tokens")},
            },
        )
        if stats["key_ce"] is not None:
            key_ces.append(stats["key_ce"])
        if stats.get("key_bits_per_byte") is not None:
            key_bpbytes.append(stats["key_bits_per_byte"])
            common.record(
                ctx, f"{prefix}.key_bits_per_byte", cluster, stats["key_bits_per_byte"]
            )
        for b, v in stats["buckets"].items():
            bucket_vals.setdefault(b, []).append(v)
    return ces, key_ces, bucket_vals, bpbytes, key_bpbytes


def run(model, ctx):
    tok = ctx["tokenizer"]
    device = ctx["device"]
    budget = common.Budget(common.group_budget_s("g1"))
    out = {}

    # 1) Frozen val cut — v1 bpb semantics (per-doc mean CE / ln 2) plus the
    # tokenizer-neutral bits-per-byte (see the module docstring).
    val = list(ctx.get("val_texts") or [])
    ces, key_ces, buckets, bpbytes, key_bpbytes = _score_texts(
        model, tok, val, device, budget, ctx, "val", out, "g1.val"
    )
    common.emit(out, "g1.bpb.val", common.bpb(common.mean(ces)))
    common.emit(out, "g1.bits_per_byte.val", common.mean(bpbytes))
    common.emit(out, "g1.bpb.key_token", common.bpb(common.mean(key_ces)))
    # Completeness is data-independent: if a tiny/public smoke cut happens to
    # contain no key tokens, emit the anchored chance floor rather than omit
    # the key and make the entire composite structurally ineligible.
    common.emit(
        out,
        "g1.bits_per_byte.key_token",
        common.mean(key_bpbytes) if key_bpbytes else _CENSORED_BITS_PER_BYTE,
    )
    for b, vals in sorted(buckets.items()):
        common.emit(out, f"g1.posloss.{b}", common.bpb(common.mean(vals)))

    # 2) Multi-domain held-out (staged public HF pack or public_dev anchors).
    # Always attempt the canonical scored names even when the staged pack
    # only listed a subset (proof 0bd93db9 scored code+news and omitted
    # prose/math). assets_path falls back to public_dev when a slice is
    # missing from the staged tree.
    dom_names = set(_CANONICAL_DOMAINS)
    for aliases in _DOMAIN_ALIASES.values():
        dom_names.update(aliases)
    for base in (ctx.get("eval_assets_dir"), common.PUBLIC_DEV_DIR):
        if not base:
            continue
        d = os.path.join(str(base), "g1", "domains")
        if os.path.isdir(d):
            dom_names.update(f[: -len(".jsonl")] for f in os.listdir(d) if f.endswith(".jsonl"))
    domain_bpb, domain_bpbyte = [], []
    cap = common.eval_asset_cap(32, 8, env_key="PRISM_EVAL_G1_CAP")
    for name in sorted(dom_names):
        path = common.assets_path(ctx, f"g1/domains/{name}.jsonl")
        if path is None:
            continue
        if not budget.ok():
            out["g1.partial"] = 1.0
            continue
        texts = [r.get("text", "") for r in common.load_jsonl(path, cap=cap)]
        texts = [t for t in texts if t]
        ces, _, _, bpbytes, _ = _score_texts(
            model, tok, texts, device, budget, ctx, f"domain/{name}", out, f"g1.domain.{name}"
        )
        v = common.bpb(common.mean(ces))
        common.emit(out, f"g1.bpb.domain.{name}", v)
        vb = common.mean(bpbytes)
        common.emit(out, f"g1.bits_per_byte.domain.{name}", vb)
        if v is not None:
            domain_bpb.append(v)
        if vb is not None:
            domain_bpbyte.append(vb)
    if domain_bpb:
        common.emit(out, "g1.bpb.multidomain", common.mean(domain_bpb))
    if domain_bpbyte:
        common.emit(out, "g1.bits_per_byte.multidomain", common.mean(domain_bpbyte))
    out["g1.assets.domains_present"] = float(len(domain_bpb))
    _alias_canonical_domains(out)

    # 3) Fresh held-out FineWeb dump, then news/prose aliases, then chance.
    _ensure_fresh(model, tok, device, budget, ctx, cap, out)
    return out


def _alias_canonical_domains(out):
    """Copy alias slices into scored names; fail-closed chance if still absent."""
    for canon in _CANONICAL_DOMAINS:
        key = f"g1.bits_per_byte.domain.{canon}"
        if out.get(key) is not None:
            continue
        for alias in _DOMAIN_ALIASES.get(canon, ()):
            src = f"g1.bits_per_byte.domain.{alias}"
            if alias != canon and out.get(src) is not None:
                out[key] = float(out[src])
                if out.get(f"g1.bpb.domain.{alias}") is not None:
                    out[f"g1.bpb.domain.{canon}"] = float(out[f"g1.bpb.domain.{alias}"])
                out[f"g1.alias.{canon}"] = 1.0
                break
        if out.get(key) is None:
            common.emit(out, key, _CENSORED_BITS_PER_BYTE)
            out[f"g1.missing.{canon}"] = 1.0


def _ensure_fresh(model, tok, device, budget, ctx, cap, out):
    if out.get("g1.bits_per_byte.fresh") is not None:
        return
    for rel in _FRESH_RELS:
        path = common.assets_path(ctx, rel)
        if path is None:
            continue
        if not budget.ok():
            out["g1.partial"] = 1.0
            break
        already = rel.startswith("g1/domains/") and out.get(
            f"g1.bits_per_byte.domain.{rel.rsplit('/', 1)[-1][: -len('.jsonl')]}"
        )
        if already is not None and rel != "g1/fresh.jsonl":
            common.emit(out, "g1.bits_per_byte.fresh", already)
            common.emit(out, "g1.bpb.fresh", out.get("g1.bpb.domain.news") or out.get("g1.bpb.domain.prose"))
            out["g1.alias.fresh"] = 1.0
            return
        texts = [r.get("text", "") for r in common.load_jsonl(path, cap=cap)]
        texts = [t for t in texts if t]
        ces, _, _, bpbytes, _ = _score_texts(
            model, tok, texts, device, budget, ctx, "fresh", out, "g1.fresh"
        )
        common.emit(out, "g1.bpb.fresh", common.bpb(common.mean(ces)))
        common.emit(out, "g1.bits_per_byte.fresh", common.mean(bpbytes))
        if out.get("g1.bits_per_byte.fresh") is not None:
            if rel != "g1/fresh.jsonl":
                out["g1.alias.fresh"] = 1.0
            return
    if out.get("g1.bits_per_byte.fresh") is None:
        common.emit(out, "g1.bits_per_byte.fresh", _CENSORED_BITS_PER_BYTE)
        out["g1.missing.fresh"] = 1.0
