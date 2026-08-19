"""FLOPs attestation — the budget currency (dual cap).

Recipe 2.1.0 makes this attestation part of the miner-facing pinned contract;
the harness bytes also move `recipe_pin_hex()`.

The budget a Prism submission spends is **attested FLOPs**, and it is
measured here. Wall-clock survives only as an anti-DoS bound. Nothing in
this module reads a number the miner supplies:

    f_tok, cv, samples = probe_flops_per_token(model, stream, secret_seed)
    C_attested         = f_tok * stream.tokens_seen     # both harness-owned

`f_tok` is the median over `FLOPS_PROBE_SAMPLES` harness-driven
forward+backward passes under `torch.utils.flop_counter.FlopCounterMode`,
on batches drawn from the real train stream at **secret indices**. The
batches are indistinguishable from training batches (same stream, same
permutation, same shapes), so a model cannot be cheap on probes and
expensive on training without also being unstable across probes -- which
`flops_probe_cv` exposes. When the CV exceeds `FLOPS_PROBE_CV_MAX` the
estimate switches from the median to the **max**: if cost is
input-dependent, the budget charges the expensive branch.

### Where the probe secret comes from, precisely

The eval secret (`PRISM_EVAL_SECRET_SEED`) is staged **after** training and
is deliberately withheld from the train-phase child, so the probe cannot
use it. [`probe_secret`] therefore resolves, in order:

1. `PRISM_FLOPS_PROBE_SECRET` — operator-set, for reproducing a run.
2. `PRISM_EVAL_SECRET_SEED` — only present on local operator runs that
   preset it; on a real pod the train child never sees it.
3. A fresh 128-bit `os.urandom` secret generated in the train child.

(3) is the production path, and it is *unpredictable* rather than *secret*:
the miner cannot know the indices in advance, which is what defeats
branching on batch content. It is not hidden from an adversary who
introspects the harness frame in-process -- cheatguard screens for that
separately. Saying otherwise would overstate the guarantee.

Determinism is preserved where it matters: `secret_indices(secret, n, span)`
is a pure function, and a probe replayed with the same secret on the same
checkpoint reproduces `flops_per_token` exactly (a dispatch counter has no
kernel-selection noise). The secret is recorded so a run can be replayed.

Why this is measurement and not modelling: a dispatch counter sees the
computation that actually ran, so it gets MoE routing sparsity, loop
factors and the attention quadratic term correct for free, on an
architecture nobody has to understand in advance.

## The hole this module cannot close by counting

`FlopCounterMode` intercepts `__torch_dispatch__`. A fused Triton/CUDA
kernel registered as ONE opaque op contributes whatever the counter knows
about that op -- typically nothing. Recipe-v10 lets miners install their
own dependencies, so shipping such a kernel is easy. A model whose matmuls
all hide inside custom ops would attest near-zero FLOPs and train
effectively unbudgeted.

So the counter is cross-checked against an **analytic** FLOPs/token model
built from the module graph ([`analytic_flops_per_token`]):

    F_tok = 6*N_body*r_eff  +  6*d*V  +  12*L*d*S
            (body matmuls)     (lm_head)  (attention quadratic)

The `lm_head` term matters far more than the usual `C = 6ND` shorthand
admits: at `d=512, V=32768` the head alone is ~36% of FLOPs/token and
`6*N_body` captures only ~55% of the true cost, so `6ND` overstates the
affordable token count by 1.3-1.8x at Prism's scale. MoE counts **active**
experts only, because that is what runs.

When counter and analytic estimate disagree by more than
`FLOPS_ANALYTIC_GAP_MAX`, `flops_analytic_ratio` records the ratio and
`flops_analytic_mismatch` is set. Neither is silent, and neither is a
guess: both numbers are emitted so an operator (and the agentic reviewer)
sees the disagreement rather than a passing run.

When `FlopCounterMode` cannot produce a sample at all — OOM even on a
single row (LoopMoE / fused kernels on a resident model), or the counter
class is missing — [`analytic_fallback_attestation`] still arms the
FLOPs cap from the graph formula. That path is loud
(`estimator="analytic_fallback"`) and never a silent disarm.

A full-graph counted fwd+bwd on an 850M–1B model fills a 32 GB card
(~31/32 GiB). `empty_cache()` does not give that reservation back, so the
train parent must **not** run that probe before `mp.spawn`: skip to
analytic 6N (`skip_dispatch_probe`) and reclaim CUDA so workers can load.

Residual risk, stated plainly: the analytic model reads `nn.Linear`-shaped
weights and declared attention/MoE attributes. An architecture that is
genuinely novel in a way the model does not recognize produces a wide gap
for an honest reason. The gap is therefore **evidence**, not an automatic
fail -- `flops_analytic_mismatch` flags it for review, while the run itself
is bounded by the dual cap either way.
"""

import hashlib
import os

# Fwd+bwd multiplier on a matmul's 2*m*n*k: forward once, backward twice
# (grad wrt input and wrt weight). Recompute, if the miner uses it, shows up
# in the counter on its own -- it is a real extra forward.
FWD_BWD = 3.0

# Analytic-model constants (documented in the module docstring).
_BODY_COEFF = 6.0
_HEAD_COEFF = 6.0
_ATTN_COEFF = 12.0

FLOPS_PROBE_SAMPLES = 8
FLOPS_PROBE_CV_MAX = 0.15
FLOPS_ANALYTIC_GAP_MAX = 0.25

# RTX 5090 dense bf16 with fp32 accumulate, per GPU. Used only for the
# physical-plausibility bound and the MFU telemetry, never for the budget.
PEAK_FLOPS_PER_GPU = 209.5e12

# Slack on the physical bound: kernel-selection noise and clock boost make an
# exact comparison meaningless, but a 5% margin still catches an attestation
# that claims more FLOPs than the hardware could have executed.
PHYSICAL_BOUND_SLACK = 1.05

# Skip FlopCounterMode on the full resident graph above this unique-param
# count. A counted 1B fwd+bwd pins ~31 GiB on GPU0; the caching allocator
# keeps it after empty_cache() and 4× spawn ranks SIGKILL at payload load.
DISPATCH_PROBE_PARAM_CEILING = 400_000_000
# Free HBM the parent must leave for one replica after any CUDA work.
PARENT_HBM_HEADROOM_BYTES = 12 * (1 << 30)
# Reserved bytes allowed on GPU0 after reclaim (CUDA context, not a replica).
PARENT_RESERVED_MAX_BYTES = 1 * (1 << 30)


class DispatchProbeSkipped(Exception):
    """Full-graph FlopCounterMode skipped so parent HBM stays spawnable."""

    def __init__(self, reason):
        self.reason = str(reason)
        super().__init__(self.reason)


class BudgetExhausted(Exception):
    """A budget cap was reached. **Expected**, not a failure.

    Carries which cap bound so `train_v3` can route to the same graceful
    checkpoint-then-eval path as `FinishEvaluation`. The predecessor
    (`_CapExceeded`) was caught by nothing and lost the whole run, which
    made reaching your own budget a way to score zero.
    """

    def __init__(self, cap, spent=None, limit=None):
        super().__init__(f"budget exhausted: {cap}")
        self.cap = str(cap)
        self.spent = spent
        self.limit = limit


def probe_secret(explicit=None):
    """Resolve the probe secret (see module docs for the ordering).

    Returns `(secret, source)` so the source is recorded alongside the
    attestation: an operator reading metrics must be able to tell a
    reproducible seeded run from the fresh-random production path.
    """
    if explicit:
        return str(explicit), "explicit"
    for env, source in (
        ("PRISM_FLOPS_PROBE_SECRET", "env_probe_secret"),
        ("PRISM_EVAL_SECRET_SEED", "env_eval_secret"),
    ):
        val = os.environ.get(env, "").strip()
        if val:
            return val, source
    # Production path: the train child has no operator secret, so draw a
    # fresh unpredictable one. Unpredictable-in-advance is the property that
    # defeats probe detection; it is not secrecy from in-process inspection.
    return os.urandom(16).hex(), "urandom"


def secret_indices(secret_seed, n, span):
    """`n` distinct batch indices in `[0, span)`, derived from the secret.

    Pure function of `(secret_seed, n, span)`: replaying a probe with the
    same secret probes the same batches. SHA-256 keyed by the secret so a
    miner cannot predict which batches will be probed -- branching on batch
    content is the natural way to be cheap on probes and expensive on
    training.
    """
    n = max(1, int(n))
    span = max(1, int(span))
    seed = str(secret_seed or "").encode("utf-8")
    out, i = [], 0
    seen = set()
    # Draw until n distinct indices or the space is exhausted.
    while len(out) < min(n, span) and i < 64 * n + 64:
        digest = hashlib.sha256(seed + b"|flops|" + str(i).encode("ascii")).digest()
        idx = int.from_bytes(digest[:8], "big") % span
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
        i += 1
    return out or [0]


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def coefficient_of_variation(xs):
    """Sample CV (std/mean); 0.0 for degenerate input."""
    vals = [float(x) for x in xs if x is not None]
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    if mean <= 0.0:
        return 0.0
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return (var**0.5) / mean


def _fwd_bwd(model, input_ids, labels):
    """Harness-owned forward+backward. Miner code never runs here.

    Gradients are zeroed afterwards so the probe cannot perturb training:
    the probe is a measurement, not a step.
    """
    import torch

    out = model(input_ids)
    logits = out.logits if hasattr(out, "logits") else out
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        reduction="mean",
    )
    loss.backward()
    model.zero_grad(set_to_none=True)
    return float(loss.detach().item())


def _is_oom(exc):
    """Whether `exc` is a CUDA/host out-of-memory error."""
    return isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()


def unique_param_count(model):
    """Tied embeddings counted once (matches the miner param floor/cap)."""
    seen, n = set(), 0
    for p in model.parameters():
        key = id(p)
        if key in seen:
            continue
        seen.add(key)
        n += int(p.numel())
    return int(n)


def cuda_hbm_snapshot():
    """Per-device free/total/reserved bytes. Empty when CUDA is absent."""
    out = {"cuda": False, "devices": []}
    try:
        import torch

        if not torch.cuda.is_available():
            return out
        out["cuda"] = True
        for i in range(int(torch.cuda.device_count())):
            reserved = 0
            allocated = 0
            try:
                reserved = int(torch.cuda.memory_reserved(i))
                allocated = int(torch.cuda.memory_allocated(i))
            except Exception:  # noqa: BLE001
                pass
            try:
                free, total = torch.cuda.mem_get_info(i)
            except Exception:  # noqa: BLE001
                free, total = 0, 0
            out["devices"].append(
                {
                    "index": i,
                    "free": int(free),
                    "total": int(total),
                    "reserved": reserved,
                    "allocated": allocated,
                }
            )
    except Exception:  # noqa: BLE001
        return out
    return out


def _cpu_module_tensors(obj):
    import torch

    if obj is None or not hasattr(obj, "modules"):
        return
    for child in obj.modules():
        for name, val in list(vars(child).items()):
            if torch.is_tensor(val) and val.is_cuda:
                setattr(child, name, val.detach().cpu())
    for p in obj.parameters():
        p.grad = None
        if p.data.is_cuda:
            p.data = p.data.cpu()
    for b in obj.buffers():
        if b.is_cuda:
            b.data = b.data.cpu()


def reclaim_cuda(model=None, stream=None):
    """Move parent tensors to CPU, drop the caching allocator, snapshot HBM."""
    import gc

    import torch

    if model is not None:
        try:
            model.to("cpu")
        except Exception:  # noqa: BLE001
            pass
        _cpu_module_tensors(model)
    if stream is not None:
        if hasattr(stream, "device"):
            stream.device = "cpu"
        for name in ("_buf", "_last", "input_ids", "labels"):
            val = getattr(stream, name, None)
            try:
                import torch as _t

                if _t.is_tensor(val) and val.is_cuda:
                    setattr(stream, name, val.detach().cpu())
            except Exception:  # noqa: BLE001
                pass
    gc.collect()
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return cuda_hbm_snapshot()


def parent_hbm_ready_for_replica(snapshot, headroom=None):
    """True when GPU0 free bytes meet the one-replica headroom (or no CUDA)."""
    headroom = PARENT_HBM_HEADROOM_BYTES if headroom is None else int(headroom)
    if not snapshot or not snapshot.get("cuda") or not snapshot.get("devices"):
        return True
    free = int(snapshot["devices"][0].get("free") or 0)
    return free >= headroom


def assert_parent_cuda_released(snapshot, max_reserved=None):
    """Fail closed if GPU0 still holds a replica-sized reservation."""
    max_reserved = PARENT_RESERVED_MAX_BYTES if max_reserved is None else int(
        max_reserved
    )
    if not snapshot or not snapshot.get("cuda") or not snapshot.get("devices"):
        return snapshot
    reserved = int(snapshot["devices"][0].get("reserved") or 0)
    free = int(snapshot["devices"][0].get("free") or 0)
    total = int(snapshot["devices"][0].get("total") or 0)
    if reserved > max_reserved:
        raise RuntimeError(
            f"parent GPU0 still reserved {reserved / (1 << 30):.2f} GiB "
            f"(free {free / (1 << 30):.2f}/{total / (1 << 30):.2f} GiB) after "
            f"reclaim; aborting before spawn SIGKILL (ceiling "
            f"{max_reserved / (1 << 30):.2f} GiB)"
        )
    return snapshot


def skip_dispatch_probe(model, cfg=None):
    """Whether to arm FLOPs from analytic 6N *before* a full-graph probe.

    Returns `(skip, reason)`. Skip when the operator asks, unique params
    exceed `DISPATCH_PROBE_PARAM_CEILING`, or free HBM cannot host the
    probe plus one replica.
    """
    cfg = cfg or {}
    if cfg.get("flops_probe_skip"):
        return True, "cfg_flops_probe_skip"
    env = os.environ.get("PRISM_FLOPS_PROBE_SKIP", "").strip().lower()
    if env in {"1", "true", "yes"}:
        return True, "env_PRISM_FLOPS_PROBE_SKIP"
    n = unique_param_count(model)
    if n >= DISPATCH_PROBE_PARAM_CEILING:
        return True, f"param_ceiling n={n}>={DISPATCH_PROBE_PARAM_CEILING}"
    snap = cuda_hbm_snapshot()
    if snap.get("cuda") and snap.get("devices"):
        free = int(snap["devices"][0].get("free") or 0)
        # Counted fwd+bwd activations + counter bookkeeping ≫ param bytes.
        need = int(n) * 2 * 8 + PARENT_HBM_HEADROOM_BYTES
        if free and free < need:
            return True, f"hbm_headroom free={free} need={need}"
    return False, ""


def _free_cuda():
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _probe_once(model, stream, idx, rows, counter_cls):
    """One counted fwd+bwd at `idx`, on at most `rows` sequences.

    Returns `(flops, n_tokens, rows_used)`. A smaller row count is a valid
    measurement because the quantity is FLOPs **per token** at a FIXED
    sequence length: body, head and attention terms are all per-token at
    fixed `S`, so trimming rows changes efficiency, not cost per token.
    """
    input_ids, labels = stream.peek_batch(idx)
    if rows and rows < int(input_ids.shape[0]):
        input_ids = input_ids[:rows].contiguous()
        labels = labels[:rows].contiguous()
    n_tok = int(labels.numel())
    if n_tok <= 0:
        return None, 0, 0
    with counter_cls(display=False) as counter:
        _fwd_bwd(model, input_ids, labels)
    return float(counter.get_total_flops()), n_tok, int(input_ids.shape[0])


def probe_flops_per_token(model, stream, secret_seed=None, n=None, log=None):
    """Attest FLOPs/token with `FlopCounterMode` on secret-index batches.

    Returns `{flops_per_token, cv, samples, unstable, estimator, ...}`.

    `estimator` is `"median"` normally and `"max"` when `cv` exceeds
    `FLOPS_PROBE_CV_MAX` -- input-dependent cost is charged at its
    expensive branch, so probe-detection cannot buy compute.

    ## Out-of-memory degrades, it does not disarm

    The probe adds a fwd+bwd of its own on top of a model already resident on
    the GPU, so it can OOM where training itself would not -- measured on the
    hybrid delta-net baseline at 341M params on a 32 GB card. That matters
    more than it looks: a probe that simply fails leaves the FLOPs cap
    DISARMED, which (a) loses attestation on exactly the memory-heavy
    architectures where the budget matters most, and (b) is an escape a miner
    could induce deliberately, since OOM-ing the probe converts the budget
    back to wall-clock.

    So an OOM halves the probe's row count and retries, down to a single
    sequence, freeing the cache between attempts. `probe_rows` and
    `probe_rows_reduced` are reported so a reduced-batch attestation is
    visible rather than silently equated with a full-batch one.
    """
    skip, reason = skip_dispatch_probe(model)
    if skip:
        raise DispatchProbeSkipped(reason)

    from torch.utils.flop_counter import FlopCounterMode

    n = int(n or FLOPS_PROBE_SAMPLES)
    secret_seed, secret_source = probe_secret(secret_seed)
    was_training = model.training
    model.train()
    samples, tokens = [], []
    span = getattr(stream, "probe_span", 4096)
    rows = int(getattr(stream, "batch_size", 1) or 1)
    full_rows = rows
    last_oom = None
    try:
        for idx in secret_indices(secret_seed, n, span):
            while True:
                try:
                    total, n_tok, used = _probe_once(
                        model, stream, idx, rows, FlopCounterMode
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    if not _is_oom(exc) or rows <= 1:
                        raise
                    last_oom = str(exc)[:160]
                    rows = max(1, rows // 2)
                    _free_cuda()
                    if log:
                        log(f"flops probe OOM at idx {idx}; retrying with {rows} row(s)")
            if total is None:
                continue
            if total <= 0.0:
                # A zero total is itself the F1 signature: every matmul hid
                # inside ops the dispatcher does not attribute. Keep the
                # sample so the analytic cross-check sees a ~0 ratio.
                if log:
                    log(f"flops probe {idx}: counter saw 0 FLOPs (opaque kernels?)")
            samples.append(total / n_tok)
            tokens.append(n_tok)
            del total
    finally:
        if not was_training:
            model.eval()
        _free_cuda()

    if not samples:
        raise RuntimeError("flops probe: no scored batches")

    cv = coefficient_of_variation(samples)
    unstable = cv > FLOPS_PROBE_CV_MAX
    f_tok = max(samples) if unstable else _median(samples)
    if log:
        log(
            f"flops/token attested: {f_tok:.4g} (n={len(samples)} cv={cv:.4f} "
            f"estimator={'max' if unstable else 'median'})"
        )
    return {
        "flops_per_token": float(f_tok),
        "cv": float(cv),
        "samples": [float(s) for s in samples],
        "n_samples": len(samples),
        "unstable": bool(unstable),
        "estimator": "max" if unstable else "median",
        "tokens_per_batch": int(_median(tokens)) if tokens else 0,
        # Reported so a reduced-batch attestation is never silently equated
        # with a full-batch one.
        "probe_rows": int(rows),
        "probe_rows_full": int(full_rows),
        "probe_rows_reduced": bool(rows < full_rows),
        "oom_retry": last_oom,
        # Recorded so a run can be replayed; `secret_source` distinguishes a
        # reproducible seeded probe from the fresh-random production path.
        "secret_source": secret_source,
        "secret": secret_seed,
    }


def _vocab_size(model):
    """Vocabulary from the largest embedding table (or the config)."""
    import torch

    vocab = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.Embedding):
            vocab = max(vocab, int(mod.num_embeddings))
    cfg = getattr(model, "config", None)
    val = getattr(cfg, "vocab_size", None) if cfg is not None else None
    if isinstance(val, int) and val > vocab:
        vocab = val
    return int(vocab)


def _split_params(model):
    """Partition parameters into `(body, embed, head)`.

    Three-way, not two-way, because the analytic formula charges the output
    head **separately** (`6*d*V`): folding it into `N_body` and then adding
    the head term again double-counts it, which is a ~2x error on a small-`d`
    model where the head is a third of all FLOPs.

    - `embed`: `nn.Embedding` tables. A lookup costs ~no FLOPs per token, so
      including them would let a large vocabulary buy a discount.
    - `head`: the output projection — a `Linear` whose `out_features` equals
      the vocabulary. Weight-tied heads share the embedding tensor and are
      therefore already excluded from `body`, and their FLOPs still land in
      `head_term`.
    - `body`: everything else — the transformer blocks that `r_eff` loops.
    """
    import torch

    vocab = _vocab_size(model)
    embed_ids, head_ids = set(), set()
    for mod in model.modules():
        if isinstance(mod, torch.nn.Embedding):
            embed_ids.add(id(mod.weight))
        elif isinstance(mod, torch.nn.Linear) and vocab and mod.out_features == vocab:
            head_ids.add(id(mod.weight))
            if mod.bias is not None:
                head_ids.add(id(mod.bias))
    body = embed = head = 0
    for p in model.parameters():
        if id(p) in embed_ids:
            embed += p.numel()
        elif id(p) in head_ids:
            head += p.numel()
        else:
            body += p.numel()
    return int(body), int(embed), int(head)


def _has_attention(model):
    """Whether the graph plausibly contains softmax attention.

    The `12*L*d*S` quadratic term is real for attention and a **phantom**
    for anything else (a pure-MLP, SSM or delta-net block pays no such
    cost). Charging it unconditionally would manufacture a disagreement on
    exactly the non-transformer architectures Prism exists to evaluate, so
    it is applied only when attention is detected — and whether it was
    applied is reported.

    Detection is deliberately shallow (class-name / module-type based): a
    novel attention implementation may not be recognized, which widens the
    gap for an honest reason. That is why the gap is review evidence rather
    than an automatic fail.
    """
    import torch

    for mod in model.modules():
        if isinstance(mod, torch.nn.MultiheadAttention):
            return True
        name = type(mod).__name__.lower()
        if "attention" in name or name.endswith("attn") or "attn" in name:
            return True
    return False


def _geometry(model):
    """Best-effort `(d_model, n_layers)` from the module graph."""
    import torch

    d_model, layers = 0, 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.Embedding):
            d_model = max(d_model, int(mod.embedding_dim))
        if isinstance(mod, (torch.nn.ModuleList, torch.nn.Sequential)):
            layers = max(layers, len(mod))
    cfg = getattr(model, "config", None)
    for attr in ("d_model", "hidden_size", "n_embd"):
        val = getattr(cfg, attr, None) if cfg is not None else None
        if isinstance(val, int) and val > 0:
            d_model = val
            break
    for attr in ("n_layer", "num_hidden_layers", "n_layers"):
        val = getattr(cfg, attr, None) if cfg is not None else None
        if isinstance(val, int) and val > 0:
            layers = val
            break
    return int(d_model), int(layers)


def analytic_flops_per_token(
    model, seq_len, r_eff=None, active_fraction=None, has_attention=None
):
    """Analytic FLOPs/token from the module graph — the cross-check.

    `F_tok = 6*N_body*r_eff*active + 6*d*V + 12*L*d*S`

    - `N_body` **excludes** embeddings and the output head (see
      [`_split_params`]); the head is charged once, by `6*d*V`.
    - `r_eff` (loop factor) is read from `model.prism_loop_factor` when the
      architecture declares one, else 1.0. Only the body loops: the head and
      the attention score matmuls do not, which is why a model looped `r=4`
      pays ~3.3x rather than 4x.
    - `active_fraction` scales the body for MoE: only the experts that run
      cost anything. Read from `model.prism_active_param_fraction`.
    - the quadratic attention term is charged only when attention is
      detected (see [`_has_attention`]).

    All three hooks are advisory and **cannot buy budget**: the dispatch
    counter remains the meter, and this estimate only ever produces a
    visible disagreement.
    """
    body, embed, head = _split_params(model)
    d_model, layers = _geometry(model)
    vocab = _vocab_size(model)
    if r_eff is None:
        r_eff = float(getattr(model, "prism_loop_factor", 1.0) or 1.0)
    if active_fraction is None:
        active_fraction = float(getattr(model, "prism_active_param_fraction", 1.0) or 1.0)
    if has_attention is None:
        has_attention = _has_attention(model)
    r_eff = max(0.0, r_eff)
    active_fraction = min(1.0, max(0.0, active_fraction))

    body_term = _BODY_COEFF * body * r_eff * active_fraction
    # Prefer the head's ACTUAL parameter count over `d*V`: a factorized or
    # bias-carrying head is then charged for what it is, not for a guess.
    head_params = head if head else (d_model * vocab)
    head_term = _HEAD_COEFF * head_params
    attn_term = (
        _ATTN_COEFF * layers * d_model * int(seq_len)
        if has_attention and d_model and layers
        else 0.0
    )
    total = body_term + head_term + attn_term
    return {
        "flops_per_token": float(total),
        "body_term": float(body_term),
        "head_term": float(head_term),
        "attn_term": float(attn_term),
        "head_share": float(head_term / total) if total > 0 else 0.0,
        "n_params_body": int(body),
        "n_params_embed": int(embed),
        "n_params_head": int(head),
        "d_model": int(d_model),
        "n_layers": int(layers),
        "vocab_size": int(vocab),
        "r_eff": float(r_eff),
        "active_fraction": float(active_fraction),
        "attention_detected": bool(has_attention),
    }


def analytic_gap(counted, analytic):
    """Relative disagreement between counter and analytic estimate.

    Normalized by the **larger** of the two so the ratio is symmetric and
    bounded in `[0, 1]`: a counter that saw nothing (the fused-kernel
    signature) yields 1.0 rather than dividing by ~0.
    """
    counted, analytic = float(counted), float(analytic)
    hi = max(abs(counted), abs(analytic))
    if hi <= 0.0:
        return 0.0
    return abs(counted - analytic) / hi


def analytic_fallback_attestation(model, seq_len, reason=None, log=None):
    """Arm the dual cap from the analytic graph when the dispatch probe dies.

    Used when `FlopCounterMode` OOMs even at one row (LoopMoE / fused
    kernels on a resident model) or the counter cannot run. Loud on
    purpose: `estimator` is `analytic_fallback` and `reason` is kept.
    Returns `None` only when the graph estimate is non-positive — that
    path cannot honestly meter a run.
    """
    est = analytic_flops_per_token(model, seq_len)
    f_tok = float(est["flops_per_token"])
    if f_tok <= 0.0:
        return None
    note = (reason or "probe unavailable").strip()[:200]
    if log:
        log(
            f"flops probe failed ({note}); analytic fallback {f_tok:.4g} "
            "FLOPs/token — dual cap stays armed"
        )
    return {
        "flops_per_token": f_tok,
        "cv": 0.0,
        "samples": [f_tok],
        "n_samples": 1,
        "unstable": False,
        "estimator": "analytic_fallback",
        "tokens_per_batch": 0,
        "probe_rows": 0,
        "probe_rows_full": 0,
        "probe_rows_reduced": False,
        "oom_retry": note,
        "secret_source": "analytic_fallback",
        "analytic_fallback": True,
        "analytic_breakdown": est,
    }


def cross_check(counted_per_token, model, seq_len, gap_max=None):
    """Counter vs analytic model; returns the visible disagreement record."""
    gap_max = FLOPS_ANALYTIC_GAP_MAX if gap_max is None else float(gap_max)
    est = analytic_flops_per_token(model, seq_len)
    analytic = est["flops_per_token"]
    gap = analytic_gap(counted_per_token, analytic)
    ratio = (counted_per_token / analytic) if analytic > 0 else 0.0
    return {
        "analytic_flops_per_token": float(analytic),
        "analytic_ratio": float(ratio),
        "analytic_gap": float(gap),
        "mismatch": bool(gap > gap_max),
        "gap_max": float(gap_max),
        "breakdown": est,
    }


def n_gpus_visible():
    """GPU count the run may use (attested, not declared).

    Recorded because a single-GPU run would otherwise silently get the
    physical headroom of a 4-GPU allowance.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return max(1, int(torch.cuda.device_count()))
    except Exception:  # noqa: BLE001
        pass
    return 1


def physically_possible(flops_attested, wall_s, n_gpu=None, peak=None):
    """`flops_attested <= peak * n_gpu * wall_s * slack`.

    A violation means the attestation is internally inconsistent (the
    cheat taxonomy already carries `inconsistent_metrics`), not that the
    model is fast.
    """
    n_gpu = n_gpus_visible() if n_gpu is None else max(1, int(n_gpu))
    peak = PEAK_FLOPS_PER_GPU if peak is None else float(peak)
    ceiling = peak * n_gpu * max(0.0, float(wall_s)) * PHYSICAL_BOUND_SLACK
    ok = float(flops_attested) <= ceiling if ceiling > 0 else True
    return bool(ok), float(ceiling)


def mfu(flops_attested, wall_s, n_gpu=None, peak=None):
    """Achieved model-FLOPs utilization — the Phase-0 unknown.

    Every FLOPs constant scales linearly with this. At 15% real MFU the
    3.0e18 cap becomes wall-bound and the dual cap partly reverts to the
    status quo, which is why it is measured rather than assumed.
    """
    n_gpu = n_gpus_visible() if n_gpu is None else max(1, int(n_gpu))
    peak = PEAK_FLOPS_PER_GPU if peak is None else float(peak)
    denom = peak * n_gpu * max(1e-9, float(wall_s))
    return float(flops_attested) / denom if denom > 0 else 0.0


def secret_seed_from_env():
    """The operator eval secret the probe indices derive from."""
    return os.environ.get("PRISM_EVAL_SECRET_SEED", "") or ""
