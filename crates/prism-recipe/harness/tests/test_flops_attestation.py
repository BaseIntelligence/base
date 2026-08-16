"""FLOPs attestation + dual-cap enforcement regressions.

Covers the properties the budget currency depends on:

1. probe determinism given a secret, and independence from stream state,
2. the analytic cross-check catching an opaque fused kernel (the F1 hole),
3. dual-cap binding: whichever cap binds first stops the run, and which one
   bound is recorded,
4. the underspend guard's inputs (`spend_fraction`),
5. `peek_batch` not spending the budget it measures,
6. byte accounting on the probe curve (the bits/byte contract).

Runs on CPU with a tiny model; no GPU, no pod.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from prismlib import flops as F  # noqa: E402
from prismlib.stream import SeededTrainStream  # noqa: E402

VOCAB = 128
D_MODEL = 32
SEQ = 16
BATCH = 2


class TinyTok:
    """Deterministic tokenizer: one token per character (ASCII ⇒ 1 B/token)."""

    eos_token_id = 0

    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
    ):
        ids = [(ord(c) % (VOCAB - 1)) + 1 for c in text]
        limit = max_length if (truncation and max_length) else 4 * SEQ
        ids = ids[:limit] or [1]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


class TinyModel(torch.nn.Module):
    """Dense reference: embedding -> linear body -> lm_head."""

    def __init__(self, layers=2):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D_MODEL)
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(D_MODEL, D_MODEL) for _ in range(layers)]
        )
        self.lm_head = torch.nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for blk in self.blocks:
            h = torch.relu(blk(h))
        return self.lm_head(h)


class OpaqueModel(TinyModel):
    """The F1 attack shape: the body matmul hides from the dispatcher.

    A real submission would register a fused Triton kernel as one opaque
    custom op. We emulate the *observable consequence* — a body whose FLOPs
    the counter does not attribute — by routing the body through an
    autograd.Function whose forward does the matmul under a no-dispatch
    detour. The parameters still exist, so the ANALYTIC model still charges
    for them: that asymmetry is exactly what the cross-check detects.
    """

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for blk in self.blocks:
            # Skip the linear entirely: the counter sees no matmul, while
            # blk.weight is still a body parameter analytically.
            h = torch.relu(h + blk.bias)
        return self.lm_head(h)


def texts(n=64):
    return [f"document {i} " + ("abcdefgh " * 12) for i in range(n)]


def make_stream(flops_cap=0.0, wall_cap_s=0.0, t0=None):
    return SeededTrainStream(
        texts(),
        TinyTok(),
        "cpu",
        seq_len=SEQ,
        batch_size=BATCH,
        seed=1234,
        flops_cap=flops_cap,
        wall_cap_s=wall_cap_s,
        t0=t0,
    )


# ---------------------------------------------------------------- indices


def test_secret_indices_are_deterministic_and_distinct():
    a = F.secret_indices("secret-a", 8, 64)
    b = F.secret_indices("secret-a", 8, 64)
    c = F.secret_indices("secret-b", 8, 64)
    assert a == b, "same secret must select the same batches (replayable)"
    assert a != c, "a different secret must select different batches"
    assert len(set(a)) == len(a) == 8, f"indices must be distinct: {a}"
    assert all(0 <= i < 64 for i in a)
    # Degenerate spans must not loop forever or crash.
    assert F.secret_indices("s", 8, 1) == [0]
    assert len(F.secret_indices("s", 100, 3)) == 3


def test_probe_secret_resolution_order():
    os.environ.pop("PRISM_FLOPS_PROBE_SECRET", None)
    os.environ.pop("PRISM_EVAL_SECRET_SEED", None)
    s, src = F.probe_secret("explicit-value")
    assert (s, src) == ("explicit-value", "explicit")
    os.environ["PRISM_EVAL_SECRET_SEED"] = "12345"
    assert F.probe_secret() == ("12345", "env_eval_secret")
    os.environ["PRISM_FLOPS_PROBE_SECRET"] = "abcdef"
    assert F.probe_secret() == ("abcdef", "env_probe_secret")
    os.environ.pop("PRISM_FLOPS_PROBE_SECRET")
    os.environ.pop("PRISM_EVAL_SECRET_SEED")
    # Production path: fresh, unpredictable, and not empty.
    s1, src1 = F.probe_secret()
    s2, _ = F.probe_secret()
    assert src1 == "urandom" and len(s1) == 32 and s1 != s2


# ------------------------------------------------------------------ probe


def test_probe_is_deterministic_for_a_fixed_secret():
    model = TinyModel()
    a = F.probe_flops_per_token(model, make_stream(), "fixed-secret", n=4)
    b = F.probe_flops_per_token(model, make_stream(), "fixed-secret", n=4)
    assert a["flops_per_token"] == b["flops_per_token"], (
        "a dispatch counter has no kernel-selection noise: replaying a probe "
        f"must reproduce it exactly ({a['flops_per_token']} vs {b['flops_per_token']})"
    )
    assert a["n_samples"] == 4
    assert a["flops_per_token"] > 0.0
    assert a["estimator"] == "median", a
    assert a["cv"] <= F.FLOPS_PROBE_CV_MAX
    assert a["secret_source"] == "explicit"


def test_probe_does_not_spend_or_advance_the_stream():
    """Measuring the budget must not consume it."""
    stream = make_stream(flops_cap=1e12)
    stream.set_flops_per_token(10.0)
    for _ in range(3):
        stream.next_batch()
    before = (stream.tokens_seen, stream.bytes_seen, stream.flops_spent)
    F.probe_flops_per_token(TinyModel(), stream, "s", n=4)
    after = (stream.tokens_seen, stream.bytes_seen, stream.flops_spent)
    assert before == after, f"probe moved stream state {before} -> {after}"


def test_peek_batch_shape_matches_training_batches():
    stream = make_stream()
    train_ids, train_labels = stream.next_batch()
    peek_ids, peek_labels = stream.peek_batch(17)
    assert peek_ids.shape == train_ids.shape
    assert peek_labels.shape == train_labels.shape
    assert peek_labels.numel() == BATCH * SEQ
    # Same index ⇒ same batch (replayable probes).
    again_ids, _ = stream.peek_batch(17)
    assert torch.equal(peek_ids, again_ids)


def test_high_cv_switches_to_max_estimator():
    """Input-dependent cost is charged at its expensive branch."""

    class Erratic(TinyModel):
        """Cost depends on batch content: cheap on some batches."""

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, input_ids):
            self.calls += 1
            h = self.embed(input_ids)
            # Every other probe runs the body twice as much work.
            reps = 1 if self.calls % 2 else 6
            for _ in range(reps):
                for blk in self.blocks:
                    h = torch.relu(blk(h))
            return self.lm_head(h)

    out = F.probe_flops_per_token(Erratic(), make_stream(), "s", n=6)
    assert out["cv"] > F.FLOPS_PROBE_CV_MAX, f"expected unstable, got cv={out['cv']}"
    assert out["unstable"] is True
    assert out["estimator"] == "max"
    assert out["flops_per_token"] == max(out["samples"]), (
        "an unstable probe must charge the MAX, not the median: otherwise "
        "being cheap on half the probes buys compute"
    )


# --------------------------------------------------------- cross-check F1


def test_analytic_model_includes_the_lm_head_term():
    """`C = 6ND` is wrong at this scale: the head is a large share."""
    est = F.analytic_flops_per_token(TinyModel(), SEQ)
    assert est["head_term"] > 0.0, "lm_head must be charged"
    assert est["n_params_body"] > 0 and est["n_params_embed"] > 0
    # Embeddings are excluded from the body: a lookup costs ~no FLOPs/token,
    # so counting them would hand a discount to a big vocabulary.
    assert est["n_params_embed"] == VOCAB * D_MODEL, est
    # The head is charged ONCE, via head_term — not folded into N_body and
    # then added again (that double-count was a ~2x error at small `d`).
    assert est["n_params_head"] == D_MODEL * VOCAB, est
    assert est["n_params_body"] < est["n_params_body"] + est["n_params_head"]
    body_only = 6.0 * est["n_params_body"]
    assert est["flops_per_token"] > body_only, (
        f"6*N_body must NOT be the whole estimate — the head is "
        f"{est['head_share']*100:.0f}% of FLOPs/token here"
    )
    # At small d the head dominates, which is the whole reason `6ND` fails.
    assert est["head_share"] > 0.3, est["head_share"]


def test_attention_term_is_not_charged_to_non_attention_models():
    """A phantom quadratic term would manufacture a false mismatch on
    exactly the SSM/delta-net architectures Prism exists to evaluate."""
    plain = F.analytic_flops_per_token(TinyModel(), SEQ)
    assert plain["attention_detected"] is False
    assert plain["attn_term"] == 0.0, plain

    class Attn(TinyModel):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.MultiheadAttention(D_MODEL, 2, batch_first=True)

    withattn = F.analytic_flops_per_token(Attn(), SEQ)
    assert withattn["attention_detected"] is True
    assert withattn["attn_term"] > 0.0
    # And it scales with sequence length (the quadratic term).
    longer = F.analytic_flops_per_token(Attn(), SEQ * 4)
    assert longer["attn_term"] > withattn["attn_term"]


def test_loop_factor_and_moe_active_fraction_scale_the_body():
    model = TinyModel()
    base = F.analytic_flops_per_token(model, SEQ)
    looped = F.analytic_flops_per_token(model, SEQ, r_eff=4.0)
    assert abs(looped["body_term"] - 4.0 * base["body_term"]) < 1e-6
    # Head and attention do NOT loop, so total is not 4x.
    assert looped["flops_per_token"] < 4.0 * base["flops_per_token"]
    # MoE: only the experts that run cost anything.
    moe = F.analytic_flops_per_token(model, SEQ, active_fraction=0.25)
    assert abs(moe["body_term"] - 0.25 * base["body_term"]) < 1e-6


def test_cross_check_detects_an_opaque_body():
    """The single largest residual risk must be VISIBLE, not silent."""
    honest = F.probe_flops_per_token(TinyModel(), make_stream(), "s", n=4)
    honest_cc = F.cross_check(honest["flops_per_token"], TinyModel(), SEQ)
    assert not honest_cc["mismatch"], (
        "a dense reference model must agree with its analytic estimate: "
        f"ratio={honest_cc['analytic_ratio']:.3f} gap={honest_cc['analytic_gap']:.3f}"
    )

    opaque = F.probe_flops_per_token(OpaqueModel(), make_stream(), "s", n=4)
    opaque_cc = F.cross_check(opaque["flops_per_token"], OpaqueModel(), SEQ)
    assert opaque["flops_per_token"] < honest["flops_per_token"], (
        "the emulated fused kernel must under-count vs the honest model"
    )
    assert opaque_cc["mismatch"], (
        "an uncounted body must trip the analytic cross-check: "
        f"ratio={opaque_cc['analytic_ratio']:.3f} gap={opaque_cc['analytic_gap']:.3f} "
        f"threshold={opaque_cc['gap_max']}"
    )
    assert opaque_cc["analytic_ratio"] < 1.0, "counter saw less than the model implies"


def test_analytic_gap_is_symmetric_and_bounded():
    assert F.analytic_gap(100.0, 100.0) == 0.0
    assert F.analytic_gap(0.0, 100.0) == 1.0, "a counter that saw nothing ⇒ gap 1"
    assert F.analytic_gap(100.0, 0.0) == 1.0
    assert F.analytic_gap(0.0, 0.0) == 0.0, "both zero must not divide by zero"
    assert abs(F.analytic_gap(75.0, 100.0) - 0.25) < 1e-12
    assert F.analytic_gap(100.0, 75.0) == F.analytic_gap(75.0, 100.0)
    assert 0.0 <= F.analytic_gap(1e-9, 1e18) <= 1.0


# -------------------------------------------------------------- dual cap


def test_flops_cap_binds_and_is_recorded():
    stream = make_stream(flops_cap=1e6)
    stream.set_flops_per_token(1000.0)  # 1e6/1000 = 1000 tokens of budget
    batches = 0
    try:
        while batches < 10_000:
            stream.next_batch()
            batches += 1
    except F.BudgetExhausted as exc:
        assert exc.cap == "flops", exc.cap
        assert exc.limit == 1e6
        assert exc.spent >= 1e6
    else:
        raise AssertionError("flops cap never bound")
    assert stream.binding_cap == "flops"
    assert stream.spend_fraction >= 1.0
    # The cap must bind at roughly the budgeted token count, not far past it.
    assert stream.tokens_seen <= 1000 + BATCH * SEQ, stream.tokens_seen
    assert stream.budget_report()["binding_cap"] == "flops"


def test_wall_cap_binds_when_flops_would_not():
    import time

    # Wall already exhausted, FLOPs effectively unlimited.
    stream = make_stream(flops_cap=1e30, wall_cap_s=0.001, t0=time.time() - 10.0)
    stream.set_flops_per_token(1.0)
    try:
        stream.next_batch()
    except F.BudgetExhausted as exc:
        assert exc.cap == "wall", exc.cap
    else:
        raise AssertionError("wall bound never fired")
    assert stream.binding_cap == "wall"
    rep = stream.budget_report()
    assert rep["binding_cap"] == "wall"
    assert rep["spend_fraction"] < 1.0, "a wall-bound run under-spends by definition"


def test_whichever_cap_binds_first_wins():
    """Dual cap: the binding one is the tighter one, and it is recorded."""
    import time

    # FLOPs tight, wall loose.
    a = make_stream(flops_cap=1e5, wall_cap_s=3600.0, t0=time.time())
    a.set_flops_per_token(1000.0)
    try:
        for _ in range(10_000):
            a.next_batch()
    except F.BudgetExhausted:
        pass
    assert a.binding_cap == "flops"

    # Wall tight, FLOPs loose.
    b = make_stream(flops_cap=1e30, wall_cap_s=0.001, t0=time.time() - 5.0)
    b.set_flops_per_token(1.0)
    try:
        b.next_batch()
    except F.BudgetExhausted:
        pass
    assert b.binding_cap == "wall"


def test_uncapped_stream_never_raises():
    """A disarmed probe must not brick training: the wall still contains it."""
    stream = make_stream()
    for _ in range(20):
        stream.next_batch()
    assert stream.tokens_seen == 20 * BATCH * SEQ
    assert stream.binding_cap == "none"
    assert stream.spend_fraction == 0.0


def test_spend_fraction_feeds_the_underspend_guard():
    stream = make_stream(flops_cap=1e6)
    stream.set_flops_per_token(1000.0)
    assert stream.spend_fraction == 0.0
    for _ in range(3):
        stream.next_batch()
    expected = 1000.0 * stream.tokens_seen / 1e6
    assert abs(stream.spend_fraction - expected) < 1e-12
    # MIN_SPEND_FRACTION = 0.5 is evaluated Rust-side on this number.
    assert 0.0 < stream.spend_fraction < 1.0


# ----------------------------------------------------------------- bytes


def test_stream_counts_bytes_for_the_bits_per_byte_contract():
    stream = make_stream()
    for _ in range(6):
        stream.next_batch()
    assert stream.tokens_seen > 0
    assert stream.bytes_seen > 0, "bytes must be counted for a bits/byte curve"
    bpt = stream.bytes_per_token()
    assert bpt > 0.0
    # TinyTok emits one token per character, so bytes/token is ~1 for ASCII.
    assert 0.2 < bpt < 8.0, f"implausible bytes/token: {bpt}"
    rep = stream.budget_report()
    assert rep["bytes_seen"] == stream.bytes_seen
    assert rep["bytes_per_token"] == bpt


def test_probe_curve_carries_bytes_and_bits_per_byte():
    from prismlib.probes import ProbeRunner

    stream = make_stream()
    for _ in range(4):
        stream.next_batch()
    model = TinyModel()
    runner = ProbeRunner(
        model=model,
        stream=stream,
        tok=TinyTok(),
        texts=texts(4),
        device="cpu",
        seq_len=SEQ,
        every=1,
        time_budget_s=60.0,
        log=lambda m: None,
    )
    state = {"reports": 1, "t0": 0.0, "probe_curve": []}
    runner.maybe_probe(state, step=10)
    assert len(state["probe_curve"]) == 1, state
    pt = state["probe_curve"][0]
    for key in ("tokens_seen", "bytes_seen", "bytes_per_token", "flops_spent", "probe_loss"):
        assert key in pt, f"probe point missing {key}: {pt}"
    assert pt["bytes_seen"] > 0
    assert "probe_bits_per_byte" in pt, (
        "the G6 bits/byte key requires byte counts on the curve — without it "
        "an `auc_log_bytes` anchor would be naming a quantity that does not exist"
    )
    # bits/byte = nats/token / (ln2 * bytes/token)
    expected = pt["probe_loss"] / (0.6931471805599453 * pt["bytes_per_token"])
    assert abs(pt["probe_bits_per_byte"] - expected) < 1e-9


# --------------------------------------------------------------- physical


def test_physical_bound_and_mfu():
    # 1 GPU, 1 second: anything above peak*1.05 is impossible.
    ok, ceiling = F.physically_possible(1e12, 1.0, n_gpu=1)
    assert ok and ceiling > 0
    bad, _ = F.physically_possible(1e18, 1.0, n_gpu=1)
    assert not bad, "an attestation above hardware peak must be flagged"
    # MFU scales as expected: half of peak over the wall is 50%.
    half = F.PEAK_FLOPS_PER_GPU * 0.5 * 10.0
    assert abs(F.mfu(half, 10.0, n_gpu=1) - 0.5) < 1e-9
    assert F.mfu(0.0, 10.0, n_gpu=1) == 0.0
    assert F.mfu(1e12, 0.0, n_gpu=1) >= 0.0, "zero wall must not divide by zero"


# ------------------------------------------------------- metrics plumbing


def _with_diag():
    """Import `main._with_diag` without running the harness's main()."""
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "prism_main_under_test", os.path.join(root, "main.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._with_diag  # noqa: SLF001


def test_diag_metrics_are_merged_into_an_existing_battery():
    """The attestation is measured in the TRAIN child but the composite reads
    `org.*` out of `battery.metrics`, which the EVAL child writes. Without the
    merge the whole Phase-0 deliverable is emitted and then dropped."""
    merge = _with_diag()
    battery = {"metrics": {"org.g1.bits_per_byte_prose": 1.10}, "tier": "battery-v1"}
    tpayload = {
        "diag_metrics": {
            "org.diag.mfu_achieved": 0.23,
            "org.diag.flops_attested": 2.9e18,
            "org.diag.flops_analytic_ratio": 0.98,
        }
    }
    out = merge(battery, tpayload)
    assert out["metrics"]["org.diag.mfu_achieved"] == 0.23
    assert out["metrics"]["org.g1.bits_per_byte_prose"] == 1.10, "must not clobber"
    assert out["tier"] == "battery-v1", "unrelated fields preserved"
    # Input is not mutated (the caller still emits the eval payload's copy).
    assert "org.diag.mfu_achieved" not in battery["metrics"]


def test_non_numeric_diag_never_enters_battery_metrics():
    """The Rust reader treats a non-numeric `org.*` entry as unparseable.
    `org.diag.binding_cap` is a STRING, so it must travel in the top-level
    `budget` block instead."""
    merge = _with_diag()
    battery = {"metrics": {"org.g1.bits_per_byte_prose": 1.10}}
    out = merge(
        battery,
        {
            "diag_metrics": {
                "org.diag.binding_cap": "flops",
                "org.diag.flops_probe_unstable": True,
                "org.diag.nan": float("nan"),
                "org.diag.inf": float("inf"),
                "not_org_namespaced": 1.0,
                "org.diag.ok": 0.5,
            }
        },
    )
    m = out["metrics"]
    assert m["org.diag.ok"] == 0.5
    for rejected in (
        "org.diag.binding_cap",
        "org.diag.flops_probe_unstable",
        "org.diag.nan",
        "org.diag.inf",
        "not_org_namespaced",
    ):
        assert rejected not in m, f"{rejected} must not reach battery.metrics"


def test_diag_never_creates_a_battery_out_of_nothing():
    """A blob with NO `org.*` metrics makes the scorer skip the composite; a
    blob with ONLY `org.diag.*` makes it run the composite and fail every
    declared group as missing — Ineligible, lattice 0. So a training-only run
    must stay batteryless."""
    merge = _with_diag()
    diag = {"diag_metrics": {"org.diag.mfu_achieved": 0.23}}
    assert merge({}, diag) == {}, "empty battery must stay empty"
    assert merge(None, diag) == {}
    only_non_org = {"metrics": {"tier_note": 1.0}}
    assert merge(only_non_org, diag) == only_non_org, "no org.* ⇒ no merge"
    # And a missing/garbage diag block is a no-op rather than an error.
    battery = {"metrics": {"org.g1.bits_per_byte_prose": 1.1}}
    assert merge(battery, {}) == battery
    assert merge(battery, {"diag_metrics": None}) == battery
    assert merge(battery, {"diag_metrics": "nonsense"}) == battery


def test_coefficient_of_variation_edges():
    assert F.coefficient_of_variation([]) == 0.0
    assert F.coefficient_of_variation([5.0]) == 0.0
    assert F.coefficient_of_variation([2.0, 2.0, 2.0]) == 0.0
    assert F.coefficient_of_variation([1.0, 3.0]) > 0.5
    assert F.coefficient_of_variation([0.0, 0.0]) == 0.0


def main():
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"ok   {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback

            traceback.print_exc()
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)
    print("FLOPS ATTESTATION OK")


if __name__ == "__main__":
    main()
