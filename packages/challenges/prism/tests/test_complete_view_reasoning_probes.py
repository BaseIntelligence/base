"""P10 logic probe suite unit fixtures (VAL-REASON-002..008).

CPU-only / pure-torch paths. No lm-eval, no Lium, no REAL TEE claim.
Proves all MUST probe metrics are computable for dual-family fixtures and
that multihop_binding is role/composition (distinct from MQAR).
"""

from __future__ import annotations

import math

import pytest

from prism_challenge.evaluator.benchmarks.logic_suite import (
    DEFAULT_TRIALS_PER_PROBE,
    GENERATOR_BY_PROBE,
    LOGIC_PROBE_KEYS,
    closed_choice_rank_preds,
    documented_logic_suite,
    encode_answer_ids,
    fixture_forced_ce_from_accuracy,
    gen_boolean_parity_xor,
    gen_count_stream,
    gen_multihop_binding,
    gen_reverse_edit,
    gen_sort_order,
    generate_logic_suite,
    generate_probe_trials,
    logic_trial_seed,
    oracle_predictions,
    probe_forced_answer_ce,
    pure_torch_fixture_model,
    score_logic_from_predictions,
    score_probe_fixture,
    score_probe_with_logits,
    score_suite_fixture,
    tokenize_simple,
)
from prism_challenge.evaluator.complete_view import (
    COMPLETE_VIEW_REASONING_CHANCE_TABLE,
    COMPLETE_VIEW_SCHEMA,
    COMPLETE_VIEW_SCORECARD_ID,
    REASONING_SUITE_ID,
    assert_complete_view_document,
    validate_complete_view_document,
)
from prism_challenge.evaluator.complete_view_reasoning import (
    assert_probe_dual_channel,
    build_complete_view_with_reasoning,
    dual_family_reasoning_fixture,
    family_reasoning_fixture,
    fill_reasoning_panel,
    probe_metric_bundle,
)
from prism_challenge.evaluator.official_comparison import OfficialScoreRecord
from prism_challenge.evaluator.scorecard_suite import relative_to_chance


def _rec(*, label: str, heldout_delta: float = 0.5) -> OfficialScoreRecord:
    return OfficialScoreRecord(
        label=label,
        bpb=1.2,
        primary_form="heldout_delta",
        heldout_delta=heldout_delta,
        valid=True,
        seed_count=3,
        bpb_std=0.01,
        heldout_std=0.02,
        stop_token_budget=True,
        finite_bpb=True,
        param_cap_ok=True,
        matched_pin=True,
        force_instrument=True,
    )


def _acc_profile(base: float, *, bump: dict[str, float] | None = None) -> dict[str, float]:
    """Per-probe accuracy profile clamped to [0,1]."""
    out = {k: float(base) for k in LOGIC_PROBE_KEYS}
    if bump:
        for k, v in bump.items():
            out[k] = max(0.0, min(1.0, float(v)))
    return out


# --- Suite identity / authorship -------------------------------------------------


def test_logic_suite_identity_no_lm_eval() -> None:
    doc = documented_logic_suite()
    assert doc["suite_id"] == REASONING_SUITE_ID == "logic_synthetic.v1"
    assert set(doc["probes"]) == set(LOGIC_PROBE_KEYS)
    assert doc["chance_table"] == dict(COMPLETE_VIEW_REASONING_CHANCE_TABLE)
    assert doc["no_lm_eval_dependency"] is True
    assert "mqar" in doc["distinct_from"]
    assert "gsm8k" in doc["distinct_from"]["gsm8k_mmlu_lm_eval"].lower()
    # Generators are challenge-owned callables, not external kits.
    assert set(GENERATOR_BY_PROBE) == set(LOGIC_PROBE_KEYS)


def test_logic_trial_seed_deterministic() -> None:
    a = logic_trial_seed(probe="boolean_parity_xor", trial_i=3)
    b = logic_trial_seed(probe="boolean_parity_xor", trial_i=3)
    c = logic_trial_seed(probe="boolean_parity_xor", trial_i=4)
    assert a == b
    assert a != c


# --- VAL-REASON-002: boolean / parity / XOR --------------------------------------


def test_val_reason_002_boolean_parity_xor_scoreable_both_sides() -> None:
    trials = generate_probe_trials("boolean_parity_xor", n_trials=24)
    assert len(trials) == 24
    for t in trials:
        assert t.probe == "boolean_parity_xor"
        assert t.gold in t.candidates
        assert set(t.candidates) == {"0", "1"}
        assert "XOR" in t.prompt or "parity" in t.prompt

    dual = dual_family_reasoning_fixture(
        a_acc=_acc_profile(0.70),
        b_acc=_acc_profile(0.55),
        n_trials=24,
    )
    sa = dual.a.probes["boolean_parity_xor"]
    sb = dual.b.probes["boolean_parity_xor"]
    assert_probe_dual_channel(sa)
    assert_probe_dual_channel(sb)
    assert sa.chance == pytest.approx(0.5)
    assert sb.chance == pytest.approx(0.5)
    assert sa.accuracy == pytest.approx(0.70, abs=1 / 24 + 1e-9)
    assert sb.accuracy == pytest.approx(0.55, abs=1 / 24 + 1e-9)
    assert sa.forced_ce is not None and sb.forced_ce is not None
    # Higher accuracy → lower forced CE (fixture mapping)
    assert sa.forced_ce < sb.forced_ce

    panel = dual.panel()
    entry = panel["probes"]["boolean_parity_xor"]
    assert entry["status"] == "filled"
    assert entry["acc"]["a"] == pytest.approx(sa.accuracy)
    assert entry["acc"]["b"] == pytest.approx(sb.accuracy)
    assert entry["forced_ce"]["a"] is not None
    assert entry["forced_ce"]["b"] is not None
    assert entry["rel_to_chance"]["a"] == pytest.approx(relative_to_chance(sa.accuracy, 0.5))


# --- VAL-REASON-003: arithmetic digit/mod ----------------------------------------


def test_val_reason_003_arith_digit_mod_scoreable_both_sides() -> None:
    trials = generate_probe_trials("arith_digit_mod", n_trials=16)
    for t in trials:
        assert t.gold in DIGIT_RANGE(t)
        assert t.gold in t.candidates
        assert t.prompt.startswith("arith:")
    dual = dual_family_reasoning_fixture(
        a_acc=_acc_profile(0.20, bump={"arith_digit_mod": 0.40}),
        b_acc=_acc_profile(0.20, bump={"arith_digit_mod": 0.25}),
        n_trials=20,
    )
    sa = dual.a.probes["arith_digit_mod"]
    sb = dual.b.probes["arith_digit_mod"]
    assert_probe_dual_channel(sa)
    assert_probe_dual_channel(sb)
    assert sa.chance == pytest.approx(0.1)
    bundle = probe_metric_bundle(dual.a, dual.b)
    assert "arith_digit_mod" in bundle
    assert bundle["arith_digit_mod"]["acc"]["a"] == pytest.approx(sa.accuracy)


def DIGIT_RANGE(t) -> set[str]:  # noqa: N802 - helper name kept close to test use
    return set(str(i) for i in range(10))


# --- VAL-REASON-004: transitive comparison ---------------------------------------


def test_val_reason_004_transitive_compare_scoreable_both_sides() -> None:
    trials = generate_probe_trials("transitive_compare", n_trials=12)
    for t in trials:
        assert t.gold in {">", "<", "=", "?"}
        assert "rel:" in t.prompt
        assert t.meta.get("hops") in (1, 2, 3, 4)
    dual = dual_family_reasoning_fixture(
        a_acc=_acc_profile(0.30, bump={"transitive_compare": 0.50}),
        b_acc=_acc_profile(0.30, bump={"transitive_compare": 0.40}),
    )
    sa = dual.a.probes["transitive_compare"]
    sb = dual.b.probes["transitive_compare"]
    assert_probe_dual_channel(sa)
    assert_probe_dual_channel(sb)
    assert sa.chance == pytest.approx(0.25)
    panel = dual.panel()
    assert panel["probes"]["transitive_compare"]["acc"]["a"] is not None
    assert panel["probes"]["transitive_compare"]["acc"]["b"] is not None


# --- VAL-REASON-005: multihop binding ≠ MQAR -------------------------------------


def test_val_reason_005_multihop_binding_distinct_from_mqar() -> None:
    trials = generate_probe_trials("multihop_binding", n_trials=16)
    assert trials
    for t in trials:
        assert t.meta.get("distinct_from_mqar") is True
        assert t.meta.get("binding_style") == "function_composition"
        assert "P(" in t.prompt  # composition formula
        # Not MQAR-style key=value associative grid text
        assert "mqar" not in t.prompt.lower()
        assert "needle_key" not in t.prompt
        assert t.gold in t.candidates

    # Explicit structural contrast to retrieval templates
    t0 = gen_multihop_binding(0)
    assert "bind:" in t0.prompt
    assert "start=a=" in t0.prompt

    dual = dual_family_reasoning_fixture(
        a_acc=_acc_profile(0.20, bump={"multihop_binding": 0.35}),
        b_acc=_acc_profile(0.20, bump={"multihop_binding": 0.30}),
    )
    sa = dual.a.probes["multihop_binding"]
    sb = dual.b.probes["multihop_binding"]
    assert_probe_dual_channel(sa)
    assert_probe_dual_channel(sb)
    assert sa.chance == pytest.approx(0.125)
    panel = dual.panel()
    mh = panel["probes"]["multihop_binding"]
    assert mh["status"] == "filled"
    assert mh["acc"]["a"] is not None and mh["acc"]["b"] is not None
    # Panel still documents MQAR separation
    assert "mqar" in panel["distinct_from"]["P3_long_ctx_mqar"].lower()


# --- VAL-REASON-006: sort/order + reverse/edit -----------------------------------


def test_val_reason_006_sort_and_reverse_edit_reported_separately() -> None:
    dual = dual_family_reasoning_fixture(
        a_acc=_acc_profile(
            0.40,
            bump={"sort_order": 0.55, "reverse_edit": 0.45},
        ),
        b_acc=_acc_profile(
            0.40,
            bump={"sort_order": 0.50, "reverse_edit": 0.35},
        ),
    )
    sort_a = dual.a.probes["sort_order"]
    rev_a = dual.a.probes["reverse_edit"]
    sort_b = dual.b.probes["sort_order"]
    rev_b = dual.b.probes["reverse_edit"]
    for s in (sort_a, rev_a, sort_b, rev_b):
        assert_probe_dual_channel(s)
    assert sort_a.chance == pytest.approx(0.25)
    assert rev_a.chance == pytest.approx(0.25)

    # reverse/edit must be transform, not identity copy
    t_rev = gen_reverse_edit(0)
    assert t_rev.meta.get("distinct_from_copy") is True
    assert t_rev.gold != t_rev.meta.get("src")  # transformation applied

    panel = dual.panel()
    assert panel["probes"]["sort_order"]["acc"]["a"] is not None
    assert panel["probes"]["sort_order"]["acc"]["b"] is not None
    assert panel["probes"]["reverse_edit"]["acc"]["a"] is not None
    assert panel["probes"]["reverse_edit"]["acc"]["b"] is not None
    # Separate fields (not fused)
    assert "sort_order" in panel["probes"]
    assert "reverse_edit" in panel["probes"]
    assert panel["probes"]["sort_order"] is not panel["probes"]["reverse_edit"]


# --- VAL-REASON-007: count stream + dyck nesting ---------------------------------


def test_val_reason_007_count_stream_and_dyck_reported() -> None:
    dual = dual_family_reasoning_fixture(
        a_acc=_acc_profile(0.25, bump={"count_stream": 0.35, "dyck_nesting": 0.60}),
        b_acc=_acc_profile(0.25, bump={"count_stream": 0.30, "dyck_nesting": 0.55}),
    )
    ca = dual.a.probes["count_stream"]
    da = dual.a.probes["dyck_nesting"]
    cb = dual.b.probes["count_stream"]
    db = dual.b.probes["dyck_nesting"]
    for s in (ca, da, cb, db):
        assert_probe_dual_channel(s)
    assert ca.chance == pytest.approx(0.1)
    assert da.chance == pytest.approx(0.5)

    trials_c = generate_probe_trials("count_stream", n_trials=8)
    assert all("count:" in t.prompt for t in trials_c)
    trials_d = generate_probe_trials("dyck_nesting", n_trials=9)
    assert all("dyck:" in t.prompt for t in trials_d)

    panel = dual.panel()
    assert panel["probes"]["count_stream"]["forced_ce"]["a"] is not None
    assert panel["probes"]["dyck_nesting"]["acc"]["b"] is not None


# --- VAL-REASON-008: instruction-toy + contradiction -----------------------------


def test_val_reason_008_instruction_and_contradiction_scored() -> None:
    dual = dual_family_reasoning_fixture(
        a_acc=_acc_profile(
            0.30,
            bump={"instruction_toy": 0.40, "contradiction_detect": 0.65},
        ),
        b_acc=_acc_profile(
            0.30,
            bump={"instruction_toy": 0.35, "contradiction_detect": 0.55},
        ),
    )
    ia = dual.a.probes["instruction_toy"]
    ca = dual.a.probes["contradiction_detect"]
    ib = dual.b.probes["instruction_toy"]
    cb = dual.b.probes["contradiction_detect"]
    for s in (ia, ca, ib, cb):
        assert_probe_dual_channel(s)
    assert ia.chance == pytest.approx(0.2)
    assert ca.chance == pytest.approx(0.5)

    trials_i = generate_probe_trials("instruction_toy", n_trials=8)
    assert any(t.prompt.startswith("FMTA|") or t.prompt.startswith("instr:") for t in trials_i)
    trials_c = generate_probe_trials("contradiction_detect", n_trials=8)
    assert all(t.gold in ("consistent", "inconsistent") for t in trials_c)

    panel = dual.panel()
    assert panel["probes"]["instruction_toy"]["status"] == "filled"
    assert panel["probes"]["contradiction_detect"]["status"] == "filled"
    assert panel["probes"]["instruction_toy"]["acc"]["a"] is not None
    assert panel["probes"]["contradiction_detect"]["acc"]["b"] is not None


# --- Dual channel + panel assembly + pure-torch hooks ----------------------------


def test_all_must_probes_fixture_suite_and_panel_fill() -> None:
    scores = score_suite_fixture(n_trials=12, accuracy_by_probe=_acc_profile(0.5))
    assert set(scores) == set(LOGIC_PROBE_KEYS)
    for key, sc in scores.items():
        assert_probe_dual_channel(sc)
        assert sc.probe == key

    fam_a = family_reasoning_fixture(
        accuracy_by_probe=_acc_profile(0.6),
        n_trials=12,
        seeds=(1337, 2027, 4242),
    )
    fam_b = family_reasoning_fixture(
        accuracy_by_probe=_acc_profile(0.4),
        n_trials=12,
        suite_seed=99,
        seeds=(1337, 2027, 4242),
    )
    panel = fill_reasoning_panel(a=fam_a, b=fam_b)
    assert panel["status"] == "filled"
    assert panel["suite_id"] == REASONING_SUITE_ID
    assert panel["scoring"]["closed_choice_accuracy"] is True
    assert panel["scoring"]["forced_ce"] is True
    assert panel["scoring"]["chance_baselines"] is True
    for key in LOGIC_PROBE_KEYS:
        p = panel["probes"][key]
        assert p["status"] == "filled"
        assert p["acc"]["a"] is not None and p["acc"]["b"] is not None
        assert p["forced_ce"]["a"] is not None and p["forced_ce"]["b"] is not None
        assert p["rel_to_chance"]["a"] is not None

    # Full complete_view document with P10 probe metrics
    doc = build_complete_view_with_reasoning(
        _rec(label="transformer-tiny-1m", heldout_delta=0.4),
        _rec(label="mamba-tiny-1m", heldout_delta=0.55),
        family_a=fam_a,
        family_b=fam_b,
        score_class="fixture",
    )
    assert doc["schema"] == COMPLETE_VIEW_SCHEMA == "complete_view.v1.3"
    assert doc["scorecard_id"] == COMPLETE_VIEW_SCORECARD_ID
    assert "real_provider_tee" not in doc
    assert doc["labels"]["provider_trust"] == "PROVIDER_TRUST"
    assert doc["non_claims"]["prism_tee_product"] is False
    p10 = doc["panels"]["P10_reasoning_logic"]
    assert p10["status"] == "filled"
    for key in LOGIC_PROBE_KEYS:
        assert p10["probes"][key]["acc"]["a"] is not None
        assert p10["probes"][key]["acc"]["b"] is not None
    problems = validate_complete_view_document(doc)
    assert problems == []
    assert_complete_view_document(doc)


def test_oracle_vs_chance_relative_to_chance() -> None:
    trials = generate_probe_trials("boolean_parity_xor", n_trials=40)
    oracle = score_logic_from_predictions(trials, oracle_predictions(trials))
    assert oracle.accuracy == pytest.approx(1.0)
    assert oracle.relative == pytest.approx(1.0)
    # Chance ~0.5 binary: fixture forced ce mapping uses relative
    ce_hi = fixture_forced_ce_from_accuracy(1.0, chance=0.5)
    ce_lo = fixture_forced_ce_from_accuracy(0.5, chance=0.5)
    assert ce_hi < ce_lo


def test_pure_torch_cpu_scoring_path() -> None:
    """Prefer pure-torch CPU path for model-facing hooks (no trained_state load)."""
    bundle = pure_torch_fixture_model(vocab_size=256, seed=7)
    trials = generate_probe_trials("boolean_parity_xor", n_trials=8)
    # Forced CE path alone
    ce = probe_forced_answer_ce(bundle.nll_fn, trials[0].prompt, trials[0].gold)
    assert math.isfinite(ce) and ce > 0
    # Closed-choice logits path (may be near chance; must still produce dual metrics)
    score = score_probe_with_logits(
        trials,
        bundle.logits_fn,
        probe="boolean_parity_xor",
        nll_fn=bundle.nll_fn,
        device="cpu",
    )
    assert score.probe == "boolean_parity_xor"
    assert score.trials == 8
    assert 0.0 <= score.accuracy <= 1.0
    assert score.chance == pytest.approx(0.5)
    assert score.forced_ce is not None and math.isfinite(score.forced_ce)
    assert score.device == "cpu"
    # Tokenizer is deterministic single-byte map
    assert tokenize_simple("AB") == [ord("A"), ord("B")]


def test_suite_generate_covers_all_probes() -> None:
    suite = generate_logic_suite(n_trials=4)
    assert set(suite) == set(LOGIC_PROBE_KEYS)
    for _key, trials in suite.items():
        assert len(trials) == 4
        assert all(t.gold in t.candidates for t in trials)


def test_score_probe_fixture_default_trials() -> None:
    sc = score_probe_fixture("contradiction_detect")
    assert sc.trials == DEFAULT_TRIALS_PER_PROBE
    assert_probe_dual_channel(sc)


def test_boolean_generator_modes_include_xor_and_parity() -> None:
    modes = {gen_boolean_parity_xor(i).meta["mode"] for i in range(8)}
    assert "xor" in modes and "parity" in modes


# --- Scoring hygiene (first-byte collapse, count domain, sort chance) -------------


def test_encode_answer_ids_keeps_multihop_entities_distinct() -> None:
    """e0..e7 must not collapse under default encode (hygiene: no first-byte only)."""
    cands = ("e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7")
    seqs = [tuple(encode_answer_ids(c)) for c in cands]
    assert len(set(seqs)) == len(cands)
    # First-byte alone would all be ord('e') — ensure multi-id sequences.
    assert all(len(s) >= 2 for s in seqs)


def test_score_probe_with_logits_multihop_not_first_byte_collapsed() -> None:
    """With nll_fn, multihop closed rank must distinguish e0..e7, not shared first-byte."""
    trials = generate_probe_trials("multihop_binding", n_trials=6)
    # Prove candidate first-bytes collide under naive collapse.
    for t in trials:
        firsts = [tokenize_simple(c)[0] for c in t.candidates]
        assert len(set(firsts)) < len(firsts)

    # Score each trial with a gold-local NLL teacher so entities stay distinguished.
    outcomes: list[bool] = []
    methods: list[str] = []
    for t in trials:
        gold = t.gold

        def nll_prefer_this_gold(
            full: list[int] | tuple[int, ...],
            *,
            g: str = gold,
        ) -> float:
            chars = "".join(chr(i) for i in full if 1 <= i < 256)
            return 0.1 if chars.endswith(g) else 3.0

        sc = score_probe_with_logits(
            [t],
            logits_fn=lambda _ctx: [0.0] * 256,  # unused when multi-token NLL path
            probe="multihop_binding",
            nll_fn=nll_prefer_this_gold,
            device="cpu",
        )
        assert sc.detail is not None
        methods.append(str(sc.detail.get("closed_rank_method")))
        outcomes.append(sc.accuracy == 1.0)

    assert all(outcomes)
    assert all(m in {"forced_nll_rank", "forced_nll_rank_collapsed_fallback"} for m in methods)
    # Suite-level path also reports multi-token method and no first-byte collapse.
    suite = score_probe_with_logits(
        trials,
        logits_fn=lambda _ctx: [0.0] * 256,
        probe="multihop_binding",
        nll_fn=lambda full: float(len(full)),  # uniform — only proves method selection
        device="cpu",
    )
    assert suite.detail is not None
    assert suite.detail.get("closed_rank_method") in {
        "forced_nll_rank",
        "forced_nll_rank_collapsed_fallback",
    }
    assert suite.chance == pytest.approx(0.125)


def test_score_probe_with_logits_reverse_edit_multi_char() -> None:
    """reverse_edit multi-char golds must score via multi-token rank (not first-byte)."""
    trials = generate_probe_trials("reverse_edit", n_trials=4)
    assert all(len(t.gold) > 1 for t in trials)

    for t in trials:
        gold = t.gold

        def nll_prefer_this_gold(
            full: list[int] | tuple[int, ...],
            *,
            g: str = gold,
        ) -> float:
            chars = "".join(chr(i) for i in full if 1 <= i < 256)
            return 0.05 if chars.endswith(g) else 4.0

        score = score_probe_with_logits(
            [t],
            logits_fn=lambda _ctx: [0.0] * 256,
            probe="reverse_edit",
            nll_fn=nll_prefer_this_gold,
            device="cpu",
        )
        assert score.detail is not None
        assert "forced_nll" in str(score.detail.get("closed_rank_method", ""))
        assert score.accuracy == pytest.approx(1.0)


def test_closed_choice_rank_preds_selects_lowest_nll() -> None:
    trials = [gen_multihop_binding(0)]
    gold = trials[0].gold

    def nll_fn(full: list[int] | tuple[int, ...]) -> float:
        chars = "".join(chr(i) for i in full if 1 <= i < 256)
        return 0.0 if chars.endswith(gold) else 5.0

    preds = closed_choice_rank_preds(trials, nll_fn=nll_fn)
    assert preds == [gold]


def test_gen_count_stream_gold_matches_true_count_domain() -> None:
    """Gold domain always equals true stream count in 0..9 (no clamp mismatch)."""
    for i in range(200):
        t = gen_count_stream(i)
        assert t.gold in t.candidates
        assert t.gold == str(t.meta["count"])
        assert 0 <= int(t.meta["count"]) <= 9
        assert t.meta.get("clamped") is False
        assert t.meta.get("gold_domain") == "0..9"
        # Reconstruct count from prompt stream body.
        # prompt form: count: stream=[...] ; target=X ; q=count → ?
        body = t.prompt.split("stream=[", 1)[1].split("]", 1)[0]
        tokens = body.split()
        target = t.meta["target"]
        observed = sum(1 for x in tokens if x == target)
        assert observed == int(t.meta["count"])
        assert observed == int(t.gold)
        assert len(tokens) == int(t.meta["stream_len"])


def test_sort_order_mode_chance_labels_consistent() -> None:
    """is_sorted uses 0.5 in meta; table chance stays 0.25 on scores."""
    trials = [gen_sort_order(i) for i in range(8)]
    modes = {t.meta["mode"] for t in trials}
    assert "is_sorted" in modes
    for t in trials:
        assert "chance" in t.meta
        assert "protocol_chance" in t.meta
        assert t.meta["protocol_chance"] == pytest.approx(0.25)
        if t.meta["mode"] == "is_sorted":
            assert t.meta["chance"] == pytest.approx(0.5)
            assert t.candidates == ("yes", "no")
        else:
            assert t.meta["chance"] == pytest.approx(1.0 / len(t.candidates))
    # Protocol score chance remains table 0.25; detail carries mode mean.
    preds = oracle_predictions(trials)
    sc = score_logic_from_predictions(trials, preds, probe="sort_order")
    assert sc.chance == pytest.approx(0.25)
    assert sc.detail is not None
    assert sc.detail.get("chance_source") == "protocol_table"
    assert "mode_chance_mean" in sc.detail
    assert sc.detail["mode_chance_mean"] > 0.25  # mix pulls mean above pure 0.25
