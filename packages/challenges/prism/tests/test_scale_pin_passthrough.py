"""P1 ProtocolPin seq/token_budget pass-through (VAL-SCALE-006).

Product/config/eval path must honor seq_len ≥256 (512 target) and
token_budget ≥1_000_000 under a matched pin without hardcoding seq=128-only.
No emission change. Fixture/unit only — no Lium spend.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.multi_family_compare import explore_protocol_pin
from prism_challenge.evaluator.official_compare_harness import (
    default_protocol_pin,
    protocol_pin_hash,
)
from prism_challenge.evaluator.official_comparison import ProtocolPin
from prism_challenge.evaluator.scale_eval import (
    SCALE_P0_SEQ_LEN,
    SCALE_P0_TOKEN_BUDGET,
    SCALE_P1_SEQ_LEN,
    SCALE_P1_SEQ_LEN_TARGET,
    SCALE_P1_TOKEN_BUDGET,
    SCALE_P1_TOKEN_BUDGET_HIGH,
    SCALE_P2_CROWN_FAMILY_IDS,
    SCALE_P2_PARAM_CAP,
    SCALE_P2_PARAM_STAGE,
    assert_public_multi_seed_pin,
    assert_scale_p1_pin_floor,
    assert_scale_p2_pin_floor,
    densify_entrypoints,
    prism_context_from_protocol_pin,
    promote_protocol_pin,
    protocol_pin_context_fields,
    scale_p0_protocol_pin,
    scale_p1_protocol_pin,
    scale_p2_protocol_pin,
    scale_pin_fields,
    scale_pin_public_ok,
    scale_product_snapshot,
    tee_package_absent,
)


def test_scale_p1_constants_meet_val_scale_006_floors() -> None:
    """P1 defaults: seq ≥256 (target 512) and token_budget ≥1_000_000."""
    assert SCALE_P1_SEQ_LEN >= 256
    assert SCALE_P1_SEQ_LEN_TARGET >= 512
    assert SCALE_P1_SEQ_LEN_TARGET >= SCALE_P1_SEQ_LEN
    assert SCALE_P1_TOKEN_BUDGET >= 1_000_000
    assert SCALE_P1_TOKEN_BUDGET_HIGH >= SCALE_P1_TOKEN_BUDGET
    # P0 remains the short-ctx baseline (not silently rewritten).
    assert SCALE_P0_SEQ_LEN == 128
    assert SCALE_P0_TOKEN_BUDGET == 500_000


def test_scale_p1_protocol_pin_defaults_and_public_k() -> None:
    pin = scale_p1_protocol_pin()
    assert pin.seq_len == SCALE_P1_SEQ_LEN
    assert pin.token_budget == SCALE_P1_TOKEN_BUDGET
    assert pin.param_ladder_stage == "explore"
    assert pin.primary_form == "heldout_delta"
    assert pin.tokenizer == "gpt2"
    assert len(pin.seeds) >= 3
    assert pin.force_iter_train_batches is True
    assert_public_multi_seed_pin(pin)
    assert_scale_p1_pin_floor(pin)
    fields = scale_pin_fields(pin)
    assert fields["seq_len"] == SCALE_P1_SEQ_LEN
    assert fields["token_budget"] == SCALE_P1_TOKEN_BUDGET
    assert fields["seed_count"] >= 3


def test_scale_p1_protocol_pin_target_512_and_2m_budget() -> None:
    """Operators can raise to target 512 / 2M without traps."""
    pin = scale_p1_protocol_pin(
        seq_len=SCALE_P1_SEQ_LEN_TARGET,
        token_budget=SCALE_P1_TOKEN_BUDGET_HIGH,
    )
    assert pin.seq_len == 512
    assert pin.token_budget == 2_000_000
    assert_scale_p1_pin_floor(pin)
    guard = scale_pin_public_ok(pin)
    assert guard.ok is True


def test_scale_p1_pin_rejects_sub_floor_when_required() -> None:
    with pytest.raises(ValueError, match="seq_len"):
        assert_scale_p1_pin_floor(scale_p1_protocol_pin(seq_len=128, require_p1_floor=False))
    with pytest.raises(ValueError, match="token_budget"):
        assert_scale_p1_pin_floor(
            scale_p1_protocol_pin(token_budget=500_000, require_p1_floor=False)
        )
    # Constructor default path enforces floor.
    with pytest.raises(ValueError, match="P1 scale pin"):
        scale_p1_protocol_pin(seq_len=64, require_p1_floor=True)


def test_explore_protocol_pin_passes_seq_and_budget() -> None:
    """explore_protocol_pin must not trap at seq=128 / 500k when kwargs set."""
    pin = explore_protocol_pin(seq_len=256, token_budget=1_000_000, seeds=(1337, 2027, 4242))
    assert pin.seq_len == 256
    assert pin.token_budget == 1_000_000
    assert pin.seeds == (1337, 2027, 4242)
    # default still P0-style for arXiv residual continuity
    base = explore_protocol_pin()
    assert base.seq_len == 128
    assert base.token_budget == 500_000


def test_protocol_pin_replace_seq_budget_survives_hash_and_as_dict() -> None:
    pin = ProtocolPin(seq_len=512, token_budget=1_500_000, seeds=(1337, 2027, 4242))
    d = pin.as_dict()
    assert d["seq_len"] == 512
    assert d["token_budget"] == 1_500_000
    h1 = protocol_pin_hash(pin)
    h2 = protocol_pin_hash(replace(pin, seq_len=256))
    assert h1 != h2
    # default pin is still short-ctx Official default (not broken by P1 helpers)
    default = default_protocol_pin()
    assert default.seq_len == 128
    assert default.token_budget == 500_000


def test_prism_context_from_protocol_pin_passes_seq_and_budget() -> None:
    pin = scale_p1_protocol_pin(seq_len=512, token_budget=1_000_000)
    ctx = prism_context_from_protocol_pin(pin)
    assert isinstance(ctx, PrismContext)
    assert ctx.sequence_length == 512
    assert ctx.max_seq_len == 512
    assert ctx.token_budget == 1_000_000
    fields = protocol_pin_context_fields(pin)
    assert fields["sequence_length"] == 512
    assert fields["token_budget"] == 1_000_000


def test_settings_allow_seq_ge_256_and_token_budget() -> None:
    """Worker-plane config knobs accept P1 values (no 128-only hard trap)."""
    settings = PrismSettings(
        shared_token="test-token-not-secret",
        docker_backend="cli",
        sequence_length=256,
        max_sequence_length=512,
        token_budget=1_000_000,
    )
    assert settings.sequence_length == 256
    assert settings.max_sequence_length == 512
    assert settings.token_budget == 1_000_000
    assert settings.resolved_eval_sequence_length() == 256
    assert settings.resolved_eval_token_budget() == 1_000_000

    settings_512 = PrismSettings(
        shared_token="test-token-not-secret",
        docker_backend="cli",
        sequence_length=512,
        max_sequence_length=512,
        token_budget=2_000_000,
    )
    assert settings_512.resolved_eval_sequence_length() == 512
    assert settings_512.resolved_eval_token_budget() == 2_000_000


def test_settings_reject_sequence_above_max() -> None:
    with pytest.raises(ValueError, match="sequence_length"):
        PrismSettings(
            shared_token="test-token-not-secret",
            docker_backend="cli",
            sequence_length=1024,
            max_sequence_length=512,
        )


def test_settings_to_prism_context_kwargs_pass_through() -> None:
    settings = PrismSettings(
        shared_token="test-token-not-secret",
        docker_backend="cli",
        sequence_length=256,
        max_sequence_length=512,
        token_budget=1_000_000,
        max_layers=48,
        max_parameters=124_000_000,
        param_ladder_stage="explore",
    )
    kwargs = settings.prism_context_kwargs()
    ctx = PrismContext(**kwargs)
    assert ctx.sequence_length == 256
    assert ctx.token_budget == 1_000_000
    assert ctx.max_layers == 48
    assert ctx.param_ladder_stage == "explore"


def test_p0_pin_unchanged_when_p1_helpers_present() -> None:
    p0 = scale_p0_protocol_pin()
    assert p0.seq_len == 128
    assert p0.token_budget == 500_000
    p1 = scale_p1_protocol_pin()
    assert p1.seq_len >= 256
    assert p1.token_budget >= 1_000_000
    assert protocol_pin_hash(p0) != protocol_pin_hash(p1)


def test_densify_entrypoints_document_p1_pin_knobs() -> None:
    ep = densify_entrypoints()
    helpers = ep["scale_helpers"]
    assert helpers["p1_pin"] == "scale_p1_protocol_pin"
    assert "pin_to_context" in helpers or helpers.get("context_from_pin")
    snap = scale_product_snapshot()
    assert "p1_pin" in snap or snap["pin"]["seq_len"] in (128, SCALE_P1_SEQ_LEN)
    assert tee_package_absent() is True
    # Snapshot includes documented P1 ladder knobs for operators.
    assert snap.get("p1_ladder") is not None
    assert snap["p1_ladder"]["seq_len_min"] >= 256
    assert snap["p1_ladder"]["token_budget_min"] >= 1_000_000


def test_scale_p2_protocol_pin_is_promote_350m() -> None:
    """P2 cup pin: stage=promote, param_cap=350M, K≥3, P1 seq/budget floors."""
    pin = scale_p2_protocol_pin()
    assert pin.param_ladder_stage == "promote"
    assert pin.param_ladder_stage == SCALE_P2_PARAM_STAGE
    assert pin.param_cap == 350_000_000
    assert pin.param_cap == SCALE_P2_PARAM_CAP
    assert pin.seq_len >= 256
    assert pin.token_budget >= 1_000_000
    assert pin.primary_form == "heldout_delta"
    assert len(pin.seeds) >= 3
    assert_public_multi_seed_pin(pin)
    assert_scale_p2_pin_floor(pin)
    fields = scale_pin_fields(pin)
    assert fields["param_ladder_stage"] == "promote"
    assert fields["param_cap"] == 350_000_000


def test_promote_protocol_pin_does_not_silent_explore() -> None:
    pin = promote_protocol_pin(seq_len=256, token_budget=1_000_000, seeds=(1337, 2027, 4242))
    assert pin.param_ladder_stage == "promote"
    assert pin.param_cap == 350_000_000
    explore = explore_protocol_pin(seq_len=256, token_budget=1_000_000, seeds=(1337, 2027, 4242))
    assert explore.param_ladder_stage == "explore"
    assert explore.param_cap == 124_000_000
    assert protocol_pin_hash(pin) != protocol_pin_hash(explore)


def test_scale_p2_pin_rejects_explore_stage() -> None:
    explore = scale_p1_protocol_pin()
    with pytest.raises(ValueError, match="param_ladder_stage"):
        assert_scale_p2_pin_floor(explore)


def test_scale_p2_crown_families_include_deeploop_runner_transformer() -> None:
    fams = set(SCALE_P2_CROWN_FAMILY_IDS)
    assert "deeploop-tiny-1m" in fams
    assert "transformer-tiny-1m" in fams
    # P1 explore crown was mamba; crown cup keeps runner-up / crown lineage.
    assert "mamba-tiny-1m" in fams
    snap = scale_product_snapshot()
    assert snap["p2_pin"]["param_ladder_stage"] == "promote"
    assert snap["p2_ladder"]["param_cap"] == 350_000_000
    assert "deeploop-tiny-1m" in snap["crown_families_p2"]
    ep = densify_entrypoints()
    assert ep["scale_helpers"]["p2_pin"] == "scale_p2_protocol_pin"
    assert ep["p2_ladder"]["param_ladder_stage"] == "promote"
