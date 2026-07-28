"""Fail-closed shape messaging, plan-derived instance defaults, loopback HTTP opt-in.

Covers three production eval-flow defects:

1. ``format_eval_shape_mismatch_error`` must exist and emit a single-line
   operator-actionable message (never full digests).
2. When ``--eval-instance-type`` is omitted, the CLI binds to the plan shape;
   an explicit mismatch still fails closed with that message.
3. ``SelfDeployRouteClient`` may use ``http://`` only for loopback hosts when
   ``SELFDEPLOY_ALLOW_INSECURE_LOOPBACK=1`` is set — nothing else.
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from agent_challenge.selfdeploy import cli
from agent_challenge.selfdeploy import measurements as measure
from agent_challenge.selfdeploy.client import RouteClientError, SelfDeployRouteClient

_RTMR0_FULL = "02" * 48  # 96 hex chars (sha384-width register)


# --------------------------------------------------------------------------- #
# DEFECT 1 — shape mismatch formatter
# --------------------------------------------------------------------------- #


def test_format_eval_shape_mismatch_error_names_types_and_flag_hint():
    msg = measure.format_eval_shape_mismatch_error(
        plan_instance_type="tdx.xlarge",
        requested_instance_type="tdx.small",
        plan_vm_shape="tdx.xlarge",
        plan_rtmr0=None,
    )
    assert isinstance(msg, str)
    assert "\n" not in msg
    assert "tdx.small" in msg
    assert "tdx.xlarge" in msg
    assert "--eval-instance-type" in msg
    assert "tdx.xlarge" in msg.split("--eval-instance-type", 1)[1]


def test_format_eval_shape_mismatch_error_truncates_rtmr0_never_full_digest():
    msg = measure.format_eval_shape_mismatch_error(
        plan_instance_type="tdx.medium",
        requested_instance_type="tdx.small",
        plan_vm_shape="tdx.medium",
        plan_rtmr0=_RTMR0_FULL,
    )
    assert _RTMR0_FULL not in msg
    prefix = _RTMR0_FULL[:16]
    assert prefix in msg
    # Convention used elsewhere (keyrelease client): 16 hex + ellipsis.
    assert f"{prefix}…" in msg or f"{prefix}..." in msg


# --------------------------------------------------------------------------- #
# DEFECT 2 — plan-derived default; explicit mismatch fail-closed
# --------------------------------------------------------------------------- #


def _eval_plan(
    *,
    instance_type: str = "tdx.xlarge",
    vm_shape: str | None = None,
) -> SimpleNamespace:
    shape = vm_shape if vm_shape is not None else instance_type.replace(".", "-")
    return SimpleNamespace(
        instance_type=instance_type,
        measurement={
            "vm_shape": shape,
            "rtmr0": _RTMR0_FULL,
            "mrtd": "01" * 48,
        },
    )


def test_assert_eval_shape_uses_plan_when_flag_omitted():
    """Operator did not pass --eval-instance-type → bind to plan (no raise)."""
    plan = _eval_plan(instance_type="tdx.xlarge")
    args = Namespace(eval_instance_type=None, expected_measurement=None)
    cli._assert_eval_deploy_shape_and_measurement_pin(plan, args)


def test_assert_eval_shape_uses_plan_when_flag_empty_string():
    plan = _eval_plan(instance_type="tdx.medium")
    args = Namespace(eval_instance_type="", expected_measurement=None)
    cli._assert_eval_deploy_shape_and_measurement_pin(plan, args)


def test_assert_eval_shape_explicit_mismatch_fails_closed_with_formatter_message():
    plan = _eval_plan(instance_type="tdx.xlarge")
    args = Namespace(eval_instance_type="tdx.small", expected_measurement=None)
    with pytest.raises(RouteClientError) as excinfo:
        cli._assert_eval_deploy_shape_and_measurement_pin(plan, args)
    msg = str(excinfo.value)
    assert "tdx.small" in msg
    assert "tdx.xlarge" in msg
    assert "--eval-instance-type" in msg
    assert _RTMR0_FULL not in msg


def test_eval_deploy_parser_default_eval_instance_type_is_unset():
    """Hardcoded tdx.small default is the root cause of plan/CLI shape fights."""
    parser = cli.build_parser()
    # Minimal argv that reaches eval deploy defaults without executing deploy.
    ns = parser.parse_args(
        [
            "eval",
            "deploy",
            "--base-url",
            "https://challenge.example",
            "--hotkey",
            "hk",
            "--submission-id",
            "1",
        ]
    )
    assert getattr(ns, "eval_instance_type", "MISSING") in (None, "")


def test_review_deploy_parser_default_review_instance_type_is_unset():
    parser = cli.build_parser()
    ns = parser.parse_args(
        [
            "review",
            "deploy",
            "--base-url",
            "https://challenge.example",
            "--hotkey",
            "hk",
            "--submission-id",
            "1",
        ]
    )
    assert getattr(ns, "review_instance_type", "MISSING") in (None, "")


# --------------------------------------------------------------------------- #
# DEFECT 3 — loopback http opt-in only
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("base_url", "env_value", "should_allow"),
    [
        ("https://challenge.example", None, True),
        ("https://127.0.0.1:18081", None, True),
        ("http://127.0.0.1:18081", "1", True),
        ("http://localhost:18081", "1", True),
        ("http://[::1]:18081", "1", True),
        ("http://127.0.0.1:18081", None, False),
        ("http://127.0.0.1:18081", "0", False),
        ("http://127.0.0.1:18081", "", False),
        ("http://localhost:18081", None, False),
        ("http://[::1]:18081", None, False),
        ("http://challenge.example", "1", False),
        ("http://10.0.0.5:18081", "1", False),
        ("http://192.168.1.1", "1", False),
        ("ftp://127.0.0.1", "1", False),
    ],
)
def test_route_client_http_policy_loopback_opt_in(
    monkeypatch: pytest.Monkeypatch,
    base_url: str,
    env_value: str | None,
    should_allow: bool,
) -> None:
    monkeypatch.delenv("SELFDEPLOY_ALLOW_INSECURE_LOOPBACK", raising=False)
    if env_value is not None:
        monkeypatch.setenv("SELFDEPLOY_ALLOW_INSECURE_LOOPBACK", env_value)

    if should_allow:
        client = SelfDeployRouteClient(base_url)
        assert client._base_url == base_url.strip().rstrip("/")
        return

    with pytest.raises(RouteClientError) as excinfo:
        SelfDeployRouteClient(base_url)
    assert str(excinfo.value) == "challenge endpoint must use https://"
