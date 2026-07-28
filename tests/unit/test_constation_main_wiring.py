"""T9: master lifespan wires custody + pod_binding + orchestrator + hook.

Regression guards:
- build_constation_router receives pod_binding= (register_miner_key not 503)
- custody master key load fail-closed
- WorkerReconciliationService invokes constation_hook before forward_result
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from cryptography.fernet import Fernet

from base.compute.constation_custody import generate_custody_master_key
from base.config.settings import ConstationSettings, Settings
from base.db.models import WorkAssignment, WorkerAssignment
from base.master.constation.custody_keys import (
    build_constation_runtime,
    load_custody_master_key,
    poller_config_from_settings,
)
from base.master.worker_reconciliation import WorkerReconciliationService


def test_main_passes_pod_binding_into_build_constation_router() -> None:
    """Given main.py call site, When AST-inspected, Then pod_binding kwarg present."""
    import base.cli_app.main as main_mod

    source = inspect.getsource(main_mod)
    tree = ast.parse(source)
    hits: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name != "build_constation_router":
            continue
        hits.append(node)
    assert hits, "build_constation_router call missing from main"
    for call in hits:
        kw_names = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert "pod_binding" in kw_names, (
            "build_constation_router must receive pod_binding= "
            f"(got keywords {sorted(kw_names)})"
        )


def test_load_custody_master_key_missing_returns_none() -> None:
    """Given no custody key, When load, Then None (fail-closed)."""
    assert load_custody_master_key(Settings()) is None
    assert (
        load_custody_master_key(Settings(constation=ConstationSettings(enabled=True)))
        is None
    )
    assert (
        load_custody_master_key(
            Settings(constation=ConstationSettings(custody_master_key=""))
        )
        is None
    )
    assert (
        load_custody_master_key(
            Settings(constation=ConstationSettings(custody_master_key="   "))
        )
        is None
    )


def test_load_custody_master_key_inline_and_file(tmp_path: Path) -> None:
    """Given inline or file Fernet key, When load, Then raw key bytes."""
    key = generate_custody_master_key()
    text = key.decode("ascii")
    settings_inline = Settings(constation=ConstationSettings(custody_master_key=text))
    assert load_custody_master_key(settings_inline) == key

    key_path = tmp_path / "custody.key"
    key_path.write_text(text, encoding="utf-8")
    settings_file = Settings(
        constation=ConstationSettings(custody_master_key_file=key_path)
    )
    assert load_custody_master_key(settings_file) == key


def test_build_constation_runtime_disabled_skips_services() -> None:
    """Given enabled=False, When build runtime, Then binding/orch None."""
    key = generate_custody_master_key().decode("ascii")
    runtime = build_constation_runtime(
        Settings(constation=ConstationSettings(enabled=False, custody_master_key=key)),
        nonce_service=object(),
        bundle_store=object(),
    )
    assert runtime.pod_binding is None
    assert runtime.orchestrator is None
    assert runtime.enabled is False


def test_build_constation_runtime_enabled_missing_key_fail_closed() -> None:
    """Given enabled=True without key, When build, Then None services (no crash)."""
    runtime = build_constation_runtime(
        Settings(constation=ConstationSettings(enabled=True)),
        nonce_service=object(),
        bundle_store=object(),
    )
    assert runtime.enabled is True
    assert runtime.pod_binding is None
    assert runtime.orchestrator is None


def test_build_constation_runtime_enabled_with_key() -> None:
    """Given enabled + valid key, When build, Then binding + orchestrator ready."""
    key = generate_custody_master_key().decode("ascii")
    nonce = object()
    store = object()
    runtime = build_constation_runtime(
        Settings(
            constation=ConstationSettings(
                enabled=True,
                custody_master_key=key,
                gap_budget_seconds=12.0,
                sidecar_internal_port=9999,
            )
        ),
        nonce_service=nonce,
        bundle_store=store,
    )
    assert runtime.pod_binding is not None
    assert runtime.orchestrator is not None
    assert runtime.orchestrator.nonce_service is nonce
    assert runtime.orchestrator.bundle_store is store
    assert runtime.orchestrator.sidecar_internal_port == 9999
    assert runtime.orchestrator.poller_config.gap_budget_seconds == 12.0
    # Fernet accepts the loaded master key
    Fernet(runtime.pod_binding.custody.master_key)


def test_poller_config_from_settings_maps_fields() -> None:
    """Given ConstationSettings, When map, Then PollerConfig fields match."""
    cs = ConstationSettings(
        gap_budget_seconds=11.0,
        min_interval_seconds=2.0,
        max_interval_seconds=9.0,
        max_polls=40,
    )
    cfg = poller_config_from_settings(cs)
    assert cfg.gap_budget_seconds == 11.0
    assert cfg.min_interval_seconds == 2.0
    assert cfg.max_interval_seconds == 9.0
    assert cfg.max_polls == 40
    assert cfg.max_cost_units == 40.0


class _FakePrimary:
    challenge_slug = "prism"
    work_unit_id = "wu-hook-1"
    submission_ref = "sub-1"
    payload: dict[str, Any] = {"required_digest": "sha256:" + ("a" * 64)}


class _FakeWinner:
    miner_hotkey = "hk-hook"
    result_payload: dict[str, Any] = {"ok": True}


class _Forwarder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def forward_result(
        self,
        *,
        challenge_slug: str,
        work_unit_id: str,
        submission_ref: str,
        result_payload: Mapping[str, Any],
    ) -> None:
        self.calls.append(
            {
                "challenge_slug": challenge_slug,
                "work_unit_id": work_unit_id,
                "submission_ref": submission_ref,
                "result_payload": dict(result_payload),
            }
        )


@pytest.mark.asyncio
async def test_reconciliation_constation_hook_called_before_forward() -> None:
    """Given hook, When _forward, Then hook runs before forward_result."""
    order: list[str] = []
    seen: dict[str, Any] = {}

    async def _hook(
        *,
        work_unit_id: str,
        miner_hotkey: str,
        metadata: Mapping[str, Any],
    ) -> None:
        order.append("hook")
        seen["work_unit_id"] = work_unit_id
        seen["miner_hotkey"] = miner_hotkey
        seen["metadata"] = dict(metadata)

    class _OrderedForwarder:
        async def forward_result(
            self,
            *,
            challenge_slug: str,
            work_unit_id: str,
            submission_ref: str,
            result_payload: Mapping[str, Any],
        ) -> None:
            order.append("forward")
            seen["forward_unit"] = work_unit_id

    svc = WorkerReconciliationService(
        session_factory=AsyncMock(),  # unused by _forward
        result_forwarder=_OrderedForwarder(),
        constation_hook=_hook,
    )
    ok = await svc._forward(  # noqa: SLF001
        cast(WorkAssignment, _FakePrimary()),
        cast(WorkerAssignment, _FakeWinner()),
    )
    assert ok is True
    assert order == ["hook", "forward"]
    assert seen["work_unit_id"] == "wu-hook-1"
    assert seen["miner_hotkey"] == "hk-hook"
    assert seen["metadata"]["required_digest"].startswith("sha256:")
    assert seen["forward_unit"] == "wu-hook-1"


@pytest.mark.asyncio
async def test_reconciliation_without_hook_still_forwards() -> None:
    """Given no hook, When _forward, Then forward still succeeds."""
    forwarder = _Forwarder()
    svc = WorkerReconciliationService(
        session_factory=AsyncMock(),
        result_forwarder=forwarder,
    )
    ok = await svc._forward(  # noqa: SLF001
        cast(WorkAssignment, _FakePrimary()),
        cast(WorkerAssignment, _FakeWinner()),
    )
    assert ok is True
    assert len(forwarder.calls) == 1


def test_main_passes_constation_pin_source_into_orchestration_driver() -> None:
    """Given main.py call site, When AST-inspected, Then pin source is wired."""
    import base.cli_app.main as main_mod

    source = inspect.getsource(main_mod)
    tree = ast.parse(source)
    hits: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name != "_master_orchestration_driver":
            continue
        hits.append(node)
    assert hits, "_master_orchestration_driver call missing from main"
    for call in hits:
        kw_names = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert "constation_pin_source" in kw_names, (
            "_master_orchestration_driver must receive constation_pin_source= "
            f"(got keywords {sorted(kw_names)})"
        )
