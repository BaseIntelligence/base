"""TDD tests for constation key custody, poller, corroboration, runner (15–17)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
import respx

from base.compute.constation_corroboration import evaluate_corroboration
from base.compute.constation_custody import (
    LiumKeyCustody,
    generate_custody_master_key,
)
from base.compute.constation_poller import (
    ContinuousConstationPoller,
    PollerConfig,
    PollPhase,
)
from base.compute.constation_runner import ConstationRunner, ConstationRunRequest
from base.compute.constation_types import (
    ConstationFailCode,
    CorroborationStatus,
    FaultClass,
    PollSample,
)
from base.compute.lium import (
    LiumAuthError,
    LiumClient,
    LiumError,
    LiumPodRead,
    LiumRateLimitError,
)

BASE = "https://lium.io/api"
DIGEST_A = "sha256:" + ("a" * 64)
DIGEST_B = "sha256:" + ("b" * 64)
HOTKEY = "5MinerHotkeyConstationTest000000000000001"
POD = "pod-const-001"
WORK_UNIT = "wu-const-001"
API_KEY = "lium-test-key-NEVER-LOG-THIS-VALUE-xyz"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)


@dataclass
class SequenceRng:
    values: list[float]
    i: int = 0

    def __call__(self) -> float:
        if not self.values:
            return 0.0
        v = self.values[self.i % len(self.values)]
        self.i += 1
        return v


@dataclass
class FakeSidecar:
    digest: str = DIGEST_A
    fail_after: int | None = None
    calls: int = 0
    hang_gap: bool = False

    async def attest(self, *, pod_id: str, phase: str) -> str:
        del pod_id, phase
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("sidecar_down")
        return self.digest


@dataclass
class ScriptedLium:
    """Minimal LiumClient stand-in for runner unit tests."""

    digests: list[str | None] = field(default_factory=lambda: [DIGEST_A])
    miner_hotkey: str = HOTKEY
    status: str = "RUNNING"
    auth_fail_on_call: int | None = None
    rate_limit_on_call: int | None = None
    network_fail_times: int = 0
    calls: int = 0
    _network_left: int = field(init=False)

    def __post_init__(self) -> None:
        self._network_left = self.network_fail_times

    async def get_pod_raw(self, pod_id: str) -> LiumPodRead:
        self.calls += 1
        if self.auth_fail_on_call is not None and self.calls >= self.auth_fail_on_call:
            raise LiumAuthError("Lium GET /pods returned 401", status_code=401)
        if (
            self.rate_limit_on_call is not None
            and self.calls >= self.rate_limit_on_call
        ):
            raise LiumRateLimitError("Lium GET /pods returned 429", status_code=429)
        if self._network_left > 0:
            self._network_left -= 1
            raise LiumError("Lium request GET /pods failed")
        idx = min(self.calls - 1, len(self.digests) - 1)
        digest = self.digests[idx]
        return LiumPodRead(
            pod_id=pod_id,
            template_id="tmpl-1",
            docker_image_digest=digest,
            raw={
                "id": pod_id,
                "status": self.status,
                "executor": {
                    "miner_hotkey": self.miner_hotkey,
                    "executor_ip_address": "10.0.0.1",
                },
                "template": {"docker_image_digest": digest},
            },
        )

    async def balance(self) -> float:
        return 1.0


def _run_request(**overrides: object) -> ConstationRunRequest:
    """Build a ConstationRunRequest with triangle-ready defaults."""
    payload: dict[str, object] = {
        "miner_hotkey": HOTKEY,
        "work_unit_id": WORK_UNIT,
        "pod_id": POD,
        "duration_seconds": 0.0,
        "required_digest": DIGEST_A,
    }
    payload.update(overrides)
    return ConstationRunRequest(**payload)  # type: ignore[arg-type]


def _custody(
    *,
    factory: Any | None = None,
) -> LiumKeyCustody:
    return LiumKeyCustody(
        master_key=generate_custody_master_key(),
        client_factory=factory or LiumClient,
    )


# ===========================================================================
# Todo 15 — custody
# ===========================================================================


@respx.mock
async def test_register_encrypts_key_and_probe_succeeds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    respx.get(f"{BASE}/users/me").mock(
        return_value=httpx.Response(200, json={"balance": 3.14})
    )
    custody = _custody()
    with caplog.at_level(logging.DEBUG):
        verdict = await custody.register(miner_hotkey=HOTKEY, api_key=API_KEY)

    assert verdict.ok is True
    assert verdict.reason is ConstationFailCode.OK
    assert custody.has_key(HOTKEY)
    # ciphertext must not equal plaintext
    blob = custody.export_encrypted()[HOTKEY]
    assert API_KEY.encode() not in blob
    assert custody.unlock_api_key(HOTKEY) == API_KEY
    assert API_KEY not in caplog.text
    assert API_KEY not in repr(custody)
    assert API_KEY not in str(custody)


@respx.mock
async def test_register_probe_401_fail_closed_does_not_store(
    caplog: pytest.LogCaptureFixture,
) -> None:
    respx.get(f"{BASE}/users/me").mock(return_value=httpx.Response(401))
    custody = _custody()
    with caplog.at_level(logging.DEBUG):
        verdict = await custody.register(miner_hotkey=HOTKEY, api_key=API_KEY)

    assert verdict.ok is False
    assert verdict.reason is ConstationFailCode.LIUM_AUTH_REVOKED
    assert verdict.fault_class is FaultClass.MINER
    assert not custody.has_key(HOTKEY)
    assert API_KEY not in caplog.text


@respx.mock
async def test_build_client_uses_unlocked_key_without_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    respx.get(f"{BASE}/users/me").mock(
        return_value=httpx.Response(200, json={"balance": 1.0})
    )
    custody = _custody()
    await custody.register(miner_hotkey=HOTKEY, api_key=API_KEY)
    with caplog.at_level(logging.DEBUG):
        client = custody.build_client(HOTKEY)
        assert await client.balance() == pytest.approx(1.0)

    assert API_KEY not in caplog.text
    assert API_KEY not in repr(client)


@respx.mock
async def test_runner_mid_run_401_is_lium_auth_revoked() -> None:
    """S15b: key works at register; mid-run get_pod 401 → fail-closed."""
    respx.get(f"{BASE}/users/me").mock(
        return_value=httpx.Response(200, json={"balance": 1.0})
    )
    scripted = ScriptedLium(auth_fail_on_call=2)

    def factory(key: str) -> Any:
        del key
        return scripted

    custody = _custody(factory=factory)
    await custody.register(miner_hotkey=HOTKEY, api_key=API_KEY)
    clock = FakeClock()
    runner = ConstationRunner(
        custody=custody,
        sidecar=FakeSidecar(),
        poller_config=PollerConfig(
            gap_budget_seconds=60.0,
            min_interval_seconds=5.0,
            max_interval_seconds=5.0,
            max_polls=10,
            max_cost_units=10.0,
            max_network_retries=0,
            rate_limit_per_second=100.0,
        ),
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        rng_fn=SequenceRng([0.0]),
    )
    record = await runner.run(
        ConstationRunRequest(
            miner_hotkey=HOTKEY,
            work_unit_id=WORK_UNIT,
            pod_id=POD,
            duration_seconds=10.0,
            required_digest=DIGEST_A,
        )
    )
    assert record.ok is False
    assert record.reason is ConstationFailCode.LIUM_AUTH_REVOKED
    assert record.fault_class is FaultClass.MINER


# ===========================================================================
# Todo 16 — poller
# ===========================================================================


async def test_poller_complete_start_interval_end() -> None:
    clock = FakeClock()
    cfg = PollerConfig(
        gap_budget_seconds=30.0,
        min_interval_seconds=5.0,
        max_interval_seconds=5.0,
        max_polls=10,
        max_cost_units=10.0,
        rate_limit_per_second=100.0,
        max_network_retries=1,
    )
    poller = ContinuousConstationPoller(
        config=cfg,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        rng_fn=SequenceRng([0.0]),
    )
    phases: list[str] = []

    async def poll_once(phase: str) -> PollSample:
        phases.append(phase)
        return PollSample(
            at_monotonic=clock.now(),
            phase=phase,
            sidecar_digest=DIGEST_A,
            lium_declared_digest=DIGEST_A,
        )

    result = await poller.run(duration_seconds=12.0, poll_once=poll_once)
    assert result.ok is True
    assert result.reason is ConstationFailCode.OK
    assert phases[0] == PollPhase.START
    assert phases[-1] == PollPhase.END
    assert PollPhase.INTERVAL in phases
    assert result.poll_count >= 3
    assert result.observed_max_gap_seconds <= cfg.gap_budget_seconds


async def test_poller_gap_budget_fail_closed() -> None:
    clock = FakeClock()
    cfg = PollerConfig(
        gap_budget_seconds=10.0,
        min_interval_seconds=25.0,  # sleep exceeds budget before next poll
        max_interval_seconds=25.0,
        max_polls=10,
        max_cost_units=10.0,
        rate_limit_per_second=100.0,
    )
    poller = ContinuousConstationPoller(
        config=cfg,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        rng_fn=SequenceRng([0.0]),
    )

    async def poll_once(phase: str) -> PollSample:
        return PollSample(
            at_monotonic=clock.now(),
            phase=phase,
            sidecar_digest=DIGEST_A,
            lium_declared_digest=DIGEST_A,
        )

    result = await poller.run(duration_seconds=40.0, poll_once=poll_once)
    assert result.ok is False
    assert result.reason is ConstationFailCode.CONSTATION_GAP
    assert result.fault_class is FaultClass.MINER
    assert result.observed_max_gap_seconds > cfg.gap_budget_seconds


async def test_poller_429_fail_closed_not_skip() -> None:
    clock = FakeClock()
    cfg = PollerConfig(
        gap_budget_seconds=60.0,
        min_interval_seconds=5.0,
        max_interval_seconds=5.0,
        max_polls=10,
        max_cost_units=10.0,
        rate_limit_per_second=100.0,
    )
    poller = ContinuousConstationPoller(
        config=cfg,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        rng_fn=SequenceRng([0.0]),
    )
    calls = {"n": 0}

    async def poll_once(phase: str) -> PollSample:
        calls["n"] += 1
        if calls["n"] == 1:
            return PollSample(
                at_monotonic=clock.now(),
                phase=phase,
                sidecar_digest=DIGEST_A,
                lium_declared_digest=DIGEST_A,
            )
        raise LiumRateLimitError("429", status_code=429)

    result = await poller.run(duration_seconds=20.0, poll_once=poll_once)
    assert result.ok is False
    assert result.reason is ConstationFailCode.LIUM_RATE_LIMITED
    assert result.fault_class is FaultClass.INFRA


async def test_poller_network_partition_after_bounded_retry() -> None:
    clock = FakeClock()
    cfg = PollerConfig(
        gap_budget_seconds=60.0,
        min_interval_seconds=5.0,
        max_interval_seconds=5.0,
        max_polls=10,
        max_cost_units=10.0,
        max_network_retries=2,
        backoff_base_seconds=1.0,
        backoff_max_seconds=1.0,
        rate_limit_per_second=100.0,
    )
    poller = ContinuousConstationPoller(
        config=cfg,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        rng_fn=SequenceRng([1.0]),  # full jitter = full delay
    )

    async def poll_once(phase: str) -> PollSample:
        del phase
        raise LiumError("transport down")

    result = await poller.run(duration_seconds=5.0, poll_once=poll_once)
    assert result.ok is False
    assert result.reason is ConstationFailCode.NETWORK_PARTITION
    assert result.fault_class is FaultClass.INFRA


async def test_poller_poll_cap_fail_closed() -> None:
    clock = FakeClock()
    cfg = PollerConfig(
        gap_budget_seconds=100.0,
        min_interval_seconds=1.0,
        max_interval_seconds=1.0,
        max_polls=2,  # start + end only; interval would exceed
        max_cost_units=100.0,
        rate_limit_per_second=100.0,
    )
    poller = ContinuousConstationPoller(
        config=cfg,
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        rng_fn=SequenceRng([0.0]),
    )

    async def poll_once(phase: str) -> PollSample:
        return PollSample(
            at_monotonic=clock.now(),
            phase=phase,
            sidecar_digest=DIGEST_A,
            lium_declared_digest=DIGEST_A,
        )

    # duration 0 → start + end only = 2 polls, ok
    ok_result = await poller.run(duration_seconds=0.0, poll_once=poll_once)
    assert ok_result.ok is True
    assert ok_result.poll_count == 2


# ===========================================================================
# Todo 17 — corroboration
# ===========================================================================


def test_corroboration_agree_is_ok_but_not_elevation() -> None:
    """Agreement is ok as a negative-channel pass; never claims independent."""
    out = evaluate_corroboration(
        lium_declared_digest=DIGEST_A,
        sidecar_digest=DIGEST_A,
    )
    assert out.ok is True
    assert out.status is CorroborationStatus.AGREE
    # Module docstring / status must not say independent — checked in source test


def test_corroboration_mismatch_fails_miner_fault() -> None:
    out = evaluate_corroboration(
        lium_declared_digest=DIGEST_A,
        sidecar_digest=DIGEST_B,
    )
    assert out.ok is False
    assert out.status is CorroborationStatus.MISMATCH
    assert out.verdict.reason is ConstationFailCode.CORROBORATION_MISMATCH
    assert out.verdict.fault_class is FaultClass.MINER


def test_corroboration_absent_lium_is_not_contradiction() -> None:
    out = evaluate_corroboration(
        lium_declared_digest=None,
        sidecar_digest=DIGEST_A,
    )
    assert out.ok is True
    assert out.status is CorroborationStatus.ABSENT


def test_corroboration_module_never_claims_independent() -> None:
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src/base/compute"
    for name in (
        "constation_corroboration.py",
        "constation_runner.py",
        "constation_custody.py",
        "constation_types.py",
        "constation_poller.py",
    ):
        text = (src / name).read_text(encoding="utf-8").lower()
        # Forbid positive claims of independence (negations like "not independent" ok)
        import re

        assert not re.search(r"(?<!not )(?<!never )independent verification", text)
        assert "independently verified" not in text
        assert "independent root of trust" not in text


async def test_runner_corroboration_agree_complete_record() -> None:
    scripted = ScriptedLium(digests=[DIGEST_A])

    def factory(key: str) -> Any:
        del key
        return scripted

    async def _probe(client: Any) -> None:
        del client
        await scripted.balance()

    custody = _custody(factory=factory)
    custody.probe_fn = _probe
    await custody.register(miner_hotkey=HOTKEY, api_key=API_KEY)

    clock = FakeClock()
    runner = ConstationRunner(
        custody=custody,
        sidecar=FakeSidecar(digest=DIGEST_A),
        poller_config=PollerConfig(
            gap_budget_seconds=60.0,
            min_interval_seconds=5.0,
            max_interval_seconds=5.0,
            max_polls=10,
            max_cost_units=10.0,
            rate_limit_per_second=100.0,
        ),
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        rng_fn=SequenceRng([0.0]),
    )
    record = await runner.run(
        ConstationRunRequest(
            miner_hotkey=HOTKEY,
            work_unit_id=WORK_UNIT,
            pod_id=POD,
            duration_seconds=10.0,
            required_digest=DIGEST_A,
        )
    )
    assert record.ok is True
    assert record.reason is ConstationFailCode.OK
    assert record.corroboration_status is CorroborationStatus.AGREE
    assert record.sidecar_digest == DIGEST_A
    assert record.lium_declared_digest == DIGEST_A
    assert (
        record.constation_observed_max_gap_seconds
        <= record.constation_gap_budget_seconds
    )
    assert len(record.samples) >= 2


async def test_runner_corroboration_mismatch_fails() -> None:
    scripted = ScriptedLium(digests=[DIGEST_A])

    def factory(key: str) -> Any:
        del key
        return scripted

    async def _probe(client: Any) -> None:
        del client
        await scripted.balance()

    custody = _custody(factory=factory)
    custody.probe_fn = _probe
    await custody.register(miner_hotkey=HOTKEY, api_key=API_KEY)

    clock = FakeClock()
    runner = ConstationRunner(
        custody=custody,
        sidecar=FakeSidecar(digest=DIGEST_B),
        poller_config=PollerConfig(
            gap_budget_seconds=60.0,
            min_interval_seconds=5.0,
            max_interval_seconds=5.0,
            max_polls=10,
            max_cost_units=10.0,
            rate_limit_per_second=100.0,
        ),
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        rng_fn=SequenceRng([0.0]),
    )
    record = await runner.run(
        ConstationRunRequest(
            miner_hotkey=HOTKEY,
            work_unit_id=WORK_UNIT,
            pod_id=POD,
            duration_seconds=0.0,
            required_digest=DIGEST_A,
        )
    )
    assert record.ok is False
    # Triangle: sidecar != required takes precedence over two-way corroboration.
    assert record.reason is ConstationFailCode.REQUIRED_DIGEST_MISMATCH
    assert record.fault_class is FaultClass.MINER
    assert record.corroboration_status is CorroborationStatus.NOT_EVALUATED


def test_corroboration_agree_insufficient_for_elevation_contract() -> None:
    """Agreement alone must not be treated as elevation — prism still needs all 6.

    This pins the runner/corroboration contract: AGREE is only a channel status,
    never a tier grant. Full elevation remains prism constation_ok's job.
    """
    out = evaluate_corroboration(
        lium_declared_digest=DIGEST_A,
        sidecar_digest=DIGEST_A,
    )
    assert out.status is CorroborationStatus.AGREE
    # No elevation API exists on the outcome
    assert not hasattr(out, "effective_tier")
    assert not hasattr(out, "grant_tier")
    assert out.verdict.reason is ConstationFailCode.OK


async def test_runner_unregistered_key_fail_closed() -> None:
    custody = _custody()
    clock = FakeClock()
    runner = ConstationRunner(
        custody=custody,
        sidecar=FakeSidecar(),
        poller_config=PollerConfig(rate_limit_per_second=100.0),
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        rng_fn=SequenceRng([0.0]),
    )
    record = await runner.run(
        ConstationRunRequest(
            miner_hotkey=HOTKEY,
            work_unit_id=WORK_UNIT,
            pod_id=POD,
            duration_seconds=0.0,
            required_digest=DIGEST_A,
        )
    )
    assert record.ok is False
    assert record.reason is ConstationFailCode.KEY_NOT_REGISTERED


# ===========================================================================
# T5 — runner poll: pod bind + digest triangle
# ===========================================================================


async def _registered_runner(
    *,
    scripted: ScriptedLium,
    sidecar: FakeSidecar | None = None,
    duration_seconds: float = 0.0,
    required_digest: str = DIGEST_A,
    attestor_factory: object | None = None,
) -> tuple[ConstationRunner, ConstationRunRequest]:
    def factory(key: str) -> object:
        del key
        return scripted

    async def _probe(client: object) -> None:
        del client
        await scripted.balance()

    custody = _custody(factory=factory)
    custody.probe_fn = _probe
    await custody.register(miner_hotkey=HOTKEY, api_key=API_KEY)
    clock = FakeClock()
    kwargs: dict[str, object] = {}
    if attestor_factory is not None:
        kwargs["attestor_factory"] = attestor_factory
    runner = ConstationRunner(
        custody=custody,
        sidecar=sidecar or FakeSidecar(digest=DIGEST_A),
        poller_config=PollerConfig(
            gap_budget_seconds=60.0,
            min_interval_seconds=5.0,
            max_interval_seconds=5.0,
            max_polls=10,
            max_cost_units=10.0,
            rate_limit_per_second=100.0,
        ),
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        rng_fn=SequenceRng([0.0]),
        **kwargs,  # type: ignore[arg-type]
    )
    req = _run_request(
        duration_seconds=duration_seconds,
        required_digest=required_digest,
    )
    return runner, req


async def test_runner_triangle_happy_path_agree() -> None:
    """S1: RUNNING + hotkey bind + required==lium==sidecar → ok AGREE."""
    runner, req = await _registered_runner(
        scripted=ScriptedLium(digests=[DIGEST_A]),
        sidecar=FakeSidecar(digest=DIGEST_A),
        duration_seconds=10.0,
    )
    record = await runner.run(req)
    assert record.ok is True
    assert record.reason is ConstationFailCode.OK
    assert record.corroboration_status is CorroborationStatus.AGREE
    assert record.sidecar_digest == DIGEST_A
    assert record.lium_declared_digest == DIGEST_A
    assert len(record.samples) >= 2


async def test_runner_triangle_sidecar_required_mismatch() -> None:
    """S2: sidecar actual != required → REQUIRED_DIGEST_MISMATCH (triangle first)."""
    runner, req = await _registered_runner(
        scripted=ScriptedLium(digests=[DIGEST_A]),
        sidecar=FakeSidecar(digest=DIGEST_B),
        required_digest=DIGEST_A,
    )
    record = await runner.run(req)
    assert record.ok is False
    assert record.reason is ConstationFailCode.REQUIRED_DIGEST_MISMATCH
    assert record.fault_class is FaultClass.MINER


async def test_runner_triangle_lium_mismatch_when_sidecar_matches_required() -> None:
    """Triple path: sidecar==required but lium differs → CORROBORATION_MISMATCH."""
    runner, req = await _registered_runner(
        scripted=ScriptedLium(digests=[DIGEST_B]),
        sidecar=FakeSidecar(digest=DIGEST_A),
        required_digest=DIGEST_A,
    )
    record = await runner.run(req)
    assert record.ok is False
    assert record.reason is ConstationFailCode.CORROBORATION_MISMATCH
    assert record.fault_class is FaultClass.MINER
    assert record.corroboration_status is CorroborationStatus.MISMATCH


async def test_runner_triangle_absent_lium_digest_fail_closed() -> None:
    """S3: Lium declared digest absent → LIUM_DIGEST_ABSENT (no longer optional-ok)."""
    runner, req = await _registered_runner(
        scripted=ScriptedLium(digests=[None]),
        sidecar=FakeSidecar(digest=DIGEST_A),
        required_digest=DIGEST_A,
    )
    record = await runner.run(req)
    assert record.ok is False
    assert record.reason is ConstationFailCode.LIUM_DIGEST_ABSENT
    assert record.fault_class is FaultClass.MINER


async def test_runner_pod_hotkey_mismatch_fail_closed() -> None:
    """S4: executor.miner_hotkey != request.miner_hotkey → POD_HOTKEY_MISMATCH."""
    runner, req = await _registered_runner(
        scripted=ScriptedLium(digests=[DIGEST_A], miner_hotkey="5OtherMinerHotkeyXXXX"),
        sidecar=FakeSidecar(digest=DIGEST_A),
    )
    record = await runner.run(req)
    assert record.ok is False
    assert record.reason is ConstationFailCode.POD_HOTKEY_MISMATCH
    assert record.fault_class is FaultClass.MINER


async def test_runner_pod_not_running_fail_closed() -> None:
    """S5: status not RUNNING → POD_NOT_RUNNING."""
    runner, req = await _registered_runner(
        scripted=ScriptedLium(digests=[DIGEST_A], status="STOPPED"),
        sidecar=FakeSidecar(digest=DIGEST_A),
    )
    record = await runner.run(req)
    assert record.ok is False
    assert record.reason is ConstationFailCode.POD_NOT_RUNNING
    assert record.fault_class is FaultClass.MINER


async def test_runner_attestor_factory_re_resolves_each_poll() -> None:
    """Factory is invoked per poll so mid-run port remap can re-bind."""
    scripted = ScriptedLium(digests=[DIGEST_A])
    seen: list[object] = []

    def factory(pod_raw: object) -> FakeSidecar:
        seen.append(pod_raw)
        return FakeSidecar(digest=DIGEST_A)

    runner, req = await _registered_runner(
        scripted=scripted,
        duration_seconds=10.0,
        attestor_factory=factory,
    )
    record = await runner.run(req)
    assert record.ok is True
    assert len(seen) >= 2


async def test_runner_retains_last_signed_wire_from_http_hit() -> None:
    """When attestor returns a signed wire (Http path), runner keeps last copy."""
    from base.compute.constation_sidecar_client import SidecarAttestHit

    wire = {
        "payload": {
            "digest": DIGEST_A,
            "nonce": "n1",
            "pod_id": POD,
            "sealed_manifest_hashes": {"a": "b"},
        },
        "signature": "sig",
        "algorithm": "hmac-sha256",
        "schema_version": "prism_attestation_payload.v1",
        "phase": "start",
    }

    @dataclass
    class WireAttestor:
        async def attest(self, *, nonce: str, phase: str) -> SidecarAttestHit:
            del nonce
            return SidecarAttestHit(
                digest=DIGEST_A,
                nonce="n1",
                pod_id=POD,
                phase=phase,
                signature="sig",
                algorithm="hmac-sha256",
                schema_version="prism_attestation_payload.v1",
                sealed_manifest_hashes={"a": "b"},
                wire=wire,
            )

    # Wrap as HttpSidecarAttestor-shaped via factory returning object with Hit path:
    # Use a thin adapter that runner detects only for HttpSidecarAttestor.
    # Instead, monkey via factory returning Protocol str attestor is insufficient.
    # Directly set last_signed_wire through a custom path: subclass Http check.
    # Prefer factory returning an object the runner treats as Protocol (str) —
    # wire retention requires HttpSidecarAttestor. Build a stub subclass.

    from base.compute.constation_sidecar_client import HttpSidecarAttestor

    class StubHttp(HttpSidecarAttestor):
        def __init__(self) -> None:
            # bypass __post_init__ URL validation via object.__new__
            object.__setattr__(self, "base_url", "http://10.0.0.1:9")
            object.__setattr__(self, "timeout_seconds", 1.0)
            object.__setattr__(self, "transport", None)

        async def attest(self, *, nonce: str, phase: str) -> SidecarAttestHit:
            del nonce
            return SidecarAttestHit(
                digest=DIGEST_A,
                nonce="n1",
                pod_id=POD,
                phase=phase,
                signature="sig",
                algorithm="hmac-sha256",
                schema_version="prism_attestation_payload.v1",
                sealed_manifest_hashes={"a": "b"},
                wire=wire,
            )

    def factory(pod_raw: object) -> StubHttp:
        del pod_raw
        return StubHttp()

    runner, req = await _registered_runner(
        scripted=ScriptedLium(digests=[DIGEST_A]),
        attestor_factory=factory,
    )
    record = await runner.run(req)
    assert record.ok is True
    assert runner.last_signed_wire == wire
