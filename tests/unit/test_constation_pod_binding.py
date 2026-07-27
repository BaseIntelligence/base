"""TDD tests for miner Lium API key + instance_id pod binding (T6a domain).

Fail-closed: probe → get_pod_raw → assert_pod_bound → store encrypted key + id.
Never logs api_key. No HTTP routes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest

from base.compute.constation_custody import (
    LiumKeyCustody,
    generate_custody_master_key,
)
from base.compute.constation_types import ConstationFailCode, FaultClass
from base.compute.lium import (
    LiumAuthError,
    LiumError,
    LiumNotFoundError,
    LiumPodRead,
    LiumRateLimitError,
)
from base.master.constation.pod_binding import MinerPodBinding

HOTKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
OTHER = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
INSTANCE = "pod-bind-001"
API_KEY = "lium-bind-test-key-NEVER-LOG-THIS-VALUE-abc"


def _pod_raw(
    *,
    miner_hotkey: object = HOTKEY,
    status: object = "RUNNING",
    pod_id: str = INSTANCE,
) -> dict[str, Any]:
    return {
        "id": pod_id,
        "status": status,
        "executor": {
            "executor_ip_address": "1.2.3.4",
            "miner_hotkey": miner_hotkey,
        },
    }


@dataclass
class ScriptedBindClient:
    """LiumClient stand-in: balance probe + get_pod_raw."""

    pod_raw: dict[str, Any] = field(default_factory=_pod_raw)
    probe_error: BaseException | None = None
    pod_error: BaseException | None = None
    balance_calls: int = 0
    pod_calls: int = 0
    last_pod_id: str | None = None
    api_key_seen: str | None = None

    async def balance(self) -> float:
        self.balance_calls += 1
        if self.probe_error is not None:
            raise self.probe_error
        return 1.0

    async def get_pod_raw(self, pod_id: str) -> LiumPodRead:
        self.pod_calls += 1
        self.last_pod_id = pod_id
        if self.pod_error is not None:
            raise self.pod_error
        raw = self.pod_raw
        return LiumPodRead(
            pod_id=str(raw.get("id") or pod_id),
            template_id="tmpl-1",
            docker_image_digest=None,
            raw=raw,
        )


def _binding(
    client: ScriptedBindClient,
) -> MinerPodBinding:
    def factory(api_key: str) -> ScriptedBindClient:
        client.api_key_seen = api_key
        return client

    custody = LiumKeyCustody(
        master_key=generate_custody_master_key(),
        client_factory=factory,  # type: ignore[arg-type]
    )
    return MinerPodBinding(custody=custody)


# ---------------------------------------------------------------------------
# S1 happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_binds_key_and_instance_when_pod_running(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given a valid key and RUNNING pod owned by miner hotkey
    client = ScriptedBindClient(pod_raw=_pod_raw())
    binding = _binding(client)

    # When register runs
    with caplog.at_level(logging.DEBUG):
        verdict = await binding.register(
            miner_hotkey=HOTKEY,
            api_key=API_KEY,
            instance_id=INSTANCE,
        )

    # Then ok + encrypted key + instance_id stored; api_key never logged
    assert verdict.ok is True
    assert verdict.reason is ConstationFailCode.OK
    assert binding.has_binding(HOTKEY) is True
    assert binding.get_instance_id(HOTKEY) == INSTANCE
    assert binding.custody.has_key(HOTKEY) is True
    assert binding.custody.unlock_api_key(HOTKEY) == API_KEY
    assert client.balance_calls >= 1
    assert client.pod_calls == 1
    assert client.last_pod_id == INSTANCE
    assert API_KEY not in caplog.text
    assert API_KEY not in repr(binding)
    assert API_KEY not in str(binding)
    blob = binding.custody.export_encrypted()[HOTKEY]
    assert API_KEY.encode() not in blob


@pytest.mark.asyncio
async def test_register_strips_whitespace_on_inputs() -> None:
    client = ScriptedBindClient(pod_raw=_pod_raw())
    binding = _binding(client)

    verdict = await binding.register(
        miner_hotkey=f"  {HOTKEY}  ",
        api_key=f"  {API_KEY}  ",
        instance_id=f"  {INSTANCE}  ",
    )

    assert verdict.ok is True
    assert binding.get_instance_id(HOTKEY) == INSTANCE
    assert binding.custody.unlock_api_key(HOTKEY) == API_KEY
    assert client.last_pod_id == INSTANCE


# ---------------------------------------------------------------------------
# S2 probe failures — fail closed, store nothing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_probe_401_fail_closed_stores_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = ScriptedBindClient(
        probe_error=LiumAuthError("401", status_code=401),
    )
    binding = _binding(client)

    with caplog.at_level(logging.DEBUG):
        verdict = await binding.register(
            miner_hotkey=HOTKEY,
            api_key=API_KEY,
            instance_id=INSTANCE,
        )

    assert verdict.ok is False
    assert verdict.reason is ConstationFailCode.LIUM_AUTH_REVOKED
    assert verdict.fault_class is FaultClass.MINER
    assert binding.has_binding(HOTKEY) is False
    assert binding.custody.has_key(HOTKEY) is False
    assert client.pod_calls == 0
    assert API_KEY not in caplog.text


@pytest.mark.asyncio
async def test_register_probe_other_error_is_probe_failed() -> None:
    client = ScriptedBindClient(probe_error=LiumError("timeout"))
    binding = _binding(client)

    verdict = await binding.register(
        miner_hotkey=HOTKEY,
        api_key=API_KEY,
        instance_id=INSTANCE,
    )

    assert verdict.ok is False
    assert verdict.reason is ConstationFailCode.PROBE_FAILED
    assert binding.has_binding(HOTKEY) is False
    assert binding.custody.has_key(HOTKEY) is False
    assert client.pod_calls == 0


# ---------------------------------------------------------------------------
# S3 / S4 pod bind failures — key must not be stored
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_pod_hotkey_mismatch_stores_nothing() -> None:
    client = ScriptedBindClient(pod_raw=_pod_raw(miner_hotkey=OTHER))
    binding = _binding(client)

    verdict = await binding.register(
        miner_hotkey=HOTKEY,
        api_key=API_KEY,
        instance_id=INSTANCE,
    )

    assert verdict.ok is False
    assert verdict.reason is ConstationFailCode.POD_HOTKEY_MISMATCH
    assert binding.has_binding(HOTKEY) is False
    assert binding.custody.has_key(HOTKEY) is False


@pytest.mark.asyncio
async def test_register_pod_not_running_stores_nothing() -> None:
    client = ScriptedBindClient(pod_raw=_pod_raw(status="STOPPED"))
    binding = _binding(client)

    verdict = await binding.register(
        miner_hotkey=HOTKEY,
        api_key=API_KEY,
        instance_id=INSTANCE,
    )

    assert verdict.ok is False
    assert verdict.reason is ConstationFailCode.POD_NOT_RUNNING
    assert binding.has_binding(HOTKEY) is False
    assert binding.custody.has_key(HOTKEY) is False


@pytest.mark.asyncio
async def test_register_mismatch_precedes_not_running() -> None:
    client = ScriptedBindClient(
        pod_raw=_pod_raw(miner_hotkey=OTHER, status="STOPPED"),
    )
    binding = _binding(client)

    verdict = await binding.register(
        miner_hotkey=HOTKEY,
        api_key=API_KEY,
        instance_id=INSTANCE,
    )

    assert verdict.ok is False
    assert verdict.reason is ConstationFailCode.POD_HOTKEY_MISMATCH


# ---------------------------------------------------------------------------
# S5 get_pod transport / not-found — fail closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_pod_not_found_fail_closed() -> None:
    client = ScriptedBindClient(
        pod_error=LiumNotFoundError("404", status_code=404),
    )
    binding = _binding(client)

    verdict = await binding.register(
        miner_hotkey=HOTKEY,
        api_key=API_KEY,
        instance_id=INSTANCE,
    )

    assert verdict.ok is False
    assert verdict.reason is ConstationFailCode.POD_HOTKEY_MISMATCH
    assert binding.has_binding(HOTKEY) is False
    assert binding.custody.has_key(HOTKEY) is False


@pytest.mark.asyncio
async def test_register_pod_401_is_lium_auth_revoked() -> None:
    client = ScriptedBindClient(
        pod_error=LiumAuthError("401", status_code=401),
    )
    binding = _binding(client)

    verdict = await binding.register(
        miner_hotkey=HOTKEY,
        api_key=API_KEY,
        instance_id=INSTANCE,
    )

    assert verdict.ok is False
    assert verdict.reason is ConstationFailCode.LIUM_AUTH_REVOKED
    assert binding.has_binding(HOTKEY) is False
    assert binding.custody.has_key(HOTKEY) is False


@pytest.mark.asyncio
async def test_register_pod_rate_limited() -> None:
    client = ScriptedBindClient(
        pod_error=LiumRateLimitError("429", status_code=429),
    )
    binding = _binding(client)

    verdict = await binding.register(
        miner_hotkey=HOTKEY,
        api_key=API_KEY,
        instance_id=INSTANCE,
    )

    assert verdict.ok is False
    assert verdict.reason is ConstationFailCode.LIUM_RATE_LIMITED
    assert binding.has_binding(HOTKEY) is False
    assert binding.custody.has_key(HOTKEY) is False


@pytest.mark.asyncio
async def test_register_pod_network_error_is_probe_failed() -> None:
    client = ScriptedBindClient(pod_error=LiumError("network down"))
    binding = _binding(client)

    verdict = await binding.register(
        miner_hotkey=HOTKEY,
        api_key=API_KEY,
        instance_id=INSTANCE,
    )

    assert verdict.ok is False
    assert verdict.reason is ConstationFailCode.PROBE_FAILED
    assert binding.has_binding(HOTKEY) is False
    assert binding.custody.has_key(HOTKEY) is False


# ---------------------------------------------------------------------------
# Blank inputs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_rejects_blank_fields() -> None:
    client = ScriptedBindClient()
    binding = _binding(client)

    with pytest.raises(ValueError, match="miner_hotkey"):
        await binding.register(
            miner_hotkey="  ",
            api_key=API_KEY,
            instance_id=INSTANCE,
        )
    with pytest.raises(ValueError, match="api_key"):
        await binding.register(
            miner_hotkey=HOTKEY,
            api_key="",
            instance_id=INSTANCE,
        )
    with pytest.raises(ValueError, match="instance_id"):
        await binding.register(
            miner_hotkey=HOTKEY,
            api_key=API_KEY,
            instance_id="   ",
        )
    assert binding.has_binding(HOTKEY) is False


# ---------------------------------------------------------------------------
# Lookup / overwrite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_instance_id_missing_returns_none() -> None:
    binding = _binding(ScriptedBindClient())
    assert binding.get_instance_id(HOTKEY) is None
    assert binding.has_binding(HOTKEY) is False


@pytest.mark.asyncio
async def test_register_overwrites_prior_binding() -> None:
    client = ScriptedBindClient(pod_raw=_pod_raw(pod_id="pod-a"))
    binding = _binding(client)

    first = await binding.register(
        miner_hotkey=HOTKEY,
        api_key=API_KEY,
        instance_id="pod-a",
    )
    assert first.ok is True
    assert binding.get_instance_id(HOTKEY) == "pod-a"

    client.pod_raw = _pod_raw(pod_id="pod-b")
    second = await binding.register(
        miner_hotkey=HOTKEY,
        api_key=API_KEY + "-rotated",
        instance_id="pod-b",
    )
    assert second.ok is True
    assert binding.get_instance_id(HOTKEY) == "pod-b"
    assert binding.custody.unlock_api_key(HOTKEY) == API_KEY + "-rotated"
