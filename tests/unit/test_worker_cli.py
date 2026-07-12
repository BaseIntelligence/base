"""CLI tests for the top-level ``base worker`` app (VAL-AGENT-001/010/011/012).

Exercises the Typer surface with :class:`CliRunner`: the ``base worker`` group is
DISTINCT from the legacy ``base master worker`` Swarm-node group; ``deploy``
enforces the provider key before any network/config work; provider planning bounds
selection by ``--max-price`` (preferring an exact GPU-count executor) and NEVER
places a provider key in the provisioned pod env; ``local`` reports the active
worker; and ``status`` renders the fleet.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import base.cli_app.main as cli_main
from base.compute.provider import Instance, InstanceSpec, Offer

runner = CliRunner()

_SENTINEL_KEY = "SENTINEL-PROVIDER-KEY-DO-NOT-LEAK"


class _FakeKeypair:
    def __init__(self, address: str) -> None:
        self.ss58_address = address

    def sign(self, message: bytes) -> bytes:
        return b"sig:" + message


@pytest.fixture
def _fake_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_main, "create_worker_keypair", lambda _s: _FakeKeypair("worker-ss58")
    )
    monkeypatch.setattr(
        cli_main,
        "create_worker_miner_keypair",
        lambda _s: _FakeKeypair("miner-ss58"),
    )
    monkeypatch.setattr(cli_main, "_configure_observability", lambda _s: None)


# -- VAL-AGENT-001: command surface -------------------------------------------


def test_worker_help_lists_agent_deploy_status() -> None:
    result = runner.invoke(cli_main.app, ["worker", "--help"])
    assert result.exit_code == 0
    assert "agent" in result.stdout
    assert "deploy" in result.stdout
    assert "status" in result.stdout


def test_legacy_master_worker_group_intact() -> None:
    """VAL-CROSS-065: residual Swarm master worker is historical/non-target."""
    result = runner.invoke(cli_main.app, ["master", "worker", "--help"])
    assert result.exit_code == 0
    for legacy in ("token", "list", "label", "drain", "rm", "inspect"):
        assert legacy in result.stdout
    help_blob = (result.stdout or "") + (result.stderr or "")
    assert "HISTORICAL" in help_blob or "NON-TARGET" in help_blob
    assert "install-master.sh" in help_blob


def test_legacy_master_worker_token_warns_historical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subcommands still exist but emit a historical/non-target warning first."""
    calls: list[list[str]] = []

    def _capture(args: list[str]) -> None:
        calls.append(list(args))

    monkeypatch.setattr(cli_main, "_docker_cli", _capture)
    result = runner.invoke(
        cli_main.app, ["master", "worker", "token", "--role", "worker"]
    )
    assert result.exit_code == 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "historical" in combined.lower() or "non-target" in combined.lower()
    assert "install-master.sh" in combined
    assert calls and calls[0][:2] == ["swarm", "join-token"]


# -- VAL-AGENT-010: provider-key refusal --------------------------------------


@pytest.mark.parametrize(
    ("provider", "env_var"),
    [("lium", "LIUM_API_KEY"), ("targon", "TARGON_API_KEY")],
)
def test_deploy_without_provider_key_refuses_before_any_work(
    provider: str, env_var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(env_var, raising=False)

    def _fail_load(*_a: object, **_k: object) -> object:
        raise AssertionError("load_settings must not run before the key check")

    monkeypatch.setattr(cli_main, "load_settings", _fail_load)
    result = runner.invoke(cli_main.app, ["worker", "deploy", "--provider", provider])
    assert result.exit_code == 2
    assert env_var in result.stderr
    assert "No provider or master call was made" in result.stderr


def test_deploy_rejects_unknown_provider() -> None:
    result = runner.invoke(cli_main.app, ["worker", "deploy", "--provider", "aws"])
    assert result.exit_code == 2
    assert "unsupported provider" in result.stderr


# -- VAL-AGENT-012 + 011: planning bounds + provider-key hygiene --------------


def test_deploy_provider_selects_in_budget_offer_without_leaking_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object, _fake_keys: None
) -> None:
    captured: dict[str, object] = {}

    class _FakeLiumClient:
        def __init__(self, api_key: str) -> None:
            captured["api_key"] = api_key

        async def list_offers(
            self, *, max_price_per_hour: float | None = None
        ) -> list[Offer]:
            captured["max_price"] = max_price_per_hour
            return [
                Offer(
                    id="multi-cheap",
                    gpu_type="H100",
                    gpu_count=8,
                    price_per_hour=0.4,
                ),
                Offer(
                    id="single-fit",
                    gpu_type="H100",
                    gpu_count=1,
                    price_per_hour=0.7,
                ),
                Offer(
                    id="over-cap",
                    gpu_type="H100",
                    gpu_count=1,
                    price_per_hour=5.0,
                ),
            ]

        async def provision(self, spec: object, *, offer: Offer) -> Instance:
            captured["spec"] = spec
            captured["offer"] = offer
            return Instance(id="pod-xyz", status="PENDING")

    monkeypatch.setenv("LIUM_API_KEY", _SENTINEL_KEY)
    monkeypatch.setenv("BASE_WORKER__DEPLOY__IMAGE", "ghcr.io/public/base-worker")
    monkeypatch.setenv("BASE_WORKER__DEPLOY__IMAGE_DIGEST", "sha256:" + "a" * 64)
    monkeypatch.setattr(cli_main, "LiumClient", _FakeLiumClient)

    result = runner.invoke(
        cli_main.app,
        [
            "worker",
            "deploy",
            "--provider",
            "lium",
            "--max-price",
            "1.0",
            "--config",
            "config/worker.example.yaml",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # In-budget + exact-gpu-count offer preferred over the cheaper 8-GPU node.
    assert captured["max_price"] == 1.0
    offer = captured["offer"]
    assert isinstance(offer, Offer)
    assert offer.id == "single-fit"
    assert "single-fit" in result.stdout
    assert "pod-xyz" in result.stdout
    # VAL-AGENT-011: the provider key authenticated the client but is absent from
    # the provisioned pod env handed to the worker agent.
    assert captured["api_key"] == _SENTINEL_KEY
    spec = captured["spec"]
    assert isinstance(spec, InstanceSpec)
    env = dict(spec.env)
    assert _SENTINEL_KEY not in repr(env)
    assert "LIUM_API_KEY" not in env
    assert env["BASE_WORKER__IDENTITY__MINER_HOTKEY"] == "miner-ss58"
    # The explicitly-configured worker image is pinned (no silent placeholder).
    assert spec.image == "ghcr.io/public/base-worker"
    assert spec.image_digest == "sha256:" + "a" * 64
    # The loopback master_url from the example config never reaches the pod env
    # (Lium's edge WAF 403s on loopback URLs baked into the template body).
    assert "127.0.0.1" not in repr(env)


def test_deploy_provider_all_over_cap_provisions_nothing(
    monkeypatch: pytest.MonkeyPatch, _fake_keys: None
) -> None:
    provisioned = {"count": 0}

    class _FakeLiumClient:
        def __init__(self, api_key: str) -> None:
            pass

        async def list_offers(
            self, *, max_price_per_hour: float | None = None
        ) -> list[Offer]:
            return [
                Offer(id="a", gpu_type="H100", gpu_count=1, price_per_hour=9.0),
            ]

        async def provision(self, *a: object, **k: object) -> Instance:
            provisioned["count"] += 1
            return Instance(id="never", status="PENDING")

    monkeypatch.setenv("LIUM_API_KEY", _SENTINEL_KEY)
    monkeypatch.setenv("BASE_WORKER__DEPLOY__IMAGE", "ghcr.io/public/base-worker")
    monkeypatch.setenv("BASE_WORKER__DEPLOY__IMAGE_DIGEST", "sha256:" + "a" * 64)
    monkeypatch.setattr(cli_main, "LiumClient", _FakeLiumClient)

    result = runner.invoke(
        cli_main.app,
        [
            "worker",
            "deploy",
            "--provider",
            "lium",
            "--max-price",
            "1.0",
            "--config",
            "config/worker.example.yaml",
        ],
    )
    assert result.exit_code == 1
    assert provisioned["count"] == 0
    assert "no rentable offer within budget" in result.stderr


def test_deploy_provider_without_worker_image_refuses(
    monkeypatch: pytest.MonkeyPatch, _fake_keys: None
) -> None:
    """A provider deploy refuses when no explicit worker image + digest is set.

    The M1 placeholder image is a PRIVATE-namespace GHCR image that fails Lium pod
    creation, so the deploy must not silently pin it: an unset image is a clear,
    actionable refusal that provisions nothing.
    """

    listed = {"count": 0}

    class _FakeLiumClient:
        def __init__(self, api_key: str) -> None:
            pass

        async def list_offers(
            self, *, max_price_per_hour: float | None = None
        ) -> list[Offer]:
            listed["count"] += 1
            return [Offer(id="a", gpu_type="H100", gpu_count=1, price_per_hour=0.5)]

        async def provision(self, *a: object, **k: object) -> Instance:
            raise AssertionError("provision must not run without a configured image")

    monkeypatch.setenv("LIUM_API_KEY", _SENTINEL_KEY)
    monkeypatch.delenv("BASE_WORKER__DEPLOY__IMAGE", raising=False)
    monkeypatch.delenv("BASE_WORKER__DEPLOY__IMAGE_DIGEST", raising=False)
    monkeypatch.setattr(cli_main, "LiumClient", _FakeLiumClient)

    result = runner.invoke(
        cli_main.app,
        [
            "worker",
            "deploy",
            "--provider",
            "lium",
            "--max-price",
            "1.0",
            "--config",
            "config/worker.example.yaml",
        ],
    )
    assert result.exit_code == 1
    # Refused before any provider network call (image check is fail-fast).
    assert listed["count"] == 0
    assert "worker.deploy.image" in result.stderr
    assert "BASE_WORKER__DEPLOY__IMAGE" in result.stderr


# -- VAL-AGENT-009 (unit-level): local deploy reports the active worker --------


def test_deploy_local_reports_active_worker(
    monkeypatch: pytest.MonkeyPatch, _fake_keys: None
) -> None:
    monkeypatch.setattr(
        cli_main,
        "_spawn_worker_agent_process",
        lambda _config: SimpleNamespace(pid=4321, terminate=lambda: None),
    )

    async def _fake_wait(*_a: object, **_k: object) -> object:
        return SimpleNamespace(
            worker_id="wrk-1",
            worker_pubkey="worker-ss58",
            miner_hotkey="miner-ss58",
            provider="local",
            status="active",
        )

    monkeypatch.setattr(cli_main, "_wait_worker_active", _fake_wait)

    result = runner.invoke(
        cli_main.app,
        [
            "worker",
            "deploy",
            "--provider",
            "local",
            "--config",
            "config/worker.example.yaml",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "pid=4321" in result.stdout
    assert "wrk-1" in result.stdout
    assert "active" in result.stdout


# -- status renders the fleet -------------------------------------------------


def test_status_renders_fleet_from_list_workers(
    monkeypatch: pytest.MonkeyPatch, _fake_keys: None
) -> None:
    class _FakeClient:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        async def list_workers(self, *, hotkey: str | None = None) -> list[object]:
            return [
                SimpleNamespace(
                    worker_id="wrk-1",
                    miner_hotkey="miner-ss58",
                    provider="lium",
                    status="active",
                    last_heartbeat_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            ]

    monkeypatch.setattr(cli_main, "WorkerCoordinationClient", _FakeClient)
    result = runner.invoke(
        cli_main.app, ["worker", "status", "--config", "config/worker.example.yaml"]
    )
    assert result.exit_code == 0, result.stdout
    assert "wrk-1" in result.stdout
    assert "miner-ss58" in result.stdout
    assert "lium" in result.stdout
    assert "active" in result.stdout


def test_status_surfaces_fault_count_agreeing_with_fleet_view(
    monkeypatch: pytest.MonkeyPatch, _fake_keys: None
) -> None:
    """The CLI fleet view shows each worker's attributed fault count (VAL-CROSS-009)."""

    class _FakeClient:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        async def list_workers(self, *, hotkey: str | None = None) -> list[object]:
            return [
                SimpleNamespace(
                    worker_id="clean-1",
                    miner_hotkey="owner-a",
                    provider="local",
                    status="active",
                    last_heartbeat_at=datetime(2026, 1, 1, tzinfo=UTC),
                    faults=[],
                ),
                SimpleNamespace(
                    worker_id="faulted-1",
                    miner_hotkey="owner-b",
                    provider="local",
                    status="active",
                    last_heartbeat_at=datetime(2026, 1, 1, tzinfo=UTC),
                    faults=[
                        SimpleNamespace(work_unit_id="u1"),
                        SimpleNamespace(work_unit_id="u2"),
                    ],
                ),
            ]

    monkeypatch.setattr(cli_main, "WorkerCoordinationClient", _FakeClient)
    result = runner.invoke(
        cli_main.app, ["worker", "status", "--config", "config/worker.example.yaml"]
    )
    assert result.exit_code == 0, result.stdout
    assert "FAULTS" in result.stdout
    lines = {
        line.split()[0]: line for line in result.stdout.splitlines() if line.split()
    }
    assert " 0 " in f" {lines['clean-1'].split()[4]} "
    assert lines["faulted-1"].split()[4] == "2"


def test_status_reports_empty_fleet(
    monkeypatch: pytest.MonkeyPatch, _fake_keys: None
) -> None:
    class _FakeClient:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        async def list_workers(self, *, hotkey: str | None = None) -> list[object]:
            return []

    monkeypatch.setattr(cli_main, "WorkerCoordinationClient", _FakeClient)
    result = runner.invoke(
        cli_main.app, ["worker", "status", "--config", "config/worker.example.yaml"]
    )
    assert result.exit_code == 0
    assert "No workers registered." in result.stdout
