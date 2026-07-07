"""Offline unit tests for worker deploy planning (VAL-AGENT-010/011/012).

Pure, network-free coverage of :mod:`base.worker.deploy`: provider-key
enforcement, in-budget offer selection with the GPU-count preference + fallback,
the miner-signed binding, and the pod env's provider-key hygiene.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from base.compute.provider import Offer
from base.compute.worker_deployment import is_metachar_free
from base.worker.deploy import (
    PROVIDER_KEY_ENV,
    MissingProviderKeyError,
    NoOfferWithinBudgetError,
    UnsupportedProviderError,
    WorkerImageNotConfiguredError,
    build_signed_binding,
    build_worker_pod_env,
    normalize_provider,
    plan_provider_deployment,
    rank_worker_offers,
    require_provider_api_key,
    require_worker_image,
    select_worker_offer,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.worker"


def _offer(offer_id: str, *, price: float, gpu_count: int = 1) -> Offer:
    return Offer(
        id=offer_id,
        gpu_type="H100",
        gpu_count=gpu_count,
        price_per_hour=price,
        provider="lium",
    )


@dataclass
class _FakeSigner:
    hotkey: str

    def sign(self, message: bytes) -> str:
        return "0x" + message.hex()


class _FakeProviderClient:
    def __init__(self, offers: list[Offer]) -> None:
        self._offers = offers
        self.list_calls: list[float | None] = []
        self.provision_calls = 0

    async def list_offers(
        self, *, max_price_per_hour: float | None = None
    ) -> list[Offer]:
        self.list_calls.append(max_price_per_hour)
        return [
            offer
            for offer in self._offers
            if max_price_per_hour is None or offer.price_per_hour <= max_price_per_hour
        ]

    async def provision(self, *_: object, **__: object) -> None:
        self.provision_calls += 1


# -- provider normalization + key enforcement (VAL-AGENT-010) -----------------


def test_normalize_provider_accepts_supported() -> None:
    assert normalize_provider("Lium") == "lium"
    assert normalize_provider(" TARGON ") == "targon"
    assert normalize_provider("local") == "local"


def test_normalize_provider_rejects_unknown() -> None:
    with pytest.raises(UnsupportedProviderError):
        normalize_provider("aws")


@pytest.mark.parametrize("provider", ["lium", "targon"])
def test_require_provider_api_key_raises_when_missing(provider: str) -> None:
    with pytest.raises(MissingProviderKeyError) as exc:
        require_provider_api_key(provider, environ={})
    assert exc.value.env_var == PROVIDER_KEY_ENV[provider]
    assert exc.value.provider == provider
    assert PROVIDER_KEY_ENV[provider] in str(exc.value)


@pytest.mark.parametrize("provider", ["lium", "targon"])
def test_require_provider_api_key_raises_when_blank(provider: str) -> None:
    with pytest.raises(MissingProviderKeyError):
        require_provider_api_key(provider, environ={PROVIDER_KEY_ENV[provider]: "  "})


def test_require_provider_api_key_returns_value() -> None:
    assert require_provider_api_key("lium", environ={"LIUM_API_KEY": "k"}) == "k"


# -- offer selection (VAL-AGENT-012 + live gpu-count constraint) --------------


def test_rank_worker_offers_filters_by_max_price() -> None:
    offers = [_offer("a", price=0.5), _offer("b", price=2.0), _offer("c", price=1.0)]
    ranked = rank_worker_offers(offers, gpu_count=1, max_price=1.0)
    assert [o.id for o in ranked] == ["a", "c"]
    assert all(o.price_per_hour <= 1.0 for o in ranked)


def test_rank_worker_offers_prefers_matching_gpu_count_over_cheaper() -> None:
    # A cheaper 8-GPU node would 400 on a gpu_count=1 rent, so the exact
    # single-GPU node is preferred even though it is pricier.
    offers = [
        _offer("multi", price=0.5, gpu_count=8),
        _offer("single", price=1.0, gpu_count=1),
    ]
    ranked = rank_worker_offers(offers, gpu_count=1, max_price=1.5)
    assert ranked[0].id == "single"
    assert ranked[1].id == "multi"  # next-cheapest fallback retained in order


def test_select_worker_offer_returns_best_in_budget() -> None:
    offers = [_offer("a", price=0.9), _offer("b", price=0.3)]
    assert select_worker_offer(offers, gpu_count=1, max_price=1.0).id == "b"


def test_select_worker_offer_raises_when_all_over_cap() -> None:
    offers = [_offer("a", price=2.0), _offer("b", price=3.0)]
    with pytest.raises(NoOfferWithinBudgetError):
        select_worker_offer(offers, gpu_count=1, max_price=1.0)


def test_select_worker_offer_raises_when_empty() -> None:
    with pytest.raises(NoOfferWithinBudgetError):
        select_worker_offer([], gpu_count=1, max_price=1.0)


async def test_plan_provider_deployment_selects_in_budget_offer() -> None:
    client = _FakeProviderClient(
        [_offer("cheap", price=0.5), _offer("pricey", price=5.0)]
    )
    offer = await plan_provider_deployment(client, gpu_count=1, max_price=1.0)
    assert offer.id == "cheap"
    assert client.provision_calls == 0  # planning never rents


async def test_plan_provider_deployment_all_over_cap_provisions_nothing() -> None:
    client = _FakeProviderClient([_offer("a", price=2.0), _offer("b", price=3.0)])
    with pytest.raises(NoOfferWithinBudgetError):
        await plan_provider_deployment(client, gpu_count=1, max_price=1.0)
    assert client.provision_calls == 0


# -- binding + pod env hygiene (VAL-AGENT-011) --------------------------------


def test_build_signed_binding_signs_pinned_message() -> None:
    binding = build_signed_binding(
        worker_pubkey="worker-pk",
        miner_signer=_FakeSigner("miner-hk"),
        nonce="n1",
    )
    assert binding.miner_hotkey == "miner-hk"
    assert binding.nonce == "n1"
    expected = b"worker-binding:worker-pk:miner-hk:n1".hex()
    assert binding.signature == "0x" + expected


def test_build_signed_binding_generates_fresh_nonce() -> None:
    first = build_signed_binding(worker_pubkey="wp", miner_signer=_FakeSigner("mh"))
    second = build_signed_binding(worker_pubkey="wp", miner_signer=_FakeSigner("mh"))
    assert first.nonce != second.nonce


def test_build_worker_pod_env_never_carries_provider_key() -> None:
    binding = build_signed_binding(
        worker_pubkey="wp", miner_signer=_FakeSigner("mh"), nonce="n1"
    )
    env = build_worker_pod_env(
        master_url="http://master:3100",
        provider="lium",
        binding=binding,
        worker_key_uri="//Worker",
        broker_url="http://127.0.0.1:8082",
        extra={
            "LIUM_API_KEY": "SENTINEL-KEY-12345",
            "TARGON_API_KEY": "SENTINEL-TARGON",
            "SAFE_VALUE": "ok",
        },
    )
    blob = repr(env)
    assert "SENTINEL-KEY-12345" not in blob
    assert "SENTINEL-TARGON" not in blob
    assert "LIUM_API_KEY" not in env
    assert "TARGON_API_KEY" not in env
    assert env["BASE_WORKER__AGENT__MASTER_URL"] == "http://master:3100"
    assert env["BASE_WORKER__IDENTITY__BINDING_SIGNATURE"] == binding.signature
    assert env["BASE_WORKER__IDENTITY__MINER_HOTKEY"] == "mh"
    assert env["SAFE_VALUE"] == "ok"


# -- loopback master_url hygiene (Lium edge-WAF 403; VAL follow-up) ------------


def test_build_worker_pod_env_omits_loopback_urls() -> None:
    # Lium's edge WAF 403s on any request body carrying a loopback URL, and this
    # env is baked into the WAF-sensitive POST /templates body. A loopback
    # coordination URL is also redundant (the pod config defaults to loopback and
    # the agent resolves it at runtime), so it must not travel in the env.
    binding = build_signed_binding(
        worker_pubkey="wp", miner_signer=_FakeSigner("mh"), nonce="n1"
    )
    env = build_worker_pod_env(
        master_url="http://127.0.0.1:8081",
        provider="lium",
        binding=binding,
        broker_url="http://127.0.0.1:8082",
        gateway_url="http://localhost:8081",
    )
    blob = repr(env)
    assert "127.0.0.1" not in blob
    assert "localhost" not in blob
    assert "BASE_WORKER__AGENT__MASTER_URL" not in env
    assert "BASE_WORKER__AGENT__BROKER_URL" not in env
    assert "BASE_WORKER__AGENT__GATEWAY_URL" not in env
    # The binding + provider still travel (they are not loopback URLs).
    assert env["BASE_WORKER__IDENTITY__MINER_HOTKEY"] == "mh"
    assert env["BASE_WORKER__DEPLOY__PROVIDER"] == "lium"


def test_build_worker_pod_env_keeps_public_urls() -> None:
    binding = build_signed_binding(
        worker_pubkey="wp", miner_signer=_FakeSigner("mh"), nonce="n1"
    )
    env = build_worker_pod_env(
        master_url="https://master.example.com",
        provider="lium",
        binding=binding,
        broker_url="http://broker.internal:8082",
        gateway_url="https://gateway.example.com",
    )
    assert env["BASE_WORKER__AGENT__MASTER_URL"] == "https://master.example.com"
    assert env["BASE_WORKER__AGENT__BROKER_URL"] == "http://broker.internal:8082"
    assert env["BASE_WORKER__AGENT__GATEWAY_URL"] == "https://gateway.example.com"


# -- worker image config (no silent private-image pin; VAL follow-up) ----------


def test_require_worker_image_returns_configured_image_and_digest() -> None:
    digest = "sha256:" + "a" * 64
    assert require_worker_image(
        image="ghcr.io/public/base-worker", image_digest=digest, provider="lium"
    ) == ("ghcr.io/public/base-worker", digest)


@pytest.mark.parametrize("provider", ["lium", "targon"])
def test_require_worker_image_raises_when_unset(provider: str) -> None:
    with pytest.raises(WorkerImageNotConfiguredError) as exc:
        require_worker_image(image=None, image_digest=None, provider=provider)
    message = str(exc.value)
    assert "worker.deploy.image" in message
    assert "BASE_WORKER__DEPLOY__IMAGE" in message


def test_require_worker_image_requires_both_image_and_digest() -> None:
    digest = "sha256:" + "a" * 64
    with pytest.raises(WorkerImageNotConfiguredError):
        require_worker_image(
            image="ghcr.io/public/base-worker", image_digest=None, provider="lium"
        )
    with pytest.raises(WorkerImageNotConfiguredError):
        require_worker_image(image=None, image_digest=digest, provider="lium")


def test_require_worker_image_rejects_malformed_digest() -> None:
    with pytest.raises(WorkerImageNotConfiguredError):
        require_worker_image(
            image="ghcr.io/public/base-worker",
            image_digest="latest",
            provider="lium",
        )


# -- worker image entrypoint (VAL-AGENT-013/014, live metachar constraint) -----


def _dockerfile_cmd_tokens() -> list[str]:
    for line in _DOCKERFILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("CMD ["):
            tokens = ast.literal_eval(stripped[len("CMD ") :])
            assert isinstance(tokens, list)
            return [str(token) for token in tokens]
    raise AssertionError("Dockerfile.worker has no exec-form CMD")


def test_worker_dockerfile_entrypoint_starts_the_agent() -> None:
    tokens = _dockerfile_cmd_tokens()
    assert tokens[:3] == ["base", "worker", "agent"]


def test_worker_dockerfile_entrypoint_is_metachar_free() -> None:
    for token in _dockerfile_cmd_tokens():
        assert is_metachar_free(token), token
