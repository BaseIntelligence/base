"""Tests for the provider-agnostic compute contract (:mod:`base.compute.provider`)."""

from __future__ import annotations

from base.compute import (
    CostGuardrailError,
    Instance,
    InstanceSpec,
    LiumClient,
    Offer,
    ProviderClient,
    ProviderError,
)


def test_cost_guardrail_error_is_provider_error() -> None:
    assert issubclass(CostGuardrailError, ProviderError)


def test_instance_spec_defaults_are_conservative() -> None:
    spec = InstanceSpec(name="pod")
    # Guardrail fields default to unset so provision() can reject an unbounded spec.
    assert spec.max_lifetime_hours is None
    assert spec.max_price_per_hour is None
    assert spec.ports == (22,)
    assert spec.gpu_count == 1
    assert spec.ssh_public_keys == ()


def test_offer_and_instance_are_frozen_dataclasses() -> None:
    offer = Offer(id="o", gpu_type="H100", gpu_count=1, price_per_hour=2.0)
    instance = Instance(id="p", status="RUNNING")
    assert offer.price_per_hour == 2.0
    assert instance.status == "RUNNING"


def test_lium_client_satisfies_provider_protocol() -> None:
    client = LiumClient("k")
    # runtime_checkable protocol: verifies the full method surface is present.
    assert isinstance(client, ProviderClient)
    typed: ProviderClient = client
    assert typed is client
