"""Offline respx tests for :class:`base.compute.lium.LiumClient`.

Every test mocks Lium HTTP via respx; no credentials and no real network are
required. These pin the provider contract assertions VAL-PROV-001/003/004/005/
011/017/018 plus secret hygiene for the Lium client.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

from base.compute import (
    CostGuardrailError,
    Instance,
    InstanceSpec,
    LiumClient,
    LiumError,
    Offer,
)
from base.compute.lium import (
    _as_list,
    _extract_gpu_count,
    _extract_gpu_type,
    _extract_price,
    _parse_instance,
    _parse_offer,
)

BASE = "https://lium.io/api"

# GET /executors payload shaped like the real API (price_per_gpu + machine_name),
# mixing offers above and below a $1.00/GPU/hr bound.
EXECUTORS = [
    {"id": "a", "machine_name": "RTX 5090", "gpu_count": 8, "price_per_gpu": 0.95},
    {"id": "b", "machine_name": "H100", "gpu_count": 1, "price_per_gpu": 2.0},
    {"id": "c", "machine_name": "RTX 4090", "gpu_count": 2, "price_per_gpu": 0.5},
]


def _spec(**overrides: object) -> InstanceSpec:
    base: dict[str, object] = {
        "name": "mission-pod",
        "template_ref": "prism-worker",
        "image": "ghcr.io/base/worker",
        "ssh_public_keys": ("ssh-ed25519 AAAA",),
        "max_lifetime_hours": 1,
        "max_price_per_hour": 1.5,
    }
    base.update(overrides)
    return InstanceSpec(**base)  # type: ignore[arg-type]


def _offer(price: float = 0.95, offer_id: str = "exec-1") -> Offer:
    return Offer(id=offer_id, gpu_type="RTX 5090", gpu_count=1, price_per_hour=price)


# -- VAL-PROV-001 -------------------------------------------------------------


@respx.mock
async def test_list_offers_filters_by_max_price() -> None:
    respx.get(f"{BASE}/executors").mock(
        return_value=httpx.Response(200, json=EXECUTORS)
    )
    offers = await LiumClient("k").list_offers(max_price_per_hour=1.0)
    assert {o.id for o in offers} == {"a", "c"}
    for offer in offers:
        assert offer.price_per_hour <= 1.0
        assert offer.gpu_type
        assert offer.gpu_count >= 1


@respx.mock
async def test_list_offers_without_bound_returns_all_priced() -> None:
    respx.get(f"{BASE}/executors").mock(
        return_value=httpx.Response(200, json=EXECUTORS)
    )
    offers = await LiumClient("k").list_offers()
    assert {o.id for o in offers} == {"a", "b", "c"}


@respx.mock
async def test_list_offers_accepts_price_per_hour_field() -> None:
    payload = [{"id": "x", "gpu_type": "H100", "gpu_count": 1, "price_per_hour": 0.8}]
    respx.get(f"{BASE}/executors").mock(return_value=httpx.Response(200, json=payload))
    offers = await LiumClient("k").list_offers(max_price_per_hour=1.0)
    assert offers[0].id == "x"
    assert offers[0].price_per_hour == 0.8


@respx.mock
async def test_list_offers_skips_offers_without_price() -> None:
    payload = [{"id": "x", "machine_name": "H100", "gpu_count": 1}]
    respx.get(f"{BASE}/executors").mock(return_value=httpx.Response(200, json=payload))
    assert await LiumClient("k").list_offers() == []


# -- VAL-PROV-003 -------------------------------------------------------------


@respx.mock
@pytest.mark.parametrize(
    "overrides",
    [
        {"max_lifetime_hours": None},
        {"max_lifetime_hours": 0},
        {"max_lifetime_hours": -1},
        {"max_price_per_hour": None},
    ],
)
async def test_provision_refuses_unbounded_spec_without_network(
    overrides: dict[str, object],
) -> None:
    client = LiumClient("k")
    with pytest.raises(CostGuardrailError):
        await client.provision(_spec(**overrides))
    assert respx.calls.call_count == 0


# -- VAL-PROV-004 -------------------------------------------------------------


def _mock_happy_path(*, template: list | None = None, keys: list | None = None) -> dict:
    routes = {
        "ssh_get": respx.get(f"{BASE}/ssh-keys").mock(
            return_value=httpx.Response(
                200,
                json=keys
                if keys is not None
                else [{"id": "k1", "public_key": "ssh-ed25519 AAAA"}],
            )
        ),
        "ssh_post": respx.post(f"{BASE}/ssh-keys").mock(
            return_value=httpx.Response(200, json={"id": "k-new"})
        ),
        "tpl_get": respx.get(f"{BASE}/templates").mock(
            return_value=httpx.Response(
                200,
                json=template
                if template is not None
                else [{"id": "tpl-1", "name": "prism-worker"}],
            )
        ),
        "tpl_post": respx.post(f"{BASE}/templates").mock(
            return_value=httpx.Response(200, json={"id": "tpl-new"})
        ),
        "rent": respx.post(f"{BASE}/executors/exec-1/rent").mock(
            return_value=httpx.Response(200, json={"id": "pod-1", "status": "PENDING"})
        ),
        "status": respx.get(f"{BASE}/pods/pod-1").mock(
            return_value=httpx.Response(200, json={"id": "pod-1", "status": "RUNNING"})
        ),
    }
    return routes


@respx.mock
async def test_provision_sends_termination_hours_and_ssh_key() -> None:
    routes = _mock_happy_path()
    instance = await LiumClient("k").provision(
        _spec(max_lifetime_hours=2), offer=_offer()
    )
    assert isinstance(instance, Instance)
    assert instance.id == "pod-1"
    assert instance.status == "RUNNING"
    body = json.loads(routes["rent"].calls.last.request.content)
    assert body["termination_hours"] == 2
    assert body["pod_name"] == "mission-pod"
    assert body["user_public_key"] == ["ssh-ed25519 AAAA"]
    assert body["template_id"] == "tpl-1"


@respx.mock
async def test_provision_rejects_overpriced_offer_without_rent() -> None:
    rent = respx.post(f"{BASE}/executors/exec-1/rent")
    client = LiumClient("k")
    with pytest.raises(CostGuardrailError):
        await client.provision(_spec(max_price_per_hour=1.0), offer=_offer(price=2.0))
    assert rent.call_count == 0
    assert respx.calls.call_count == 0


@respx.mock
async def test_provision_selects_cheapest_within_budget_when_no_offer() -> None:
    respx.get(f"{BASE}/executors").mock(
        return_value=httpx.Response(200, json=EXECUTORS)
    )
    _mock_happy_path()
    # cheapest under the bound is "c" at 0.5 -> rent goes to /executors/c/rent
    rent_c = respx.post(f"{BASE}/executors/c/rent").mock(
        return_value=httpx.Response(200, json={"id": "pod-1", "status": "PENDING"})
    )
    await LiumClient("k").provision(_spec(max_price_per_hour=1.0))
    assert rent_c.call_count == 1


@respx.mock
async def test_provision_raises_guardrail_when_no_offer_within_budget() -> None:
    respx.get(f"{BASE}/executors").mock(
        return_value=httpx.Response(200, json=EXECUTORS)
    )
    rent = respx.post(f"{BASE}/executors/b/rent")
    with pytest.raises(CostGuardrailError):
        await LiumClient("k").provision(_spec(max_price_per_hour=0.1))
    assert rent.call_count == 0


# -- VAL-PROV-005 -------------------------------------------------------------


@respx.mock
async def test_terminate_is_idempotent() -> None:
    route = respx.delete(f"{BASE}/pods/pod-1").mock(
        side_effect=[httpx.Response(200), httpx.Response(404)]
    )
    client = LiumClient("k")
    await client.terminate("pod-1")
    await client.terminate("pod-1")
    assert route.call_count == 2


@respx.mock
async def test_terminate_raises_on_non_404_error() -> None:
    respx.delete(f"{BASE}/pods/pod-1").mock(return_value=httpx.Response(500))
    with pytest.raises(LiumError):
        await LiumClient("k").terminate("pod-1")


@respx.mock
async def test_verify_terminated_reflects_pod_presence() -> None:
    respx.get(f"{BASE}/pods").mock(
        side_effect=[
            httpx.Response(200, json=[{"id": "pod-1", "status": "RUNNING"}]),
            httpx.Response(200, json=[]),
        ]
    )
    client = LiumClient("k")
    assert await client.verify_terminated("pod-1") is False
    assert await client.verify_terminated("pod-1") is True


# -- VAL-PROV-011 -------------------------------------------------------------


@respx.mock
async def test_watchtower_digest_returned_verbatim() -> None:
    digest = "sha256:" + "b" * 64
    respx.get(f"{BASE}/watchtower/digest").mock(
        return_value=httpx.Response(
            200, json={"digest": digest, "signature": "0xabc", "timestamp": 1}
        )
    )
    assert await LiumClient("k").watchtower_digest() == digest


@respx.mock
async def test_watchtower_digest_missing_field_raises() -> None:
    respx.get(f"{BASE}/watchtower/digest").mock(
        return_value=httpx.Response(200, json={"signature": "0xabc"})
    )
    with pytest.raises(LiumError):
        await LiumClient("k").watchtower_digest()


# -- VAL-PROV-017 -------------------------------------------------------------


@respx.mock
async def test_provision_failure_path_terminates_and_verifies() -> None:
    respx.get(f"{BASE}/ssh-keys").mock(
        return_value=httpx.Response(200, json=[{"public_key": "ssh-ed25519 AAAA"}])
    )
    respx.get(f"{BASE}/templates").mock(
        return_value=httpx.Response(200, json=[{"id": "tpl-1", "name": "prism-worker"}])
    )
    respx.post(f"{BASE}/executors/exec-1/rent").mock(
        return_value=httpx.Response(200, json={"id": "pod-1"})
    )
    # The post-rent status poll fails mid-provision.
    respx.get(f"{BASE}/pods/pod-1").mock(return_value=httpx.Response(500))
    delete = respx.delete(f"{BASE}/pods/pod-1").mock(return_value=httpx.Response(200))
    pods = respx.get(f"{BASE}/pods").mock(return_value=httpx.Response(200, json=[]))

    with pytest.raises(LiumError):
        await LiumClient("k").provision(_spec(), offer=_offer())

    assert delete.call_count == 1
    assert pods.called  # verify_terminated polled GET /pods after the DELETE


# -- VAL-PROV-018 -------------------------------------------------------------


@respx.mock
async def test_ensure_helpers_idempotent_when_present() -> None:
    routes = _mock_happy_path()
    client = LiumClient("k")
    # Repeated planning runs never re-create an existing template/key.
    await client.provision(_spec(), offer=_offer())
    await client.provision(_spec(), offer=_offer())
    assert routes["ssh_post"].call_count == 0
    assert routes["tpl_post"].call_count == 0
    body = json.loads(routes["rent"].calls.last.request.content)
    assert body["template_id"] == "tpl-1"


@respx.mock
async def test_ensure_helpers_create_once_when_absent() -> None:
    routes = _mock_happy_path(template=[], keys=[])
    await LiumClient("k").provision(_spec(), offer=_offer())
    assert routes["ssh_post"].call_count == 1
    assert routes["tpl_post"].call_count == 1
    body = json.loads(routes["rent"].calls.last.request.content)
    assert body["template_id"] == "tpl-new"


@respx.mock
async def test_ensure_template_returns_existing_id() -> None:
    respx.get(f"{BASE}/templates").mock(
        return_value=httpx.Response(200, json=[{"id": "tpl-9", "name": "prism-worker"}])
    )
    post = respx.post(f"{BASE}/templates")
    result = await LiumClient("k").ensure_template(
        name="prism-worker", docker_image="img"
    )
    assert result == "tpl-9"
    assert post.call_count == 0


@respx.mock
async def test_ensure_ssh_key_creates_when_absent() -> None:
    respx.get(f"{BASE}/ssh-keys").mock(return_value=httpx.Response(200, json=[]))
    post = respx.post(f"{BASE}/ssh-keys").mock(
        return_value=httpx.Response(200, json={"id": "k-new", "public_key": "ssh x"})
    )
    result = await LiumClient("k").ensure_ssh_key(public_key="ssh x", name="deploy")
    assert result["id"] == "k-new"
    assert post.call_count == 1
    assert json.loads(post.calls.last.request.content)["public_key"] == "ssh x"


@respx.mock
async def test_ensure_template_body_pins_digest_tag_env_and_ports() -> None:
    respx.get(f"{BASE}/templates").mock(return_value=httpx.Response(200, json=[]))
    post = respx.post(f"{BASE}/templates").mock(
        return_value=httpx.Response(200, json={"id": "tpl-new"})
    )
    digest = "sha256:" + "d" * 64
    template_id = await LiumClient("k").ensure_template(
        name="prism-worker",
        docker_image="ghcr.io/base/worker",
        docker_image_digest=digest,
        docker_image_tag="v1",
        internal_ports=(22, 8080),
        environment={"ROLE": "worker"},
    )
    assert template_id == "tpl-new"
    body = json.loads(post.calls.last.request.content)
    assert body["docker_image_digest"] == digest
    assert body["docker_image_tag"] == "v1"
    assert body["environment"] == {"ROLE": "worker"}
    assert body["internal_ports"] == [22, 8080]
    assert body["is_private"] is True


# -- status / logs / balance -------------------------------------------------


@respx.mock
async def test_status_parses_pod_detail() -> None:
    respx.get(f"{BASE}/pods/pod-1").mock(
        return_value=httpx.Response(200, json={"id": "pod-1", "status": "RUNNING"})
    )
    instance = await LiumClient("k").status("pod-1")
    assert instance.id == "pod-1"
    assert instance.status == "RUNNING"
    assert instance.provider == "lium"


@respx.mock
async def test_stream_logs_yields_lines() -> None:
    respx.get(f"{BASE}/pods/pod-1/logs").mock(
        return_value=httpx.Response(200, text="line-1\nline-2\n")
    )
    lines = [line async for line in LiumClient("k").stream_logs("pod-1")]
    assert lines == ["line-1", "line-2"]


@respx.mock
async def test_stream_logs_raises_on_error() -> None:
    respx.get(f"{BASE}/pods/pod-1/logs").mock(return_value=httpx.Response(404))
    with pytest.raises(LiumError):
        async for _ in LiumClient("k").stream_logs("pod-1"):
            pass


@respx.mock
async def test_balance_returns_float() -> None:
    respx.get(f"{BASE}/users/me").mock(
        return_value=httpx.Response(200, json={"balance": 9.99})
    )
    assert await LiumClient("k").balance() == pytest.approx(9.99)


@respx.mock
async def test_request_error_raises_lium_error() -> None:
    respx.get(f"{BASE}/users/me").mock(return_value=httpx.Response(500))
    with pytest.raises(LiumError) as exc_info:
        await LiumClient("k").balance()
    assert exc_info.value.status_code == 500


# -- secret hygiene -----------------------------------------------------------


@respx.mock
async def test_api_key_sent_only_in_header() -> None:
    route = respx.get(f"{BASE}/users/me").mock(
        return_value=httpx.Response(200, json={"balance": 1.0})
    )
    await LiumClient("MY-SECRET").balance()
    assert route.calls.last.request.headers["X-API-Key"] == "MY-SECRET"


@respx.mock
async def test_api_key_never_in_repr_str_logs_or_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel = "SENTINEL-LIUM-KEY-XYZ"
    client = LiumClient(sentinel)
    assert sentinel not in repr(client)
    assert sentinel not in str(client)
    respx.get(f"{BASE}/users/me").mock(return_value=httpx.Response(500))
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(LiumError) as exc_info:
            await client.balance()
    assert sentinel not in str(exc_info.value)
    assert sentinel not in caplog.text


# -- provision fallback / cleanup edge paths ---------------------------------


@respx.mock
async def test_provision_falls_back_to_pod_lookup_by_name() -> None:
    respx.get(f"{BASE}/ssh-keys").mock(
        return_value=httpx.Response(200, json=[{"public_key": "ssh-ed25519 AAAA"}])
    )
    respx.get(f"{BASE}/templates").mock(
        return_value=httpx.Response(200, json=[{"id": "tpl-1", "name": "prism-worker"}])
    )
    # Rent response carries no id -> the client resolves it via GET /pods by name.
    respx.post(f"{BASE}/executors/exec-1/rent").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.get(f"{BASE}/pods").mock(
        return_value=httpx.Response(
            200, json=[{"id": "pod-7", "pod_name": "mission-pod"}]
        )
    )
    respx.get(f"{BASE}/pods/pod-7").mock(
        return_value=httpx.Response(200, json={"id": "pod-7", "status": "RUNNING"})
    )
    instance = await LiumClient("k").provision(_spec(), offer=_offer())
    assert instance.id == "pod-7"


@respx.mock
async def test_provision_raises_when_pod_id_undeterminable() -> None:
    respx.get(f"{BASE}/ssh-keys").mock(
        return_value=httpx.Response(200, json=[{"public_key": "ssh-ed25519 AAAA"}])
    )
    respx.get(f"{BASE}/templates").mock(
        return_value=httpx.Response(200, json=[{"id": "tpl-1", "name": "prism-worker"}])
    )
    respx.post(f"{BASE}/executors/exec-1/rent").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.get(f"{BASE}/pods").mock(return_value=httpx.Response(200, json=[]))
    with pytest.raises(LiumError):
        await LiumClient("k").provision(_spec(), offer=_offer())


@respx.mock
async def test_provision_cleanup_swallows_cleanup_errors_and_reraises() -> None:
    respx.get(f"{BASE}/ssh-keys").mock(
        return_value=httpx.Response(200, json=[{"public_key": "ssh-ed25519 AAAA"}])
    )
    respx.get(f"{BASE}/templates").mock(
        return_value=httpx.Response(200, json=[{"id": "tpl-1", "name": "prism-worker"}])
    )
    respx.post(f"{BASE}/executors/exec-1/rent").mock(
        return_value=httpx.Response(200, json={"id": "pod-1"})
    )
    respx.get(f"{BASE}/pods/pod-1").mock(return_value=httpx.Response(500))
    # Cleanup itself fails, but the original provisioning error must still surface.
    delete = respx.delete(f"{BASE}/pods/pod-1").mock(return_value=httpx.Response(500))
    respx.get(f"{BASE}/pods").mock(return_value=httpx.Response(500))
    with pytest.raises(LiumError):
        await LiumClient("k").provision(_spec(), offer=_offer())
    assert delete.call_count == 1


@respx.mock
async def test_provision_with_dockerfile_content_omits_template() -> None:
    respx.get(f"{BASE}/ssh-keys").mock(
        return_value=httpx.Response(200, json=[{"public_key": "ssh-ed25519 AAAA"}])
    )
    rent = respx.post(f"{BASE}/executors/exec-1/rent").mock(
        return_value=httpx.Response(200, json={"id": "pod-1", "status": "PENDING"})
    )
    respx.get(f"{BASE}/pods/pod-1").mock(
        return_value=httpx.Response(200, json={"id": "pod-1", "status": "RUNNING"})
    )
    templates = respx.get(f"{BASE}/templates")
    spec = InstanceSpec(
        name="mission-pod",
        template_ref=None,
        dockerfile_content="FROM ubuntu:22.04",
        ssh_public_keys=("ssh-ed25519 AAAA",),
        max_lifetime_hours=1,
        max_price_per_hour=1.5,
    )
    await LiumClient("k").provision(spec, offer=_offer())
    body = json.loads(rent.calls.last.request.content)
    assert body["dockerfile_content"] == "FROM ubuntu:22.04"
    assert "template_id" not in body
    assert templates.call_count == 0  # no template ensure for the dockerfile path


@respx.mock
async def test_provision_requires_template_or_dockerfile() -> None:
    respx.get(f"{BASE}/ssh-keys").mock(
        return_value=httpx.Response(200, json=[{"public_key": "ssh-ed25519 AAAA"}])
    )
    rent = respx.post(f"{BASE}/executors/exec-1/rent")
    spec = InstanceSpec(
        name="mission-pod",
        template_ref=None,
        dockerfile_content=None,
        ssh_public_keys=("ssh-ed25519 AAAA",),
        max_lifetime_hours=1,
        max_price_per_hour=1.5,
    )
    with pytest.raises(LiumError):
        await LiumClient("k").provision(spec, offer=_offer())
    assert rent.call_count == 0


@respx.mock
async def test_provision_requires_ssh_key() -> None:
    spec = InstanceSpec(
        name="mission-pod",
        template_ref="prism-worker",
        ssh_public_keys=(),
        max_lifetime_hours=1,
        max_price_per_hour=1.5,
    )
    with pytest.raises(LiumError):
        await LiumClient("k").provision(spec, offer=_offer())
    assert respx.calls.call_count == 0


@respx.mock
async def test_transport_error_wrapped_as_lium_error() -> None:
    respx.get(f"{BASE}/users/me").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(LiumError):
        await LiumClient("k").balance()


@respx.mock
async def test_balance_missing_field_raises() -> None:
    respx.get(f"{BASE}/users/me").mock(return_value=httpx.Response(200, json={}))
    with pytest.raises(LiumError):
        await LiumClient("k").balance()


@respx.mock
async def test_watchtower_digest_accepts_plain_string() -> None:
    respx.get(f"{BASE}/watchtower/digest").mock(
        return_value=httpx.Response(200, json="sha256:" + "c" * 64)
    )
    assert await LiumClient("k").watchtower_digest() == "sha256:" + "c" * 64


@respx.mock
async def test_list_offers_accepts_wrapped_payload() -> None:
    respx.get(f"{BASE}/executors").mock(
        return_value=httpx.Response(200, json={"executors": EXECUTORS})
    )
    offers = await LiumClient("k").list_offers(max_price_per_hour=1.0)
    assert {o.id for o in offers} == {"a", "c"}


# -- parsing helpers ----------------------------------------------------------


def test_extract_price_prefers_explicit_then_per_gpu() -> None:
    assert _extract_price({"price_per_hour": 1.5}) == 1.5
    assert _extract_price({"price_per_gpu": 0.9}) == 0.9
    assert _extract_price({"pending_price_per_hour": 0.7}) == 0.7
    assert _extract_price({}) is None
    assert _extract_price({"price_per_gpu": "not-a-number"}) is None


def test_extract_gpu_type_falls_back_to_specs_details() -> None:
    item = {"specs": {"gpu": {"details": [{"name": "NVIDIA H100"}]}}}
    assert _extract_gpu_type(item) == "NVIDIA H100"
    assert _extract_gpu_type({}) == ""


def test_extract_gpu_count_falls_back_to_specs_count() -> None:
    assert _extract_gpu_count({"gpu_count": 4}) == 4
    assert _extract_gpu_count({"specs": {"gpu": {"count": 2}}}) == 2
    assert _extract_gpu_count({}) == 0
    assert _extract_gpu_count({"gpu_count": "x"}) == 0


def test_parse_offer_skips_items_without_id_or_price() -> None:
    assert _parse_offer({"machine_name": "H100", "gpu_count": 1}) is None
    parsed = _parse_offer({"id": "z", "price_per_gpu": 1.0})
    assert parsed is not None
    assert parsed.id == "z"


def test_parse_instance_rejects_non_mapping() -> None:
    with pytest.raises(LiumError):
        _parse_instance(["not", "a", "mapping"])


def test_as_list_handles_unexpected_shapes() -> None:
    assert _as_list("nope", "executors") == []
    assert _as_list({"executors": "nope"}, "executors") == []
    assert _as_list([{"a": 1}, "skip"], "executors") == [{"a": 1}]
