"""Offline respx tests for :class:`base.compute.targon.TargonClient`.

Every test mocks Targon HTTP via respx; no credentials and no real network are
required. These pin the provider contract assertions VAL-PROV-002/003/006/007/008
for the Targon client.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from base.compute import (
    BalanceUnavailableError,
    CostGuardrailError,
    Instance,
    InstanceSpec,
    InsufficientCreditsError,
    ProviderError,
    TargonClient,
    TargonError,
)
from base.compute.targon import (
    _as_list,
    _is_insufficient_credits,
    _parse_inventory_offer,
    _parse_workload_instance,
)

BASE = "https://api.targon.com/tha/v2"

# GET /inventory?type=rental&gpu=true payload shaped like the real API: a bare
# list of fixed GPU shapes carrying cost_per_hour + availability, mixing zero and
# non-zero availability and prices above/below a $3.00/GPU/hr bound.
INVENTORY = [
    {
        "name": "h100",
        "display_name": "H100 x1",
        "type": "rental",
        "gpu": True,
        "spec": {"gpu_type": "H100", "gpu_count": 1},
        "cost_per_hour": 2.5,
        "available": 4,
    },
    {
        "name": "h200",
        "display_name": "H200 x1",
        "type": "rental",
        "gpu": True,
        "spec": {"gpu_type": "H200", "gpu_count": 1},
        "cost_per_hour": 3.29,
        "available": 2,
    },
    {
        "name": "h100x8",
        "display_name": "H100 x8",
        "type": "rental",
        "gpu": True,
        "spec": {"gpu_type": "H100", "gpu_count": 8},
        "cost_per_hour": 20.0,
        "available": 0,
    },
]


def _spec(**overrides: object) -> InstanceSpec:
    base: dict[str, object] = {
        "name": "mission-workload",
        "template_ref": "h100",
        "image": "ghcr.io/base/worker",
        "ssh_public_keys": ("ssh-ed25519 AAAA",),
        "max_lifetime_hours": 1,
        "max_price_per_hour": 3.0,
    }
    base.update(overrides)
    return InstanceSpec(**base)  # type: ignore[arg-type]


# -- VAL-PROV-002 -------------------------------------------------------------


@respx.mock
async def test_list_offers_excludes_zero_availability_and_over_price() -> None:
    respx.get(f"{BASE}/inventory").mock(
        return_value=httpx.Response(200, json=INVENTORY)
    )
    offers = await TargonClient("k").list_offers(max_price_per_hour=3.0)
    assert {o.id for o in offers} == {"h100"}
    only = offers[0]
    assert only.price_per_hour == pytest.approx(2.5)
    assert isinstance(only.price_per_hour, float)
    assert only.gpu_type == "H100"
    assert only.gpu_count == 1
    assert only.provider == "targon"
    assert only.raw["cost_per_hour"] == pytest.approx(2.5)


@respx.mock
async def test_list_offers_without_bound_excludes_only_zero_availability() -> None:
    respx.get(f"{BASE}/inventory").mock(
        return_value=httpx.Response(200, json=INVENTORY)
    )
    offers = await TargonClient("k").list_offers()
    assert {o.id for o in offers} == {"h100", "h200"}
    for offer in offers:
        assert offer.price_per_hour > 0


@respx.mock
async def test_list_offers_sends_rental_gpu_query_params() -> None:
    route = respx.get(f"{BASE}/inventory").mock(
        return_value=httpx.Response(200, json=INVENTORY)
    )
    await TargonClient("k").list_offers()
    request = route.calls.last.request
    assert request.url.params["type"] == "rental"
    assert request.url.params["gpu"] == "true"


@respx.mock
async def test_list_offers_accepts_wrapped_payload() -> None:
    respx.get(f"{BASE}/inventory").mock(
        return_value=httpx.Response(200, json={"items": INVENTORY})
    )
    offers = await TargonClient("k").list_offers()
    assert {o.id for o in offers} == {"h100", "h200"}


@respx.mock
async def test_list_offers_normalizes_price_to_per_gpu() -> None:
    inventory = [
        {
            "name": "h100x4",
            "display_name": "H100 x4",
            "type": "rental",
            "gpu": True,
            "spec": {"gpu_type": "H100", "gpu_count": 4},
            "cost_per_hour": 10.0,
            "available": 2,
        }
    ]
    respx.get(f"{BASE}/inventory").mock(
        return_value=httpx.Response(200, json=inventory)
    )
    offers = await TargonClient("k").list_offers()
    assert len(offers) == 1
    offer = offers[0]
    # Whole-shape cost is 10.0 for 4 GPUs -> per-GPU price is 2.5.
    assert offer.price_per_hour == pytest.approx(2.5)
    assert offer.gpu_count == 4
    assert offer.raw["cost_per_hour"] == pytest.approx(10.0)


@respx.mock
async def test_multi_gpu_offer_filtered_by_per_gpu_max_price() -> None:
    inventory = [
        {
            "name": "h100x4",
            "spec": {"gpu_type": "H100", "gpu_count": 4},
            "cost_per_hour": 10.0,
            "available": 2,
        },
        {
            "name": "h200x8",
            "spec": {"gpu_type": "H200", "gpu_count": 8},
            "cost_per_hour": 40.0,
            "available": 2,
        },
    ]
    respx.get(f"{BASE}/inventory").mock(
        return_value=httpx.Response(200, json=inventory)
    )
    # Per-GPU: h100x4 -> 2.5/gpu (kept), h200x8 -> 5.0/gpu (dropped by 3.0 cap).
    # A whole-shape comparison (10 and 40) would have wrongly dropped BOTH.
    offers = await TargonClient("k").list_offers(max_price_per_hour=3.0)
    assert {o.id for o in offers} == {"h100x4"}
    assert offers[0].price_per_hour == pytest.approx(2.5)


@respx.mock
async def test_multi_gpu_offer_count_from_numeric_field_filters_correctly() -> None:
    inventory = [
        {
            "name": "h100x8",
            # ``spec`` is present but carries no usable gpu_count; the count lives
            # ONLY in the top-level numeric field. Per-GPU price must still be
            # cost/count so the per-GPU cap filters the shape correctly.
            "spec": {"gpu_type": "H100", "gpu_count": 0},
            "gpu_count": 8,
            "cost_per_hour": 16.0,
            "available": 2,
        }
    ]
    respx.get(f"{BASE}/inventory").mock(
        return_value=httpx.Response(200, json=inventory)
    )
    # Per-GPU: 16.0 / 8 = 2.0 (within the 3.0/GPU cap). A whole-shape fallback
    # (16.0 > 3.0) would have WRONGLY dropped this in-budget multi-GPU shape.
    offers = await TargonClient("k").list_offers(max_price_per_hour=3.0)
    assert len(offers) == 1
    assert offers[0].gpu_count == 8
    assert offers[0].price_per_hour == pytest.approx(2.0)


# -- VAL-PROV-003 (provision guardrails, no network) --------------------------


@respx.mock
@pytest.mark.parametrize(
    "overrides",
    [
        {"max_lifetime_hours": None},
        {"max_lifetime_hours": 0},
        {"max_lifetime_hours": -1},
        {"max_lifetime_hours": 0.5},
        {"max_price_per_hour": None},
        {"max_price_per_hour": 0},
    ],
)
async def test_provision_refuses_unbounded_spec_without_network(
    overrides: dict[str, object],
) -> None:
    create = respx.post(f"{BASE}/workloads")
    with pytest.raises(CostGuardrailError):
        await TargonClient("k").provision(_spec(**overrides))
    assert create.call_count == 0
    assert respx.calls.call_count == 0


async def test_provision_sub_hour_lifetime_message_mentions_truncation() -> None:
    with pytest.raises(CostGuardrailError) as exc_info:
        await TargonClient("k").provision(_spec(max_lifetime_hours=0.5))
    message = str(exc_info.value)
    assert "at least 1 hour" in message
    assert "termination_hours" in message


# -- deploy call shape (two-step create-then-deploy) --------------------------


@respx.mock
async def test_provision_creates_then_deploys_with_workload_body() -> None:
    create = respx.post(f"{BASE}/workloads").mock(
        return_value=httpx.Response(200, json={"uid": "wl-1"})
    )
    deploy = respx.post(f"{BASE}/workloads/wl-1/deploy").mock(
        return_value=httpx.Response(
            200, json={"uid": "wl-1", "state": {"status": "PENDING"}}
        )
    )
    instance = await TargonClient("k").provision(_spec(max_lifetime_hours=2))
    assert isinstance(instance, Instance)
    assert instance.id == "wl-1"
    assert instance.status == "PENDING"
    assert instance.provider == "targon"
    assert create.call_count == 1
    assert deploy.call_count == 1
    body = json.loads(create.calls.last.request.content)
    assert body["name"] == "mission-workload"
    assert body["type"] == "RENTAL"
    assert body["resource_name"] == "h100"
    assert body["image"] == "ghcr.io/base/worker"
    assert body["termination_hours"] == 2
    assert body["envs"] == [] or isinstance(body["envs"], list)
    # The deploy step carries no body (matches the SDK / live route).
    assert not deploy.calls.last.request.content


@respx.mock
async def test_deploy_uses_create_uid_when_deploy_response_omits_it() -> None:
    respx.post(f"{BASE}/workloads").mock(
        return_value=httpx.Response(200, json={"uid": "wl-42"})
    )
    respx.post(f"{BASE}/workloads/wl-42/deploy").mock(
        return_value=httpx.Response(200, json={"state": {"status": "DEPLOYING"}})
    )
    instance = await TargonClient("k").deploy({"name": "custom"})
    assert instance.id == "wl-42"
    assert instance.status == "DEPLOYING"


@respx.mock
async def test_deploy_passes_explicit_payload_verbatim_to_create() -> None:
    create = respx.post(f"{BASE}/workloads").mock(
        return_value=httpx.Response(200, json={"uid": "wl-9"})
    )
    respx.post(f"{BASE}/workloads/wl-9/deploy").mock(
        return_value=httpx.Response(200, json={"uid": "wl-9", "status": "RUNNING"})
    )
    payload = {"name": "custom", "resource_name": "h200", "image": "img"}
    instance = await TargonClient("k").deploy(payload)
    assert instance.id == "wl-9"
    assert instance.status == "RUNNING"
    assert json.loads(create.calls.last.request.content) == payload


@respx.mock
async def test_deploy_raises_when_create_returns_no_uid() -> None:
    create = respx.post(f"{BASE}/workloads").mock(
        return_value=httpx.Response(200, json={"state": {"status": "PENDING"}})
    )
    deploy = respx.post(f"{BASE}/workloads//deploy")
    with pytest.raises(TargonError):
        await TargonClient("k").deploy({"name": "x"})
    assert create.call_count == 1
    assert deploy.call_count == 0


@respx.mock
async def test_provision_includes_env_and_ports() -> None:
    create = respx.post(f"{BASE}/workloads").mock(
        return_value=httpx.Response(200, json={"uid": "wl-1"})
    )
    respx.post(f"{BASE}/workloads/wl-1/deploy").mock(
        return_value=httpx.Response(200, json={"uid": "wl-1"})
    )
    spec = _spec(env={"ROLE": "worker"}, ports=(22, 8080))
    await TargonClient("k").provision(spec)
    body = json.loads(create.calls.last.request.content)
    assert {"name": "ROLE", "value": "worker"} in body["envs"]
    assert body["ports"] == [{"port": 22}, {"port": 8080}]
    assert body["ssh_public_keys"] == ["ssh-ed25519 AAAA"]


# -- VAL-PROV-007 (insufficient credits, typed, no retry) ---------------------


@respx.mock
async def test_deploy_402_on_create_raises_insufficient_credits_no_retry() -> None:
    create = respx.post(f"{BASE}/workloads").mock(
        return_value=httpx.Response(402, json={"error": "payment required"})
    )
    deploy = respx.post(url__regex=rf"{BASE}/workloads/.+/deploy")
    with pytest.raises(InsufficientCreditsError) as exc_info:
        await TargonClient("k").provision(_spec())
    assert isinstance(exc_info.value, ProviderError)
    assert isinstance(exc_info.value, TargonError)
    assert create.call_count == 1
    # A credit failure on create never proceeds to (or retries) the deploy step.
    assert deploy.call_count == 0


@respx.mock
async def test_deploy_402_on_deploy_step_raises_insufficient_credits_no_retry() -> None:
    create = respx.post(f"{BASE}/workloads").mock(
        return_value=httpx.Response(200, json={"uid": "wl-1"})
    )
    deploy = respx.post(f"{BASE}/workloads/wl-1/deploy").mock(
        return_value=httpx.Response(402, json={"error": "insufficient credits"})
    )
    with pytest.raises(InsufficientCreditsError):
        await TargonClient("k").provision(_spec())
    assert create.call_count == 1
    assert deploy.call_count == 1


@respx.mock
async def test_deploy_403_raises_insufficient_credits_no_retry() -> None:
    create = respx.post(f"{BASE}/workloads").mock(
        return_value=httpx.Response(403, text="forbidden")
    )
    with pytest.raises(InsufficientCreditsError):
        await TargonClient("k").deploy({"name": "x"})
    assert create.call_count == 1


@respx.mock
async def test_deploy_credit_body_raises_insufficient_credits() -> None:
    create = respx.post(f"{BASE}/workloads").mock(
        return_value=httpx.Response(400, json={"error": "insufficient credits"})
    )
    with pytest.raises(InsufficientCreditsError):
        await TargonClient("k").deploy({"name": "x"})
    assert create.call_count == 1


@respx.mock
async def test_deploy_generic_error_raises_targon_error_not_credits() -> None:
    create = respx.post(f"{BASE}/workloads").mock(
        return_value=httpx.Response(500, text="boom")
    )
    with pytest.raises(TargonError) as exc_info:
        await TargonClient("k").deploy({"name": "x"})
    assert not isinstance(exc_info.value, InsufficientCreditsError)
    assert create.call_count == 1


# -- VAL-PROV-008 (balance unavailable, typed, zero HTTP) ---------------------


@respx.mock
async def test_balance_raises_typed_error_with_no_http() -> None:
    with pytest.raises(BalanceUnavailableError) as exc_info:
        await TargonClient("k").balance()
    assert isinstance(exc_info.value, ProviderError)
    assert isinstance(exc_info.value, TargonError)
    assert respx.calls.call_count == 0


# -- apps / workloads listing -------------------------------------------------


@respx.mock
async def test_list_apps_returns_items() -> None:
    respx.get(f"{BASE}/apps").mock(
        return_value=httpx.Response(200, json={"items": [{"uid": "app-1"}]})
    )
    apps = await TargonClient("k").list_apps()
    assert apps == [{"uid": "app-1"}]


@respx.mock
async def test_list_apps_treats_410_gone_as_empty() -> None:
    """Targon retired GET /apps with 410 Gone; inventory still works via /workloads.

    Live smoke (BASE_LIVE_PROVIDER_TESTS Lium path) calls list_apps in a shared
    read-only preflight. Raising on 410 blocked Lium rent entirely; empty list is
    the correct retired-endpoint semantics (no invent, no retry cascade).
    """
    route = respx.get(f"{BASE}/apps").mock(
        return_value=httpx.Response(410, text="Gone")
    )
    apps = await TargonClient("k").list_apps()
    assert apps == []
    assert route.call_count == 1


@respx.mock
async def test_list_apps_still_raises_on_non_gone_errors() -> None:
    respx.get(f"{BASE}/apps").mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(TargonError) as exc_info:
        await TargonClient("k").list_apps()
    assert exc_info.value.status_code == 500


@respx.mock
async def test_create_app_posts_name() -> None:
    route = respx.post(f"{BASE}/apps").mock(
        return_value=httpx.Response(200, json={"uid": "app-2", "name": "worker"})
    )
    result = await TargonClient("k").create_app("worker")
    assert result["uid"] == "app-2"
    assert json.loads(route.calls.last.request.content)["name"] == "worker"


@respx.mock
async def test_list_workloads_returns_items() -> None:
    respx.get(f"{BASE}/workloads").mock(
        return_value=httpx.Response(200, json={"items": [{"uid": "wl-1"}]})
    )
    workloads = await TargonClient("k").list_workloads()
    assert workloads == [{"uid": "wl-1"}]


@respx.mock
async def test_status_parses_workload_detail() -> None:
    respx.get(f"{BASE}/workloads/wl-1").mock(
        return_value=httpx.Response(
            200, json={"uid": "wl-1", "state": {"status": "RUNNING"}}
        )
    )
    instance = await TargonClient("k").status("wl-1")
    assert instance.id == "wl-1"
    assert instance.status == "RUNNING"


@respx.mock
async def test_workload_state_and_events() -> None:
    respx.get(f"{BASE}/workloads/wl-1/state").mock(
        return_value=httpx.Response(200, json={"uid": "wl-1", "status": "RUNNING"})
    )
    respx.get(f"{BASE}/workloads/wl-1/events").mock(
        return_value=httpx.Response(200, json={"items": [{"event_type": "created"}]})
    )
    client = TargonClient("k")
    state = await client.workload_state("wl-1")
    events = await client.workload_events("wl-1")
    assert state["status"] == "RUNNING"
    assert events["items"][0]["event_type"] == "created"


# -- logs ---------------------------------------------------------------------


@respx.mock
async def test_stream_logs_yields_lines() -> None:
    respx.get(f"{BASE}/workloads/wl-1/logs").mock(
        return_value=httpx.Response(200, text="line-1\nline-2\n")
    )
    lines = [line async for line in TargonClient("k").stream_logs("wl-1")]
    assert lines == ["line-1", "line-2"]


@respx.mock
async def test_stream_logs_raises_on_error() -> None:
    respx.get(f"{BASE}/workloads/wl-1/logs").mock(return_value=httpx.Response(404))
    with pytest.raises(TargonError):
        async for _ in TargonClient("k").stream_logs("wl-1"):
            pass


# -- terminate / verify_terminated -------------------------------------------


@respx.mock
async def test_terminate_is_idempotent() -> None:
    route = respx.delete(f"{BASE}/workloads/wl-1").mock(
        side_effect=[httpx.Response(200), httpx.Response(404)]
    )
    client = TargonClient("k")
    await client.terminate("wl-1")
    await client.terminate("wl-1")
    assert route.call_count == 2


@respx.mock
async def test_terminate_raises_on_non_404_error() -> None:
    respx.delete(f"{BASE}/workloads/wl-1").mock(return_value=httpx.Response(500))
    with pytest.raises(TargonError):
        await TargonClient("k").terminate("wl-1")


@respx.mock
async def test_verify_terminated_reflects_absence() -> None:
    respx.get(f"{BASE}/workloads/wl-1").mock(
        side_effect=[
            httpx.Response(200, json={"uid": "wl-1", "state": {"status": "RUNNING"}}),
            httpx.Response(404),
        ]
    )
    client = TargonClient("k")
    assert await client.verify_terminated("wl-1") is False
    assert await client.verify_terminated("wl-1") is True


@respx.mock
async def test_verify_terminated_true_for_deleted_status() -> None:
    respx.get(f"{BASE}/workloads/wl-1").mock(
        return_value=httpx.Response(
            200, json={"uid": "wl-1", "state": {"status": "deleted"}}
        )
    )
    assert await TargonClient("k").verify_terminated("wl-1") is True


@respx.mock
async def test_verify_terminated_raises_on_non_404_error() -> None:
    respx.get(f"{BASE}/workloads/wl-1").mock(return_value=httpx.Response(500))
    with pytest.raises(TargonError):
        await TargonClient("k").verify_terminated("wl-1")


@respx.mock
async def test_create_app_sends_project_id() -> None:
    route = respx.post(f"{BASE}/apps").mock(
        return_value=httpx.Response(200, json={"uid": "app-3"})
    )
    await TargonClient("k").create_app("worker", project_id="proj-1")
    assert json.loads(route.calls.last.request.content)["project_id"] == "proj-1"


@respx.mock
async def test_workload_state_and_create_app_tolerate_non_dict() -> None:
    respx.get(f"{BASE}/workloads/wl-1/state").mock(
        return_value=httpx.Response(200, json=["unexpected"])
    )
    respx.post(f"{BASE}/apps").mock(
        return_value=httpx.Response(200, json=["unexpected"])
    )
    client = TargonClient("k")
    assert await client.workload_state("wl-1") == {}
    assert await client.create_app("worker") == {}


# -- transport / parsing edge paths ------------------------------------------


@respx.mock
async def test_request_error_raises_targon_error() -> None:
    respx.get(f"{BASE}/workloads").mock(return_value=httpx.Response(500))
    with pytest.raises(TargonError) as exc_info:
        await TargonClient("k").list_workloads()
    assert exc_info.value.status_code == 500


@respx.mock
async def test_transport_error_wrapped_as_targon_error() -> None:
    respx.get(f"{BASE}/workloads").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(TargonError):
        await TargonClient("k").list_workloads()


@respx.mock
async def test_api_key_sent_only_in_authorization_header() -> None:
    route = respx.get(f"{BASE}/workloads").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    await TargonClient("MY-SECRET").list_workloads()
    assert route.calls.last.request.headers["Authorization"] == "Bearer MY-SECRET"


def test_is_insufficient_credits_classifies_codes_and_bodies() -> None:
    assert _is_insufficient_credits(402, "") is True
    assert _is_insufficient_credits(403, "") is True
    assert _is_insufficient_credits(400, "insufficient credits") is True
    assert _is_insufficient_credits(400, "no credit left") is True
    assert _is_insufficient_credits(500, "boom") is False
    assert _is_insufficient_credits(401, "unauthorized") is False


def test_parse_inventory_offer_skips_zero_and_missing() -> None:
    zero = {"name": "x", "available": 0, "cost_per_hour": 1}
    assert _parse_inventory_offer(zero) is None
    assert _parse_inventory_offer({"available": 1, "cost_per_hour": 1}) is None
    assert _parse_inventory_offer({"name": "x", "available": 1}) is None
    offer = _parse_inventory_offer(
        {"name": "x", "available": 2, "cost_per_hour": "1.5"}
    )
    assert offer is not None
    assert offer.price_per_hour == pytest.approx(1.5)


def test_parse_inventory_offer_normalizes_price_per_gpu() -> None:
    multi = _parse_inventory_offer(
        {
            "name": "h100x8",
            "spec": {"gpu_count": 8},
            "cost_per_hour": 20.0,
            "available": 3,
        }
    )
    assert multi is not None
    assert multi.price_per_hour == pytest.approx(2.5)
    assert multi.gpu_count == 8
    # Missing gpu_count falls back to the whole-shape cost (no ZeroDivisionError).
    no_count = _parse_inventory_offer(
        {"name": "x", "cost_per_hour": 1.5, "available": 1}
    )
    assert no_count is not None
    assert no_count.price_per_hour == pytest.approx(1.5)
    assert no_count.gpu_count == 0


def test_parse_workload_instance_rejects_non_mapping() -> None:
    with pytest.raises(TargonError):
        _parse_workload_instance(["not", "a", "mapping"])


def test_as_list_handles_shapes() -> None:
    assert _as_list("nope", "items") == []
    assert _as_list({"items": "nope"}, "items") == []
    assert _as_list([{"a": 1}, "skip"], "items") == [{"a": 1}]
    assert _as_list({"items": [{"a": 1}]}, "items") == [{"a": 1}]
