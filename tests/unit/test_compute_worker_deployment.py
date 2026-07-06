"""Offline tests for the worker deployment definitions (VAL-PROV-009/010/016).

These pin the well-formedness of the declarative deploy definitions the miner's
worker image ships as:

* the Lium ``CustomTemplateRequest`` payload (VAL-PROV-009), and
* the Targon app definition (VAL-PROV-010),

each pinning the docker image BY DIGEST, plus an explicit assertion that the whole
compute package makes NO real network calls under respx strict mode and needs no
provider credentials (VAL-PROV-016).
"""

from __future__ import annotations

import json
import re

import httpx
import pytest
import respx

from base.compute import (
    LiumClient,
    TargonClient,
    build_lium_worker_template,
    build_targon_worker_app,
    pinned_image_reference,
)
from base.compute.worker_deployment import (
    WORKER_IMAGE,
    WORKER_IMAGE_DIGEST,
    WORKER_INTERNAL_PORTS,
    is_pinned_digest,
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

# Substrings that would indicate a leaked credential or secret in a definition.
_SECRET_MARKERS = (
    "X-API-Key",
    "Authorization",
    "Bearer",
    "LIUM_API_KEY",
    "TARGON_API_KEY",
    "SENTINEL",
    "password",
    "secret",
)


def _assert_no_secrets(definition: object) -> None:
    blob = json.dumps(definition)
    for marker in _SECRET_MARKERS:
        assert marker not in blob


# -- shared placeholder digest ------------------------------------------------


def test_placeholder_image_pinned_by_digest() -> None:
    assert WORKER_IMAGE == "ghcr.io/baseintelligence/prism-evaluator"
    assert _DIGEST_RE.match(WORKER_IMAGE_DIGEST)
    assert is_pinned_digest(WORKER_IMAGE_DIGEST) is True
    assert is_pinned_digest("sha256:short") is False
    assert is_pinned_digest("latest") is False


def test_pinned_image_reference_is_fully_qualified_and_immutable() -> None:
    ref = pinned_image_reference(WORKER_IMAGE, WORKER_IMAGE_DIGEST, tag="latest")
    assert ref == f"{WORKER_IMAGE}:latest@{WORKER_IMAGE_DIGEST}"
    assert f"@{WORKER_IMAGE_DIGEST}" in ref
    # Tag is optional; the digest pin is what makes the reference immutable.
    ref_no_tag = pinned_image_reference(WORKER_IMAGE, WORKER_IMAGE_DIGEST)
    assert ref_no_tag == f"{WORKER_IMAGE}@{WORKER_IMAGE_DIGEST}"


def test_pinned_image_reference_rejects_unpinned_digest() -> None:
    with pytest.raises(ValueError):
        pinned_image_reference(WORKER_IMAGE, "latest")


# -- VAL-PROV-009 : Lium worker template --------------------------------------


def test_lium_template_is_well_formed_and_pins_digest() -> None:
    template = build_lium_worker_template()
    assert template["name"]
    assert template["docker_image"] == WORKER_IMAGE
    assert _DIGEST_RE.match(template["docker_image_digest"])
    assert 22 in template["internal_ports"]
    assert template["is_private"] is True
    _assert_no_secrets(template)


def test_lium_template_digest_is_sha256_shaped() -> None:
    template = build_lium_worker_template()
    assert _DIGEST_RE.match(template["docker_image_digest"]) is not None


def test_lium_template_plumbs_environment() -> None:
    template = build_lium_worker_template(
        environment={"BASE_MASTER_URL": "http://master:8000", "WORKER_ROLE": "gpu"}
    )
    assert template["environment"]["BASE_MASTER_URL"] == "http://master:8000"
    assert template["environment"]["WORKER_ROLE"] == "gpu"


def test_lium_template_default_environment_is_empty_dict() -> None:
    template = build_lium_worker_template()
    assert template["environment"] == {}


def test_lium_template_accepts_swapped_image_and_digest() -> None:
    digest = "sha256:" + "a" * 64
    template = build_lium_worker_template(
        image="ghcr.io/baseintelligence/base-worker",
        image_digest=digest,
        image_tag="v2",
    )
    assert template["docker_image"] == "ghcr.io/baseintelligence/base-worker"
    assert template["docker_image_digest"] == digest
    assert template["docker_image_tag"] == "v2"


def test_lium_template_custom_ports_still_include_ssh() -> None:
    template = build_lium_worker_template(internal_ports=(22, 8082))
    assert template["internal_ports"] == [22, 8082]


def test_lium_template_rejects_ports_without_ssh() -> None:
    with pytest.raises(ValueError):
        build_lium_worker_template(internal_ports=(8082,))


def test_lium_template_rejects_malformed_digest() -> None:
    with pytest.raises(ValueError):
        build_lium_worker_template(image_digest="not-a-digest")


# -- VAL-PROV-010 : Targon app definition -------------------------------------


def test_targon_app_is_well_formed_and_references_pinned_image() -> None:
    app = build_targon_worker_app()
    assert app["name"]
    # Fully qualified image reference with an immutable digest pin.
    assert app["image"].startswith(f"{WORKER_IMAGE}:")
    assert f"@{WORKER_IMAGE_DIGEST}" in app["image"]
    assert _DIGEST_RE.match(app["image_digest"]) is not None
    _assert_no_secrets(app)


def test_targon_app_declares_gpu_resource_shape() -> None:
    app = build_targon_worker_app()
    assert app["resource"]
    assert app["gpu_type"]
    assert isinstance(app["gpu_count"], int)
    assert app["gpu_count"] >= 1


def test_targon_app_default_gpu_shape_is_live_valid_suffixed_id() -> None:
    # Real Targon inventory ids carry a size suffix (h100-small, b200-large per
    # library/targon-api.md); a bare 'h100' would be rejected by a live deploy.
    from base.compute.worker_deployment import WORKER_GPU_SHAPE

    assert WORKER_GPU_SHAPE == "h100-small"
    assert build_targon_worker_app()["resource"] == "h100-small"


def test_targon_app_plumbs_environment_as_name_value_pairs() -> None:
    app = build_targon_worker_app(environment={"BASE_MASTER_URL": "http://master:8000"})
    assert {"name": "BASE_MASTER_URL", "value": "http://master:8000"} in app["envs"]


def test_targon_app_default_environment_is_empty_list() -> None:
    app = build_targon_worker_app()
    assert app["envs"] == []


def test_targon_app_ports_include_ssh() -> None:
    app = build_targon_worker_app()
    assert 22 in app["ports"]


def test_targon_app_accepts_swapped_image_and_digest() -> None:
    digest = "sha256:" + "b" * 64
    app = build_targon_worker_app(
        image="ghcr.io/baseintelligence/base-worker",
        image_digest=digest,
        image_tag="v2",
        gpu_shape="h200",
        gpu_type="H200",
        gpu_count=2,
    )
    assert app["image"] == f"ghcr.io/baseintelligence/base-worker:v2@{digest}"
    assert app["image_digest"] == digest
    assert app["resource"] == "h200"
    assert app["gpu_type"] == "H200"
    assert app["gpu_count"] == 2


def test_targon_app_rejects_malformed_digest() -> None:
    with pytest.raises(ValueError):
        build_targon_worker_app(image_digest="nope")


def test_targon_app_rejects_non_positive_gpu_count() -> None:
    with pytest.raises(ValueError):
        build_targon_worker_app(gpu_count=0)


# -- VAL-PROV-016 : offline, no credentials, respx strict mode ----------------


@respx.mock(assert_all_mocked=True)
async def test_respx_strict_mode_blocks_unmocked_request() -> None:
    # No route is registered: a real call would egress to lium.io. Under respx
    # strict mode the request is refused BEFORE leaving the process, proving the
    # compute suite performs zero real network I/O.
    with pytest.raises(AssertionError):
        await LiumClient("k").balance()
    assert respx.calls.call_count == 0


@respx.mock(assert_all_mocked=True)
async def test_targon_respx_strict_mode_blocks_unmocked_request() -> None:
    with pytest.raises(AssertionError):
        await TargonClient("k").list_workloads()
    assert respx.calls.call_count == 0


@respx.mock
async def test_clients_need_no_provider_credentials_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LIUM_API_KEY", raising=False)
    monkeypatch.delenv("TARGON_API_KEY", raising=False)
    monkeypatch.delenv("BASE_LIVE_PROVIDER_TESTS", raising=False)
    respx.get("https://lium.io/api/users/me").mock(
        return_value=httpx.Response(200, json={"balance": 1.0})
    )
    respx.get("https://api.targon.com/tha/v2/workloads").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    assert await LiumClient("dummy").balance() == pytest.approx(1.0)
    assert await TargonClient("dummy").list_workloads() == []


def test_default_internal_ports_include_ssh() -> None:
    assert 22 in WORKER_INTERNAL_PORTS
