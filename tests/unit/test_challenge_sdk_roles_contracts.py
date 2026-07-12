from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from base.challenge_sdk.api_manifest import API_MANIFEST, API_MANIFEST_DIGEST
from base.challenge_sdk.roles import (
    CAPABILITY_REGISTRY_VERSION,
    ROLE_REGISTRY,
    Capability,
    Role,
    activate_role,
    capabilities_for_role,
    role_contract,
)
from base.challenge_sdk.schemas import (
    AssignmentView,
    RawWeightPushRequest,
    VersionResponse,
)


def test_registry_has_exact_roles_and_capabilities() -> None:
    assert {role.value for role in Role} == {
        "master",
        "validator",
        "challenge",
        "worker",
    }
    assert {capability.value for capability in Capability} == {
        "master.coordination",
        "master.registry",
        "master.persistence",
        "master.raw_weight_ingress",
        "master.aggregation",
        "master.vector_read",
        "master.watcher",
        "validator.registration",
        "validator.heartbeat",
        "validator.assignment_pull",
        "validator.assignment_progress",
        "validator.assignment_result",
        "validator.vector_read",
        "validator.own_set_weights",
        "challenge.scoring",
        "challenge.ordinary_proof",
        "challenge.tee_verification",
        "challenge.state",
        "challenge.raw_weight_push",
        "worker.assignment_execution",
        "worker.result_reporting",
    }
    assert ROLE_REGISTRY.version == CAPABILITY_REGISTRY_VERSION
    assert "validator.own_set_weights" not in {
        item.token for item in ROLE_REGISTRY.for_role(Role.MASTER)
    }
    assert "challenge.tee_verification" not in {
        item for item in capabilities_for_role(Role.CHALLENGE, tee_verification=False)
    }
    with pytest.raises(TypeError):
        ROLE_REGISTRY.capabilities[Capability.MASTER_COORDINATION] = (  # type: ignore[index]
            ROLE_REGISTRY.capabilities[Capability.MASTER_COORDINATION]
        )


def test_role_contract_metadata_and_runtime_guard() -> None:
    @role_contract(role=Role.VALIDATOR, capability=Capability.VALIDATOR_VECTOR_READ)
    def read_vector() -> str:
        return "ok"

    with activate_role(Role.VALIDATOR):
        assert read_vector() == "ok"
    metadata = read_vector.__base_role_contract__  # type: ignore[attr-defined]
    assert metadata["role"] == Role.VALIDATOR
    assert metadata["capability"] == Capability.VALIDATOR_VECTOR_READ.value


async def test_async_role_contract_preserves_coroutine_surface() -> None:
    @role_contract(
        role=Role.VALIDATOR,
        capability=Capability.VALIDATOR_VECTOR_READ,
    )
    async def read_vector_async() -> str:
        return "ok"

    assert inspect.iscoroutinefunction(read_vector_async)
    with activate_role(Role.VALIDATOR):
        assert await read_vector_async() == "ok"


def test_strict_assignment_schema_rejects_coercion_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AssignmentView.model_validate(
            {
                "api_version": "1.0",
                "assignment_id": "a-1",
                "work_unit_id": "u-1",
                "submission_ref": "s-1",
                "challenge_slug": "prism",
                "payload": {},
                "payload_digest": "0" * 64,
                "required_capability": "gpu",
                "revision": 1,
                "attempt": 1,
                "status": "assigned",
                "lease_deadline": None,
                "checkpoint_ref": None,
                "unknown": True,
            }
        )
    with pytest.raises(ValidationError):
        AssignmentView.model_validate(
            {
                "api_version": "1.0",
                "assignment_id": "a-1",
                "work_unit_id": "u-1",
                "submission_ref": "s-1",
                "challenge_slug": "prism",
                "payload": {},
                "payload_digest": "0" * 64,
                "required_capability": "gpu",
                "revision": True,
                "attempt": 1,
                "status": "assigned",
                "lease_deadline": None,
                "checkpoint_ref": None,
            }
        )


def test_raw_weight_digest_and_strict_values() -> None:
    computed_at = datetime.now(UTC).replace(microsecond=0)
    expires_at = computed_at + timedelta(minutes=5)
    body = {
        "protocol_version": "1.0",
        "challenge_slug": "prism",
        "epoch": 42,
        "revision": 1,
        "computed_at": computed_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "nonce": "n-1",
        "weights": {"5CkeyABC": 1.5},
    }
    digest = RawWeightPushRequest.compute_digest(body)
    payload = RawWeightPushRequest.model_validate({**body, "payload_digest": digest})
    assert payload.payload_digest == digest
    assert json.loads(payload.canonical_bytes())["weights"] == {"5CkeyABC": 1.5}
    with pytest.raises(ValidationError):
        RawWeightPushRequest.model_validate(
            {
                **body,
                "payload_digest": digest,
                "weights": {"5CkeyABC": True},
            }
        )


def test_version_response_rejects_caller_capability_claims() -> None:
    response = VersionResponse(
        distribution_name="base",
        artifact_version="3.1.2",
        release_id="v3.1.2",
        api_version="1.0",
        challenge_slug="prism",
        challenge_version="0.1.0",
        sdk_contract_version="1.0.0",
        sdk_version="1.0.0",
        role="challenge",
        capabilities=capabilities_for_role(Role.CHALLENGE),
    )
    assert response.capabilities == capabilities_for_role(Role.CHALLENGE)
    with pytest.raises(ValidationError):
        VersionResponse.model_validate({**response.model_dump(), "x_role": "master"})


def test_api_manifest_is_immutable_and_self_describing() -> None:
    assert API_MANIFEST.version == "1"
    assert API_MANIFEST_DIGEST == API_MANIFEST.digest()
    assert API_MANIFEST.routes and API_MANIFEST.cli
