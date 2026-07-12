from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import APIRouter
from fastapi.routing import APIRoute
from pydantic import ValidationError
from typer.main import get_command

from base.challenge_sdk.api_manifest import API_MANIFEST, API_MANIFEST_DIGEST
from base.challenge_sdk.app_factory import create_challenge_app
from base.challenge_sdk.config import ChallengeSettings
from base.challenge_sdk.roles import (
    CAPABILITY_REGISTRY_VERSION,
    ROLE_REGISTRY,
    Capability,
    Role,
    RoleContractError,
    activate_role,
    capabilities_for_role,
    role_contract,
)
from base.challenge_sdk.schemas import (
    AssignmentView,
    RawWeightPushRequest,
    VersionResponse,
)
from base.cli_app.main import app as cli_app
from base.master.orchestration import MasterChallengeReconciler
from base.master.service import MasterWeightService
from base.validator.weight_submitter import ValidatorWeightSubmitter


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
    assert ("POST", "/internal/v1/work_units/result") in API_MANIFEST.route_keys()
    assert (
        "POST",
        "/internal/v1/challenges/{slug}/raw-weights",
    ) in API_MANIFEST.route_keys()
    assert "base master weights" in API_MANIFEST.cli_names()


def test_api_manifest_matches_live_challenge_fastapi_surface(tmp_path) -> None:
    class _Database:
        async def init(self) -> None:
            return None

        async def close(self) -> None:
            return None

    secret = tmp_path / "token"
    secret.write_text("test-shared-token", encoding="utf-8")
    app = create_challenge_app(
        settings=ChallengeSettings(
            shared_token=None,
            shared_token_file=str(secret),
        ),
        database=_Database(),
        public_router=APIRouter(),
        get_weights_fn=_async_weights,
    )
    live_routes: set[tuple[str, str]] = set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods or ():
            if method in {"HEAD", "OPTIONS"}:
                continue
            live_routes.add((method, route.path))
    # Canonical factory-owned challenge routes from the release-owned manifest
    # must exist on the live FastAPI surface. Target raw-weight + external result
    # are inventory ownership now; result is attached by Prism and raw-weight
    # ingress is owned by the weights milestone runtime wiring.
    required_live = {
        ("GET", "/health"),
        ("GET", "/ready"),
        ("GET", "/version"),
        ("GET", "/internal/v1/get_weights"),
    }
    assert required_live.issubset(live_routes)
    inventory = API_MANIFEST.route_keys()
    assert required_live.issubset(inventory)
    assert ("POST", "/internal/v1/work_units/result") in inventory
    assert ("POST", "/internal/v1/challenges/{slug}/raw-weights") in inventory


def test_api_manifest_cli_commands_exist_on_typer_surface() -> None:
    command = get_command(cli_app)

    def _names(cmd: object, prefix: str = "base") -> set[str]:
        found: set[str] = set()
        commands = getattr(cmd, "commands", None) or {}
        for name, child in commands.items():
            full = f"{prefix} {name}"
            found.add(full)
            found |= _names(child, full)
        return found

    live_cli = _names(command)
    for name in API_MANIFEST.cli_names():
        assert name in live_cli, f"manifest CLI missing from live Typer: {name}"


async def _async_weights() -> dict[str, float]:
    return {"5Ctest": 1.0}


async def test_production_side_effect_entrypoints_are_role_gated() -> None:
    """Capability-bearing side effects fail before mock effects under wrong role."""

    score_calls: list[str] = []

    @role_contract(
        role=Role.CHALLENGE, capability=Capability.CHALLENGE_ORDINARY_PROOF
    )
    async def challenge_result_ingest(value: str) -> str:
        score_calls.append(value)
        return value

    service = MasterWeightService(metagraph_cache=object())  # type: ignore[arg-type]
    reconciler = MasterChallengeReconciler(
        registry=object(),  # type: ignore[arg-type]
        orchestrator=object(),  # type: ignore[arg-type]
    )
    submitter = ValidatorWeightSubmitter(
        submit_enabled=True,
        netuid=1,
        weights_client=object(),  # type: ignore[arg-type]
        weight_setter_factory=lambda: None,  # type: ignore[arg-type, return-value]
    )

    assert hasattr(service.compute_weights, "__base_role_contract__")
    assert hasattr(service.collect_weights, "__base_role_contract__")
    assert hasattr(reconciler.reconcile_once, "__base_role_contract__")
    assert hasattr(submitter.run_once, "__base_role_contract__")
    assert hasattr(challenge_result_ingest, "__base_role_contract__")

    with activate_role(Role.VALIDATOR):
        with pytest.raises(RoleContractError, match="master"):
            await service.compute_weights([], {})
        with pytest.raises(RoleContractError, match="master"):
            await reconciler.reconcile_once()
        with pytest.raises(RoleContractError, match="challenge"):
            await challenge_result_ingest("scored")

    with activate_role(Role.MASTER):
        with pytest.raises(RoleContractError, match="validator"):
            await submitter.run_once()
        with pytest.raises(RoleContractError, match="challenge"):
            await challenge_result_ingest("scored")

    assert score_calls == []
    with activate_role(Role.CHALLENGE):
        assert await challenge_result_ingest("ok") == "ok"
    assert score_calls == ["ok"]
