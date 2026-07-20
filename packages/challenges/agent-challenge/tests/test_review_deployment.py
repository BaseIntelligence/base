"""Offline contract tests for the isolated encrypted attested-review deployment.

These tests intentionally exercise only deterministic local seams.  They do
not claim that a CVM was deployed or that encrypted-env confinement was
observed on hardware.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from agent_challenge.core.models import AgentSubmission
from agent_challenge.review import compose as review_compose
from agent_challenge.review.build import validate_review_build_definition
from agent_challenge.review.canonical import canonical_sha256
from agent_challenge.review.deployment import (
    ReviewDeploymentError,
    build_review_deployed_acknowledgement,
    review_input_config_from_settings,
    validate_review_deployed_acknowledgement,
)
from agent_challenge.review.schemas import ReviewInputConfig, build_review_assignment
from agent_challenge.review.sessions import (
    ReviewConflict,
    create_review_session,
    mark_review_deployed,
)
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.selfdeploy.review import (
    REVIEW_ALLOWED_ENVS,
    ReviewPhalaDeployment,
    build_review_deployment_plan,
    encrypt_review_secrets,
)

REVIEW_IMAGE = "docker.io/example/agent-challenge-review@sha256:" + ("a" * 64)
EVAL_IMAGE = "docker.io/example/agent-challenge-canonical@sha256:" + ("b" * 64)
MEASUREMENT = {
    "mrtd": "01" * 48,
    "rtmr0": "02" * 48,
    "rtmr1": "03" * 48,
    "rtmr2": "04" * 48,
    "os_image_hash": "05" * 32,
    "key_provider": "phala",
    "vm_shape": "tdx.small",
}
EVAL_MEASUREMENT = {
    "mrtd": "11" * 48,
    "rtmr0": "12" * 48,
    "rtmr1": "13" * 48,
    "rtmr2": "14" * 48,
    "compose_hash": "15" * 32,
    "os_image_hash": "16" * 32,
}


def _review_compose() -> dict[str, object]:
    return review_compose.generate_review_app_compose(
        review_image=REVIEW_IMAGE,
        app_identity="agent-challenge-review-v1",
    )


def _allowlisted(compose_hash: str | None = None) -> dict[str, str]:
    return {
        "mrtd": MEASUREMENT["mrtd"],
        "rtmr0": MEASUREMENT["rtmr0"],
        "rtmr1": MEASUREMENT["rtmr1"],
        "rtmr2": MEASUREMENT["rtmr2"],
        "compose_hash": compose_hash or review_compose.review_app_compose_hash(_review_compose()),
        "os_image_hash": MEASUREMENT["os_image_hash"],
    }


def _review_config(public_key_hex: str) -> ReviewInputConfig:
    compose_hash = review_compose.review_app_compose_hash(_review_compose())
    entries = (_allowlisted(compose_hash),)
    return ReviewInputConfig(
        image_ref=REVIEW_IMAGE,
        compose_hash=compose_hash,
        app_identity="agent-challenge-review-v1",
        kms_public_key_hex=public_key_hex,
        measurement=MEASUREMENT,
        measurement_allowlist=entries,
        measurement_allowlist_sha256=canonical_sha256({"entries": list(entries)}),
    )


def _assignment(*, public_key_hex: str) -> tuple[dict[str, object], str]:
    config = _review_config(public_key_hex)
    token = "review-session-token-sentinel"
    assignment, _bytes, _digest = build_review_assignment(
        session_id="rs-review",
        assignment_id="ra-review",
        attempt=1,
        submission_id="17",
        artifact={
            "agent_hash": "10" * 32,
            "zip_sha256": "20" * 32,
            "zip_size_bytes": 1,
            "manifest_sha256": "30" * 32,
            "manifest_entries_sha256": "40" * 32,
            "fetch_path": "/review/v1/assignments/ra-review/artifact",
        },
        rules_snapshot_sha256_value="50" * 32,
        rules_revision_id="rules-v1",
        review_nonce="rn-review",
        issued_at_ms=1,
        expires_at_ms=2,
        session_token_sha256=hashlib.sha256(token.encode()).hexdigest(),
        config=config,
    )
    return assignment, token


def _decrypt_ciphertext(ciphertext: str, private_key: X25519PrivateKey) -> dict[str, object]:
    raw = bytes.fromhex(ciphertext)
    ephemeral = raw[:32]
    nonce = raw[32:44]
    encrypted = raw[44:]
    shared = private_key.exchange(X25519PublicKey.from_public_bytes(ephemeral))
    return json.loads(AESGCM(shared).decrypt(nonce, encrypted, None))


def _test_settings(public_key_hex: str) -> ChallengeSettings:
    compose_hash = review_compose.review_app_compose_hash(_review_compose())
    allowlisted = _allowlisted(compose_hash)
    return ChallengeSettings(
        attested_review_enabled=True,
        phala_attestation_enabled=True,
        review_app_image_ref=REVIEW_IMAGE,
        review_app_compose_hash=compose_hash,
        review_app_identity="agent-challenge-review-v1",
        review_app_kms_public_key_hex=public_key_hex,
        review_app_measurement=MEASUREMENT,
        review_app_measurement_allowlist=(allowlisted,),
        eval_app_image_ref=EVAL_IMAGE,
        eval_app_compose_hash=EVAL_MEASUREMENT["compose_hash"],
        eval_app_identity="agent-challenge-eval-v1",
        eval_app_kms_public_key_hex="e" * 64,
        eval_app_measurement_allowlist=(EVAL_MEASUREMENT,),
    )


def test_review_compose_is_deterministic_digest_pinned_and_capability_confined() -> None:
    first = _review_compose()
    second = _review_compose()

    assert review_compose.render_review_app_compose(
        first
    ) == review_compose.render_review_app_compose(second)
    assert review_compose.review_app_compose_hash(first) == review_compose.review_app_compose_hash(
        second
    )
    assert first["name"] == "agent-challenge-review-v1"
    assert first["allowed_envs"] == [
        "OPENROUTER_API_KEY",
        "REVIEW_API_BASE_URL",
        "REVIEW_SESSION_TOKEN",
    ]
    assert first["gateway_enabled"] is False
    assert first["public_logs"] is True
    assert first["public_sysinfo"] is False

    service = yaml.safe_load(str(first["docker_compose_file"]))["services"]["reviewer"]
    assert set(service) == review_compose.REVIEWER_SERVICE_KEYS
    assert service["image"] == REVIEW_IMAGE
    assert service["environment"] == [
        "OPENROUTER_API_KEY",
        "REVIEW_API_BASE_URL",
        "REVIEW_SESSION_TOKEN",
    ]
    assert service["volumes"] == ["/var/run/dstack.sock:/var/run/dstack.sock:ro"]
    inventory = json.dumps(first, sort_keys=True)
    for forbidden in (
        "docker.sock",
        "golden",
        "task-cache",
        "BASE_LLM_GATEWAY_URL",
        "BASE_GATEWAY_TOKEN",
        "KEY_RELEASE",
        "EVAL_RUN",
        "weight",
    ):
        assert forbidden not in inventory


@pytest.mark.parametrize(
    "extra_key",
    [
        "privileged",
        "cap_add",
        "devices",
        "network_mode",
        "ports",
        "secrets",
        "configs",
        "pid",
        "ipc",
        "security_opt",
    ],
)
def test_review_compose_rejects_unexpected_service_capability_keys(extra_key: str) -> None:
    compose = _review_compose()
    services = yaml.safe_load(compose["docker_compose_file"])
    services["services"]["reviewer"][extra_key] = True if extra_key != "cap_add" else ["SYS_ADMIN"]
    if extra_key == "ports":
        services["services"]["reviewer"][extra_key] = ["8080:8080"]
    if extra_key == "devices":
        services["services"]["reviewer"][extra_key] = ["/dev/null:/dev/null"]
    if extra_key in {"secrets", "configs"}:
        services["services"]["reviewer"][extra_key] = ["evil"]
    if extra_key in {"pid", "ipc", "network_mode"}:
        services["services"]["reviewer"][extra_key] = "host"
    forged = dict(compose)
    # Intentionally inject unauthorized keys into the measured service inventory.
    forged["docker_compose_file"] = yaml.safe_dump(services, sort_keys=False)
    with pytest.raises(review_compose.ReviewComposeError, match="schema-closed"):
        review_compose.validate_review_app_compose(forged)


def test_review_build_definition_is_separate_and_contains_only_review_runtime() -> None:
    definition = review_compose.review_build_definition()
    assert definition.dockerfile.is_file()
    assert definition.requirements.is_file()
    assert definition.dockerfile != review_compose.eval_build_definition().dockerfile

    dockerfile = definition.dockerfile.read_text(encoding="utf-8")
    assert "@sha256:" in dockerfile
    assert "review_runtime.py" in dockerfile
    assert "review/openrouter.py" in dockerfile
    assert "review/policy.py" in dockerfile
    for forbidden in (
        "COPY golden",
        "COPY src/agent_challenge/evaluation",
        "own_runner",
        "Dockerfile",
    ):
        assert forbidden not in dockerfile
    assert validate_review_build_definition().digest_pinned


def test_review_runtime_only_calls_bounded_get_quote_and_never_executes_input() -> None:
    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    class QuoteClient:
        def __init__(self) -> None:
            self.report_data: list[bytes] = []

        def get_quote(self, report_data: bytes) -> object:
            self.report_data.append(report_data)
            return type("Quote", (), {"quote": "beef", "event_log": [], "vm_config": {}})()

    client = QuoteClient()
    assert runtime._quote("ab" * 64, client=client)["quote"] == "beef"
    assert client.report_data == [bytes.fromhex("ab" * 64)]

    source = runtime_path.read_text(encoding="utf-8")
    for forbidden in ("subprocess", "exec(", "eval(", "get_key(", "extend_rtmr"):
        assert forbidden not in source


def test_validator_configuration_binds_allowlist_and_rejects_shared_identities() -> None:
    private_key = X25519PrivateKey.generate()
    public_key_hex = (
        private_key.public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        .hex()
    )
    config = review_input_config_from_settings(_test_settings(public_key_hex))

    assert config.image_ref == REVIEW_IMAGE
    assert config.app_identity == "agent-challenge-review-v1"
    assert config.kms_public_key_hex == public_key_hex
    assert config.resolved_measurement() == MEASUREMENT
    assert config.measurement_allowlist
    assert config.measurement_allowlist_sha256 == canonical_sha256(
        {"entries": list(config.measurement_allowlist)}
    )
    assignment, _ = _assignment(public_key_hex=public_key_hex)
    review_app = assignment["assignment_core"]["review_app"]
    assert review_app["measurement_allowlist_sha256"] == config.measurement_allowlist_sha256
    assert review_app["measurement_allowlist"] == list(config.measurement_allowlist)

    shared_allowlist = _allowlisted()
    with pytest.raises(ReviewDeploymentError, match="disjoint"):
        review_input_config_from_settings(
            _test_settings(public_key_hex).model_copy(
                update={"eval_app_measurement_allowlist": (shared_allowlist,)}
            )
        )
    with pytest.raises(ReviewDeploymentError, match="app identities must be disjoint"):
        review_input_config_from_settings(
            _test_settings(public_key_hex).model_copy(
                update={"eval_app_identity": "agent-challenge-review-v1"}
            )
        )
    with pytest.raises(ReviewDeploymentError, match="image refs must be disjoint"):
        review_input_config_from_settings(
            _test_settings(public_key_hex).model_copy(update={"eval_app_image_ref": REVIEW_IMAGE})
        )
    with pytest.raises(ReviewDeploymentError, match="compose hashes must be disjoint"):
        review_input_config_from_settings(
            _test_settings(public_key_hex).model_copy(
                update={"eval_app_compose_hash": config.compose_hash}
            )
        )
    with pytest.raises(ReviewDeploymentError, match="KMS public keys must be disjoint"):
        review_input_config_from_settings(
            _test_settings(public_key_hex).model_copy(
                update={"eval_app_kms_public_key_hex": public_key_hex}
            )
        )


def test_review_deployment_encrypts_and_transmits_only_exact_secret_names() -> None:
    private_key = X25519PrivateKey.generate()
    public_key_hex = (
        private_key.public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        .hex()
    )
    assignment, token = _assignment(public_key_hex=public_key_hex)
    plan = build_review_deployment_plan(
        {
            "assignment": assignment,
            "review_session_token": token,
        }
    )
    sentinel_key = "review-openrouter-secret-sentinel"
    encrypted = encrypt_review_secrets(
        plan,
        {
            "OPENROUTER_API_KEY": sentinel_key,
            "REVIEW_API_BASE_URL": "https://chain.joinbase.ai/challenges/agent-challenge",
            "REVIEW_SESSION_TOKEN": token,
        },
    )
    payload = _decrypt_ciphertext(encrypted.ciphertext, private_key)

    assert payload == {
        "env": [
            {"key": "OPENROUTER_API_KEY", "value": sentinel_key},
            {
                "key": "REVIEW_API_BASE_URL",
                "value": "https://chain.joinbase.ai/challenges/agent-challenge",
            },
            {"key": "REVIEW_SESSION_TOKEN", "value": token},
        ]
    }
    assert encrypted.env_keys == REVIEW_ALLOWED_ENVS
    assert (
        encrypted.measurement_allowlist_sha256
        == assignment["assignment_core"]["review_app"]["measurement_allowlist_sha256"]
    )
    assert sentinel_key not in repr(plan)
    assert sentinel_key not in repr(encrypted)

    deployment = ReviewPhalaDeployment(
        provision_response={
            "app_id": "agent-challenge-review-v1",
            "compose_hash": plan.compose_hash,
            "app_env_encrypt_pubkey": public_key_hex,
            "os_image_hash": MEASUREMENT["os_image_hash"],
        },
        create_response={
            "id": "cvm-review-1",
            "request_id": "req-review-1",
            "created_at_ms": 1000,
            "receipt": "created",
        },
    )
    acknowledgement = deployment.deploy(plan, encrypted)
    assert deployment.provision_requests == [
        {
            "app_id": "agent-challenge-review-v1",
            "name": "agent-challenge-review-v1",
            "instance_type": "tdx.small",
            "region": "us-west-1",
            "compose_file": plan.compose,
            "env_keys": ["OPENROUTER_API_KEY", "REVIEW_API_BASE_URL", "REVIEW_SESSION_TOKEN"],
            "image": "dstack-0.5.9",
        }
    ]
    create_request = deployment.create_requests[0]
    assert create_request["app_id"] == "agent-challenge-review-v1"
    assert create_request["compose_hash"] == plan.compose_hash
    assert create_request["env_keys"] == [
        "OPENROUTER_API_KEY",
        "REVIEW_API_BASE_URL",
        "REVIEW_SESSION_TOKEN",
    ]
    assert create_request["encrypted_env"] == encrypted.ciphertext
    assert not {"env", "environment", "args", "files"} & set(create_request)
    assert sentinel_key not in json.dumps(create_request)
    assert set(acknowledgement) == {
        "schema_version",
        "assignment_id",
        "cvm_id",
        "phala_create_receipt",
        "compose_identity",
    }
    assert acknowledgement["schema_version"] == 1
    assert acknowledgement["compose_identity"] == {
        "image_ref": assignment["assignment_core"]["review_app"]["image_ref"],
        "compose_hash": assignment["assignment_core"]["review_app"]["compose_hash"],
        "app_kms_public_key_sha256": assignment["assignment_core"]["review_app"][
            "kms_public_key_sha256"
        ],
    }
    assert acknowledgement["phala_create_receipt"]["cvm_id"] == "cvm-review-1"
    assert acknowledgement["phala_create_receipt"]["app_id"] == "agent-challenge-review-v1"

    second = ReviewPhalaDeployment(
        provision_response={
            "app_id": "agent-challenge-review-v1",
            "compose_hash": plan.compose_hash,
            "app_env_encrypt_pubkey": public_key_hex,
        },
        create_response={"id": "cvm-review-2", "request_id": "req-2", "created_at_ms": 1},
    )
    with pytest.raises(ReviewDeploymentError, match="bound"):
        second.deploy(plan, replace(encrypted, assignment_id="ra-other"))
    assert second.provision_requests == []
    assert second.create_requests == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda assignment: assignment["assignment_core"]["review_app"].update(
            {"image_ref": EVAL_IMAGE}
        ),
        lambda assignment: assignment["assignment_core"]["review_app"].update(
            {"compose_hash": "f" * 64}
        ),
        lambda assignment: assignment["assignment_core"]["review_app"].update(
            {"app_identity": "evil-review-app"}
        ),
        lambda assignment: assignment["assignment_core"]["review_app"].update(
            {"kms_public_key_hex": "e" * 64}
        ),
        lambda assignment: assignment["assignment_core"]["review_app"].update(
            {"measurement_allowlist_sha256": "a" * 64}
        ),
    ],
)
def test_bad_assignment_identity_or_missing_or_extra_secret_fails_before_create(mutate) -> None:
    private_key = X25519PrivateKey.generate()
    public_key_hex = (
        private_key.public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        .hex()
    )
    assignment, token = _assignment(public_key_hex=public_key_hex)
    tampered = copy.deepcopy(assignment)
    mutate(tampered)

    with pytest.raises(ReviewDeploymentError):
        build_review_deployment_plan({"assignment": tampered, "review_session_token": token})

    plan = build_review_deployment_plan({"assignment": assignment, "review_session_token": token})
    with pytest.raises(ReviewDeploymentError, match="non-empty"):
        encrypt_review_secrets(
            plan,
            {
                "OPENROUTER_API_KEY": "",
                "REVIEW_API_BASE_URL": "https://chain.joinbase.ai/challenges/agent-challenge",
                "REVIEW_SESSION_TOKEN": token,
            },
        )
    with pytest.raises(ReviewDeploymentError, match="exactly"):
        encrypt_review_secrets(
            plan,
            {
                "OPENROUTER_API_KEY": "key",
                "REVIEW_API_BASE_URL": "https://chain.joinbase.ai/challenges/agent-challenge",
                "REVIEW_SESSION_TOKEN": token,
                "UNTRUSTED_EXTRA": "no",
            },
        )


async def test_nested_deployed_acknowledgement_is_bound_before_running_transition(
    database_session,
) -> None:
    private_key = X25519PrivateKey.generate()
    public_key_hex = (
        private_key.public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        .hex()
    )
    config = _review_config(public_key_hex)
    submission_bytes = b"review-deployment-artifact"
    submission = AgentSubmission(
        miner_hotkey="review-miner",
        name="review-agent",
        agent_hash=hashlib.sha256(submission_bytes).hexdigest(),
        artifact_uri="/tmp/review-deployment.zip",
        artifact_path="/tmp/review-deployment.zip",
        zip_sha256=hashlib.sha256(submission_bytes).hexdigest(),
        zip_size_bytes=len(submission_bytes),
        raw_status="review_queued",
        effective_status="queued",
    )
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=submission_bytes,
            rules_files={".rules/policy.md": b"review"},
            rules_revision_id="rules-v1",
            settings=ChallengeSettings(shared_token="review-token"),
            input_config=config,
            now=datetime(2026, 7, 10, tzinfo=UTC),
        )
        assignment = json.loads(created.assignment.assignment_bytes)
        acknowledgement = build_review_deployed_acknowledgement(
            assignment=assignment,
            cvm_id="cvm-review-1",
            request_id="req-review-1",
            receipt_sha256="6" * 64,
            created_at_ms=1_000,
        )
        validate_review_deployed_acknowledgement(assignment, acknowledgement)
        deployed = await mark_review_deployed(
            session,
            session_row=created.session,
            expected_assignment_id=created.assignment.assignment_id,
            deployed_receipt=acknowledgement,
            now=datetime(2026, 7, 10, 0, 0, 1, tzinfo=UTC),
        )
        assert deployed.phase == "review_cvm_running"

        flat_legacy = {
            "assignment_id": created.assignment.assignment_id,
            "phala_create_receipt_sha256": "6" * 64,
            "cvm_id": "cvm-review-1",
            "app_identity": assignment["assignment_core"]["review_app"]["app_identity"],
            "image_ref": assignment["assignment_core"]["review_app"]["image_ref"],
            "compose_hash": assignment["assignment_core"]["review_app"]["compose_hash"],
            "kms_public_key_sha256": assignment["assignment_core"]["review_app"][
                "kms_public_key_sha256"
            ],
        }
        with pytest.raises(ReviewDeploymentError, match="schema-closed"):
            validate_review_deployed_acknowledgement(assignment, flat_legacy)

        changed = copy.deepcopy(acknowledgement)
        changed["compose_identity"]["compose_hash"] = "7" * 64
        with pytest.raises(ReviewConflict):
            await mark_review_deployed(
                session,
                session_row=created.session,
                expected_assignment_id=created.assignment.assignment_id,
                deployed_receipt=changed,
                now=datetime(2026, 7, 10, 0, 0, 2, tzinfo=UTC),
            )
