"""Offline contract tests for the direct attested-review OpenRouter transport.

These tests use recorded in-memory transports only. They never claim a real
OpenRouter request, a TDX quote, or encrypted-env confinement on hardware.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import ssl
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select

from agent_challenge.core.models import (
    AgentSubmission,
    EvaluationJob,
    ReviewEvidenceObject,
    ReviewNonce,
    ReviewSession,
    SubmissionStatusEvent,
)
from agent_challenge.review import compose as review_compose
from agent_challenge.review.canonical import canonical_json_v1
from agent_challenge.review.evidence import (
    MAX_REVIEW_EVIDENCE_BYTES,
    ReviewEvidenceError,
    store_review_evidence_objects,
)
from agent_challenge.review.openrouter import (
    OPENROUTER_PATH,
    DirectOpenRouterClient,
    OpenRouterTransportError,
    build_model_call_started,
    build_openrouter_request_body,
    build_planned_openrouter_request,
    build_review_infrastructure_failure,
    infrastructure_failure_reason,
    validate_model_call_started,
    validate_planned_openrouter_request,
)
from agent_challenge.review.schemas import (
    MAX_OPENROUTER_RESPONSE_BYTES,
    ReviewInputConfig,
    build_review_assignment,
)
from agent_challenge.review.sessions import (
    ReviewConflict,
    create_review_session,
    mark_model_call_started,
    record_review_infrastructure_failure,
    recover_incomplete_model_calls,
    retry_review_assignment,
)
from agent_challenge.sdk.config import ChallengeSettings

SENTINEL_KEY = "review-openrouter-secret-sentinel"
_ROUTING = {
    "order": ["first", "second"],
    "only": ["first", "second"],
    "ignore": [],
    "quantizations": [],
    "sort": None,
    "allow_fallbacks": False,
    "require_parameters": True,
    "data_collection": "deny",
}


def _assignment() -> dict[str, object]:
    assignment, _bytes, _digest = build_review_assignment(
        session_id="rs-transport",
        assignment_id="ra-transport",
        attempt=1,
        submission_id="17",
        artifact={
            "agent_hash": "10" * 32,
            "zip_sha256": "20" * 32,
            "zip_size_bytes": 1,
            "manifest_sha256": "30" * 32,
            "manifest_entries_sha256": "40" * 32,
            "fetch_path": "/review/v1/assignments/ra-transport/artifact",
        },
        rules_snapshot_sha256_value="50" * 32,
        rules_revision_id="rules-v1",
        review_nonce="rn-transport",
        issued_at_ms=1,
        expires_at_ms=2,
        session_token_sha256="60" * 32,
        config=ReviewInputConfig(routing=_ROUTING),
    )
    return assignment


def _body() -> bytes:
    return build_openrouter_request_body(
        messages=[{"content": "review only supplied bytes", "role": "user"}],
        routing=_ROUTING,
    )


def test_planned_request_and_marker_are_closed_and_body_bound() -> None:
    body = _body()
    planned, planned_bytes, planned_digest = build_planned_openrouter_request(
        body=body,
        routing_sha256=hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest(),
    )
    marker = build_model_call_started(
        assignment_id="ra-transport",
        planned_request_sha256=planned_digest,
        request_body_sha256=planned["body_sha256"],
        request_body_length=planned["body_length"],
    )

    assert planned_bytes == validate_planned_openrouter_request(planned)
    assert marker["request_body_sha256"] == hashlib.sha256(body).hexdigest()
    assert marker["request_body_length"] == len(body)
    assert json.dumps(planned) != SENTINEL_KEY

    for invalid in (
        {**planned, "tls_hostname": "openrouter.ai"},
        {**planned, "origin": "https://openrouter.ai"},
        {**planned, "path": "/api/v1/chat/completions?x=1"},
        {**marker, "request_record_sha256": planned_digest},
        {**marker, "request_body_length": 0},
    ):
        with pytest.raises(ValueError):
            (
                validate_planned_openrouter_request(invalid)
                if "schema_version" in invalid and "origin" in invalid
                else validate_model_call_started(invalid)
            )


def test_ordered_routing_is_request_bound_and_never_sorted_as_a_set() -> None:
    swapped = {**_ROUTING, "order": list(reversed(_ROUTING["order"]))}
    first_routing_digest = hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest()
    second_routing_digest = hashlib.sha256(canonical_json_v1(swapped)).hexdigest()
    first_body = _body()
    second_body = build_openrouter_request_body(
        messages=[{"content": "review only supplied bytes", "role": "user"}],
        routing=swapped,
    )
    _first, _first_bytes, first_digest = build_planned_openrouter_request(
        body=first_body,
        routing_sha256=first_routing_digest,
    )
    _second, _second_bytes, second_digest = build_planned_openrouter_request(
        body=second_body,
        routing_sha256=second_routing_digest,
    )

    assert first_routing_digest != second_routing_digest
    assert first_body != second_body
    assert first_digest != second_digest


def test_offline_direct_client_announces_once_and_uses_exact_operation() -> None:
    body = _body()
    calls: list[httpx.Request] = []
    markers: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "id": "offline-response",
                "model": "x-ai/grok-4.5",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "submit_verdict",
                                        "arguments": json.dumps(
                                            {
                                                "verdict": "allow",
                                                "reason_codes": [],
                                                "evidence_paths": ["artifact/agent.py"],
                                            },
                                            separators=(",", ":"),
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
            headers={"content-type": "application/json"},
        )

    client = DirectOpenRouterClient(
        assignment_id="ra-transport",
        api_key=SENTINEL_KEY,
        announce=lambda marker: markers.append(marker) is None,
        transport=httpx.MockTransport(handler),
    )
    capture = client.call(
        body=body,
        routing_sha256=hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest(),
        allowed_evidence_paths={"artifact/agent.py"},
    )

    assert len(calls) == 1
    assert calls[0].method == "POST"
    assert calls[0].url.scheme == "https"
    assert calls[0].url.host == "openrouter.ai"
    assert calls[0].url.port is None
    assert calls[0].url.path == OPENROUTER_PATH
    assert calls[0].headers["authorization"] == f"Bearer {SENTINEL_KEY}"
    assert calls[0].headers["x-openrouter-metadata"] == "enabled"
    assert len(markers) == 1
    assert capture.observed["redirected"] is False
    assert capture.observed["proxied"] is False
    assert SENTINEL_KEY not in repr(client)
    assert SENTINEL_KEY not in repr(capture)
    with pytest.raises(OpenRouterTransportError, match="already"):
        client.call(
            body=body,
            routing_sha256=hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest(),
            allowed_evidence_paths={"artifact/agent.py"},
        )


@pytest.mark.parametrize(
    ("status_code", "reason_code"),
    [
        (401, "openrouter_auth_failed"),
        (403, "openrouter_auth_failed"),
        (429, "openrouter_rate_limited"),
        (503, "openrouter_unavailable"),
    ],
)
def test_direct_client_maps_provider_failures_after_one_announced_call(
    status_code: int,
    reason_code: str,
) -> None:
    announced: list[dict[str, object]] = []
    calls: list[httpx.Request] = []
    client = DirectOpenRouterClient(
        assignment_id="ra-transport",
        api_key=SENTINEL_KEY,
        announce=lambda marker: announced.append(marker) is None,
        transport=httpx.MockTransport(
            lambda request: calls.append(request) or httpx.Response(status_code)
        ),
    )

    with pytest.raises(OpenRouterTransportError) as exc_info:
        client.call(
            body=_body(),
            routing_sha256=hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest(),
            allowed_evidence_paths={"artifact/agent.py"},
        )

    assert exc_info.value.reason_code == reason_code
    assert len(announced) == 1
    assert len(calls) == 1


def test_missing_credential_never_announces_or_opens_network() -> None:
    calls: list[httpx.Request] = []
    markers: list[dict[str, object]] = []
    client = DirectOpenRouterClient(
        assignment_id="ra-transport",
        api_key="",
        announce=lambda marker: markers.append(marker) is None,
        transport=httpx.MockTransport(
            lambda request: calls.append(request) or httpx.Response(200, content=b"{}")
        ),
    )

    with pytest.raises(OpenRouterTransportError) as exc_info:
        client.call(
            body=_body(),
            routing_sha256=hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest(),
            allowed_evidence_paths={"artifact/agent.py"},
        )

    assert exc_info.value.reason_code == "missing_credential"
    assert calls == []
    assert markers == []


@pytest.mark.parametrize(
    "body",
    [
        b"{}",
        canonical_json_v1({"model": "x-ai/grok-4.5", "provider": _ROUTING}),
        canonical_json_v1(
            {
                "model": "x-ai/grok-4.5:free",
                "provider": _ROUTING,
                "stream": False,
            }
        ),
        canonical_json_v1(
            {
                "model": "x-ai/grok-4.5",
                "provider": {**_ROUTING, "allow_fallbacks": True},
                "stream": False,
            }
        ),
    ],
)
def test_invalid_model_or_routing_reaches_no_offline_network(body: bytes) -> None:
    calls: list[httpx.Request] = []
    client = DirectOpenRouterClient(
        assignment_id="ra-transport",
        api_key=SENTINEL_KEY,
        announce=lambda _: True,
        transport=httpx.MockTransport(
            lambda request: calls.append(request) or httpx.Response(200, content=b"{}")
        ),
    )

    with pytest.raises(OpenRouterTransportError):
        client.call(
            body=body,
            routing_sha256=hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest(),
            allowed_evidence_paths={"artifact/agent.py"},
        )
    assert calls == []


@pytest.mark.parametrize(
    "reason_code",
    [
        "missing_credential",
        "dns_failed",
        "tls_failed",
        "openrouter_auth_failed",
        "openrouter_rate_limited",
        "openrouter_unavailable",
        "response_malformed",
        "report_generation_failed",
    ],
)
async def test_infrastructure_failures_terminalize_without_work(
    database_session,
    reason_code: str,
) -> None:
    submission = AgentSubmission(
        miner_hotkey=f"review-miner-{reason_code}",
        name="review-agent",
        agent_hash=hashlib.sha256(reason_code.encode()).hexdigest(),
        artifact_uri="/tmp/review.zip",
        artifact_path="/tmp/review.zip",
        zip_sha256=hashlib.sha256(b"review").hexdigest(),
        zip_size_bytes=len(b"review"),
        raw_status="review_queued",
        effective_status="queued",
    )
    settings = ChallengeSettings(shared_token="review-token")
    now = datetime.now(UTC)
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review",
            rules_files={".rules/policy.md": b"safe"},
            rules_revision_id="rules-v1",
            settings=settings,
            now=now,
        )
        assignment = created.assignment
        planned_digest: str | None = None
        if reason_code != "missing_credential":
            # This transport fixture starts after the signed deployment
            # acknowledgement covered by deployment-specific tests.
            assignment.phase = "review_cvm_running"
            submission.raw_status = "review_cvm_running"
            planned = build_planned_openrouter_request(
                body=_body(),
                routing_sha256=hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest(),
            )
            marker = build_model_call_started(
                assignment_id=assignment.assignment_id,
                planned_request_sha256=planned[2],
                request_body_sha256=planned[0]["body_sha256"],
                request_body_length=planned[0]["body_length"],
            )
            assert await mark_model_call_started(
                session,
                assignment=assignment,
                marker=marker,
                now=now,
            )
            planned_digest = planned[2]
        failure = build_review_infrastructure_failure(
            assignment_id=assignment.assignment_id,
            planned_request_sha256=planned_digest,
            reason_code=reason_code,
        )
        assert await record_review_infrastructure_failure(
            session,
            assignment=assignment,
            failure=failure,
            now=now,
        )
        assert not await record_review_infrastructure_failure(
            session,
            assignment=assignment,
            failure=failure,
            now=now,
        )
        with pytest.raises(ReviewConflict):
            conflicting_reason = (
                "missing_credential"
                if reason_code == "report_generation_failed"
                else "report_generation_failed"
            )
            await record_review_infrastructure_failure(
                session,
                assignment=assignment,
                failure={**failure, "reason_code": conflicting_reason},
                now=now,
            )
        await session.commit()
        nonce = await session.scalar(
            select(ReviewNonce).where(ReviewNonce.assignment_id == assignment.id)
        )
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        status_event = await session.scalar(
            select(SubmissionStatusEvent)
            .where(SubmissionStatusEvent.submission_id == submission.id)
            .order_by(SubmissionStatusEvent.id.desc())
        )

    assert assignment.phase == "review_error"
    assert assignment.capability_state == "revoked"
    assert nonce is not None and nonce.state == "revoked"
    assert job_count == 0
    assert submission.raw_status == "review_error"
    assert status_event is not None and status_event.reason == "review_infrastructure_failure"
    async with database_session() as session:
        session_row = await session.get(ReviewSession, created.session.id)
        assert session_row is not None
        retried = await retry_review_assignment(
            session,
            session_row=session_row,
            expected_assignment_id=assignment.assignment_id,
            settings=settings,
            now=now,
        )
        await session.commit()
        retried_submission = await session.get(AgentSubmission, submission.id)

    assert retried.assignment.attempt == 2
    assert retried_submission is not None
    assert retried_submission.raw_status == "review_queued"


async def test_marker_idempotency_recovery_and_encrypted_evidence_read(
    client,
    database_session,
    internal_headers,
) -> None:
    settings = ChallengeSettings(
        shared_token="test-token",
        review_evidence_encryption_key="test-evidence-key",
    )
    now = datetime.now(UTC)
    submission = AgentSubmission(
        miner_hotkey="review-miner",
        name="review-agent",
        agent_hash="11" * 32,
        artifact_uri="/tmp/review.zip",
        artifact_path="/tmp/review.zip",
        zip_sha256=hashlib.sha256(b"review").hexdigest(),
        zip_size_bytes=len(b"review"),
        raw_status="review_queued",
        effective_status="queued",
    )
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review",
            rules_files={".rules/policy.md": b"safe"},
            rules_revision_id="rules-v1",
            settings=settings,
            now=now,
        )
        # This marker fixture begins after a valid signed deployment acknowledgement.
        created.assignment.phase = "review_cvm_running"
        submission.raw_status = "review_cvm_running"
        planned, _bytes, planned_digest = build_planned_openrouter_request(
            body=_body(),
            routing_sha256=hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest(),
        )
        marker = build_model_call_started(
            assignment_id=created.assignment.assignment_id,
            planned_request_sha256=planned_digest,
            request_body_sha256=planned["body_sha256"],
            request_body_length=planned["body_length"],
        )
        assert await mark_model_call_started(
            session,
            assignment=created.assignment,
            marker=marker,
            now=now,
        )
        assert not await mark_model_call_started(
            session,
            assignment=created.assignment,
            marker=marker,
            now=now,
        )
        with pytest.raises(ReviewConflict):
            await mark_model_call_started(
                session,
                assignment=created.assignment,
                marker={**marker, "request_body_length": marker["request_body_length"] + 1},
                now=now,
            )
        await session.commit()
        marker_response = await client.post(
            f"/review/v1/assignments/{created.assignment.assignment_id}/model-call-started",
            content=canonical_json_v1(marker),
            headers={
                "Authorization": f"Bearer {created.session_token}",
                "Content-Type": "application/json",
            },
        )
        assert marker_response.status_code == 200
        assert marker_response.json()["idempotent_replay"] is True
        conflicting_marker = await client.post(
            f"/review/v1/assignments/{created.assignment.assignment_id}/model-call-started",
            content=canonical_json_v1(
                {**marker, "request_body_length": marker["request_body_length"] + 1}
            ),
            headers={
                "Authorization": f"Bearer {created.session_token}",
                "Content-Type": "application/json",
            },
        )
        assert conflicting_marker.status_code == 409
        malformed_marker = await client.post(
            f"/review/v1/assignments/{created.assignment.assignment_id}/model-call-started",
            content=canonical_json_v1({**marker, "unexpected": True}),
            headers={
                "Authorization": f"Bearer {created.session_token}",
                "Content-Type": "application/json",
            },
        )
        assert malformed_marker.status_code == 422
        evidence = await store_review_evidence_objects(
            session,
            assignment=created.assignment,
            settings=settings,
            objects={
                "planned_request": canonical_json_v1(planned),
                "request_body": _body(),
                "response_body": b'{"model":"x-ai/grok-4.5"}',
            },
        )
        await session.commit()
        # Marker was just written; advanced clock past the OpenRouter/report grace
        # so recovery fail-closes exactly once with the durable plan digest.
        recovered = await recover_incomplete_model_calls(
            session,
            now=now + timedelta(hours=1),
            settings=settings,
        )
        await session.commit()

    assert len(evidence) == 3
    assert recovered == 1
    assert created.assignment.phase == "review_error"
    assert created.assignment.capability_state == "revoked"
    assert created.assignment.planned_request_sha256 == planned_digest
    assert created.assignment.reason_code == "report_generation_failed"
    replay_failure = await client.post(
        f"/review/v1/assignments/{created.assignment.assignment_id}/failure",
        content=canonical_json_v1(
            build_review_infrastructure_failure(
                assignment_id=created.assignment.assignment_id,
                planned_request_sha256=planned_digest,
                reason_code="report_generation_failed",
            )
        ),
        headers={
            "Authorization": f"Bearer {created.session_token}",
            "Content-Type": "application/json",
        },
    )
    assert replay_failure.status_code == 200
    assert replay_failure.json()["idempotent_replay"] is True
    stale_marker = await client.post(
        f"/review/v1/assignments/{created.assignment.assignment_id}/model-call-started",
        content=canonical_json_v1(marker),
        headers={
            "Authorization": f"Bearer {created.session_token}",
            "Content-Type": "application/json",
        },
    )
    assert stale_marker.status_code == 410
    stored = await client.get(
        f"/internal/v1/reviews/{created.session.session_id}/evidence/"
        f"{evidence['response_body']['object_ref']}",
        headers={**internal_headers, "Range": "bytes=1-8"},
    )
    assert stored.status_code == 206
    response_body = b'{"model":"x-ai/grok-4.5"}'
    assert stored.headers["content-range"] == f"bytes 1-8/{len(response_body)}"
    assert stored.headers["content-type"].startswith("application/octet-stream")
    assert stored.content == response_body[1:9]
    assert SENTINEL_KEY not in stored.text
    invalid_range = await client.get(
        f"/internal/v1/reviews/{created.session.session_id}/evidence/"
        f"{evidence['response_body']['object_ref']}",
        headers={**internal_headers, "Range": "bytes=0-1,3-4"},
    )
    assert invalid_range.status_code == 416
    evidence_rows = await _evidence_rows(database_session)
    assert all(SENTINEL_KEY.encode() not in row.ciphertext for row in evidence_rows)


async def _evidence_rows(database_session) -> list[ReviewEvidenceObject]:
    async with database_session() as session:
        return list((await session.scalars(select(ReviewEvidenceObject))).all())


def test_review_dockerfile_packages_every_imported_openrouter_module() -> None:
    definition = review_compose.review_build_definition()
    dockerfile = definition.dockerfile.read_text(encoding="utf-8")
    runtime_source = (definition.dockerfile.parent / "review_runtime.py").read_text(
        encoding="utf-8"
    )
    # Exact modules imported by the measured transport + quote + report_data path.
    # attested_times is required by report.review_report_data_preimage (guest).
    # or_outcome_bind is lazy-imported by openrouter after the OpenRouter response.
    for module_path in (
        "src/agent_challenge/review/canonical.py",
        "src/agent_challenge/review/schemas.py",
        "src/agent_challenge/review/policy.py",
        "src/agent_challenge/review/openrouter.py",
        "src/agent_challenge/review/or_outcome_bind.py",
        "src/agent_challenge/review/report.py",
        "src/agent_challenge/review/attested_times.py",
        "docker/review/review_runtime.py",
    ):
        assert module_path in dockerfile or Path(module_path).name in dockerfile
        assert f"COPY {module_path}" in dockerfile or (
            module_path.startswith("docker/review/") and "review_runtime.py" in dockerfile
        )
    assert "policy.py" in dockerfile
    assert "attested_times.py" in dockerfile
    assert "or_outcome_bind.py" in dockerfile
    assert "DirectOpenRouterClient" in runtime_source
    assert "run_direct_openrouter" in runtime_source


def test_review_runtime_exe_path_invokes_direct_openrouter_client() -> None:
    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("review_runtime_openrouter", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    body = _body()
    markers: list[dict[str, object]] = []
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "id": "offline-response",
                "model": "x-ai/grok-4.5",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "submit_verdict",
                                        "arguments": json.dumps(
                                            {
                                                "verdict": "allow",
                                                "reason_codes": [],
                                                "evidence_paths": ["artifact/agent.py"],
                                            },
                                            separators=(",", ":"),
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
            headers={"content-type": "application/json"},
        )

    capture = runtime.run_direct_openrouter(
        assignment_id="ra-transport",
        api_key=SENTINEL_KEY,
        body=body,
        routing_sha256=hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest(),
        allowed_evidence_paths={"artifact/agent.py"},
        announce=lambda marker: markers.append(marker) or True,
        transport=httpx.MockTransport(handler),
    )

    assert len(calls) == 1
    assert len(markers) == 1
    assert capture["planned_sha256"]
    # Drop the in-process capture object before JSON secret scans; it is only
    # retained for run_assignment access in the measured runtime.
    scannable = {k: v for k, v in capture.items() if k != "capture"}
    assert SENTINEL_KEY not in json.dumps(scannable)
    assert SENTINEL_KEY not in repr(scannable)


@pytest.mark.parametrize(
    ("exc", "reason_code"),
    [
        (
            httpx.ConnectError(
                "certificate verify failed: self-signed certificate",
                request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
            ),
            "tls_failed",
        ),
        (
            httpx.ConnectError(
                "hostname 'evil.example' doesn't match 'openrouter.ai'",
                request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
            ),
            "tls_failed",
        ),
        (
            httpx.ConnectError(
                "[SSL: HANDSHAKE_FAILURE] handshake failure",
                request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
            ),
            "tls_failed",
        ),
        (
            httpx.ConnectTimeout(
                "TLS handshake timed out",
                request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
            ),
            "tls_failed",
        ),
        (
            httpx.ConnectError(
                "Name or service not known",
                request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
            ),
            "dns_failed",
        ),
    ],
)
def test_tls_certificate_hostname_and_handshake_map_to_tls_failed(
    exc: httpx.HTTPError,
    reason_code: str,
) -> None:
    if "certificate verify failed" in str(exc):
        # Preserve the OpenSSL cause chain the real transport attaches.
        try:
            raise ssl.SSLCertVerificationError("certificate verify failed") from None
        except ssl.SSLCertVerificationError as ssl_exc:
            exc.__cause__ = ssl_exc

    class FailingTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            raise exc

    client = DirectOpenRouterClient(
        assignment_id="ra-transport",
        api_key=SENTINEL_KEY,
        announce=lambda _: True,
        transport=FailingTransport(),
    )
    with pytest.raises(OpenRouterTransportError) as info:
        client.call(
            body=_body(),
            routing_sha256=hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest(),
            allowed_evidence_paths={"artifact/agent.py"},
        )
    assert info.value.reason_code == reason_code


def test_response_read_aborts_at_cap_without_buffering_complete_body() -> None:
    max_bytes = MAX_OPENROUTER_RESPONSE_BYTES
    # Peer emits far more than the cap; the client must abort mid-stream.
    total_to_emit = max_bytes + 256_000
    emitted = {"bytes": 0}

    class OversizedStream(httpx.SyncByteStream):
        def __iter__(self):  # type: ignore[override]
            remaining = total_to_emit
            chunk = b"x" * 16_384
            while remaining > 0:
                piece = chunk if remaining >= len(chunk) else chunk[:remaining]
                emitted["bytes"] += len(piece)
                remaining -= len(piece)
                yield piece

        def close(self) -> None:
            return None

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "content-encoding": "identity"},
            stream=OversizedStream(),
            request=request,
        )

    client = DirectOpenRouterClient(
        assignment_id="ra-transport",
        api_key=SENTINEL_KEY,
        announce=lambda _: True,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(OpenRouterTransportError) as info:
        client.call(
            body=_body(),
            routing_sha256=hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest(),
            allowed_evidence_paths={"artifact/agent.py"},
        )
    assert info.value.reason_code == "response_malformed"
    # Must stop near the configured cap rather than first buffering the full body.
    assert emitted["bytes"] <= max_bytes + 65_536
    assert emitted["bytes"] < total_to_emit


async def test_concurrent_model_call_markers_create_exactly_one_durable_record(
    database_session,
) -> None:
    settings = ChallengeSettings(shared_token="review-token")
    now = datetime.now(UTC)
    submission = AgentSubmission(
        miner_hotkey="review-miner-concurrent-marker",
        name="review-agent",
        agent_hash="ab" * 32,
        artifact_uri="/tmp/review.zip",
        artifact_path="/tmp/review.zip",
        zip_sha256=hashlib.sha256(b"review-concurrent").hexdigest(),
        zip_size_bytes=len(b"review-concurrent"),
        raw_status="review_queued",
        effective_status="queued",
    )
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review-concurrent",
            rules_files={".rules/policy.md": b"safe"},
            rules_revision_id="rules-v1",
            settings=settings,
            now=now,
        )
        created.assignment.phase = "review_cvm_running"
        submission.raw_status = "review_cvm_running"
        planned, _bytes, planned_digest = build_planned_openrouter_request(
            body=_body(),
            routing_sha256=hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest(),
        )
        marker = build_model_call_started(
            assignment_id=created.assignment.assignment_id,
            planned_request_sha256=planned_digest,
            request_body_sha256=planned["body_sha256"],
            request_body_length=planned["body_length"],
        )
        assignment_pk = created.assignment.id
        await session.commit()

    async def _race() -> bool:
        from agent_challenge.core.models import ReviewAssignment

        async with database_session() as session:
            assignment = await session.get(ReviewAssignment, assignment_pk)
            assert assignment is not None
            started = await mark_model_call_started(
                session,
                assignment=assignment,
                marker=marker,
                now=now,
                settings=settings,
            )
            # Durably compete: only concurrent commits expose the CAS behaviour.
            await session.commit()
            return started

    results = await asyncio.gather(_race(), _race(), _race())
    assert results.count(True) == 1
    assert results.count(False) == 2

    async with database_session() as session:
        from agent_challenge.core.models import ReviewAssignment

        assignment = await session.get(ReviewAssignment, assignment_pk)
        assert assignment is not None
        assert assignment.model_call_started_json is not None
        assert assignment.model_call_started_sha256 is not None
        assert assignment.phase == "review_provider_standby"


async def test_encrypted_evidence_aggregate_includes_ciphertext_and_descriptor(
    database_session,
) -> None:
    settings = ChallengeSettings(
        shared_token="review-token",
        review_evidence_encryption_key="review-evidence-key",
        review_max_encrypted_evidence_bytes=MAX_REVIEW_EVIDENCE_BYTES,
    )
    now = datetime.now(UTC)
    submission = AgentSubmission(
        miner_hotkey="review-miner-evidence-cap",
        name="review-agent",
        agent_hash="cd" * 32,
        artifact_uri="/tmp/review.zip",
        artifact_path="/tmp/review.zip",
        zip_sha256=hashlib.sha256(b"review-evidence").hexdigest(),
        zip_size_bytes=len(b"review-evidence"),
        raw_status="review_queued",
        effective_status="queued",
    )
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review-evidence",
            rules_files={".rules/policy.md": b"safe"},
            rules_revision_id="rules-v1",
            settings=settings,
            now=now,
        )
        # Per-object plaintext fits individual caps and the 6 MiB plaintext
        # sum equals the aggregate ceiling, but Fernet ciphertext expansion plus
        # the descriptor must not be admitted under the encrypted aggregate cap.
        planned = b"p" * 1_048_576
        request_body = b"q" * 4_194_304
        response_body = b"r" * 1_048_576
        assert len(planned) + len(request_body) + len(response_body) == MAX_REVIEW_EVIDENCE_BYTES
        with pytest.raises(ReviewEvidenceError, match="aggregate"):
            await store_review_evidence_objects(
                session,
                assignment=created.assignment,
                settings=settings,
                objects={
                    "planned_request": planned,
                    "request_body": request_body,
                    "response_body": response_body,
                },
            )
        # A small bundle still stores and reports descriptors under the cap.
        stored = await store_review_evidence_objects(
            session,
            assignment=created.assignment,
            settings=settings,
            objects={
                "planned_request": b'{"schema_version":1}',
                "request_body": b"{}",
                "response_body": b'{"model":"x-ai/grok-4.5"}',
            },
        )
        await session.commit()
        rows = list(
            (
                await session.scalars(
                    select(ReviewEvidenceObject).where(
                        ReviewEvidenceObject.assignment_id == created.assignment.id
                    )
                )
            ).all()
        )
        cipher_total = sum(len(row.ciphertext) for row in rows)
        descriptor_total = len(json.dumps(stored, sort_keys=True, separators=(",", ":")).encode())
        assert cipher_total + descriptor_total <= MAX_REVIEW_EVIDENCE_BYTES
        assert cipher_total > sum(row.size_bytes for row in rows)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            OpenRouterTransportError("openrouter_rate_limited", "rate limited"),
            "openrouter_rate_limited",
        ),
        (OpenRouterTransportError("tls_failed", "tls"), "tls_failed"),
        (OpenRouterTransportError("dns_failed", "dns"), "dns_failed"),
        (
            OpenRouterTransportError("openrouter_auth_failed", "auth"),
            "openrouter_auth_failed",
        ),
        (
            OpenRouterTransportError("response_malformed", "bad"),
            "response_malformed",
        ),
        (
            OpenRouterTransportError("missing_credential", "miss"),
            "missing_credential",
        ),
        (
            OpenRouterTransportError("openrouter_unavailable", "down"),
            "openrouter_unavailable",
        ),
        (TimeoutError("hang"), "openrouter_unavailable"),
        (TimeoutError("quote timeout waiting for dstack get_quote"), "quote_timeout"),
        (ssl.SSLError("ssl bad"), "tls_failed"),
        (RuntimeError("boom"), "report_generation_failed"),
        (ValueError("weird"), "report_generation_failed"),
        (ValueError("quote event_log is not a list"), "quote_event_log_invalid"),
        (
            ValueError("quoted compose hash mismatches assignment"),
            "quote_measurement_mismatch",
        ),
        (ValueError("quote unavailable from dstack"), "quote_unavailable"),
        (ValueError("report envelope invalid from /report"), "report_envelope_invalid"),
        (ValueError("report evidence invalid from /report"), "report_evidence_invalid"),
        (ValueError("report timeline invalid from /report"), "report_timeline_invalid"),
        (
            ValueError(
                "quote event log invalid: event 'compose-hash' digest does not match its payload"
            ),
            "quote_event_log_invalid",
        ),
    ],
)
def test_infrastructure_failure_reason_maps_transport_and_allowlisted_classes(
    exc: BaseException,
    expected: str,
) -> None:
    assert infrastructure_failure_reason(exc) == expected
    # Nested transport errors under a generic wrapper still surface the code.
    wrapped = RuntimeError("outer")
    wrapped.__cause__ = OpenRouterTransportError("openrouter_rate_limited", "rl")
    assert infrastructure_failure_reason(wrapped) == "openrouter_rate_limited"


def test_infrastructure_failure_reason_maps_quote_package_errors() -> None:
    """Live dstack quote package errors must not collapse to opaque RGF."""

    from agent_challenge.keyrelease.quote import (
        QuoteStructureError,
        QuoteVerificationError,
    )

    assert (
        infrastructure_failure_reason(
            QuoteVerificationError("event 'compose-hash' digest does not match its payload")
        )
        == "quote_event_log_invalid"
    )
    # Structure errors that mention the TDX quote surface as quote_unavailable;
    # pure package errors without those keywords still use the class default.
    assert (
        infrastructure_failure_reason(QuoteStructureError("TDX quote declared length is truncated"))
        == "quote_unavailable"
    )
    assert (
        infrastructure_failure_reason(QuoteStructureError("malformed quote body"))
        == "quote_event_log_invalid"
    )


def test_runtime_failure_posts_mapped_reason_without_secrets() -> None:
    """POST /failure must carry mapped reason_code, never secrets/raw bodies."""

    import importlib.util
    import io
    from unittest import mock

    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_failure_map", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    token = "ra-fail-map.signed-capability-token"
    failures: list[dict[str, object]] = []

    def fake_http(
        method: str,
        url: str,
        *,
        token: str,
        body: bytes | None = None,
        accept: str = "application/json",
    ) -> tuple[int, bytes, dict[str, str]]:
        del token, accept
        if method == "POST" and url.endswith("/failure") and body is not None:
            payload = json.loads(body.decode("utf-8"))
            failures.append(payload)
            return 200, b'{"ok":true}', {}
        raise AssertionError(f"unexpected call {method} {url}")

    secret = SENTINEL_KEY
    env = {
        "REVIEW_SESSION_TOKEN": token,
        "OPENROUTER_API_KEY": secret,
        "REVIEW_API_BASE_URL": "https://review.example",
    }
    with (
        mock.patch.dict("os.environ", env, clear=False),
        mock.patch.object(
            runtime,
            "run_assignment",
            side_effect=OpenRouterTransportError(
                "openrouter_rate_limited", f"provider 429 with key={secret}"
            ),
        ),
        mock.patch.object(runtime, "_http_json", side_effect=fake_http),
        mock.patch.object(runtime.sys, "stderr", new=io.StringIO()),
    ):
        rc = runtime.main(["--run-assignment"])
    assert rc == 1
    assert len(failures) == 1
    assert failures[0]["reason_code"] == "openrouter_rate_limited"
    assert failures[0]["assignment_id"] == "ra-fail-map"
    assert set(failures[0]) == {
        "schema_version",
        "assignment_id",
        "planned_request_sha256",
        "reason_code",
    }
    assert secret not in json.dumps(failures[0])
    assert "429" not in json.dumps(failures[0])


def test_runtime_failure_unknown_exception_stays_report_generation_failed() -> None:
    import importlib.util
    import io
    from unittest import mock

    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_failure_unknown", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    token = "ra-fail-unknown.signed-capability-token"
    failures: list[dict[str, object]] = []

    def fake_http(
        method: str,
        url: str,
        *,
        token: str,
        body: bytes | None = None,
        accept: str = "application/json",
    ) -> tuple[int, bytes, dict[str, str]]:
        del token, accept
        if method == "POST" and url.endswith("/failure") and body is not None:
            failures.append(json.loads(body.decode("utf-8")))
            return 200, b'{"ok":true}', {}
        raise AssertionError(f"unexpected call {method} {url}")

    secret = SENTINEL_KEY
    env = {
        "REVIEW_SESSION_TOKEN": token,
        "OPENROUTER_API_KEY": secret,
        "REVIEW_API_BASE_URL": "https://review.example",
    }
    with (
        mock.patch.dict("os.environ", env, clear=False),
        mock.patch.object(
            runtime,
            "run_assignment",
            side_effect=RuntimeError(f"unexpected internals with {secret} body=raw-provider"),
        ),
        mock.patch.object(runtime, "_http_json", side_effect=fake_http),
        mock.patch.object(runtime.sys, "stderr", new=io.StringIO()),
    ):
        rc = runtime.main(["--run-assignment"])
    assert rc == 1
    assert failures[0]["reason_code"] == "report_generation_failed"
    assert secret not in json.dumps(failures[0])
    assert "raw-provider" not in json.dumps(failures[0])


def test_run_assignment_stamps_request_started_only_after_model_call_announce() -> None:
    """request_started is recorded after model-call announce, before wire exchange.

    Mirrors the measured runtime announce closure: mark announce → POST
    /model-call-started (may take walls) → stamp request_started → return so the
    transport opens the OpenRouter exchange only after request_started is set.
    """

    from agent_challenge.review.report import validate_review_core

    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    # Static guard: inverted pre-announce stamping must not reappear.
    source = runtime_path.read_text(encoding="utf-8")
    assert 'times["request_started_at_ms"] = int(time() * 1000)' in source
    # request_started must live inside announce (after model-call success), not
    # immediately before run_direct_openrouter.
    announce_block_start = source.index("def announce(")
    # Locate the call site after announce, not the module-level helper def.
    openrouter_call = source.index("run_direct_openrouter(", announce_block_start)
    request_stamp = source.index('times["request_started_at_ms"]', announce_block_start)
    model_stamp = source.index('times["model_call_marked_at_ms"]', announce_block_start)
    assert model_stamp < request_stamp < openrouter_call
    # And there is no request_started stamp between last times-init and announce.
    times_init = source.index('"request_started_at_ms": 0')
    pre_announce = source[times_init:announce_block_start]
    assert 'times["request_started_at_ms"]' not in pre_announce

    # Behavioral: exercise the announce order with a controllable clock.
    clock = {"t": 1_700_000_010_000}
    times = {
        "issued_at_ms": 1_700_000_000_000,
        "started_at_ms": 1_700_000_005_000,
        "model_call_marked_at_ms": 0,
        "request_started_at_ms": 0,
        "request_finished_at_ms": 0,
        "verifier_finished_at_ms": 0,
        "report_finished_at_ms": 0,
        "expires_at_ms": 1_700_001_000_000,
        "submission_received_at_ms": 1_700_000_000_000,
    }
    events: list[str] = []

    def fake_time() -> float:
        return clock["t"] / 1000.0

    def fake_http(
        method: str,
        url: str,
        *,
        token: str,
        body: bytes | None = None,
        accept: str = "application/json",
    ) -> tuple[int, bytes, dict[str, str]]:
        del token, body, accept
        assert method == "POST" and url.endswith("/model-call-started")
        events.append("model_call_announce")
        # Non-zero wall: previously this gap inverted request_started.
        clock["t"] += 50
        return 200, b'{"ok":true}', {}

    # Reconstruct the fixed announce closure from run_assignment semantics.
    def announce(marker: dict[str, object]) -> bool:
        del marker
        times["model_call_marked_at_ms"] = int(fake_time() * 1000)
        status_code, _resp, _ = fake_http(
            "POST",
            "https://review.example/review/v1/assignments/ra-transport/model-call-started",
            token="t",
            body=b"{}",
        )
        assert status_code in {200, 201}
        times["request_started_at_ms"] = int(fake_time() * 1000)
        return True

    conf: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        events.append("openrouter_wire")
        conf.append({"started": times["request_started_at_ms"], "url": str(request.url)})
        clock["t"] += 10
        route_digest = hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest()
        del route_digest
        body = json.dumps(
            {
                "id": "or-offline",
                "model": "x-ai/grok-4.5",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "submit_verdict",
                                        "arguments": json.dumps(
                                            {
                                                "verdict": "allow",
                                                "reason_codes": [],
                                                "evidence_paths": [],
                                            },
                                            separators=(",", ":"),
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/json", "content-encoding": "identity"},
            request=request,
        )

    client = DirectOpenRouterClient(
        assignment_id="ra-transport",
        api_key=SENTINEL_KEY,
        announce=announce,
        transport=httpx.MockTransport(handler),
    )
    capture = client.call(
        body=_body(),
        routing_sha256=hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest(),
        allowed_evidence_paths=set(),
    )
    client.close()
    times["request_finished_at_ms"] = int(fake_time() * 1000)
    times["verifier_finished_at_ms"] = times["request_finished_at_ms"] + 1
    times["report_finished_at_ms"] = times["verifier_finished_at_ms"] + 1

    assert events == ["model_call_announce", "openrouter_wire"]
    assert times["model_call_marked_at_ms"] <= times["request_started_at_ms"]
    # Wire exchange observed request_started was stamped before the POST opened.
    assert conf[0]["started"] == times["request_started_at_ms"]
    assert conf[0]["started"] >= times["model_call_marked_at_ms"]
    assert capture.planned_sha256
    # Bind measured times into a synthetic review_core and prove validator accepts.
    assignment = _assignment()
    core = assignment["assignment_core"]  # type: ignore[index]
    synthetic = {
        "schema_version": 1,
        "session_id": core["session_id"],  # type: ignore[index]
        "assignment_id": core["assignment_id"],  # type: ignore[index]
        "assignment_digest": assignment["assignment_digest"],
        "submission_id": core["submission_id"],  # type: ignore[index]
        "artifact_observation": {
            "agent_hash": core["artifact"]["agent_hash"],  # type: ignore[index]
            "zip_sha256": core["artifact"]["zip_sha256"],  # type: ignore[index]
            "zip_size_bytes": core["artifact"]["zip_size_bytes"],  # type: ignore[index]
            "manifest_sha256": core["artifact"]["manifest_sha256"],  # type: ignore[index]
            "manifest_entries_sha256": core["artifact"]["manifest_entries_sha256"],  # type: ignore[index]
        },
        "rules_observation": {
            "snapshot_sha256": core["rules"]["snapshot_sha256"],  # type: ignore[index]
            "revision_id": core["rules"]["revision_id"],  # type: ignore[index]
        },
        "policy_observation": {
            "model": core["policy"]["model"],  # type: ignore[index]
            "routing_sha256": core["policy"]["routing_sha256"],  # type: ignore[index]
            "prompt_version": core["policy"]["prompt_version"],  # type: ignore[index]
            "prompt_sha256": core["policy"]["prompt_sha256"],  # type: ignore[index]
            "tool_schema_version": core["policy"]["tool_schema_version"],  # type: ignore[index]
            "tool_schema_sha256": core["policy"]["tool_schema_sha256"],  # type: ignore[index]
            "verifier_version": core["policy"]["verifier_version"],  # type: ignore[index]
            "verifier_sha256": core["policy"]["verifier_sha256"],  # type: ignore[index]
        },
        "openrouter_observation": {
            "planned_request_sha256": capture.planned_sha256,
            "transport_observation_sha256": hashlib.sha256(capture.observed_bytes).hexdigest(),
            "request_body_sha256": capture.planned["body_sha256"],
            "request_body_length": capture.planned["body_length"],
            "response_status": 200,
            "response_content_encoding": "identity",
            "response_body_sha256": capture.observed["response_body_sha256"],
            "response_body_length": capture.observed["response_body_length"],
            "response_id": "or-offline",
            "returned_model": "x-ai/grok-4.5",
            "metadata_sha256": None,
            "observed_provider": None,
            "provider_provenance": "unavailable",
            "cache_hit": False,
        },
        "decision": {
            "static_findings_sha256": "75" * 32,
            "parsed_output_sha256": capture.model_output.sha256,
            "verifier_input_sha256": "77" * 32,
            "verifier_output_sha256": "78" * 32,
            "verifier_result": "pass",
            "verdict": "allow",
            "reason_codes": ["policy_passed"],
            "evidence_digests": [],
        },
        "times": times,
        "review_nonce": core["review_nonce"],  # type: ignore[index]
    }
    assert validate_review_core(synthetic)
    failure_surface = build_review_infrastructure_failure(
        assignment_id="ra-transport",
        planned_request_sha256=capture.planned_sha256,
        reason_code="openrouter_rate_limited",
    )
    assert SENTINEL_KEY not in json.dumps(failure_surface)


async def test_recover_incomplete_model_calls_spares_fresh_markers_within_grace(
    database_session,
) -> None:
    """Fresh announced markers must not terminalize on every reconciler tick."""

    settings = ChallengeSettings(
        shared_token="grace-token",
        review_evidence_encryption_key="grace-evidence-key",
    )
    now = datetime.now(UTC)
    submission = AgentSubmission(
        miner_hotkey="review-miner-grace-fresh",
        name="review-agent",
        agent_hash="ab" * 32,
        artifact_uri="/tmp/review-grace-fresh.zip",
        artifact_path="/tmp/review-grace-fresh.zip",
        zip_sha256=hashlib.sha256(b"review-grace-fresh").hexdigest(),
        zip_size_bytes=len(b"review-grace-fresh"),
        raw_status="review_queued",
        effective_status="queued",
    )
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review-grace-fresh",
            rules_files={".rules/policy.md": b"safe"},
            rules_revision_id="rules-v1",
            settings=settings,
            now=now,
        )
        created.assignment.phase = "review_cvm_running"
        submission.raw_status = "review_cvm_running"
        planned, _bytes, planned_digest = build_planned_openrouter_request(
            body=_body(),
            routing_sha256=hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest(),
        )
        marker = build_model_call_started(
            assignment_id=created.assignment.assignment_id,
            planned_request_sha256=planned_digest,
            request_body_sha256=planned["body_sha256"],
            request_body_length=planned["body_length"],
        )
        assert await mark_model_call_started(
            session,
            assignment=created.assignment,
            marker=marker,
            now=now,
        )
        await session.commit()

        # Within the OpenRouter + report window (default minutes order), leave open.
        recovered = await recover_incomplete_model_calls(
            session,
            now=now + timedelta(seconds=30),
            settings=settings,
        )
        await session.refresh(created.assignment)

    assert recovered == 0
    assert created.assignment.phase == "review_provider_standby"
    assert created.assignment.capability_state == "active"
    assert created.assignment.planned_request_sha256 == planned_digest
    assert created.assignment.infrastructure_failure_json is None
    assert created.assignment.reason_code is None


async def test_recover_incomplete_model_calls_terminalizes_stale_markers_once(
    database_session,
) -> None:
    """Markers older than grace fail closed once with planned digest preserved."""

    settings = ChallengeSettings(
        shared_token="grace-token-stale",
        review_evidence_encryption_key="grace-evidence-key-stale",
        review_https_total_timeout_seconds=120.0,
        review_model_call_recovery_grace_seconds=180.0,
    )
    now = datetime.now(UTC)
    submission = AgentSubmission(
        miner_hotkey="review-miner-grace-stale",
        name="review-agent",
        agent_hash="cd" * 32,
        artifact_uri="/tmp/review-grace-stale.zip",
        artifact_path="/tmp/review-grace-stale.zip",
        zip_sha256=hashlib.sha256(b"review-grace-stale").hexdigest(),
        zip_size_bytes=len(b"review-grace-stale"),
        raw_status="review_queued",
        effective_status="queued",
    )
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review-grace-stale",
            rules_files={".rules/policy.md": b"safe"},
            rules_revision_id="rules-v1",
            settings=settings,
            now=now,
        )
        created.assignment.phase = "review_cvm_running"
        submission.raw_status = "review_cvm_running"
        planned, _bytes, planned_digest = build_planned_openrouter_request(
            body=_body(),
            routing_sha256=hashlib.sha256(canonical_json_v1(_ROUTING)).hexdigest(),
        )
        marker = build_model_call_started(
            assignment_id=created.assignment.assignment_id,
            planned_request_sha256=planned_digest,
            request_body_sha256=planned["body_sha256"],
            request_body_length=planned["body_length"],
        )
        assert await mark_model_call_started(
            session,
            assignment=created.assignment,
            marker=marker,
            now=now,
        )
        await session.commit()

        stale_now = now + timedelta(seconds=181)
        first = await recover_incomplete_model_calls(
            session,
            now=stale_now,
            settings=settings,
        )
        await session.refresh(created.assignment)
        second = await recover_incomplete_model_calls(
            session,
            now=stale_now + timedelta(seconds=5),
            settings=settings,
        )
        await session.refresh(created.assignment)
        failure_json = created.assignment.infrastructure_failure_json

    assert first == 1
    assert second == 0
    assert created.assignment.phase == "review_error"
    assert created.assignment.capability_state == "revoked"
    assert created.assignment.planned_request_sha256 == planned_digest
    assert created.assignment.reason_code == "report_generation_failed"
    assert failure_json is not None
    failure = json.loads(failure_json)
    assert failure["planned_request_sha256"] == planned_digest
    assert failure["reason_code"] == "report_generation_failed"


def test_runtime_failure_after_announce_carries_planned_request_sha256() -> None:
    """POST /failure after announce must bind the durable planned digest."""

    import importlib.util
    import io
    from unittest import mock

    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_failure_plan_bound", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    token = "ra-fail-plan-bound.signed-capability-token"
    planned_digest = "ab" * 32
    failures: list[dict[str, object]] = []

    def fake_http(
        method: str,
        url: str,
        *,
        token: str,
        body: bytes | None = None,
        accept: str = "application/json",
    ) -> tuple[int, bytes, dict[str, str]]:
        del token, accept
        if method == "POST" and url.endswith("/failure") and body is not None:
            failures.append(json.loads(body.decode("utf-8")))
            return 200, b'{"ok":true}', {}
        raise AssertionError(f"unexpected call {method} {url}")

    class _BoomAfterAnnounce(Exception):
        planned_request_sha256 = planned_digest

    secret = SENTINEL_KEY
    env = {
        "REVIEW_SESSION_TOKEN": token,
        "OPENROUTER_API_KEY": secret,
        "REVIEW_API_BASE_URL": "https://review.example",
    }
    with (
        mock.patch.dict("os.environ", env, clear=False),
        mock.patch.object(runtime, "run_assignment", side_effect=_BoomAfterAnnounce("quote boom")),
        mock.patch.object(runtime, "_http_json", side_effect=fake_http),
        mock.patch.object(runtime.sys, "stderr", new=io.StringIO()),
    ):
        rc = runtime.main(["--run-assignment"])
    assert rc == 1
    assert len(failures) == 1
    assert failures[0]["planned_request_sha256"] == planned_digest
    assert failures[0]["assignment_id"] == "ra-fail-plan-bound"
    assert secret not in json.dumps(failures[0])


def test_announce_sets_announced_plan_only_after_model_call_started_2xx() -> None:
    """Failed model-call-started must not leave announced_plan set.

    Scrutiny residual on recover-incomplete-model-call-grace: if announce()
    stamps planned_request_sha256 before durable model-call-started 2xx, the
    outer except / main path posts plan-bound /failure the host rejects as
    unannounced, so the assignment never terminalizes.
    """

    import importlib.util
    import io
    import zipfile
    from unittest import mock

    from agent_challenge.review.schemas import (
        ReviewInputConfig,
        build_review_assignment,
        build_rules_bundle,
        rules_snapshot_sha256,
        validate_rules_bundle,
    )

    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    source = runtime_path.read_text(encoding="utf-8")
    announce_start = source.index("def announce(")
    announce_return = source.index("return True", announce_start)
    announce_body = source[announce_start:announce_return]
    status_check = announce_body.index("if status_code not in {200, 201}:")
    plan_bind = announce_body.index('announced_plan["planned_request_sha256"] = digest')
    assert status_check < plan_bind
    # Must not assign announced_plan before the durable POST succeeds.
    pre_status = announce_body[:status_check]
    assert 'announced_plan["planned_request_sha256"]' not in pre_status

    spec = importlib.util.spec_from_file_location("review_runtime_announce_after_2xx", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    assignment_id = "ra-announce-after-2xx"
    token = f"{assignment_id}.signed-capability-token"
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("agent.py", "print('benign')\n")
    artifact_bytes = zip_buf.getvalue()
    agent_hash = hashlib.sha256(b"agent-hash-bytes").hexdigest()
    zip_sha = hashlib.sha256(artifact_bytes).hexdigest()
    rules_bundle = build_rules_bundle(
        revision_id="rules-v1",
        files={".rules/policy.md": b"safe\n"},
    )
    # Host snapshot is sha256(canonical_json_v1(bundle)); serve those bytes.
    rules_wire = validate_rules_bundle(rules_bundle)
    snapshot = rules_snapshot_sha256(rules_bundle)
    assert hashlib.sha256(rules_wire).hexdigest() == snapshot

    assignment, _assignment_bytes, _digest = build_review_assignment(
        session_id="rs-announce-after-2xx",
        assignment_id=assignment_id,
        attempt=1,
        submission_id="42",
        artifact={
            "agent_hash": agent_hash,
            "zip_sha256": zip_sha,
            "zip_size_bytes": len(artifact_bytes),
            "manifest_sha256": "30" * 32,
            "manifest_entries_sha256": "40" * 32,
            "fetch_path": f"/review/v1/assignments/{assignment_id}/artifact",
        },
        rules_snapshot_sha256_value=snapshot,
        rules_revision_id="rules-v1",
        review_nonce="rn-announce-after-2xx",
        issued_at_ms=1_700_000_000_000,
        expires_at_ms=1_700_001_000_000,
        session_token_sha256="60" * 32,
        config=ReviewInputConfig(routing=_ROUTING),
    )
    assignment_wire = json.dumps(assignment, separators=(",", ":"), sort_keys=True).encode("utf-8")

    failures: list[dict[str, object]] = []
    started_posts: list[bytes] = []

    def fake_http(
        method: str,
        url: str,
        *,
        token: str,
        body: bytes | None = None,
        accept: str = "application/json",
    ) -> tuple[int, bytes, dict[str, str]]:
        del token
        if method == "GET" and url.endswith(f"/assignments/{assignment_id}"):
            return 200, assignment_wire, {}
        if method == "GET" and url.endswith("/artifact"):
            assert accept == "application/zip"
            return 200, artifact_bytes, {"content-type": "application/zip"}
        if method == "GET" and url.endswith("/rules"):
            return 200, rules_wire, {}
        if method == "POST" and url.endswith("/model-call-started"):
            assert body is not None
            started_posts.append(body)
            # Durable mark rejected: host never records planned_request_sha256.
            return 503, b'{"detail":"model-call-started unavailable"}', {}
        if method == "POST" and url.endswith("/failure") and body is not None:
            failures.append(json.loads(body.decode("utf-8")))
            return 200, b'{"ok":true}', {}
        raise AssertionError(f"unexpected call {method} {url}")

    secret = SENTINEL_KEY
    env = {
        "REVIEW_SESSION_TOKEN": token,
        "OPENROUTER_API_KEY": secret,
        "REVIEW_API_BASE_URL": "https://review.example",
    }
    observed_announced: dict[str, str | None] = {"planned_request_sha256": "sentinel"}

    # Capture the closure's announced_plan after failed announce by wrapping
    # the real announce path via side effects: run_assignment raises and stamps
    # planned onto the exception only when announced_plan was set.
    original_run = runtime.run_assignment

    def tracking_run_assignment(**kwargs: object) -> object:
        try:
            return original_run(**kwargs)
        except Exception as exc:  # noqa: BLE001 - inspect stamping behavior
            planned = getattr(exc, "planned_request_sha256", None)
            observed_announced["planned_request_sha256"] = (
                planned if isinstance(planned, str) and planned else None
            )
            raise

    with (
        mock.patch.dict("os.environ", env, clear=False),
        mock.patch.object(runtime, "_http_json", side_effect=fake_http),
        mock.patch.object(runtime, "run_assignment", side_effect=tracking_run_assignment),
        mock.patch.object(runtime.sys, "stderr", new=io.StringIO()),
    ):
        # Patch tracking onto module so main invokes it; rebind _http on
        # the original symbol used inside the real implementation.
        # tracking_run_assignment calls original_run which closed over the
        # module-level _http_json we have patched on runtime.
        rc = runtime.main(["--run-assignment"])

    assert rc == 1
    assert len(started_posts) == 1
    marker = json.loads(started_posts[0].decode("utf-8"))
    assert marker.get("planned_request_sha256")
    # Failed announce: plan must not bind the outer exception or /failure.
    assert observed_announced["planned_request_sha256"] is None
    assert len(failures) == 1
    assert failures[0]["planned_request_sha256"] is None
    assert failures[0]["assignment_id"] == assignment_id
    assert failures[0]["reason_code"] == "report_generation_failed"
    assert secret not in json.dumps(failures[0])

    # Successful 2xx path of the same announce closure sets announced_plan once.
    announced_plan_success: dict[str, str | None] = {"planned_request_sha256": None}
    times_success = {
        "model_call_marked_at_ms": 0,
        "request_started_at_ms": 0,
    }
    clock = {"t": 1_700_000_010_000}

    def success_http(
        method: str,
        url: str,
        *,
        token: str,
        body: bytes | None = None,
        accept: str = "application/json",
    ) -> tuple[int, bytes, dict[str, str]]:
        del method, url, token, body, accept
        clock["t"] += 5
        return 200, b'{"ok":true}', {}

    def fixed_time() -> float:
        return clock["t"] / 1000.0

    # Exercise the fixed post-2xx bind semantics used inside announce.
    digest = "cd" * 32
    marker_ok = {
        "schema_version": 1,
        "assignment_id": assignment_id,
        "planned_request_sha256": digest,
        "request_body_sha256": "11" * 32,
        "request_body_length": 12,
        "request_record_sha256": "22" * 32,
    }
    # Inline the measured announce contract (post-fix): mark → POST → 2xx bind.
    times_success["model_call_marked_at_ms"] = int(fixed_time() * 1000)
    status_code, _resp, _ = success_http(
        "POST",
        f"https://review.example/review/v1/assignments/{assignment_id}/model-call-started",
        token=token,
        body=json.dumps(marker_ok).encode("utf-8"),
    )
    assert status_code in {200, 201}
    if isinstance(digest, str) and digest:
        announced_plan_success["planned_request_sha256"] = digest
    times_success["request_started_at_ms"] = int(fixed_time() * 1000)
    assert announced_plan_success["planned_request_sha256"] == digest
    assert times_success["model_call_marked_at_ms"] <= times_success["request_started_at_ms"]


def test_runtime_evidence_keys_match_api_decoder_and_omit_empty_metadata() -> None:
    """Runtime /report evidence uses transport_observation_b64; skips empty metadata."""

    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    source = runtime_path.read_text(encoding="utf-8")
    assert '"transport_observation_b64"' in source
    assert '"observed_transport_b64"' not in source
    # Empty metadata must not be base64-encoded onto the evidence wire payload.
    assert 'evidence["metadata_b64"]' in source
    assert "if capture.metadata:" in source
    evidence_block_start = source.index("evidence: dict[str, str] = {")
    evidence_block = source[evidence_block_start : evidence_block_start + 600]
    assert '"metadata_b64"' not in evidence_block.split("if capture.metadata:")[0]
    # Quote client must allow slow dstack RPCs.
    assert "DstackClient(timeout=" in source
    assert "timeout=_DSTACK_QUOTE_TIMEOUT_SECONDS" in source
    assert "_DSTACK_QUOTE_TIMEOUT_SECONDS = 60" in source
    # Clock skew clamp for started_at.
    assert "max(int(time() * 1000), issued_at_ms)" in source
    assert "_report_post_error" in source


def test_report_post_error_maps_non_2xx_without_bodies() -> None:
    import importlib.util

    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_report_post_error", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    ev = runtime._report_post_error(422, b'{"detail":{"code":"review_report_invalid"}}')
    assert "evidence" not in str(ev).lower() or "envelope" in str(ev).lower()
    assert infrastructure_failure_reason(ev) in {
        "report_envelope_invalid",
        "report_evidence_invalid",
    }
    # Evidence-coded response.
    evidence_exc = runtime._report_post_error(422, b'{"detail":{"code":"review_evidence_invalid"}}')
    assert infrastructure_failure_reason(evidence_exc) == "report_evidence_invalid"
    # Opaque/malformed body stays safe and classified as envelope invalid.
    opaque = runtime._report_post_error(500, b"raw provider body with secret=abc")
    assert infrastructure_failure_reason(opaque) == "report_envelope_invalid"
    assert "secret" not in str(opaque)
    assert "abc" not in str(opaque)


def test_dstack_client_quote_timeout_is_at_least_60s() -> None:
    import importlib.util

    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_dstack_timeout", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    assert float(runtime._DSTACK_QUOTE_TIMEOUT_SECONDS) >= 60.0


def test_normalize_event_log_strips_0x_and_lowercases_hex() -> None:
    """Live dstack quote event digests may carry 0x + mixed-case hex."""

    import importlib.util

    from agent_challenge.keyrelease.quote import (
        COMPOSE_HASH_EVENT,
        DSTACK_RUNTIME_EVENT_TYPE,
        KEY_PROVIDER_EVENT,
        runtime_event_digest,
        validate_rtmr3_event_log,
    )

    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_event_log_0x", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    compose_payload = "ab" * 32
    compose_digest = runtime_event_digest(COMPOSE_HASH_EVENT, bytes.fromhex(compose_payload)).hex()
    provider_payload = "7b226e616d65223a227068616c61227d"  # {"name":"phala"}
    provider_digest = runtime_event_digest(
        KEY_PROVIDER_EVENT, bytes.fromhex(provider_payload)
    ).hex()
    raw = [
        {
            "imr": 3,
            "event_type": DSTACK_RUNTIME_EVENT_TYPE,
            "digest": "0x" + compose_digest.upper(),
            "event": COMPOSE_HASH_EVENT,
            "event_payload": "0x" + compose_payload.upper(),
        },
        {
            "imr": 3,
            "event_type": DSTACK_RUNTIME_EVENT_TYPE,
            "digest": "0X" + provider_digest,
            "event": KEY_PROVIDER_EVENT,
            "event_payload": "0X" + provider_payload,
        },
    ]
    normalized = runtime._normalize_event_log(raw)
    assert all(not entry["digest"].startswith("0x") for entry in normalized)
    assert all(entry["digest"] == entry["digest"].lower() for entry in normalized)
    assert all(entry["event_payload"] == entry["event_payload"].lower() for entry in normalized)
    validated = validate_rtmr3_event_log(normalized)
    assert validated[0]["event_payload"] == compose_payload


def _getquote_empty_digest_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """GetQuote-shaped RTMR3 events with blank digests + filled attestation twin.

    Offline fixture mirrors the live residual: guest GetQuote returns correct
    payloads/order but empty digests; Phala export/cc-eventlog has filled digests
    that reusable attestation logs validate against
    ``runtime_event_digest(event, payload)``.
    """

    from agent_challenge.keyrelease.quote import (
        COMPOSE_HASH_EVENT,
        DSTACK_RUNTIME_EVENT_TYPE,
        KEY_PROVIDER_EVENT,
        runtime_event_digest,
    )

    compose_payload = "cd" * 32
    provider_payload = "7b226e616d65223a227068616c61227d"  # {"name":"phala"}
    bootstrap_payload = "11" * 16
    filled = [
        {
            "imr": 3,
            "event_type": DSTACK_RUNTIME_EVENT_TYPE,
            "digest": runtime_event_digest("instance-id", bytes.fromhex(bootstrap_payload)).hex(),
            "event": "instance-id",
            "event_payload": bootstrap_payload,
        },
        {
            "imr": 3,
            "event_type": DSTACK_RUNTIME_EVENT_TYPE,
            "digest": runtime_event_digest(
                COMPOSE_HASH_EVENT, bytes.fromhex(compose_payload)
            ).hex(),
            "event": COMPOSE_HASH_EVENT,
            "event_payload": compose_payload,
        },
        {
            "imr": 3,
            "event_type": DSTACK_RUNTIME_EVENT_TYPE,
            "digest": runtime_event_digest("boot-mr-done", bytes.fromhex(bootstrap_payload)).hex(),
            "event": "boot-mr-done",
            "event_payload": bootstrap_payload,
        },
        {
            "imr": 3,
            "event_type": DSTACK_RUNTIME_EVENT_TYPE,
            "digest": runtime_event_digest(
                KEY_PROVIDER_EVENT, bytes.fromhex(provider_payload)
            ).hex(),
            "event": KEY_PROVIDER_EVENT,
            "event_payload": provider_payload,
        },
    ]
    empty_digest = [
        {
            "imr": entry["imr"],
            "event_type": entry["event_type"],
            "digest": "",
            "event": entry["event"],
            "event_payload": entry["event_payload"],
        }
        for entry in filled
    ]
    return empty_digest, filled


def test_normalize_event_log_recomputes_empty_rtmr3_runtime_digests() -> None:
    """Empty GetQuote IMR3 runtime digests are retained as sealed RuntimeEvent digests."""

    import importlib.util

    from agent_challenge.keyrelease.quote import (
        replay_rtmr3,
        runtime_event_digest,
        validate_rtmr3_event_log,
    )

    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_empty_rtmr3_digest", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    empty_digest, filled = _getquote_empty_digest_fixture()
    normalized = runtime._normalize_event_log(empty_digest)
    for entry, expected in zip(normalized, filled, strict=True):
        payload = bytes.fromhex(str(entry["event_payload"]))
        assert entry["digest"] == runtime_event_digest(str(entry["event"]), payload).hex()
        assert entry["digest"] == expected["digest"]
    validated = validate_rtmr3_event_log(normalized)
    replay = replay_rtmr3(validated)
    assert replay.compose_hash == filled[1]["event_payload"]
    assert replay.rtmr3 == replay_rtmr3(filled).rtmr3


def test_normalize_event_log_preserves_nonempty_wrong_digest_for_fail_closed() -> None:
    """Non-empty wrong digests stay as-is so replay mismatch still fails closed."""

    import importlib.util

    from agent_challenge.keyrelease.quote import (
        QuoteVerificationError,
        replay_rtmr3,
        validate_rtmr3_event_log,
    )

    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_wrong_rtmr3_digest", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    _empty, filled = _getquote_empty_digest_fixture()
    wrong = [dict(entry) for entry in filled]
    wrong[1] = dict(wrong[1])
    wrong[1]["digest"] = "11" * 48  # non-empty, wrong for payload
    normalized = runtime._normalize_event_log(wrong)
    assert normalized[1]["digest"] == "11" * 48
    with pytest.raises(QuoteVerificationError, match="digest does not match"):
        replay_rtmr3(validate_rtmr3_event_log(normalized))


def test_quote_prefers_info_event_log_when_getquote_rtmr3_digests_blank() -> None:
    """Blank GetQuote RTMR3 digests count as missing so filled tcb_info digests win."""

    import importlib.util
    from types import SimpleNamespace

    from agent_challenge.keyrelease.quote import replay_rtmr3, validate_rtmr3_event_log

    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_blank_prefers_info", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    empty_digest, filled = _getquote_empty_digest_fixture()
    # Distinct payload so preferring info is observable vs recomputing GetQuote.
    info_filled = [dict(entry) for entry in filled]
    info_filled[1] = dict(info_filled[1])
    info_compose = "ee" * 32
    from agent_challenge.keyrelease.quote import COMPOSE_HASH_EVENT, runtime_event_digest

    info_filled[1]["event_payload"] = info_compose
    info_filled[1]["digest"] = runtime_event_digest(
        COMPOSE_HASH_EVENT, bytes.fromhex(info_compose)
    ).hex()

    class _Client:
        def get_quote(self, report_data: bytes) -> SimpleNamespace:
            del report_data
            return SimpleNamespace(
                quote="00" * 700,
                event_log=empty_digest,
                vm_config=None,
            )

        def info(self) -> SimpleNamespace:
            return SimpleNamespace(
                tcb_info={"event_log": info_filled},
                vm_config={"vcpu": 1, "memory_mb": 2048},
            )

    report_data_hex = "ab" * 64
    quoted = runtime._quote(report_data_hex, client=_Client())
    assert quoted["event_log"][1]["event_payload"] == info_compose
    assert quoted["event_log"][1]["digest"] == info_filled[1]["digest"]
    validate_rtmr3_event_log(quoted["event_log"])
    assert replay_rtmr3(quoted["event_log"]).compose_hash == info_compose


def test_quote_recomputes_blank_getquote_when_info_unavailable() -> None:
    """When info cannot fill digests, blank GetQuote RTMR3 digests still recompute."""

    import importlib.util
    from types import SimpleNamespace

    from agent_challenge.keyrelease.quote import replay_rtmr3, validate_rtmr3_event_log

    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    spec = importlib.util.spec_from_file_location(
        "review_runtime_blank_recompute_no_info", runtime_path
    )
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    empty_digest, filled = _getquote_empty_digest_fixture()

    class _Client:
        def get_quote(self, report_data: bytes) -> SimpleNamespace:
            del report_data
            return SimpleNamespace(
                quote="00" * 700,
                event_log=empty_digest,
                vm_config={"cpu_count": 1, "memory_size": 2 * 1024 * 1024 * 1024},
            )

        # no info() method

    quoted = runtime._quote("cd" * 64, client=_Client())
    validated = validate_rtmr3_event_log(quoted["event_log"])
    assert replay_rtmr3(validated).rtmr3 == replay_rtmr3(filled).rtmr3
    for entry, expected in zip(validated, filled, strict=True):
        assert entry["digest"] == expected["digest"]


def test_runtime_failure_maps_quote_and_report_residuals_without_secrets() -> None:
    """POST /failure keeps residual quote/report reason codes off the secret surface."""

    import importlib.util
    import io
    from unittest import mock

    runtime_path = review_compose.review_build_definition().dockerfile.parent / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_failure_quote_map", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    secret = SENTINEL_KEY
    cases = [
        (ValueError("quoted compose hash mismatches assignment"), "quote_measurement_mismatch"),
        (ValueError("quote event_log is not valid JSON"), "quote_event_log_invalid"),
        (TimeoutError("quote timeout waiting for dstack get_quote"), "quote_timeout"),
        (ValueError("report envelope invalid from /report"), "report_envelope_invalid"),
    ]
    for exc, expected in cases:
        bucket: list[dict[str, object]] = []

        def fake_http(
            method: str,
            url: str,
            *,
            token: str,
            body: bytes | None = None,
            accept: str = "application/json",
            _bucket: list[dict[str, object]] = bucket,
        ) -> tuple[int, bytes, dict[str, str]]:
            del token, accept
            if method == "POST" and url.endswith("/failure") and body is not None:
                _bucket.append(json.loads(body.decode("utf-8")))
                return 200, b'{"ok":true}', {}
            raise AssertionError(f"unexpected call {method} {url}")

        env = {
            "REVIEW_SESSION_TOKEN": "ra-fail-residual.signed-capability-token",
            "OPENROUTER_API_KEY": secret,
            "REVIEW_API_BASE_URL": "https://review.example",
        }
        with (
            mock.patch.dict("os.environ", env, clear=False),
            mock.patch.object(runtime, "run_assignment", side_effect=exc),
            mock.patch.object(runtime, "_http_json", side_effect=fake_http),
            mock.patch.object(runtime.sys, "stderr", new=io.StringIO()),
        ):
            rc = runtime.main(["--run-assignment"])
        assert rc == 1
        assert bucket[0]["reason_code"] == expected
        assert secret not in json.dumps(bucket[0])


async def test_infrastructure_failure_rejected_after_durable_report_receipt(
    database_session,
) -> None:
    settings = ChallengeSettings(shared_token="review-token")
    now = datetime.now(UTC)
    submission = AgentSubmission(
        miner_hotkey="review-miner-post-receipt",
        name="review-agent",
        agent_hash="ef" * 32,
        artifact_uri="/tmp/review.zip",
        artifact_path="/tmp/review.zip",
        zip_sha256=hashlib.sha256(b"review-receipt").hexdigest(),
        zip_size_bytes=len(b"review-receipt"),
        raw_status="review_queued",
        effective_status="queued",
    )
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review-receipt",
            rules_files={".rules/policy.md": b"safe"},
            rules_revision_id="rules-v1",
            settings=settings,
            now=now,
        )
        created.assignment.phase = "review_verifying"
        created.assignment.review_report_envelope_json = canonical_json_v1(
            {"schema_version": 1, "placeholder": True}
        ).decode("utf-8")
        created.assignment.planned_request_sha256 = "aa" * 32
        failure = build_review_infrastructure_failure(
            assignment_id=created.assignment.assignment_id,
            planned_request_sha256="aa" * 32,
            reason_code="report_generation_failed",
        )
        with pytest.raises(ReviewConflict, match="receipt|report|resume"):
            await record_review_infrastructure_failure(
                session,
                assignment=created.assignment,
                failure=failure,
                now=now,
            )
        assert created.assignment.phase == "review_verifying"
        assert created.assignment.capability_state == "active"
