"""Strict signed Eval authorization lifecycle tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from agent_challenge.core.models import (
    AgentSubmission,
    EvalNonce,
    EvalRun,
    ReviewAssignment,
    ReviewSession,
)
from agent_challenge.evaluation.authorization import (
    EvalAuthorizationConflict,
    EvalAuthorizationRequired,
    cancel_eval_run,
    create_eval_run,
    eval_status_page,
    fail_eval_run,
    mark_eval_key_granted,
    retry_eval_run,
)
from agent_challenge.review.or_outcome_bind import (
    REVIEW_REPORT_DOMAIN,
    build_decision,
    build_observed_openrouter_transport,
    build_openrouter_observation,
    build_planned_openrouter_request,
    build_policy_observation,
    build_review_core_minimal,
    planned_request_sha256,
    review_report_data_hex,
    sha256_hex,
)
from agent_challenge.review.or_outcome_bind import (
    review_digest as bound_review_digest,
)
from agent_challenge.sdk.config import ChallengeSettings

MEASUREMENT = {
    "mrtd": "01" * 48,
    "rtmr0": "02" * 48,
    "rtmr1": "03" * 48,
    "rtmr2": "04" * 48,
    "os_image_hash": "05" * 32,
    "key_provider": "validator-kms",
    "vm_shape": "tdx-small",
}

_T0 = 1_700_000_000_000
_ROUTING = sha256_hex(b'{"order":["ledger"]}')
_BODY = b'{"model":"x-ai/grok-4.5","messages":[]}'
_BODY_SHA = sha256_hex(_BODY)
_RESP = b'{"id":"gen-ledger","model":"x-ai/grok-4.5","choices":[]}'
_RESP_SHA = sha256_hex(_RESP)
_META = sha256_hex(b"meta-ledger")


def _fresh_review_envelope() -> tuple[str, str, str]:
    """Return (envelope_json, report_data_hex, review_digest) for fresh allow.

    create_eval_run now re-verifies receipted materials (VAL-ACAT-028/029), so
    fixtures must plant real report_data-bound cores — not cache allow-bits alone.
    """

    planned = build_planned_openrouter_request(
        body_sha256=_BODY_SHA,
        body_length=len(_BODY),
        routing_sha256=_ROUTING,
    )
    p_digest = planned_request_sha256(planned)
    observed = build_observed_openrouter_transport(
        planned_request_sha256_=p_digest,
        response_body_sha256=_RESP_SHA,
        response_body_length=len(_RESP),
        metadata_sha256=_META,
    )
    or_obs = build_openrouter_observation(
        planned=planned,
        observed=observed,
        request_body_sha256=_BODY_SHA,
        request_body_length=len(_BODY),
        response_id="gen-ledger",
    )
    policy = build_policy_observation(
        prompt_version="review-policy-prompt-v1",
        prompt_bytes=b"prompt-ledger",
        tool_schema_version="review-policy-tool-v1",
        tool_schema_bytes=b"tools-ledger",
        verifier_version="review-policy-verifier-v1",
        verifier_bytes=b"verifier-ledger",
        routing_sha256=_ROUTING,
    )
    core = build_review_core_minimal(
        session_id="rs-ledger",
        assignment_id="ra-ledger",
        submission_id="sub-ledger",
        review_nonce="nonce-ledger",
        assignment_digest="13" * 32,
        rules_observation={
            "rules_version": "rules-v1",
            "rules_bundle_sha256": "11" * 32,
            "rules_files": [".rules/acceptance.md"],
            "rules_file_digests": {".rules/acceptance.md": "22" * 32},
            "rules_policy_text_sha256": "33" * 32,
        },
        policy_observation=policy,
        openrouter_observation=or_obs,
        decision=build_decision(verdict="allow"),
        times={
            "issued_at_ms": _T0,
            "started_at_ms": _T0,
            "model_call_marked_at_ms": _T0 + 1,
            "request_started_at_ms": _T0 + 2,
            "request_finished_at_ms": _T0 + 3,
            "verifier_finished_at_ms": _T0 + 4,
            "report_finished_at_ms": _T0 + 5,
            "expires_at_ms": _T0 + 3_600_000,
            "submission_received_at_ms": _T0 + 60_000,
        },
    )
    digest = bound_review_digest(core)
    rd = review_report_data_hex(core)
    env = {
        "schema_version": 1,
        "domain": REVIEW_REPORT_DOMAIN,
        "review_digest": digest,
        "report_data_hex": rd,
        "review_core": core,
    }
    return json.dumps(env, sort_keys=True, separators=(",", ":")), rd, digest


def _settings() -> ChallengeSettings:
    return ChallengeSettings(
        attested_review_enabled=True,
        phala_attestation_enabled=True,
        eval_app_image_ref="registry.example/eval@sha256:" + "a" * 64,
        eval_app_compose_hash="06" * 32,
        eval_app_identity="agent-challenge-eval-v1",
        eval_app_kms_public_key_hex="07" * 32,
        eval_app_measurement=MEASUREMENT,
        eval_app_measurement_allowlist=(
            {
                "mrtd": MEASUREMENT["mrtd"],
                "rtmr0": MEASUREMENT["rtmr0"],
                "rtmr1": MEASUREMENT["rtmr1"],
                "rtmr2": MEASUREMENT["rtmr2"],
                "compose_hash": "06" * 32,
                "os_image_hash": MEASUREMENT["os_image_hash"],
            },
        ),
        eval_key_release_endpoint="validator.example:8701",
        eval_k=2,
        evaluation_task_count=2,
    )


async def _authorized_submission(database_session) -> tuple[int, int]:
    envelope_json, report_data_hex, digest = _fresh_review_envelope()
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="ledger-miner",
            name="ledger-agent",
            agent_hash=hashlib.sha256(b"agent").hexdigest(),
            artifact_uri="/tmp/agent.zip",
            artifact_path="/tmp/agent.zip",
            zip_sha256=hashlib.sha256(b"zip").hexdigest(),
            zip_size_bytes=3,
            raw_status="review_allowed",
            status="queued",
            effective_status="queued",
            version_number=1,
        )
        session.add(submission)
        await session.flush()
        review_session = ReviewSession(
            session_id="review-ledger-session",
            submission_id=submission.id,
            artifact_sha256=submission.zip_sha256,
            artifact_size_bytes=3,
            manifest_sha256="11" * 32,
            manifest_entries_sha256="12" * 32,
            authorizing_assignment_id="review-ledger-assignment",
            current_assignment_id="review-ledger-assignment",
        )
        session.add(review_session)
        await session.flush()
        assignment = ReviewAssignment(
            session_id=review_session.id,
            assignment_id="review-ledger-assignment",
            attempt=1,
            assignment_bytes="{}",
            assignment_digest="13" * 32,
            artifact_sha256=submission.zip_sha256,
            rules_snapshot_sha256="14" * 32,
            rules_revision_id="rules-1",
            review_nonce="review-nonce",
            session_token_sha256="15" * 32,
            capability_state="revoked",
            phase="review_allowed",
            issued_at=datetime.now(UTC),
            expires_at=datetime.now(UTC),
            # Full receipted envelope for re-verify at create_eval_run (not cache-only).
            review_report_envelope_json=envelope_json,
            review_report_data_hex=report_data_hex,
            review_digest=digest,
            review_verification_outcome_json=(
                '{"status":"verified_allow","terminal":true,"retryable":false,'
                '"nonce_consumed":true}'
            ),
        )
        session.add(assignment)
        await session.commit()
        return submission.id, assignment.id


async def test_preparation_requires_persisted_verified_allow(database_session) -> None:
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="blocked-miner",
            name="blocked-agent",
            agent_hash="21" * 32,
            artifact_uri="/tmp/blocked.zip",
            artifact_path="/tmp/blocked.zip",
            zip_sha256="22" * 32,
            zip_size_bytes=1,
            raw_status="review_queued",
            status="queued",
            effective_status="queued",
            version_number=1,
        )
        session.add(submission)
        await session.commit()
        with pytest.raises(EvalAuthorizationRequired):
            await create_eval_run(session, submission, settings=_settings())
        assert await session.scalar(select(func.count()).select_from(EvalRun)) == 0


async def test_preparation_refuses_cached_allow_without_envelope(
    database_session,
) -> None:
    """VAL-ACAT-029: phase/status allow bits alone cannot open Eval prepare."""

    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="cache-only-miner",
            name="cache-only-agent",
            agent_hash="31" * 32,
            artifact_uri="/tmp/cache-only.zip",
            artifact_path="/tmp/cache-only.zip",
            zip_sha256="32" * 32,
            zip_size_bytes=1,
            raw_status="review_allowed",
            status="queued",
            effective_status="queued",
            version_number=1,
        )
        session.add(submission)
        await session.flush()
        review_session = ReviewSession(
            session_id="review-cache-only-session",
            submission_id=submission.id,
            artifact_sha256=submission.zip_sha256,
            artifact_size_bytes=1,
            manifest_sha256="41" * 32,
            manifest_entries_sha256="42" * 32,
            authorizing_assignment_id="review-cache-only-assignment",
            current_assignment_id="review-cache-only-assignment",
        )
        session.add(review_session)
        await session.flush()
        assignment = ReviewAssignment(
            session_id=review_session.id,
            assignment_id="review-cache-only-assignment",
            attempt=1,
            assignment_bytes="{}",
            assignment_digest="43" * 32,
            artifact_sha256=submission.zip_sha256,
            rules_snapshot_sha256="44" * 32,
            rules_revision_id="rules-1",
            review_nonce="review-nonce-cache",
            session_token_sha256="45" * 32,
            capability_state="revoked",
            phase="review_allowed",
            issued_at=datetime.now(UTC),
            expires_at=datetime.now(UTC),
            # Cache-shaped stub: no re-verifiable core/report_data.
            review_report_envelope_json='{"schema_version":1}',
            review_digest="16" * 32,
            review_verification_outcome_json=(
                '{"status":"verified_allow","terminal":true,"retryable":false,'
                '"nonce_consumed":true}'
            ),
        )
        session.add(assignment)
        await session.commit()
        with pytest.raises(EvalAuthorizationRequired) as exc:
            await create_eval_run(session, submission, settings=_settings())
        assert "fresh review re-verify" in str(exc.value)
        assert await session.scalar(select(func.count()).select_from(EvalRun)) == 0


async def test_prepare_is_one_time_and_issues_distinct_typed_nonces(
    database_session,
    monkeypatch,
) -> None:
    submission_id, _assignment_id = await _authorized_submission(database_session)
    monkeypatch.setattr(
        "agent_challenge.evaluation.authorization.load_benchmark_tasks",
        lambda: [
            type(
                "Task",
                (),
                {
                    "task_id": "task-b",
                    "docker_image": "registry.example/task@sha256:" + "b" * 64,
                    "prompt": "",
                    "benchmark": "terminal_bench",
                    "metadata": {"content_digest_sha256": "cd" * 32},
                },
            )(),
            type(
                "Task",
                (),
                {
                    "task_id": "task-a",
                    "docker_image": "registry.example/task@sha256:" + "c" * 64,
                    "prompt": "",
                    "benchmark": "terminal_bench",
                    "metadata": {"content_digest_sha256": "ab" * 32},
                },
            )(),
        ],
    )
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        assert submission is not None
        first = await create_eval_run(session, submission, settings=_settings())
        await session.commit()

    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        assert submission is not None
        second = await create_eval_run(session, submission, settings=_settings())
        assert second.run.eval_run_id == first.run.eval_run_id
        assert second.token is None
        assert second.plan == first.plan
        nonces = list((await session.scalars(select(EvalNonce))).all())
        assert {nonce.purpose for nonce in nonces} == {"key_release", "score"}
        assert len({nonce.nonce for nonce in nonces}) == 2
        assert len(nonces[0].nonce) >= 22
        assert "run_token_sha256" in first.plan
        assert first.plan["run_token_sha256"] == first.run.token_sha256
        assert first.plan["scoring_policy"]["schema_version"] == 1
        plan_tasks = {item["task_id"]: item for item in first.plan["selected_tasks"]}
        assert plan_tasks["task-a"]["task_config_sha256"] == "ab" * 32
        assert plan_tasks["task-b"]["task_config_sha256"] == "cd" * 32
        assert first.plan["agent_hash"] == submission.agent_hash


async def test_cancel_failure_and_retry_retain_attempt_history(
    database_session,
    monkeypatch,
) -> None:
    submission_id, _assignment_id = await _authorized_submission(database_session)
    monkeypatch.setattr(
        "agent_challenge.evaluation.authorization.load_benchmark_tasks",
        lambda: [
            type(
                "Task",
                (),
                {
                    "task_id": "task-a",
                    "docker_image": "registry.example/task@sha256:" + "b" * 64,
                    "prompt": "",
                    "benchmark": "terminal_bench",
                    "metadata": {"content_digest_sha256": "aa" * 32},
                },
            )()
        ],
    )
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        assert submission is not None
        created = await create_eval_run(session, submission, settings=_settings())
        await cancel_eval_run(session, submission, created.run.eval_run_id)
        repeated_cancel = await cancel_eval_run(
            session,
            submission,
            created.run.eval_run_id,
        )
        assert repeated_cancel.id == created.run.id
        replacement = await retry_eval_run(
            session,
            submission,
            expected_run_id=created.run.eval_run_id,
            settings=_settings(),
        )
        assert replacement.run.eval_run_id != created.run.eval_run_id
        assert replacement.token is not None
        page = await eval_status_page(session, submission)
        assert page["total_count"] == 2
        assert [item["eval_run_id"] for item in page["items"]] == [
            created.run.eval_run_id,
            replacement.run.eval_run_id,
        ]


async def test_failure_reason_is_closed_and_retryable(database_session, monkeypatch) -> None:
    submission_id, _assignment_id = await _authorized_submission(database_session)
    monkeypatch.setattr(
        "agent_challenge.evaluation.authorization.load_benchmark_tasks",
        lambda: [
            type(
                "Task",
                (),
                {
                    "task_id": "task-a",
                    "docker_image": "registry.example/task@sha256:" + "b" * 64,
                    "prompt": "",
                    "benchmark": "terminal_bench",
                    "metadata": {"content_digest_sha256": "aa" * 32},
                },
            )()
        ],
    )
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        assert submission is not None
        created = await create_eval_run(session, submission, settings=_settings())
        failed = await fail_eval_run(
            session,
            submission,
            expected_run_id=created.run.eval_run_id,
            reason_code="eval_key_release_unavailable",
        )
        assert failed.phase == "eval_error"
        assert failed.retryable is True
        replay = await fail_eval_run(
            session,
            submission,
            expected_run_id=created.run.eval_run_id,
            reason_code="eval_key_release_unavailable",
        )
        assert replay.id == failed.id


async def test_key_grant_closes_cancel_and_retry(database_session, monkeypatch) -> None:
    submission_id, _assignment_id = await _authorized_submission(database_session)
    monkeypatch.setattr(
        "agent_challenge.evaluation.authorization.load_benchmark_tasks",
        lambda: [
            type(
                "Task",
                (),
                {
                    "task_id": "task-a",
                    "docker_image": "registry.example/task@sha256:" + "b" * 64,
                    "prompt": "",
                    "benchmark": "terminal_bench",
                    "metadata": {"content_digest_sha256": "aa" * 32},
                },
            )()
        ],
    )
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        assert submission is not None
        created = await create_eval_run(session, submission, settings=_settings())
        granted = await mark_eval_key_granted(
            session,
            eval_run_id=created.run.eval_run_id,
        )
        assert granted.retryable is False
        assert granted.key_granted_at is not None
        with pytest.raises(EvalAuthorizationConflict):
            await cancel_eval_run(session, submission, created.run.eval_run_id)
        with pytest.raises(EvalAuthorizationConflict):
            await retry_eval_run(
                session,
                submission,
                expected_run_id=created.run.eval_run_id,
                settings=_settings(),
            )


async def test_status_uses_normative_safe_item_and_stable_cursor(
    database_session,
    monkeypatch,
) -> None:
    submission_id, _assignment_id = await _authorized_submission(database_session)
    monkeypatch.setattr(
        "agent_challenge.evaluation.authorization.load_benchmark_tasks",
        lambda: [
            type(
                "Task",
                (),
                {
                    "task_id": "task-a",
                    "docker_image": "registry.example/task@sha256:" + "b" * 64,
                    "prompt": "",
                    "benchmark": "terminal_bench",
                    "metadata": {"content_digest_sha256": "aa" * 32},
                },
            )()
        ],
    )
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        assert submission is not None
        first = await create_eval_run(session, submission, settings=_settings())
        await cancel_eval_run(session, submission, first.run.eval_run_id)
        second = await retry_eval_run(
            session,
            submission,
            expected_run_id=first.run.eval_run_id,
            settings=_settings(),
        )
        page = await eval_status_page(session, submission, limit=1)
        expected_fields = {
            "eval_run_id",
            "attempt",
            "prior_eval_run_id",
            "receipt_id",
            "body_sha256",
            "phase",
            "terminal",
            "verified",
            "retryable",
            "reason_code",
            "key_grant_state",
            "key_release_nonce_state",
            "score_nonce_state",
            "issued_at_ms",
            "expires_at_ms",
            "received_at_ms",
            "finalized_at_ms",
            "result_available",
        }
        assert set(page["items"][0]) == expected_fields
        assert page["items"][0]["eval_run_id"] == first.run.eval_run_id
        assert page["items"][0]["prior_eval_run_id"] is None
        assert page["next_cursor"] is not None
        assert "selected_tasks" not in page["items"][0]
        assert second.run.eval_run_id not in str(page["items"][0])

        next_page = await eval_status_page(
            session,
            submission,
            cursor=page["next_cursor"],
            limit=1,
        )
        assert [item["eval_run_id"] for item in next_page["items"]] == [second.run.eval_run_id]
        with pytest.raises(EvalAuthorizationConflict, match="cursor"):
            await eval_status_page(session, submission, cursor="tampered", limit=1)
