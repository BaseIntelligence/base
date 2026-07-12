"""Production HTTP implementations of the orchestration driver's challenge seams.

The live master driver (:mod:`base.master.orchestration`) is challenge-agnostic;
these adapters realize its :class:`ChallengeWorkSource` /
:class:`ChallengeFoldTrigger` protocols against the challenge services over their
internal-token-gated HTTP routes:

- ``GET /internal/v1/work_units`` exposes each challenge's currently-assignable
  pending work units (agent-challenge: one descriptor per selected task carrying
  ``job_id``/``task_id``; prism: one descriptor per submission carrying its
  resume ``checkpoint_ref`` in the payload).
- ``POST /internal/v1/work_units/fold`` folds a permanently-failed agent-challenge
  work unit back into its EvaluationJob.

The challenge base URL + bearer token are resolved from the master challenge
registry exactly as the weight-collection path does.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import httpx

from base.challenge_sdk.schemas import ExecutionProof, ExternalResultEnvelope
from base.master.orchestration import (
    WORK_UNIT_MAX_ATTEMPTS_REASON,
    ChallengePendingWork,
)
from base.master.replay_audit import (
    ReplayAuditRequest,
    ReplayAuditResult,
    ReplayAuditWireError,
    parse_replay_json,
)

logger = logging.getLogger(__name__)

#: Payload key prism uses to carry a resume checkpoint to a reassigned unit.
RESUME_CHECKPOINT_PAYLOAD_KEY = "resume_checkpoint_ref"


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class HttpChallengeWorkSource:
    """Fetch pending work units from every active challenge over HTTP."""

    def __init__(
        self,
        registry: Any,
        *,
        timeout_seconds: float = 10.0,
        retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._registry = registry
        self._timeout_seconds = timeout_seconds
        self._retries = retries
        self._transport = transport

    async def fetch_pending_work(self) -> list[ChallengePendingWork]:
        records = await _resolve(self._registry.list(active_only=True))
        works: list[ChallengePendingWork] = []
        for record in records:
            token = await _resolve(self._registry.get_token(record.slug))
            if not token:
                logger.warning(
                    "challenge %s has no token; skipping work-unit bridge",
                    record.slug,
                )
                continue
            payload = await self._fetch_work_units(
                slug=record.slug,
                base_url=record.internal_base_url,
                token=str(token),
            )
            if payload is None:
                continue
            works.extend(_parse_work_units(record.slug, payload))
        return works

    async def _fetch_work_units(
        self, *, slug: str, base_url: str, token: str
    ) -> dict[str, Any] | None:
        url = f"{base_url.rstrip('/')}/internal/v1/work_units"
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Base-Challenge-Slug": slug,
            "Accept": "application/json",
        }
        last_error = "unknown error"
        for _attempt in range(max(self._retries, 1)):
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout_seconds, transport=self._transport
                ) as client:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    return dict(response.json())
            except Exception as exc:  # noqa: BLE001 - logged, retried, then skipped
                last_error = str(exc)
        logger.warning(
            "failed to fetch work units for challenge %s: %s", slug, last_error
        )
        return None


class HttpChallengeReplayClient:
    """Fetch labelled replay requests and post raw replay trials.

    This client is intentionally separate from :class:`HttpChallengeWorkSource`.
    The normal ``/internal/v1/work_units`` route is never used for replay audits,
    and a replay body is accepted only after the explicit protocol label and
    immutable-plan checks pass.
    """

    def __init__(
        self,
        registry: Any,
        *,
        timeout_seconds: float = 10.0,
        retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._registry = registry
        self._timeout_seconds = timeout_seconds
        self._retries = retries
        self._transport = transport

    async def fetch_request(
        self, *, challenge_slug: str, eval_run_id: str
    ) -> ReplayAuditRequest:
        record = await _resolve(self._registry.get(challenge_slug))
        token = await _resolve(self._registry.get_token(challenge_slug))
        if not token:
            raise RuntimeError(
                f"challenge {challenge_slug!r} has no token for replay request"
            )
        url = (
            f"{record.internal_base_url.rstrip('/')}/internal/v1/replay-audits/"
            f"{eval_run_id}/request"
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Base-Challenge-Slug": challenge_slug,
            "Accept": "application/json",
        }
        last_error: Exception | None = None
        for _attempt in range(max(self._retries, 1)):
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout_seconds, transport=self._transport
                ) as client:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    raw = response.content
                parsed = parse_replay_json(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("replay request response must be an object")
                request = ReplayAuditRequest.from_mapping(parsed, raw_body=raw)
                if request.eval_run_id != eval_run_id:
                    raise ValueError("replay request run identity mismatch")
                return request
            except ReplayAuditWireError:
                raise
            except (ValueError, TypeError):
                raise
            except Exception as exc:  # noqa: BLE001 - retry transport failures
                last_error = exc
        assert last_error is not None
        raise RuntimeError(
            f"failed to fetch replay request for {eval_run_id} on "
            f"{challenge_slug}: {last_error}"
        ) from last_error

    async def post_result(
        self,
        *,
        challenge_slug: str,
        result: ReplayAuditResult,
    ) -> dict[str, Any]:
        record = await _resolve(self._registry.get(challenge_slug))
        token = await _resolve(self._registry.get_token(challenge_slug))
        if not token:
            raise RuntimeError(
                f"challenge {challenge_slug!r} has no token for replay result"
            )
        url = (
            f"{record.internal_base_url.rstrip('/')}/internal/v1/replay-audits/"
            f"{result.eval_run_id}/result"
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Base-Challenge-Slug": challenge_slug,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        body = result.to_dict()
        last_error: Exception | None = None
        for _attempt in range(max(self._retries, 1)):
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout_seconds, transport=self._transport
                ) as client:
                    response = await client.post(url, json=body, headers=headers)
                    response.raise_for_status()
                    payload = parse_replay_json(response.content)
                if not isinstance(payload, dict):
                    raise ValueError("replay result response must be an object")
                return payload
            except (ReplayAuditWireError, ValueError, TypeError):
                raise
            except Exception as exc:  # noqa: BLE001 - retry transient HTTP failures
                last_error = exc
        assert last_error is not None
        raise RuntimeError(
            f"failed to post replay result {result.audit_id} on "
            f"{challenge_slug}: {last_error}"
        ) from last_error

    async def forward(
        self, *, challenge_slug: str, result: ReplayAuditResult
    ) -> dict[str, Any]:
        """Forward one validated replay result to the challenge comparator."""

        return await self.post_result(challenge_slug=challenge_slug, result=result)


class HttpChallengeReplaySource:
    """Discover sampled replay requests without touching normal work units."""

    def __init__(
        self,
        registry: Any,
        *,
        timeout_seconds: float = 10.0,
        retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._registry = registry
        self._client = HttpChallengeReplayClient(
            registry,
            timeout_seconds=timeout_seconds,
            retries=retries,
            transport=transport,
        )

    async def fetch_sampled_requests(self) -> list[ReplayAuditRequest]:
        records = await _resolve(self._registry.list(active_only=True))
        requests: list[ReplayAuditRequest] = []
        for record in records:
            if record.slug != "agent-challenge":
                continue
            token = await _resolve(self._registry.get_token(record.slug))
            if not token:
                continue
            url = (
                f"{record.internal_base_url.rstrip('/')}"
                "/internal/v1/replay-audits/requests"
            )
            headers = {
                "Authorization": f"Bearer {token}",
                "X-Base-Challenge-Slug": record.slug,
                "Accept": "application/json",
            }
            try:
                async with httpx.AsyncClient(
                    timeout=self._client._timeout_seconds,
                    transport=self._client._transport,
                ) as client:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    body = parse_replay_json(response.content)
                raw_requests = body.get("requests") if isinstance(body, dict) else None
                if not isinstance(raw_requests, list):
                    raise ValueError("replay request list must contain requests")
                for raw in raw_requests:
                    requests.append(ReplayAuditRequest.from_mapping(raw))
            except ReplayAuditWireError:
                raise
            except Exception as exc:  # noqa: BLE001 - audit is best effort
                logger.warning("failed to discover replay requests: %s", exc)
        return requests


def _parse_work_units(slug: str, payload: dict[str, Any]) -> list[ChallengePendingWork]:
    """Map a challenge ``work_units`` response into bridgeable pending work.

    agent-challenge units (which carry ``task_id``/``job_id``) are grouped per
    ``(submission, job)`` into one cpu fan-out; prism units (one per submission)
    become a single gpu unit each, surfacing any resume checkpoint ref.
    """

    units = payload.get("work_units") or []
    agent_groups: dict[tuple[str, str], dict[str, Any]] = {}
    works: list[ChallengePendingWork] = []
    for unit in units:
        task_id = unit.get("task_id")
        job_id = unit.get("job_id")
        submission_id = str(unit.get("submission_id"))
        submission_ref = str(unit.get("submission_ref") or "")
        if task_id and job_id:
            key = (submission_id, str(job_id))
            group = agent_groups.get(key)
            if group is None:
                group = {
                    "submission_ref": submission_ref,
                    "task_ids": [],
                }
                agent_groups[key] = group
            group["task_ids"].append(str(task_id))
        else:
            unit_payload = dict(unit.get("payload") or {})
            checkpoint_ref = unit_payload.pop(RESUME_CHECKPOINT_PAYLOAD_KEY, None)
            works.append(
                ChallengePendingWork(
                    challenge_slug=slug,
                    submission_id=submission_id,
                    submission_ref=submission_ref,
                    task_ids=(),
                    checkpoint_ref=str(checkpoint_ref) if checkpoint_ref else None,
                    payload=unit_payload,
                )
            )
    for (submission_id, job_id), group in agent_groups.items():
        works.append(
            ChallengePendingWork(
                challenge_slug=slug,
                submission_id=submission_id,
                submission_ref=str(group["submission_ref"]),
                task_ids=tuple(group["task_ids"]),
                job_id=job_id,
            )
        )
    return works


class HttpChallengeFoldTrigger:
    """Fold a permanently-failed agent-challenge work unit over HTTP."""

    def __init__(
        self,
        registry: Any,
        *,
        timeout_seconds: float = 10.0,
        retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._registry = registry
        self._timeout_seconds = timeout_seconds
        self._retries = retries
        self._transport = transport

    async def fold(
        self,
        *,
        challenge_slug: str,
        job_id: str,
        task_id: str,
        reason: str = WORK_UNIT_MAX_ATTEMPTS_REASON,
    ) -> None:
        record = await _resolve(self._registry.get(challenge_slug))
        token = await _resolve(self._registry.get_token(challenge_slug))
        if not token:
            raise RuntimeError(f"challenge {challenge_slug!r} has no token for fold")
        url = f"{record.internal_base_url.rstrip('/')}/internal/v1/work_units/fold"
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Base-Challenge-Slug": challenge_slug,
            "Accept": "application/json",
        }
        body = {"job_id": job_id, "task_id": task_id, "reason": reason}
        last_error = "unknown error"
        for _attempt in range(max(self._retries, 1)):
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout_seconds, transport=self._transport
                ) as client:
                    response = await client.post(url, json=body, headers=headers)
                    response.raise_for_status()
                return
            except Exception as exc:  # noqa: BLE001 - raised after retries exhausted
                last_error = str(exc)
        raise RuntimeError(
            f"failed to fold work unit {job_id}:{task_id} on {challenge_slug}: "
            f"{last_error}"
        )


class HttpChallengeResultForwarder:
    """Forward a reconciled worker result to the challenge over HTTP.

    Realizes :class:`base.master.worker_reconciliation.ChallengeResultForwarder`:
    the master posts an accepted (reconciled) worker result to the challenge's
    internal result route so the challenge folds exactly one outcome for the unit
    (architecture.md sec 3.3). The challenge base URL + bearer token are resolved
    from the registry exactly as the fold path does.
    """

    def __init__(
        self,
        registry: Any,
        *,
        timeout_seconds: float = 10.0,
        retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._registry = registry
        self._timeout_seconds = timeout_seconds
        self._retries = retries
        self._transport = transport

    async def forward_result(
        self,
        *,
        challenge_slug: str,
        work_unit_id: str,
        submission_ref: str,
        result_payload: Any,
    ) -> None:
        record = await _resolve(self._registry.get(challenge_slug))
        token = await _resolve(self._registry.get_token(challenge_slug))
        if not token:
            raise RuntimeError(
                f"challenge {challenge_slug!r} has no token for result forward"
            )
        url = f"{record.internal_base_url.rstrip('/')}/internal/v1/work_units/result"
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Base-Challenge-Slug": challenge_slug,
            "Accept": "application/json",
        }
        payload = dict(result_payload or {})
        proof = payload.get("execution_proof")
        if not isinstance(proof, dict):
            # Fail closed before any network POST: dual/legacy reduced bodies
            # without a bound execution_proof are not part of the wire contract.
            raise RuntimeError(
                f"refusing to forward result for work unit {work_unit_id} on "
                f"{challenge_slug}: execution_proof is required for "
                "ExternalResultEnvelope"
            )
        try:
            envelope = ExternalResultEnvelope(
                api_version="1.0",
                work_unit_id=work_unit_id,
                assignment_id=work_unit_id,
                submission_ref=submission_ref,
                challenge_slug=challenge_slug,
                result=payload,
                proof=ExecutionProof.model_validate(proof),
            )
        except Exception as exc:
            raise RuntimeError(
                f"refusing to forward result for work unit {work_unit_id} on "
                f"{challenge_slug}: result does not satisfy ExternalResultEnvelope "
                f"({exc})"
            ) from exc
        body = envelope.model_dump(mode="json")
        last_error = "unknown error"
        for _attempt in range(max(self._retries, 1)):
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout_seconds, transport=self._transport
                ) as client:
                    response = await client.post(url, json=body, headers=headers)
                    response.raise_for_status()
                return
            except Exception as exc:  # noqa: BLE001 - raised after retries exhausted
                last_error = str(exc)
        raise RuntimeError(
            f"failed to forward result for work unit {work_unit_id} on "
            f"{challenge_slug}: {last_error}"
        )


__all__ = [
    "HttpChallengeFoldTrigger",
    "HttpChallengeReplayClient",
    "HttpChallengeReplaySource",
    "HttpChallengeResultForwarder",
    "HttpChallengeWorkSource",
]
