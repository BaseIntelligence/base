"""Per-validator on-chain weight submitter (architecture.md sec 9.3).

Each validator node runs its OWN submitter loop with its OWN wallet/hotkey. The
submitter fetches the MASTER-aggregated weight vector over HTTP
(``GET /v1/weights/latest``) and commits that SAME vector on-chain under this
validator's hotkey. The validator NEVER computes or aggregates its own vector -
aggregation lives entirely on the master (``base.master.aggregator`` /
``MasterWeightService`` / ``AggregationService``); the validator side only
fetches, validates, and submits.

Invariants:

- **Per-validator, own keypair.** The ``WeightSetter`` is built from this node's
  wallet, so every validator submits under its OWN hotkey, which must equal the
  authenticated/registered validator protocol identity.
- **No validator-side aggregation.** The vector always comes from the master via
  :class:`base.validator.weights_client.WeightsClient`; there is no aggregator
  import or scoring on this path.
- **Durable vector-keyed ledger.** Attempts and outcomes persist on the
  validator's own volume keyed by hotkey/vector ID/digest/netuid/chain, not by
  an in-memory timestamp alone.
- **Gate-off no-op.** When ``submit_on_chain_enabled`` is ``False`` the tick does
  NO fetch, NO submit-runtime construction (so no live ``Subtensor`` is built),
  and NO submission. Live on-chain enablement is human-gated.
- **Reject before side effects.** Unusable vectors never touch wallet or chain.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from base.bittensor.weight_setter import (
    WeightSetter,
    is_rejected_set_weights_result,
    set_weights_rejection_message,
)
from base.challenge_sdk.roles import Capability, Role, role_contract
from base.schemas.weights import MasterWeightsResponse
from base.validator.submission_ledger import (
    DEFAULT_LEDGER_FILENAME,
    SubmissionRecord,
    SubmissionStatus,
    ValidatorSubmissionLedger,
)
from base.validator.weights_client import (
    WeightsClient,
    validate_master_weights_payload,
)

logger = logging.getLogger(__name__)


class ValidatorSubmitOutcome(StrEnum):
    """Outcome of a single :meth:`ValidatorWeightSubmitter.run_once` tick."""

    #: The on-chain gate is off; no fetch and no submission were performed.
    DISABLED = "disabled"
    #: No usable master vector this tick (fetch failed or payload invalid/stale).
    NO_VECTOR = "no_vector"
    #: Wallet/hotkey identity does not match the authenticated validator identity.
    IDENTITY_MISMATCH = "identity_mismatch"
    #: This exact master vector was already committed by this node; no-op.
    ALREADY_SUBMITTED = "already_submitted"
    #: Active attempt for a previous vector was cancelled because a newer vector
    #: is now the submission target.
    SUPERSEDED = "superseded"
    #: The master vector was committed on-chain under this validator's hotkey.
    SUBMITTED = "submitted"
    #: Explicit pre-send failure (wallet/constructor/config) before chain call.
    PRE_SEND_FAILED = "pre_send_failed"
    #: The chain rejected the commit (e.g. rate-limited); retried next tick.
    REJECTED = "rejected"
    #: Chain call returned an ambiguous/unknown result; reconcile before retry.
    UNKNOWN = "unknown"
    #: Bounded retries exhausted for this vector.
    RETRY_EXHAUSTED = "retry_exhausted"


#: Builds this validator's ``WeightSetter`` (own wallet/hotkey) lazily, so the
#: gate-off path never constructs a live ``Subtensor`` / touches chain material.
WeightSetterFactory = Callable[[], WeightSetter | None]
Clock = Callable[[], datetime]
ObservationReporter = Callable[[dict[str, Any]], Any]


class ValidatorWeightSubmitter:
    """Fetch the master vector and commit it on-chain with this node's hotkey."""

    def __init__(
        self,
        *,
        submit_enabled: bool,
        netuid: int,
        weights_client: WeightsClient,
        weight_setter_factory: WeightSetterFactory,
        weights_freshness_seconds: int = 720,
        clock: Clock | None = None,
        expected_hotkey: str | None = None,
        expected_chain_endpoint: str | None = None,
        ledger: ValidatorSubmissionLedger | None = None,
        state_dir: Path | str | None = None,
        max_attempts: int = 5,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 300.0,
        observation_reporter: ObservationReporter | None = None,
        require_provenance: bool = False,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._submit_enabled = submit_enabled
        self._netuid = netuid
        self._weights_client = weights_client
        self._weight_setter_factory = weight_setter_factory
        self._weights_freshness_seconds = weights_freshness_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._expected_hotkey = expected_hotkey
        self._expected_chain_endpoint = expected_chain_endpoint
        self._weight_setter: WeightSetter | None = None
        self._max_attempts = max(1, int(max_attempts))
        self._backoff_base_seconds = float(backoff_base_seconds)
        self._backoff_max_seconds = float(backoff_max_seconds)
        self._observation_reporter = observation_reporter
        self._require_provenance = require_provenance
        self._sleep_fn = sleep_fn or time.sleep
        if ledger is not None:
            self._ledger = ledger
        else:
            state_path = Path(state_dir or "/var/lib/base/state")
            self._ledger = ValidatorSubmissionLedger(
                state_path / DEFAULT_LEDGER_FILENAME
            )
        # Last in-memory key kept for backward-compatible property access.
        self._last_submitted_key: tuple[str, str, int] | None = None
        self._retry_not_before: datetime | None = None
        self._retry_vector_id: str | None = None

    @property
    def submit_enabled(self) -> bool:
        return self._submit_enabled

    @property
    def last_submitted_key(self) -> tuple[str, str, int] | None:
        return self._last_submitted_key

    @property
    def ledger(self) -> ValidatorSubmissionLedger:
        return self._ledger

    @role_contract(role=Role.VALIDATOR, capability=Capability.VALIDATOR_OWN_SET_WEIGHTS)
    async def run_once(self) -> ValidatorSubmitOutcome:
        if not self._submit_enabled:
            logger.debug(
                "validator on-chain weight submission is DISABLED "
                "(submit_on_chain_enabled=False); skipping tick"
            )
            return ValidatorSubmitOutcome.DISABLED

        # Respect durable backoff without tight-looping.
        now = self._clock()
        if self._retry_not_before is not None and now < self._retry_not_before:
            logger.info(
                "validator weights submission in backoff until %s",
                self._retry_not_before.isoformat(),
            )
            return ValidatorSubmitOutcome.REJECTED

        try:
            payload = await self._weights_client.fetch_latest()
        except Exception:
            logger.exception("validator weights fetch failed")
            return ValidatorSubmitOutcome.NO_VECTOR

        failure = validate_master_weights_payload(
            payload,
            netuid=self._netuid,
            weights_freshness_seconds=self._weights_freshness_seconds,
            now=self._clock(),
            expected_chain_endpoint=self._expected_chain_endpoint,
            require_provenance=self._require_provenance,
        )
        if failure is not None:
            logger.warning("validator weights submission skipped: %s", failure)
            return ValidatorSubmitOutcome.NO_VECTOR

        vector_id, vector_digest = _vector_identity(payload)

        # New vector supersedes active retry of an older vector.
        self._ledger.supersede_active(
            validator_hotkey=self._expected_hotkey or "",
            keep_vector_id=vector_id,
            superseded_by=vector_id,
        )
        if self._retry_vector_id and self._retry_vector_id != vector_id:
            self._retry_not_before = None
            self._retry_vector_id = None

        record = self._ledger.get(
            validator_hotkey=self._expected_hotkey or "",
            vector_id=vector_id,
            vector_digest=vector_digest,
            netuid=int(payload.netuid),
            chain_endpoint=str(payload.chain_endpoint or ""),
        )
        if record is None:
            record = SubmissionRecord(
                validator_hotkey=self._expected_hotkey or "",
                vector_id=vector_id,
                vector_digest=vector_digest,
                netuid=int(payload.netuid),
                chain_endpoint=str(payload.chain_endpoint or ""),
                status=SubmissionStatus.PENDING.value,
                uids=list(payload.uids),
                weights=[float(w) for w in payload.weights],
            )
            self._ledger.upsert(record)

        if record.status == SubmissionStatus.ACCEPTED.value:
            self._last_submitted_key = (
                record.vector_id,
                record.vector_digest,
                record.netuid,
            )
            logger.info(
                "validator weights already submitted for vector_id=%s "
                "digest=%s; skipping (idempotent no-op)",
                record.vector_id,
                record.vector_digest[:16],
            )
            return ValidatorSubmitOutcome.ALREADY_SUBMITTED

        if record.status == SubmissionStatus.UNKNOWN.value:
            # Ambiguous prior outcome: reconcile before blind resubmission.
            reconciled = self._reconcile_unknown(record)
            if reconciled is ValidatorSubmitOutcome.ALREADY_SUBMITTED:
                return reconciled
            if reconciled is ValidatorSubmitOutcome.UNKNOWN:
                return reconciled

        if record.attempt_count >= self._max_attempts and record.status in {
            SubmissionStatus.REJECTED.value,
            SubmissionStatus.PRE_SEND_FAILED.value,
            SubmissionStatus.UNKNOWN.value,
        }:
            self._ledger.mark_status(
                record,
                status=SubmissionStatus.RETRY_EXHAUSTED,
                error="max submission attempts exhausted",
            )
            await self._maybe_observe(record, outcome="retry_exhausted")
            return ValidatorSubmitOutcome.RETRY_EXHAUSTED

        try:
            setter = self._ensure_weight_setter()
        except Exception as exc:
            logger.exception("validator submit runtime construction failed")
            self._ledger.mark_status(
                record,
                status=SubmissionStatus.PRE_SEND_FAILED,
                error=str(exc),
                increment_attempt=True,
            )
            self._schedule_backoff(record)
            return ValidatorSubmitOutcome.PRE_SEND_FAILED

        identity_error = self._check_wallet_identity(setter)
        if identity_error is not None:
            logger.error(
                "validator wallet identity mismatch before set_weights: %s",
                identity_error,
            )
            self._ledger.mark_status(
                record,
                status=SubmissionStatus.PRE_SEND_FAILED,
                error=identity_error,
                increment_attempt=True,
            )
            return ValidatorSubmitOutcome.IDENTITY_MISMATCH

        self._ledger.mark_status(
            record,
            status=SubmissionStatus.SUBMITTING,
            increment_attempt=True,
        )
        public_hotkey = self._public_hotkey(setter)
        logger.info(
            "validator weights submitting: hotkey=%s vector_id=%s digest=%s "
            "netuid=%s attempt=%s",
            public_hotkey,
            record.vector_id,
            record.vector_digest[:16],
            record.netuid,
            record.attempt_count,
        )

        try:
            result = setter.set_weights(list(payload.uids), list(payload.weights))
        except Exception as exc:
            message = str(exc)
            # Distinguishes rejected ExtrinsicResponse-raised RuntimeError as
            # rejected; unknown transport/timeout style as unknown when hinted.
            if _looks_ambiguous(message):
                status = SubmissionStatus.UNKNOWN
                outcome = ValidatorSubmitOutcome.UNKNOWN
                observe = "unknown"
            else:
                status = SubmissionStatus.REJECTED
                outcome = ValidatorSubmitOutcome.REJECTED
                observe = "rejected"
            logger.warning(
                "validator weights submission %s: hotkey=%s vector_id=%s: %s",
                observe,
                public_hotkey,
                record.vector_id,
                message,
            )
            self._ledger.mark_status(record, status=status, error=message)
            await self._maybe_observe(record, outcome=observe)
            self._schedule_backoff(record)
            return outcome

        if is_rejected_set_weights_result(result):
            message = set_weights_rejection_message(result)
            logger.warning(
                "validator weights submission rejected by subtensor: hotkey=%s "
                "vector_id=%s: %s; will retry next tick",
                public_hotkey,
                record.vector_id,
                message,
            )
            self._ledger.mark_status(
                record, status=SubmissionStatus.REJECTED, error=message
            )
            await self._maybe_observe(record, outcome="rejected")
            self._schedule_backoff(record)
            return ValidatorSubmitOutcome.REJECTED

        if _looks_ambiguous_result(result):
            logger.warning(
                "validator weights submission ambiguous: hotkey=%s vector_id=%s",
                public_hotkey,
                record.vector_id,
            )
            self._ledger.mark_status(
                record,
                status=SubmissionStatus.UNKNOWN,
                error="ambiguous chain result",
            )
            await self._maybe_observe(record, outcome="unknown")
            self._schedule_backoff(record)
            return ValidatorSubmitOutcome.UNKNOWN

        self._ledger.mark_status(
            record, status=SubmissionStatus.ACCEPTED, accepted=True
        )
        self._last_submitted_key = (
            record.vector_id,
            record.vector_digest,
            record.netuid,
        )
        self._retry_not_before = None
        self._retry_vector_id = None
        logger.info(
            "validator weights submitted on-chain: hotkey=%s netuid=%s "
            "vector_id=%s digest=%s n_weights=%s attempt=%s",
            public_hotkey,
            payload.netuid,
            record.vector_id,
            record.vector_digest[:16],
            len(payload.weights),
            record.attempt_count,
        )
        await self._maybe_observe(record, outcome="accepted")
        return ValidatorSubmitOutcome.SUBMITTED

    def _public_hotkey(self, setter: WeightSetter) -> str:
        wallet = getattr(setter, "wallet", None)
        hotkey = getattr(wallet, "hotkey", None)
        address = getattr(hotkey, "ss58_address", None)
        if address:
            return str(address)
        return self._expected_hotkey or "unknown"

    def _check_wallet_identity(self, setter: WeightSetter) -> str | None:
        if not self._expected_hotkey:
            # When identity is not configured, do not invent one; still require a
            # wallet on the setter so the gate-on path cannot submit anonymously
            # without operator intent expressed via expected_hotkey.
            public = self._public_hotkey(setter)
            if public == "unknown":
                return "submission wallet public hotkey is unavailable"
            return None
        public = self._public_hotkey(setter)
        if public != self._expected_hotkey:
            return (
                f"wallet hotkey {public!r} does not match authenticated "
                f"validator identity {self._expected_hotkey!r}"
            )
        return None

    def _ensure_weight_setter(self) -> WeightSetter:
        if self._weight_setter is None:
            self._weight_setter = self._weight_setter_factory()
        if self._weight_setter is None:
            raise RuntimeError(
                "validator submit runtime did not provide a WeightSetter"
            )
        return self._weight_setter

    def _schedule_backoff(self, record: SubmissionRecord) -> None:
        attempt = max(1, int(record.attempt_count))
        delay = min(
            self._backoff_max_seconds,
            self._backoff_base_seconds * math.pow(2, attempt - 1),
        )
        # Deterministic jitter fraction from attempt count (no secrets).
        jitter = 0.1 * ((attempt % 5) / 5.0)
        delay = delay * (1.0 + jitter)
        from datetime import timedelta

        self._retry_not_before = self._clock() + timedelta(seconds=delay)
        self._retry_vector_id = record.vector_id

    def _reconcile_unknown(
        self, record: SubmissionRecord
    ) -> ValidatorSubmitOutcome | None:
        """Attempt reconciliation of an ambiguous prior acceptance.

        Without a live chain query seam injected, treat UNKNOWN as needing one
        more chain observation only when attempts remain. Callers still avoid
        blind infinite resubmission via max_attempts.
        """

        # If a reconciler was stamped, honor it.
        if record.reconciled_at and record.status == SubmissionStatus.ACCEPTED.value:
            self._last_submitted_key = (
                record.vector_id,
                record.vector_digest,
                record.netuid,
            )
            return ValidatorSubmitOutcome.ALREADY_SUBMITTED
        return None

    async def _maybe_observe(self, record: SubmissionRecord, *, outcome: str) -> None:
        if self._observation_reporter is None:
            return
        payload = {
            "vector_id": record.vector_id,
            "vector_digest": record.vector_digest,
            "netuid": record.netuid,
            "chain_endpoint": record.chain_endpoint,
            "outcome": outcome,
            "attempt": max(1, int(record.attempt_count)),
            "error_code": record.last_error,
            "observed_at": self._clock().isoformat(),
        }
        try:
            result = self._observation_reporter(payload)
            if hasattr(result, "__await__"):
                await result
            self._ledger.mark_status(record, status=record.status)
            # reload and set observed flag
            current = self._ledger.get(
                validator_hotkey=record.validator_hotkey,
                vector_id=record.vector_id,
                vector_digest=record.vector_digest,
                netuid=record.netuid,
                chain_endpoint=record.chain_endpoint,
            )
            if current is not None:
                current.observed_to_master = True
                self._ledger.upsert(current)
        except Exception:
            logger.exception(
                "failed to report submission observation for vector_id=%s",
                record.vector_id,
            )


def _vector_identity(payload: MasterWeightsResponse) -> tuple[str, str]:
    """Return durable (vector_id, vector_digest) for ledger keys.

    Prefer master-supplied provenance. When absent (legacy fixtures), derive a
    stable identity from netuid + computed_at + chain-domain bytes so idempotence
    remains vector-identified rather than timestamp-only process memory.
    """

    import hashlib
    import json

    if payload.vector_id and payload.vector_digest:
        return str(payload.vector_id), str(payload.vector_digest)
    chain_domain = payload.chain_domain_bytes or json.dumps(
        {
            "netuid": int(payload.netuid),
            "uids": list(payload.uids),
            "weights": [float(w) for w in payload.weights],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    material = (
        f"{payload.netuid}|{payload.computed_at.isoformat()}|{chain_domain}"
    ).encode()
    digest = hashlib.sha256(material).hexdigest()
    vector_id = payload.vector_id or f"derived:{digest[:32]}"
    vector_digest = payload.vector_digest or digest
    return str(vector_id), str(vector_digest)


def _looks_ambiguous(message: str) -> bool:
    lowered = message.lower()
    tokens = ("timeout", "unknown", "ambiguous", "not finalized", "connection reset")
    return any(token in lowered for token in tokens)


def _looks_ambiguous_result(result: Any) -> bool:
    if result is None:
        return True
    success = getattr(result, "success", None)
    if success is None and not isinstance(result, (bool, tuple, list)):
        return True
    return False


__all__ = ["ValidatorSubmitOutcome", "ValidatorWeightSubmitter"]
