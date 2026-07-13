"""HTTP client and chain-domain validation for master weight vectors.

Validators fetch the master's immutable vector over HTTP and re-validate every
invariant before any wallet or chain side effect.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import httpx

from base.schemas.weights import MasterWeightsResponse


def recompute_vector_digest(
    *,
    protocol_version: str,
    epoch: int,
    revision: int,
    netuid: int,
    chain_endpoint: str,
    uids: Sequence[int],
    weights: Sequence[float],
    emission_policy_version: str,
    emission_shares: Mapping[str, float],
    burn_policy_version: str,
    mapping_policy_version: str,
    source_snapshot_ids: Sequence[str],
    source_snapshot_digests: Sequence[str],
    metagraph_hash: str | None,
) -> tuple[str, str]:
    """Recompute (digest_hex, chain_domain_json) without master imports.

    Mirrors the master's ``compute_vector_digest`` chain-domain / digest
    algorithm so validators can verify provenance independently.
    """

    chain_domain = {
        "netuid": int(netuid),
        "uids": [int(uid) for uid in uids],
        "weights": [float(weight) for weight in weights],
    }
    chain_domain_bytes = json.dumps(
        chain_domain, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    body = {
        "protocol_version": protocol_version,
        "epoch": int(epoch),
        "revision": int(revision),
        "netuid": int(netuid),
        "chain_endpoint": chain_endpoint,
        "uids": chain_domain["uids"],
        "weights": chain_domain["weights"],
        "emission_policy_version": emission_policy_version,
        "emission_shares": {
            str(slug): float(share) for slug, share in sorted(emission_shares.items())
        },
        "burn_policy_version": burn_policy_version,
        "mapping_policy_version": mapping_policy_version,
        "source_snapshot_ids": list(source_snapshot_ids),
        "source_snapshot_digests": list(source_snapshot_digests),
        "metagraph_hash": metagraph_hash,
        "chain_domain": chain_domain,
    }
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest, chain_domain_bytes


def validate_master_weights_payload(
    payload: MasterWeightsResponse,
    *,
    netuid: int | None,
    weights_freshness_seconds: int,
    now: datetime | None = None,
    expected_chain_endpoint: str | None = None,
    require_provenance: bool = False,
    max_uid: int | None = None,
) -> str | None:
    """Validate a fetched master weight vector before on-chain submission.

    Returns ``None`` when the payload is safe to submit, or a human-readable
    reason string when it must be skipped. Shared by every consumer of
    ``/v1/weights/latest`` so all on-chain relays validate the master vector
    identically. Reject unusable vectors before wallet/chain side effects.
    """

    now = now or datetime.now(UTC)
    if payload.netuid != netuid:
        return f"netuid mismatch: expected {netuid}, got {payload.netuid}"
    if expected_chain_endpoint is not None:
        expected = (expected_chain_endpoint or "").strip()
        actual = (payload.chain_endpoint or "").strip()
        if expected and actual and expected != actual:
            return f"chain_endpoint mismatch: expected {expected!r}, got {actual!r}"
    if payload.expires_at <= now:
        return "payload expired"
    if (now - payload.computed_at).total_seconds() > weights_freshness_seconds:
        return "payload stale"
    if not payload.uids:
        return "uids vector is empty"
    if not payload.weights:
        return "weights vector is empty"
    if len(payload.uids) != len(payload.weights):
        return "uids and weights vector lengths differ"

    seen: set[int] = set()
    previous_uid: int | None = None
    for index, uid in enumerate(payload.uids):
        if not isinstance(uid, int) or isinstance(uid, bool):
            return f"uid at index {index} is not an int"
        if uid < 0:
            return f"uid {uid} is negative"
        if max_uid is not None and uid > max_uid:
            return f"uid {uid} exceeds max_uid {max_uid}"
        if uid in seen:
            return f"duplicate uid {uid}"
        seen.add(uid)
        if previous_uid is not None and uid < previous_uid:
            return "uids are not sorted ascending"
        previous_uid = uid

    weight_sum = 0.0
    for index, weight in enumerate(payload.weights):
        try:
            value = float(weight)
        except (TypeError, ValueError):
            return f"weight at index {index} is not numeric"
        if not math.isfinite(value):
            return f"weight at index {index} is non-finite"
        if value < 0.0:
            return f"weight at index {index} is negative"
        weight_sum += value
    if weight_sum <= 0.0:
        return "weights sum is not positive"
    if weight_sum > 1.0 + 1e-6:
        return f"weights sum {weight_sum:.8f} exceeds 1.0"

    if require_provenance:
        if not payload.vector_id:
            return "missing vector_id provenance"
        if not payload.vector_digest:
            return "missing vector_digest provenance"
        protocol = str(payload.protocol_version or "").strip()
        if not protocol:
            return "missing protocol_version"
        if payload.chain_domain_bytes:
            try:
                domain = json.loads(payload.chain_domain_bytes)
            except json.JSONDecodeError:
                return "chain_domain_bytes is not valid JSON"
            if not isinstance(domain, dict):
                return "chain_domain_bytes must be a JSON object"
            for key in ("netuid", "uids", "weights"):
                if key not in domain:
                    return f"chain_domain_bytes missing {key}"
            if int(domain["netuid"]) != int(payload.netuid):
                return "chain_domain netuid does not match payload"
            if list(domain["uids"]) != list(payload.uids):
                return "chain_domain uids do not match payload"
            domain_weights = [float(w) for w in domain["weights"]]
            if len(domain_weights) != len(payload.weights):
                return "chain_domain weights length mismatch"
            for left, right in zip(domain_weights, payload.weights, strict=True):
                if abs(float(left) - float(right)) > 1e-9:
                    return "chain_domain weights do not match payload"

        # When full seal provenance is present, recompute digest invariants.
        if (
            payload.epoch is not None
            and payload.emission_policy_version
            and payload.burn_policy_version
            and payload.mapping_policy_version
        ):
            source_ids = [ref.snapshot_id for ref in payload.source_snapshots]
            source_digests = [ref.payload_digest for ref in payload.source_snapshots]
            try:
                expected_digest, expected_chain = recompute_vector_digest(
                    protocol_version=str(payload.protocol_version),
                    epoch=int(payload.epoch),
                    revision=int(payload.revision),
                    netuid=int(payload.netuid),
                    chain_endpoint=str(payload.chain_endpoint or ""),
                    uids=payload.uids,
                    weights=payload.weights,
                    emission_policy_version=str(payload.emission_policy_version),
                    emission_shares=payload.emission_shares or {},
                    burn_policy_version=str(payload.burn_policy_version),
                    mapping_policy_version=str(payload.mapping_policy_version),
                    source_snapshot_ids=source_ids,
                    source_snapshot_digests=source_digests,
                    metagraph_hash=payload.metagraph_hash,
                )
            except Exception as exc:  # pragma: no cover - defensive
                return f"digest recompute failed: {exc}"
            if payload.vector_digest != expected_digest:
                return "vector_digest does not match recomputed digest"
            if (
                payload.chain_domain_bytes
                and payload.chain_domain_bytes != expected_chain
            ):
                return "chain_domain_bytes does not match recomputed domain"

    return None


class WeightsClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 15.0,
        retries: int = 3,
        backoff_seconds: float = 0.05,
        permanent_status_codes: frozenset[int] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.permanent_status_codes = permanent_status_codes or frozenset(
            {400, 401, 403, 404, 422}
        )

    async def fetch_latest(self) -> MasterWeightsResponse:
        last_error: Exception | None = None
        attempts = max(1, self.retries + 1)
        for attempt in range(attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.get(f"{self.base_url}/v1/weights/latest")
                    if response.status_code in self.permanent_status_codes:
                        response.raise_for_status()
                    response.raise_for_status()
                    return MasterWeightsResponse.model_validate(response.json())
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response else None
                if status_code in self.permanent_status_codes:
                    raise
                if attempt < attempts - 1:
                    await asyncio.sleep(self.backoff_seconds * (2**attempt))
            except Exception as exc:
                last_error = exc
                if attempt < attempts - 1:
                    await asyncio.sleep(self.backoff_seconds * (2**attempt))
        raise last_error or RuntimeError("weights fetch failed")

    async def report_submission_observation(
        self,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """POST a non-authoritative submission observation to the master."""

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/v1/weights/submission-observations",
                json=payload,
                headers=headers or {},
            )
            response.raise_for_status()
            return response.json()
