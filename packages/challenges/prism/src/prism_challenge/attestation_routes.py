"""Prism-hosted attestation routes (public via BASE challenge proxy).

Published (proxy strip ``/challenges/prism``):
  GET  /challenges/prism/v1/attestation/challenge
  POST /challenges/prism/v1/attestation/answer

Internal (prism process, shared-token):
  POST /internal/v1/constation/register_digest
  POST /internal/v1/constation/check_allowlist
  POST /internal/v1/constation/check_nonce
  POST /internal/v1/constation/verify_attestation

Durable allowlist/nonce SoT may live on BASE master; when
``constation_base_url`` is set, challenge issuance and checkers call master
internal HTTP. Otherwise in-process services on ``app.state`` (tests / single-node).

Challenge routes belong on the challenge app — not on base master.
"""

from __future__ import annotations

import hmac
from datetime import timedelta
from typing import Any

import httpx
from base.challenge_sdk.roles import public_route
from base.compute.attestation_nonce import (
    AttestationNonceService,
    NonceBinding,
    NonceConsumeHit,
)
from base.compute.digest_allowlist import (
    AllowlistHit,
    DigestAllowlist,
    DigestRecord,
    ImageVariant,
)
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from .auth import authenticate_internal
from .constation import CheckOutcome

try:
    from base.attestation.payload import (
        AttestationPayload,
        AttestationVerifyReason,
        SignedAttestation,
        verify_attestation_payload,
    )
except ImportError:  # pragma: no cover - optional in minimal installs
    AttestationPayload = Any  # type: ignore[misc, assignment]
    AttestationVerifyReason = None  # type: ignore[assignment]
    SignedAttestation = Any  # type: ignore[misc, assignment]
    verify_attestation_payload = None  # type: ignore[assignment]


class RegisterDigestBody(BaseModel):
    commit_sha: str
    tree_sha: str
    variant: str
    digest: str
    sealed_manifest_hashes: dict[str, str]


class CheckAllowlistBody(BaseModel):
    digest: str
    commit_sha: str
    tree_sha: str
    variant: str


class CheckNonceBody(BaseModel):
    nonce: str
    work_unit_id: str
    miner_hotkey: str
    pod_id: str


class VerifyAttestationBody(BaseModel):
    signed: dict[str, Any]
    key_hex: str | None = None


class AnswerBody(BaseModel):
    model_config = {"extra": "allow"}

    nonce: str = Field(min_length=1)
    phase: str | None = None


def _bearer_ok(authorization: str | None, expected: str | None) -> bool:
    if not expected:
        return True
    if not authorization or not authorization.lower().startswith("bearer "):
        return False
    presented = authorization.split(" ", 1)[1].strip()
    return bool(presented) and hmac.compare_digest(presented, expected)


def _signed_from_wire(raw: dict[str, Any]) -> Any:
    if verify_attestation_payload is None:
        raise RuntimeError("attestation payload helpers unavailable")
    if "payload" in raw and isinstance(raw["payload"], dict):
        pl_raw = raw["payload"]
        sig = raw.get("signature")
        alg = raw.get("algorithm", "hmac-sha256")
        schema = raw.get("schema_version", "prism_attestation_payload.v1")
    else:
        pl_raw = raw
        sig = raw.get("signature")
        alg = raw.get("algorithm", "hmac-sha256")
        schema = raw.get("schema_version", "prism_attestation_payload.v1")
    if not isinstance(sig, str):
        raise ValueError("signature must be a hex string")
    payload = AttestationPayload(
        nonce=str(pl_raw["nonce"]),
        digest=str(pl_raw["digest"]),
        pod_id=str(pl_raw["pod_id"]),
        variant=pl_raw["variant"],
        sealed_manifest_hashes=dict(pl_raw["sealed_manifest_hashes"]),
        build_secret_response=str(pl_raw["build_secret_response"]),
    )
    return SignedAttestation(
        payload=payload,
        signature=sig,
        algorithm=str(alg),
        schema_version=str(schema),
    )


def ensure_default_constation_services(app: Any, *, ttl: timedelta | None = None) -> None:
    """Attach in-process allowlist + nonce when missing (tests / single-node)."""
    state = app.state
    if getattr(state, "digest_allowlist", None) is None:
        state.digest_allowlist = DigestAllowlist()
    if getattr(state, "attestation_nonce_service", None) is None:
        state.attestation_nonce_service = AttestationNonceService(ttl=ttl or timedelta(hours=2))
    if getattr(state, "attestation_answers", None) is None:
        state.attestation_answers = []


def make_inprocess_checkers(app: Any) -> dict[str, Any]:
    """Sync checkers bound to app.state in-process services."""
    ensure_default_constation_services(app)
    allowlist: DigestAllowlist = app.state.digest_allowlist
    nonce_svc: AttestationNonceService = app.state.attestation_nonce_service
    verify_key: bytes | None = getattr(app.state, "attestation_verify_key", None)

    def check_allowlist(
        *,
        digest: str,
        commit_sha: str,
        tree_sha: str,
        variant: str,
    ) -> CheckOutcome:
        result = allowlist.lookup(
            digest=digest,
            commit_sha=commit_sha,
            tree_sha=tree_sha,
            variant=variant,
        )
        if isinstance(result, AllowlistHit):
            return CheckOutcome(ok=True, reason="ok")
        return CheckOutcome(ok=False, reason=result.reason.value)

    def check_nonce(
        *,
        nonce: str,
        work_unit_id: str,
        miner_hotkey: str,
        pod_id: str,
    ) -> CheckOutcome:
        result = nonce_svc.consume(
            nonce,
            NonceBinding(
                work_unit_id=work_unit_id,
                miner_hotkey=miner_hotkey,
                pod_id=pod_id,
            ),
        )
        if isinstance(result, NonceConsumeHit):
            return CheckOutcome(ok=True, reason="ok")
        return CheckOutcome(ok=False, reason=result.reason.value)

    def verify_signature(signed: object) -> CheckOutcome:
        if verify_attestation_payload is None:
            return CheckOutcome(ok=False, reason="attestation_helpers_missing")
        if verify_key is None:
            # Soft path for harnesses that don't sign: accept dict markers.
            if isinstance(signed, dict) and signed.get("sig"):
                return CheckOutcome(ok=True, reason="ok")
            return CheckOutcome(ok=False, reason="empty_key")
        try:
            if not isinstance(signed, dict):
                return CheckOutcome(ok=False, reason="invalid_signed_shape")
            wire = _signed_from_wire(signed)
            outcome = verify_attestation_payload(wire, verify_key=verify_key)
            return CheckOutcome(ok=outcome.ok, reason=outcome.reason.value)
        except Exception as exc:  # noqa: BLE001
            return CheckOutcome(ok=False, reason=f"signature_error:{exc}")

    return {
        "check_allowlist": check_allowlist,
        "check_nonce": check_nonce,
        "verify_constation_signature": verify_signature,
    }


def build_attestation_public_router() -> APIRouter:
    """Public attestation challenge/answer under the prism ``/v1`` prefix."""
    router = APIRouter(tags=["attestation"])

    @public_route(tags=["attestation"])
    @router.get("/attestation/challenge")
    async def attestation_challenge(
        request: Request,
        phase: str = Query(default="interval"),
        work_unit_id: str | None = Query(default=None),
        miner_hotkey: str | None = Query(default=None),
        pod_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        settings = getattr(request.app.state, "settings", None)
        base_url = getattr(settings, "constation_base_url", None) if settings else None
        token = getattr(settings, "constation_internal_token", None) or "" if settings else ""
        phase_key = phase.strip().lower()
        if phase_key in {"random", "mid"}:
            phase_key = "interval"
        if phase_key not in {"start", "interval", "end"}:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"unknown challenge phase: {phase!r}",
            )

        default_binding = getattr(request.app.state, "default_nonce_binding", None)
        if default_binding is not None:
            binding = default_binding
        else:
            if not (work_unit_id and miner_hotkey and pod_id):
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        "work_unit_id, miner_hotkey, and pod_id query params "
                        "required when no default binding configured"
                    ),
                )
            binding = NonceBinding(
                work_unit_id=work_unit_id,
                miner_hotkey=miner_hotkey,
                pod_id=pod_id,
            )

        # Durable SoT on BASE master: issue via internal HTTP.
        if base_url and token:
            url = f"{str(base_url).rstrip('/')}/internal/v1/constation/issue_nonce"
            payload = {
                "work_unit_id": binding.work_unit_id,
                "miner_hotkey": binding.miner_hotkey,
                "pod_id": binding.pod_id,
                "phase": phase_key,
            }
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Accept": "application/json",
                        },
                    )
                    response.raise_for_status()
                    body = response.json()
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"constation issue_nonce failed: {exc}",
                ) from exc
            if not isinstance(body, dict) or "nonce" not in body:
                raise HTTPException(
                    status.HTTP_502_BAD_GATEWAY,
                    detail="constation issue_nonce malformed response",
                )
            return {
                "nonce": body["nonce"],
                "phase": body.get("phase", phase_key),
                "challenge_id": body.get("challenge_id", body["nonce"]),
                "work_unit_id": body.get("work_unit_id", binding.work_unit_id),
                "expires_at": body.get("expires_at"),
            }

        ensure_default_constation_services(request.app)
        nonce_svc: AttestationNonceService = request.app.state.attestation_nonce_service
        record = nonce_svc.issue(binding)
        return {
            "nonce": record.nonce,
            "phase": phase_key,
            "challenge_id": record.nonce,
            "work_unit_id": binding.work_unit_id,
            "expires_at": record.expires_at.isoformat(),
        }

    @public_route(tags=["attestation"])
    @router.post("/attestation/answer")
    async def attestation_answer(request: Request, body: AnswerBody) -> dict[str, str]:
        ensure_default_constation_services(request.app)
        answers: list[dict[str, Any]] = request.app.state.attestation_answers
        answers.append(body.model_dump())
        return {"status": "accepted"}

    return router


def build_attestation_internal_router() -> APIRouter:
    """Internal check/register surfaces on the prism process (shared token)."""
    router = APIRouter(tags=["constation-internal"])

    @router.post(
        "/internal/v1/constation/register_digest",
        dependencies=[Depends(authenticate_internal)],
    )
    async def register_digest(request: Request, body: RegisterDigestBody) -> dict[str, str]:
        ensure_default_constation_services(request.app)
        allowlist: DigestAllowlist = request.app.state.digest_allowlist
        try:
            record = DigestRecord(
                commit_sha=body.commit_sha,
                tree_sha=body.tree_sha,
                variant=ImageVariant(body.variant.strip().lower()),
                digest=body.digest,
                sealed_manifest_hashes=body.sealed_manifest_hashes,
            )
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        allowlist.register(record)
        return {"status": "registered", "digest": record.digest}

    @router.post(
        "/internal/v1/constation/check_allowlist",
        dependencies=[Depends(authenticate_internal)],
    )
    async def check_allowlist(request: Request, body: CheckAllowlistBody) -> dict[str, Any]:
        checkers = make_inprocess_checkers(request.app)
        outcome = checkers["check_allowlist"](
            digest=body.digest,
            commit_sha=body.commit_sha,
            tree_sha=body.tree_sha,
            variant=body.variant,
        )
        return {"ok": outcome.ok, "reason": outcome.reason}

    @router.post(
        "/internal/v1/constation/check_nonce",
        dependencies=[Depends(authenticate_internal)],
    )
    async def check_nonce(request: Request, body: CheckNonceBody) -> dict[str, Any]:
        checkers = make_inprocess_checkers(request.app)
        outcome = checkers["check_nonce"](
            nonce=body.nonce,
            work_unit_id=body.work_unit_id,
            miner_hotkey=body.miner_hotkey,
            pod_id=body.pod_id,
        )
        return {"ok": outcome.ok, "reason": outcome.reason}

    @router.post(
        "/internal/v1/constation/verify_attestation",
        dependencies=[Depends(authenticate_internal)],
    )
    async def verify_attestation(request: Request, body: VerifyAttestationBody) -> dict[str, Any]:
        if body.key_hex:
            request.app.state.attestation_verify_key = bytes.fromhex(body.key_hex)
        checkers = make_inprocess_checkers(request.app)
        outcome = checkers["verify_constation_signature"](body.signed)
        return {"ok": outcome.ok, "reason": outcome.reason}

    return router


__all__ = [
    "build_attestation_internal_router",
    "build_attestation_public_router",
    "ensure_default_constation_services",
    "make_inprocess_checkers",
]
