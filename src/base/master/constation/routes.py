"""HTTP surfaces for attestation challenge/answer and constation checkers.

Sidecar convention (prism-recipe HttpxChallengeTransport):
  GET  /v1/attestation/challenge?phase=
  POST /v1/attestation/answer

Internal (prism re-verify / ops):
  POST /internal/v1/constation/check_allowlist
  POST /internal/v1/constation/check_nonce
  POST /internal/v1/constation/register_digest
  POST /internal/v1/constation/verify_attestation
  PUT  /internal/v1/constation/bundle/{work_unit_id}
  GET  /internal/v1/constation/bundle/{work_unit_id}
"""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from base.attestation.payload import (
    AttestationPayload,
    AttestationVerifyReason,
    SignedAttestation,
    verify_attestation_payload,
)
from base.compute.attestation_nonce import NonceBinding, NonceConsumeHit
from base.compute.digest_allowlist import AllowlistHit, DigestRecord, ImageVariant
from base.master.constation.allowlist_repository import DigestAllowlistRepository
from base.master.constation.bundle_store import ConstationBundleStore
from base.master.constation.nonce_repository import DurableAttestationNonceService


class RegisterDigestBody(BaseModel):
    commit_sha: str
    tree_sha: str
    variant: str
    digest: str


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


class IssueNonceBody(BaseModel):
    work_unit_id: str
    miner_hotkey: str
    pod_id: str
    phase: str = "interval"


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


def _signed_from_wire(raw: dict[str, Any]) -> SignedAttestation:
    """Parse a wire/json signed attestation into the domain object."""
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
        variant=pl_raw["variant"],  # type: ignore[arg-type]
        sealed_manifest_hashes=dict(pl_raw["sealed_manifest_hashes"]),
        build_secret_response=str(pl_raw["build_secret_response"]),
    )
    return SignedAttestation(
        payload=payload,
        signature=sig,
        algorithm=str(alg),
        schema_version=str(schema),
    )


def build_constation_router(
    *,
    allowlist_repo: DigestAllowlistRepository,
    nonce_service: DurableAttestationNonceService,
    bundle_store: ConstationBundleStore | None = None,
    internal_token: str | None = None,
    attestation_verify_key: bytes | None = None,
    default_binding: NonceBinding | None = None,
) -> APIRouter:
    """Build router; require Bearer when ``internal_token`` is set."""
    store = bundle_store or ConstationBundleStore()
    answers: list[dict[str, Any]] = []
    router = APIRouter(tags=["constation"])

    def require_internal(
        authorization: str | None = Header(default=None),
    ) -> None:
        if not _bearer_ok(authorization, internal_token):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, detail="invalid internal token"
            )

    @router.get("/v1/attestation/challenge")
    async def attestation_challenge(
        phase: str = Query(default="interval"),
        work_unit_id: str | None = Query(default=None),
        miner_hotkey: str | None = Query(default=None),
        pod_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
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
        phase_key = phase.strip().lower()
        if phase_key in {"random", "mid"}:
            phase_key = "interval"
        if phase_key not in {"start", "interval", "end"}:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"unknown challenge phase: {phase!r}",
            )
        record = await nonce_service.issue(binding)
        return {
            "nonce": record.nonce,
            "phase": phase_key,
            "challenge_id": record.nonce,
            "work_unit_id": binding.work_unit_id,
            "expires_at": record.expires_at.isoformat(),
        }

    @router.post("/v1/attestation/answer")
    async def attestation_answer(body: AnswerBody) -> dict[str, str]:
        answers.append(body.model_dump())
        return {"status": "accepted"}

    @router.post(
        "/internal/v1/constation/issue_nonce",
        dependencies=[Depends(require_internal)],
    )
    async def issue_nonce(body: IssueNonceBody) -> dict[str, Any]:
        """Issue a durable nonce for Prism public challenge (SoT on master)."""
        phase_key = body.phase.strip().lower()
        if phase_key in {"random", "mid"}:
            phase_key = "interval"
        if phase_key not in {"start", "interval", "end"}:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"unknown challenge phase: {body.phase!r}",
            )
        binding = NonceBinding(
            work_unit_id=body.work_unit_id,
            miner_hotkey=body.miner_hotkey,
            pod_id=body.pod_id,
        )
        record = await nonce_service.issue(binding)
        return {
            "nonce": record.nonce,
            "phase": phase_key,
            "challenge_id": record.nonce,
            "work_unit_id": binding.work_unit_id,
            "expires_at": record.expires_at.isoformat(),
        }

    @router.post(
        "/internal/v1/constation/register_digest",
        dependencies=[Depends(require_internal)],
    )
    async def register_digest(body: RegisterDigestBody) -> dict[str, str]:
        record = DigestRecord(
            commit_sha=body.commit_sha,
            tree_sha=body.tree_sha,
            variant=ImageVariant(body.variant.strip().lower()),
            digest=body.digest,
        )
        await allowlist_repo.register(record)
        return {"status": "registered", "digest": record.digest}

    @router.post(
        "/internal/v1/constation/check_allowlist",
        dependencies=[Depends(require_internal)],
    )
    async def check_allowlist(body: CheckAllowlistBody) -> dict[str, Any]:
        allowlist = await allowlist_repo.load_allowlist()
        result = allowlist.lookup(
            digest=body.digest,
            commit_sha=body.commit_sha,
            tree_sha=body.tree_sha,
            variant=body.variant,
        )
        if isinstance(result, AllowlistHit):
            return {"ok": True, "reason": "ok"}
        return {"ok": False, "reason": result.reason.value}

    @router.post(
        "/internal/v1/constation/check_nonce",
        dependencies=[Depends(require_internal)],
    )
    async def check_nonce(body: CheckNonceBody) -> dict[str, Any]:
        result = await nonce_service.consume(
            body.nonce,
            NonceBinding(
                work_unit_id=body.work_unit_id,
                miner_hotkey=body.miner_hotkey,
                pod_id=body.pod_id,
            ),
        )
        if isinstance(result, NonceConsumeHit):
            return {"ok": True, "reason": "ok"}
        return {"ok": False, "reason": result.reason.value}

    @router.post(
        "/internal/v1/constation/verify_attestation",
        dependencies=[Depends(require_internal)],
    )
    async def verify_attestation(body: VerifyAttestationBody) -> dict[str, Any]:
        if body.key_hex:
            key = bytes.fromhex(body.key_hex)
        elif attestation_verify_key is not None:
            key = attestation_verify_key
        else:
            return {"ok": False, "reason": AttestationVerifyReason.EMPTY_KEY.value}
        try:
            signed = _signed_from_wire(body.signed)
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"invalid signed attestation: {exc}",
            ) from exc
        outcome = verify_attestation_payload(signed, verify_key=key)
        return {"ok": outcome.ok, "reason": outcome.reason.value}

    @router.put(
        "/internal/v1/constation/bundle/{work_unit_id}",
        dependencies=[Depends(require_internal)],
    )
    async def put_bundle(work_unit_id: str, request: Request) -> dict[str, str]:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, detail="bundle must be object"
            )
        store.put(work_unit_id, payload)
        return {"status": "stored", "work_unit_id": work_unit_id}

    @router.get(
        "/internal/v1/constation/bundle/{work_unit_id}",
        dependencies=[Depends(require_internal)],
    )
    async def get_bundle(work_unit_id: str) -> dict[str, Any]:
        blob = store.get(work_unit_id)
        if blob is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="bundle not found")
        return blob

    router._answers = answers  # type: ignore[attr-defined]
    router._bundle_store = store  # type: ignore[attr-defined]
    return router


def create_constation_test_app(
    *,
    allowlist_repo: DigestAllowlistRepository,
    nonce_service: DurableAttestationNonceService,
    internal_token: str | None = "test-internal",
    default_binding: NonceBinding | None = None,
    attestation_verify_key: bytes | None = None,
    bundle_store: ConstationBundleStore | None = None,
) -> Any:
    """Minimal FastAPI app hosting only the constation router (unit tests)."""
    from fastapi import FastAPI

    app = FastAPI()
    router = build_constation_router(
        allowlist_repo=allowlist_repo,
        nonce_service=nonce_service,
        bundle_store=bundle_store,
        internal_token=internal_token,
        default_binding=default_binding,
        attestation_verify_key=attestation_verify_key,
    )
    app.include_router(router)
    app.state.constation_router = router
    return app


__all__ = [
    "build_constation_router",
    "create_constation_test_app",
]
