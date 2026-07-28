"""Token-bound download of a miner's stored submission ZIP for eval CVMs.

The guest verifies ``sha256(bytes)`` against the plan hash, so this endpoint's
job is availability plus not leaking other miners' artifacts. Access is gated
by an HMAC grant bound to ``eval_run_id`` + ``agent_hash`` with an expiry —
mirroring the review assignment capability style (domain-separated HMAC over
the internal shared secret).
"""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from agent_challenge.core.config import settings
from agent_challenge.core.db import database
from agent_challenge.core.models import EvalRun
from agent_challenge.sdk.auth import load_internal_token

logger = logging.getLogger(__name__)

router = APIRouter()

DatabaseSession = Annotated[AsyncSession, Depends(database.session_dependency)]

#: Domain separation so an artifact grant can never collide with review session
#: tokens, attempt stream tokens, or other HMACs from the shared secret.
_GRANT_CONTEXT = b"agent-challenge:eval-artifact-grant:v1:"
_GRANT_VERSION = "v1"


@dataclass(frozen=True)
class EvalArtifactGrant:
    """Parsed, cryptographically verified artifact download grant."""

    eval_run_id: str
    agent_hash: str
    expires_at: datetime


def mint_eval_artifact_grant(
    *,
    secret: str,
    eval_run_id: str,
    agent_hash: str,
    expires_at: datetime,
) -> str:
    """Mint a bearer grant bound to one eval run and submission agent_hash.

    Format: ``v1.{eval_run_id}.{agent_hash}.{exp_unix}.{mac_hex}``.

    The launch path (separate task) calls this with the server secret so the
    in-TEE orchestrator can fetch the ZIP without holding the raw shared token.
    """

    if not isinstance(secret, str) or not secret:
        raise ValueError("eval artifact grant secret is required")
    if (
        not isinstance(eval_run_id, str)
        or not eval_run_id
        or "/" in eval_run_id
        or "." in eval_run_id
    ):
        # Dots separate token fields; reject so mint/verify stay unambiguous.
        raise ValueError("eval_run_id is invalid for artifact grant")
    if not isinstance(agent_hash, str) or not agent_hash or "." in agent_hash:
        raise ValueError("agent_hash is invalid for artifact grant")
    exp = expires_at if expires_at.tzinfo is not None else expires_at.replace(tzinfo=UTC)
    exp_unix = int(exp.astimezone(UTC).timestamp())
    if exp_unix <= 0:
        raise ValueError("expires_at is invalid for artifact grant")
    body = f"{_GRANT_VERSION}.{eval_run_id}.{agent_hash}.{exp_unix}"
    mac = hmac.new(
        secret.encode("utf-8"),
        _GRANT_CONTEXT + body.encode("ascii"),
        sha256,
    ).hexdigest()
    return f"{body}.{mac}"


def verify_eval_artifact_grant(
    *,
    secret: str,
    token: str,
    eval_run_id: str,
    now: datetime | None = None,
) -> EvalArtifactGrant:
    """Verify grant MAC + binding + expiry for the path-scoped eval_run_id.

    Raises ``PermissionError`` with a stable reason code for HTTP mapping:
    - ``missing`` / ``invalid`` / ``expired`` -> 401
    - ``wrong_run`` -> 404 (token valid for a different run)
    """

    if not isinstance(secret, str) or not secret:
        raise PermissionError("invalid")
    if not isinstance(token, str) or not token:
        raise PermissionError("missing")
    parts = token.split(".")
    if len(parts) != 5:
        raise PermissionError("invalid")
    version, token_run_id, agent_hash, exp_raw, mac = parts
    if version != _GRANT_VERSION:
        raise PermissionError("invalid")
    if not token_run_id or not agent_hash or not exp_raw or not mac:
        raise PermissionError("invalid")
    try:
        exp_unix = int(exp_raw)
    except ValueError as exc:
        raise PermissionError("invalid") from exc
    body = f"{version}.{token_run_id}.{agent_hash}.{exp_unix}"
    expected = hmac.new(
        secret.encode("utf-8"),
        _GRANT_CONTEXT + body.encode("ascii"),
        sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, mac):
        raise PermissionError("invalid")
    if token_run_id != eval_run_id:
        raise PermissionError("wrong_run")
    now_utc = now or datetime.now(UTC)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    else:
        now_utc = now_utc.astimezone(UTC)
    expires_at = datetime.fromtimestamp(exp_unix, tz=UTC)
    if now_utc >= expires_at:
        raise PermissionError("expired")
    return EvalArtifactGrant(
        eval_run_id=token_run_id,
        agent_hash=agent_hash,
        expires_at=expires_at,
    )


def _bearer_token(authorization: str | None) -> str | None:
    if not isinstance(authorization, str) or not authorization:
        return None
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    presented = authorization[len(prefix) :].strip()
    return presented or None


@router.get("/eval/v1/runs/{eval_run_id}/artifact")
async def download_eval_artifact(
    eval_run_id: str,
    session: DatabaseSession,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    """Stream the submission ZIP bound to ``eval_run_id`` under a grant token."""

    secret = load_internal_token(settings)
    if not secret:
        # Misconfiguration is not a client auth failure; avoid leaking setup.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="unavailable",
        )

    presented = _bearer_token(authorization)
    if presented is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")

    try:
        grant = verify_eval_artifact_grant(
            secret=secret,
            token=presented,
            eval_run_id=eval_run_id,
        )
    except PermissionError as exc:
        reason = str(exc) or "invalid"
        if reason == "wrong_run":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found") from exc
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
        ) from exc

    run = await session.scalar(
        select(EvalRun)
        .where(EvalRun.eval_run_id == eval_run_id)
        .options(selectinload(EvalRun.submission))
    )
    if run is None or run.submission is None:
        # Valid grant for a run that is gone / never existed — do not distinguish.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")

    submission = run.submission
    if submission.agent_hash != grant.agent_hash:
        # Grant agent_hash must match the durable submission binding.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")

    artifact_path = submission.artifact_path or submission.artifact_uri
    if not artifact_path:
        logger.warning(
            "eval artifact path missing",
            extra={"eval_run_id": eval_run_id, "submission_id": submission.id},
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")

    path = Path(artifact_path)
    try:
        zip_bytes = path.read_bytes()
    except OSError:
        logger.warning(
            "eval artifact unreadable",
            extra={"eval_run_id": eval_run_id, "submission_id": submission.id},
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found") from None

    # Defense in depth: refuse to serve bytes that do not match the bound hash.
    if sha256(zip_bytes).hexdigest() != grant.agent_hash:
        logger.warning(
            "eval artifact digest mismatch",
            extra={"eval_run_id": eval_run_id, "submission_id": submission.id},
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")

    headers = {
        "Content-Length": str(len(zip_bytes)),
        "X-Agent-Hash": submission.agent_hash,
    }
    tree_sha = submission.package_tree_sha
    if isinstance(tree_sha, str) and tree_sha:
        headers["X-Package-Tree-Sha"] = tree_sha

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers=headers,
    )
