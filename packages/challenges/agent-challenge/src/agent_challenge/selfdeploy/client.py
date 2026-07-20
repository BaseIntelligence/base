"""Production-route client for the ordered self-deploy flow.

The client intentionally exposes only the signed miner routes and the one
challenge-direct result route.  It does not provide database/state-seeding
helpers, internal routes, or BASE bridge aliases.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from dataclasses import dataclass
from hashlib import sha256
from http.client import HTTPResponse
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from agent_challenge.auth.security import canonical_request_string


class RouteClientError(RuntimeError):
    """A bounded, secret-free production route failure."""


_ALLOWED_PRODUCTION_ROUTES = (
    re.compile(r"^/submissions/[0-9]+/review/(?:prepare|retry|cancel|deployed|history|report)$"),
    re.compile(r"^/submissions/[0-9]+/eval/(?:prepare|retry|cancel|failure|status)$"),
    re.compile(r"^/evaluation/v1/runs/[^/]+/result$"),
)


@dataclass(frozen=True)
class SignedIdentity:
    hotkey: str
    signature: str
    nonce: str
    timestamp: str


def _load_signing_keypair() -> Any:
    """Load a miner substrate keypair only when a signed route is used."""

    import os

    mnemonic = os.environ.get("MINER_HOTKEY_MNEMONIC", "").strip()
    uri = os.environ.get("MINER_HOTKEY_URI", "").strip()
    if not mnemonic and not uri:
        raise RouteClientError("signed route requires MINER_HOTKEY_MNEMONIC or MINER_HOTKEY_URI")
    try:
        from bittensor import Keypair
    except ImportError as exc:
        raise RouteClientError("bittensor is required for signed production routes") from exc
    try:
        return Keypair.create_from_mnemonic(mnemonic) if mnemonic else Keypair.create_from_uri(uri)
    except Exception as exc:
        raise RouteClientError("miner signing key could not be loaded") from exc


def sign_request_identity(
    *,
    hotkey: str,
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    raw_body: bytes,
    query_string: str = "",
) -> SignedIdentity:
    """Produce a fresh identity over the exact production request bytes."""

    keypair = _load_signing_keypair()
    if str(keypair.ss58_address) != hotkey:
        raise RouteClientError("configured miner hotkey does not match signing key")
    canonical = canonical_request_string(
        method=method,
        path=path,
        query_string=query_string,
        timestamp=timestamp,
        nonce=nonce,
        raw_body=raw_body,
    )
    signature = keypair.sign(canonical)
    encoded = "0x" + bytes(signature).hex() if isinstance(signature, bytes) else str(signature)
    return SignedIdentity(hotkey=hotkey, signature=encoded, nonce=nonce, timestamp=timestamp)


class SelfDeployRouteClient:
    """HTTP client restricted to the ordered production route contract."""

    def __init__(
        self,
        base_url: str,
        *,
        identity: SignedIdentity | None = None,
        auto_sign: bool = False,
        opener=urlopen,
        timeout: float = 30.0,
    ) -> None:
        base = base_url.strip().rstrip("/")
        if not base.startswith("https://"):
            raise RouteClientError("challenge endpoint must use https://")
        self._base_url = base
        self._identity = identity
        self._auto_sign = auto_sign
        self._opener = opener
        self._timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        raw_body: bytes | None = None,
        bearer: str | None = None,
        signed: bool = False,
    ) -> dict[str, Any]:
        parsed_path = urlsplit(path)
        route_path = parsed_path.path
        if not any(pattern.fullmatch(route_path) for pattern in _ALLOWED_PRODUCTION_ROUTES):
            raise RouteClientError("route is not an allowed production self-deploy route")
        if raw_body is not None and body is not None:
            raise RouteClientError("request cannot provide both decoded and raw body")
        raw = (
            raw_body
            if raw_body is not None
            else (
                b""
                if body is None
                else json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
            )
        )
        headers = {"Accept": "application/json"}
        if body is not None or raw_body is not None:
            headers["Content-Type"] = "application/json"
        if bearer is not None:
            headers["Authorization"] = f"Bearer {bearer}"
        if signed:
            identity = self._identity
            if identity is None:
                raise RouteClientError("signed route credentials are required")
            # Exact query string is part of the signed canonical request.
            # Auto-sign and explicit-header paths both bind it so history /
            # report / status cursor requests verify on the challenge host.
            query_string = parsed_path.query or ""
            if self._auto_sign:
                identity = sign_request_identity(
                    hotkey=identity.hotkey,
                    method=method,
                    path=route_path,
                    query_string=query_string,
                    timestamp=str(time.time()),
                    nonce=secrets.token_urlsafe(24),
                    raw_body=raw,
                )
            canonical = canonical_request_string(
                method=method,
                path=route_path,
                query_string=query_string,
                timestamp=identity.timestamp,
                nonce=identity.nonce,
                raw_body=raw,
            )
            # The caller supplies a substrate signature over the exact
            # canonical bytes.  Never log or return the signature itself.
            if not canonical:
                raise RouteClientError("signed request canonicalization failed")
            headers.update(
                {
                    "X-Hotkey": identity.hotkey,
                    "X-Signature": identity.signature,
                    "X-Nonce": identity.nonce,
                    "X-Timestamp": identity.timestamp,
                }
            )
        request = Request(
            f"{self._base_url}{path}",
            data=raw if (body is not None or raw_body is not None) else None,
            headers=headers,
            method=method.upper(),
        )
        try:
            response: HTTPResponse = self._opener(request, timeout=self._timeout)
            payload = response.read(16 * 1024 * 1024 + 1)
        except HTTPError as exc:
            # Do not echo origin response text, which may contain submitted
            # bytes or provider/error secrets.
            raise RouteClientError(f"challenge route returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RouteClientError("challenge route is unreachable") from exc
        if len(payload) > 16 * 1024 * 1024:
            raise RouteClientError("challenge response exceeded the bounded response size")
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RouteClientError("challenge route returned malformed JSON") from exc
        if not isinstance(decoded, dict):
            raise RouteClientError("challenge route returned a non-object response")
        return decoded

    def review_prepare(self, submission_id: int) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/submissions/{submission_id}/review/prepare",
            body={},
            signed=True,
        )

    def review_retry(
        self,
        submission_id: int,
        assignment_id: str,
        *,
        approval_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"expected_assignment_id": assignment_id}
        if approval_id is not None:
            if not isinstance(approval_id, str) or not approval_id.strip():
                raise RouteClientError("approval_id must be a non-empty string")
            body["approval_id"] = approval_id.strip()
        return self._request(
            "POST",
            f"/submissions/{submission_id}/review/retry",
            body=body,
            signed=True,
        )

    def review_cancel(self, submission_id: int, assignment_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/submissions/{submission_id}/review/cancel",
            body={"expected_assignment_id": assignment_id},
            signed=True,
        )

    def review_deployed(
        self,
        submission_id: int,
        acknowledgement: dict[str, Any],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/submissions/{submission_id}/review/deployed",
            body=acknowledgement,
            signed=True,
        )

    def review_report(self, submission_id: int, *, cursor: str | None = None) -> dict[str, Any]:
        suffix = "" if cursor is None else f"?cursor={cursor}"
        return self._request(
            "GET",
            f"/submissions/{submission_id}/review/report{suffix}",
            signed=True,
        )

    def review_history(self, submission_id: int, *, cursor: str | None = None) -> dict[str, Any]:
        suffix = "" if cursor is None else f"?cursor={cursor}"
        return self._request(
            "GET",
            f"/submissions/{submission_id}/review/history{suffix}",
            signed=True,
        )

    def eval_prepare(self, submission_id: int) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/submissions/{submission_id}/eval/prepare",
            body={},
            signed=True,
        )

    def eval_retry(self, submission_id: int, run_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/submissions/{submission_id}/eval/retry",
            body={"schema_version": 1, "eval_run_id": run_id},
            signed=True,
        )

    def eval_cancel(self, submission_id: int, run_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/submissions/{submission_id}/eval/cancel",
            body={"schema_version": 1, "eval_run_id": run_id},
            signed=True,
        )

    def eval_failure(self, submission_id: int, run_id: str, reason_code: str) -> dict[str, Any]:
        if reason_code not in {
            "eval_deploy_failed",
            "eval_tunnel_failed",
            "eval_key_release_unavailable",
            "eval_no_result",
        }:
            raise RouteClientError("unsupported Eval pre-receipt failure reason")
        return self._request(
            "POST",
            f"/submissions/{submission_id}/eval/failure",
            body={"schema_version": 1, "eval_run_id": run_id, "reason_code": reason_code},
            signed=True,
        )

    def eval_status(self, submission_id: int, *, cursor: str | None = None) -> dict[str, Any]:
        suffix = "" if cursor is None else f"?cursor={cursor}"
        return self._request(
            "GET",
            f"/submissions/{submission_id}/eval/status{suffix}",
            signed=True,
        )

    def eval_result(self, run_id: str, result: dict[str, Any], token: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/evaluation/v1/runs/{run_id}/result",
            body=result,
            bearer=token,
        )

    def eval_result_bytes(self, run_id: str, result_bytes: bytes, token: str) -> dict[str, Any]:
        """Post already canonicalized result bytes without reserialization."""

        if not isinstance(result_bytes, bytes) or not result_bytes:
            raise RouteClientError("Eval result request bytes are empty")

        return self._request(
            "POST",
            f"/evaluation/v1/runs/{run_id}/result",
            raw_body=result_bytes,
            bearer=token,
        )


def build_signed_identity(
    *,
    hotkey: str,
    signature: str,
    nonce: str,
    timestamp: str | None = None,
) -> SignedIdentity:
    """Build the exact caller-provided signed identity without exposing it."""

    values = {
        "hotkey": hotkey,
        "signature": signature,
        "nonce": nonce,
        "timestamp": timestamp or str(int(time.time())),
    }
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        raise RouteClientError("signed route identity is incomplete")
    return SignedIdentity(**{key: value.strip() for key, value in values.items()})


def body_digest(body: bytes) -> str:
    """Return a digest useful for safe receipt/result reporting."""

    return sha256(body).hexdigest()


__all__ = [
    "RouteClientError",
    "SelfDeployRouteClient",
    "SignedIdentity",
    "body_digest",
    "build_signed_identity",
    "sign_request_identity",
]
