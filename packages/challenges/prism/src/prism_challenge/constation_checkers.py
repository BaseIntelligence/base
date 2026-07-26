"""BASE-backed HTTP checkers for prism production constation_ok."""

from __future__ import annotations

from typing import Any

import httpx

from .constation import CheckOutcome


class BaseHttpConstationClient:
    """Thin client for BASE internal constation checker endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("constation base_url must be non-empty")
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout_s = timeout_s
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def check_allowlist(
        self,
        *,
        digest: str,
        commit_sha: str,
        tree_sha: str,
        variant: str,
    ) -> CheckOutcome:
        body = {
            "digest": digest,
            "commit_sha": commit_sha,
            "tree_sha": tree_sha,
            "variant": variant,
        }
        return self._post_outcome("/internal/v1/constation/check_allowlist", body)

    def check_nonce(
        self,
        *,
        nonce: str,
        work_unit_id: str,
        miner_hotkey: str,
        pod_id: str,
    ) -> CheckOutcome:
        body = {
            "nonce": nonce,
            "work_unit_id": work_unit_id,
            "miner_hotkey": miner_hotkey,
            "pod_id": pod_id,
        }
        return self._post_outcome("/internal/v1/constation/check_nonce", body)

    def verify_signature(self, signed: object) -> CheckOutcome:
        body = {"signed": signed if isinstance(signed, dict) else {"value": signed}}
        return self._post_outcome("/internal/v1/constation/verify_attestation", body)

    def _post_outcome(self, path: str, body: dict[str, Any]) -> CheckOutcome:
        url = f"{self._base_url}{path}"
        try:
            with httpx.Client(
                timeout=self._timeout_s, transport=self._transport
            ) as client:
                response = client.post(url, json=body, headers=self._headers())
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # noqa: BLE001 - map to checker fail-closed
            return CheckOutcome(ok=False, reason=f"checker_transport_error:{exc}")
        if not isinstance(data, dict):
            return CheckOutcome(ok=False, reason="checker_malformed_response")
        ok = bool(data.get("ok"))
        reason = data.get("reason", "ok" if ok else "checker_rejected")
        return CheckOutcome(ok=ok, reason=str(reason))


__all__ = ["BaseHttpConstationClient"]
