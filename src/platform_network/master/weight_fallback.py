from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from platform_network.schemas.weight_fallback import (
    SignedWeightsPayload,
    SignedWeightsResponse,
)
from platform_network.schemas.weights import ChallengeWeightsResult, FinalWeights


class WeightFallbackError(RuntimeError):
    pass


class LatestWeightsStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write_final(self, final: FinalWeights) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = SignedWeightsPayload(weights=final.hotkey_weights)
        self.path.write_text(payload.model_dump_json(), encoding="utf-8")

    def read(self, challenge_slug: str | None = None) -> SignedWeightsPayload:
        if not self.path.is_file():
            raise WeightFallbackError("latest weights are not available")
        payload = SignedWeightsPayload.model_validate_json(
            self.path.read_text(encoding="utf-8")
        )
        if challenge_slug is not None:
            payload.challenge_slug = challenge_slug
        return payload


class SignedWeightsService:
    def __init__(self, *, store: LatestWeightsStore, signing_secret: str) -> None:
        self.store = store
        self.signing_secret = signing_secret

    def latest(self, challenge_slug: str | None = None) -> SignedWeightsResponse:
        payload = self.store.read(challenge_slug)
        return SignedWeightsResponse(
            payload=payload,
            signature=sign_payload(payload, self.signing_secret),
        )


class FallbackWeightClient:
    def __init__(
        self,
        *,
        primary_url: str,
        token: str,
        signing_secret: str,
        max_age_seconds: int = 600,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.primary_url = primary_url.rstrip("/")
        self.token = token
        self.signing_secret = signing_secret
        self.max_age_seconds = max_age_seconds
        self.timeout_seconds = timeout_seconds

    async def get_weights(
        self, *, slug: str, emission_percent: float
    ) -> ChallengeWeightsResult:
        url = f"{self.primary_url}/v1/weights/latest"
        headers = {"Authorization": f"Bearer {self.token}"}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(
                url,
                headers=headers,
                params={"challenge_slug": slug},
            )
            response.raise_for_status()
        signed = SignedWeightsResponse.model_validate(response.json())
        if not verify_payload(signed.payload, signed.signature, self.signing_secret):
            raise WeightFallbackError("invalid fallback weight signature")
        age = (datetime.now(UTC) - signed.payload.computed_at).total_seconds()
        if age > self.max_age_seconds:
            raise WeightFallbackError("fallback weights are too old")
        return ChallengeWeightsResult(
            slug=slug,
            emission_percent=emission_percent,
            weights=signed.payload.weights,
            ok=True,
        )


def sign_payload(payload: SignedWeightsPayload, secret: str) -> str:
    body = _canonical(payload)
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def verify_payload(payload: SignedWeightsPayload, signature: str, secret: str) -> bool:
    return hmac.compare_digest(sign_payload(payload, secret), signature)


def _canonical(payload: SignedWeightsPayload) -> bytes:
    return json.dumps(
        payload.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
