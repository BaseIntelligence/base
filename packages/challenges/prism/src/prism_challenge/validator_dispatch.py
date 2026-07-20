"""Validator dispatch for prism assignments.

The platform validator agent (``base validator agent``) pulls the single prism
gpu work unit from the master coordination plane and dispatches it here (selected
by ``challenge_slug``). :func:`dispatch_assignment` runs the GPU re-execution on
the validator's OWN broker by driving the production :class:`PrismWorker` (built
via :func:`prism_challenge.app.create_app`, the same construction the deployed
challenge uses) through :func:`run_validator_cycle`: the eval container runs
``network=none`` mounting only the locked train split + writable artifacts (never
val/test), with concurrency 1 enforced against the validator's real in-flight
draw.

LLM gateway scoped settings are gone: admission and scoring are deterministic and
never require a gateway token or provider URL. Residual ``gateway_*`` payload keys
are rejected fail-closed before any broker dispatch (VAL-GATE-017).

The signature deliberately uses only plain types (no dependency on the platform
validator-agent package), so this runs against the published ``base`` while the
platform side maps it onto the validator agent's executor seam.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .app import create_app
from .audit import is_audit_unit_id
from .config import PrismSettings, get_settings
from .evaluator.checkpoint_publisher import CheckpointPublisher
from .proof import MANIFEST_PAYLOAD_KEY, PROOF_PAYLOAD_KEY
from .validator_executor import run_primary_execution_cycle, run_validator_audit_cycle

CHALLENGE_SLUG = "prism"

_LEGACY_GATEWAY_PAYLOAD_KEYS = frozenset(
    {
        "gateway_token",
        "gateway_url",
        "gateway_base_url",
        "BASE_GATEWAY_TOKEN",
        "BASE_GATEWAY_TOKEN_FILE",
        "BASE_LLM_GATEWAY_URL",
        "PRISM_GATEWAY_TOKEN",
        "PRISM_GATEWAY_TOKEN_FILE",
        "PRISM_LLM_GATEWAY_URL",
        "llm_gateway_url",
        "llm_gateway_token",
        "llm_gateway_token_file",
        "llm_provider",
        "llm_model",
        "llm",
        "gateway",
    }
)


class PrismGatewayConfigError(ValueError):
    """Legacy LLM-gateway assignment payload was rejected fail-closed.

    Raised BEFORE any broker dispatch so residual gateway/provider/model fields
    never reach settings construction or execution.
    """


def _reject_legacy_gateway_payload(payload: Mapping[str, Any]) -> None:
    hits: list[str] = []

    def walk(node: Mapping[str, Any], path: str) -> None:
        for key, value in node.items():
            key_str = str(key)
            here = f"{path}.{key_str}" if path else key_str
            if (
                key_str in _LEGACY_GATEWAY_PAYLOAD_KEYS
                or key_str.upper() in _LEGACY_GATEWAY_PAYLOAD_KEYS
                or key_str.startswith("gateway_")
                or key_str.startswith("GATEWAY_")
                or key_str.startswith("llm_gateway")
            ):
                hits.append(here)
                continue
            if key_str in {"provider", "Provider"} and isinstance(value, Mapping):
                llm_like = {
                    "gateway_token",
                    "gateway_url",
                    "api_key",
                    "base_url",
                    "model",
                    "openai_api_key",
                    "openrouter_api_key",
                    "token",
                    "token_file",
                }
                if any(str(nested) in llm_like for nested in value):
                    hits.append(here)
                    continue
            if isinstance(value, Mapping):
                walk(value, here)

    walk(payload, "")
    if hits:
        raise PrismGatewayConfigError(
            "unsupported removed LLM gateway assignment fields: " + ", ".join(sorted(set(hits)))
        )


async def dispatch_assignment(
    *,
    work_unit_id: str,
    payload: Mapping[str, Any],
    broker_url: str,
    broker_token: str | None = None,
    broker_token_file: str | None = None,
    settings: PrismSettings | None = None,
    checkpoint_publisher: CheckpointPublisher | None = None,
) -> dict[str, Any]:
    """Run a pulled prism assignment on the caller's own broker (architecture.md 4).

    Routes on the UNIT TYPE so the same dispatch entrypoint serves both roles that reuse it:

    * an ``audit:`` unit (only assigned to a VALIDATOR, and only when the worker plane is on) is
      replayed + resolved (see :func:`_dispatch_audit_only`); the finalized score is untouched on a
      matching replay and invalidated + fault-recorded on a divergent one (VAL-FINAL-005);
    * any other (PRIMARY) unit is GPU-re-executed via :func:`run_primary_execution_cycle` -- this is
      the miner-funded worker path when the flag is on (it emits the ExecutionProof base reconciles)
      and the legacy validator re-execution when the flag is off.

    The autonomous validator cycle's audit-only restriction lives in :func:`run_validator_cycle`;
    here a primary unit always executes because the base assignment plane only ever routes a primary
    unit to a worker (flag on) or a legacy validator (flag off), never to an audit-only validator.
    Returns the cycle counts for the platform agent to post back to the master.
    """

    _reject_legacy_gateway_payload(payload)

    base_settings = settings if settings is not None else get_settings()
    if base_settings.worker_plane.enabled and is_audit_unit_id(work_unit_id):
        return await _dispatch_audit_only(
            work_unit_id=work_unit_id,
            broker_url=broker_url,
            broker_token=broker_token,
            broker_token_file=broker_token_file,
            settings=base_settings,
            checkpoint_publisher=checkpoint_publisher,
        )

    effective = settings_with_broker(
        base_settings,
        broker_url=broker_url,
        broker_token=broker_token,
        broker_token_file=broker_token_file,
    )
    app = create_app(effective, checkpoint_publisher=checkpoint_publisher)
    database = app.state.database
    await database.init()
    try:
        summary = await run_primary_execution_cycle(
            worker=app.state.worker, work_unit_ids=[work_unit_id]
        )
    finally:
        await database.close()
    result: dict[str, Any] = {
        "pulled": summary.pulled,
        "executed": summary.executed,
        "skipped": summary.skipped,
        "completed_submissions": list(summary.completed_submissions),
    }
    # Emit the ExecutionProof IN the work-unit result payload at successful finalization
    # (architecture.md 3.4; VAL-PRISM-001). Absent when the worker plane is off or the unit did not
    # freshly finalize. The backing run manifest is forwarded alongside so the accepting plane can
    # recompute + verify the signed digest and reject a tampered manifest (VAL-PRISM-007).
    proof = summary.execution_proofs.get(work_unit_id)
    if proof is not None:
        result[PROOF_PAYLOAD_KEY] = proof
        manifest = summary.execution_manifests.get(work_unit_id)
        if manifest is not None:
            result[MANIFEST_PAYLOAD_KEY] = manifest
    return result


async def _dispatch_audit_only(
    *,
    work_unit_id: str,
    broker_url: str,
    broker_token: str | None,
    broker_token_file: str | None,
    settings: PrismSettings,
    checkpoint_publisher: CheckpointPublisher | None,
) -> dict[str, Any]:
    """Audit-only dispatch: replay + resolve a sampled ``audit:`` unit (VAL-FINAL-005).

    Audits never require a master gateway token, only the validator's own broker
    for the deterministic re-execution.
    """

    effective = settings_with_broker(
        settings,
        broker_url=broker_url,
        broker_token=broker_token,
        broker_token_file=broker_token_file,
    )
    app = create_app(effective, checkpoint_publisher=checkpoint_publisher)
    database = app.state.database
    await database.init()
    try:
        summary = await run_validator_audit_cycle(
            worker=app.state.worker, work_unit_ids=[work_unit_id]
        )
    finally:
        await database.close()
    invalidated = sum(1 for resolution in summary.audits if resolution.invalidated)
    return {
        "pulled": summary.pulled,
        "executed": summary.executed,
        "skipped": summary.skipped,
        "completed_submissions": [],
        "is_audit": is_audit_unit_id(work_unit_id),
        "audits_resolved": len(summary.audits),
        "audits_invalidated": invalidated,
        "audit_results": [resolution.to_response() for resolution in summary.audits],
    }


def settings_with_broker(
    settings: PrismSettings,
    *,
    broker_url: str,
    broker_token: str | None = None,
    broker_token_file: str | None = None,
) -> PrismSettings:
    """Return settings bound only to the validator's broker (no LLM gateway)."""

    return settings.model_copy(
        update={
            "docker_broker_url": broker_url,
            "docker_broker_token": broker_token,
            "docker_broker_token_file": broker_token_file,
        }
    )


def gateway_scoped_settings(
    settings: PrismSettings,
    payload: Mapping[str, Any],
    *,
    broker_url: str,
    broker_token: str | None = None,
    broker_token_file: str | None = None,
) -> PrismSettings:
    """Compatibility shim: reject residual gateway payloads and bind the broker only.

    Historically stamped LLM gateway settings onto Prism. Those settings are removed;
    gateway fields now fail closed. Prefer :func:`settings_with_broker`.
    """

    _reject_legacy_gateway_payload(payload)
    return settings_with_broker(
        settings,
        broker_url=broker_url,
        broker_token=broker_token,
        broker_token_file=broker_token_file,
    )


__all__ = [
    "CHALLENGE_SLUG",
    "PrismGatewayConfigError",
    "dispatch_assignment",
    "gateway_scoped_settings",
    "settings_with_broker",
]
