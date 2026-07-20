"""Miner-facing self-deploy CLI (``python -m agent_challenge.selfdeploy``).

Subcommands cover the full flow (VAL-DEPLOY-001): ``prepare`` (fetch/prepare the
canonical image + generated compose), ``measurements`` (publish/reproduce the
canonical measurement), ``verdict`` (report a measurement + its allowlist
verdict), ``deploy`` (deploy a CPU-only CVM, with a no-spend ``--dry-run`` and
GPU/over-cap/credential guards), ``run`` (run the eval against the validator
key-release endpoint), ``result`` (surface + verify the attested-result envelope),
and ``teardown`` (delete the CVM).

The two spend-capable subcommands (``deploy``, ``run``) accept injectable side
effects (the Phala deployer / backend runner / teardown runner) so the whole
surface is testable offline; only ``deploy`` (without ``--dry-run``) and
``teardown`` reach Phala, and both refuse clearly before any Phala call when a
guard fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_challenge.selfdeploy import eval as eval_deploy
from agent_challenge.selfdeploy import lifecycle
from agent_challenge.selfdeploy import measurements as measure
from agent_challenge.selfdeploy import result as result_mod
from agent_challenge.selfdeploy import review as review_deploy
from agent_challenge.selfdeploy import run as run_mod
from agent_challenge.selfdeploy.client import RouteClientError, SelfDeployRouteClient
from agent_challenge.selfdeploy.phala import (
    DEFAULT_PHALA_API,
    PhalaApiError,
    PhalaCloudClient,
)
from agent_challenge.selfdeploy.plan import (
    CredentialError,
    DeployPlan,
    PrepareError,
    build_deploy_plan,
    check_phala_credentials,
    prepare_deployment,
    render_plan,
    write_prepared,
)
from agent_challenge.selfdeploy.shapes import (
    DEFAULT_MAX_RUNTIME_HOURS,
    DEFAULT_MONEY_CAP_USD,
    DEFAULT_OS_IMAGE,
    ShapeError,
)

PROG = "agent-challenge-selfdeploy"

#: A Phala deployer: (plan, out_dir) -> arbitrary result (printed by the CLI).
Deployer = Callable[[DeployPlan, str], Any]
#: A teardown runner: cvm_id -> arbitrary result (printed by the CLI).
Teardowner = Callable[[str], Any]

#: The subcommands the CLI exposes (kept in sync with docs/miner/self-deploy.md).
SUBCOMMANDS: tuple[str, ...] = (
    "prepare",
    "measurements",
    "verdict",
    "deploy",
    "run",
    "result",
    "teardown",
)

#: Subcommands that can create/charge Phala resources (must all be documented).
SPEND_CAPABLE_SUBCOMMANDS: frozenset[str] = frozenset({"deploy", "run"})

# Attested mode uses only these ordered production stages.  The legacy
# top-level helpers remain for the feature-flag-off validator-run path.
ORDERED_SUBCOMMANDS: tuple[str, ...] = ("review", "eval")


# --------------------------------------------------------------------------- #
# Default side effects (never exercised by the offline suite)
# --------------------------------------------------------------------------- #
def default_phala_deployer(plan: DeployPlan, out_dir: str) -> dict[str, Any]:  # pragma: no cover
    """Write the exact app-compose bytes and invoke ``phala deploy`` (live, M6).

    Writes ``app-compose.json`` verbatim (so the deployed compose-hash matches the
    pinned measurement) and shells the ``phala`` CLI. Live deploy/teardown are
    validated at milestone ``self-deploy-live``; the offline suite drives an
    injected deployer instead.
    """

    compose_path = write_prepared(plan.prepared, out_dir)
    cmd = [
        "phala",
        "deploy",
        "-c",
        str(compose_path),
        "-n",
        plan.name,
        "-t",
        plan.instance_type,
        "-r",
        plan.region,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


_TEARDOWN_DIAGNOSTIC_LIMIT = 512


def _bounded_text(value: str | None, *, limit: int = _TEARDOWN_DIAGNOSTIC_LIMIT) -> str:
    """Return a size-bounded diagnostic string suitable for CLI surfaces."""

    text = value or ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def default_phala_teardown(cvm_id: str) -> dict[str, Any]:  # pragma: no cover
    """Delete a CVM via ``phala cvms delete <id> -f`` (idempotent; live, M6).

    Always returns a structured result with ``ok``/``returncode`` so callers can
    fail non-zero when deletion does not succeed. Stdout/stderr are bounded.
    """

    proc = subprocess.run(["phala", "cvms", "delete", cvm_id, "-f"], capture_output=True, text=True)
    ok = proc.returncode == 0
    return {
        "returncode": proc.returncode,
        "ok": ok,
        "stdout": _bounded_text(proc.stdout),
        "stderr": _bounded_text(proc.stderr),
        "error": None if ok else "phala cvms delete failed",
    }


def _teardown_payload(cvm_id: str, result: Any) -> tuple[dict[str, Any], int]:
    """Project teardown outcome as a miner-facing payload and exit code."""

    if isinstance(result, Mapping):
        returncode = int(result.get("returncode") or 0)
        ok = bool(result.get("ok", returncode == 0))
        diagnostics = {
            "returncode": returncode,
            "error": result.get("error"),
            "stdout": _bounded_text(str(result.get("stdout") or "")),
            "stderr": _bounded_text(str(result.get("stderr") or "")),
        }
    else:
        ok = True
        returncode = 0
        diagnostics = {"returncode": 0, "error": None, "stdout": "", "stderr": ""}
    payload = {
        "torn_down": cvm_id,
        "ok": ok,
        "diagnostics": diagnostics,
        "result": diagnostics,
    }
    return payload, 0 if ok else 1


def _delete_attributable_cvm(cvm_id: str | None) -> None:
    """Best-effort delete of a funded CVM after a post-create failure."""

    if not cvm_id:
        return
    try:
        default_phala_teardown(cvm_id)
    except Exception:  # noqa: BLE001 - cleanup must never mask the original failure
        return


def _review_token_present(prepare_response: Mapping[str, Any] | None) -> bool:
    """True when prepare/retry body still carries a non-empty one-time token."""

    if not isinstance(prepare_response, Mapping):
        return False
    token = prepare_response.get("review_session_token")
    return isinstance(token, str) and bool(token)


def _assignment_id_from_prepare(prepare_response: Mapping[str, Any]) -> str | None:
    """Extract current assignment_id without requiring a valid token."""

    assignment_id = prepare_response.get("assignment_id")
    if isinstance(assignment_id, str) and assignment_id:
        return assignment_id
    assignment = prepare_response.get("assignment")
    if isinstance(assignment, Mapping):
        core = assignment.get("assignment_core")
        if isinstance(core, Mapping):
            nested = core.get("assignment_id")
            if isinstance(nested, str) and nested:
                return nested
    return None


def _obtain_review_prepare_with_token(
    client: SelfDeployRouteClient,
    submission_id: int,
) -> dict[str, Any]:
    """Return a prepare/retry response that still delivers the session token.

    Product residual timeline:
    - tee-live-proof-v5: re-prepare after first delivery returned null token →
      cancel+retry recovered a new attempt (burned undepoyed attempts).
    - review-v9: product redelivers capability for active undepoyed assignments,
      so the common dry-run → live path never hits null and never cancels.
    Cancel+retry remains only when prepare is sticky-null (post-deploy CVM still
    holding the capability, or terminal) and a fresh assignment is required.
    Does not invent tokens or persist capabilities offline.
    """

    response = client.review_prepare(submission_id)
    if _review_token_present(response):
        return response
    assignment_id = _assignment_id_from_prepare(response)
    if not assignment_id:
        raise RouteClientError("review session has no current assignment id to refresh capability")
    # Sticky-null after deploy or terminal: only then cancel+retry for a new token.
    # Prefer not cancelling an active undepoyed assignment — product should have
    # redelivered. If prepare still null, attempt cancel (may already be terminal)
    # then issue a fresh assignment with token.
    try:
        client.review_cancel(submission_id, assignment_id)
    except RouteClientError:
        # Terminal / already cancelled: still try retry against that id.
        pass
    retried = client.review_retry(submission_id, assignment_id)
    if not _review_token_present(retried):
        raise RouteClientError(
            "review session token unavailable after prepare and retry; "
            "capability may be spent or assignment not retryable"
        )
    return retried


def _review_allowlist_verdict(plan: review_deploy.ReviewDeploymentPlan) -> str:
    """Compute a verified review-domain allowlist verdict or explicit UNKNOWN."""

    app = plan.assignment["assignment_core"]["review_app"]
    allowlist = app.get("measurement_allowlist")
    if not isinstance(allowlist, list) or not allowlist:
        return "UNKNOWN"
    measurement = {
        "mrtd": plan.measurement.get("mrtd"),
        "rtmr0": plan.measurement.get("rtmr0"),
        "rtmr1": plan.measurement.get("rtmr1"),
        "rtmr2": plan.measurement.get("rtmr2"),
        "compose_hash": plan.compose_hash,
        "os_image_hash": plan.measurement.get("os_image_hash"),
    }
    try:
        return measure.domain_allowlist_verdict(
            domain="review",
            measurement=measurement,
            review_allowlist=allowlist,
        ).as_dict()["verdict"]
    except measure.MeasurementError:
        return "UNKNOWN"


def _eval_allowlist_verdict(plan: eval_deploy.EvalDeploymentPlan) -> str:
    """Eval plan alone cannot prove allowlist membership; report UNKNOWN."""

    # The validator-owned Eval allowlist is not part of the miner prepare wrapper.
    # Never fabricate IN-LIST from compose-hash presence alone.
    allowlist = plan.plan.get("measurement_allowlist") or plan.measurement.get(
        "measurement_allowlist"
    )
    if not allowlist:
        return "UNKNOWN"
    measurement = {
        "mrtd": plan.measurement.get("mrtd"),
        "rtmr0": plan.measurement.get("rtmr0"),
        "rtmr1": plan.measurement.get("rtmr1"),
        "rtmr2": plan.measurement.get("rtmr2"),
        "compose_hash": plan.compose_hash,
        "os_image_hash": plan.measurement.get("os_image_hash"),
    }
    try:
        return measure.domain_allowlist_verdict(
            domain="eval",
            measurement=measurement,
            eval_allowlist=allowlist,
        ).as_dict()["verdict"]
    except measure.MeasurementError:
        return "UNKNOWN"


def _add_route_args(
    parser: argparse.ArgumentParser,
    *,
    include_submission: bool,
    signed: bool = True,
) -> None:
    """Add safe signed-route arguments without accepting arbitrary endpoints."""

    parser.add_argument("--base-url", required=True, help="challenge production API base URL")
    if include_submission:
        parser.add_argument("--submission-id", required=True, type=int)
    if signed:
        parser.add_argument("--hotkey", required=True)
        parser.add_argument("--signature", default=None)
        parser.add_argument("--nonce", default=None)
        parser.add_argument("--timestamp", default=None)
        parser.add_argument(
            "--auto-sign",
            action="store_true",
            help="sign each request with MINER_HOTKEY_MNEMONIC or MINER_HOTKEY_URI",
        )


def _route_client(args: argparse.Namespace) -> SelfDeployRouteClient:
    from agent_challenge.selfdeploy.client import build_signed_identity

    identity = None
    auto_sign = bool(getattr(args, "auto_sign", False))
    if hasattr(args, "hotkey"):
        if auto_sign:
            identity = build_signed_identity(
                hotkey=args.hotkey,
                signature="auto",
                nonce="auto",
                timestamp=args.timestamp,
            )
        else:
            if not args.signature or not args.nonce:
                raise RouteClientError(
                    "signed route requires --signature and --nonce, or --auto-sign"
                )
            identity = build_signed_identity(
                hotkey=args.hotkey,
                signature=args.signature,
                nonce=args.nonce,
                timestamp=args.timestamp,
            )
    return SelfDeployRouteClient(args.base_url, identity=identity, auto_sign=auto_sign)


def _safe_json_file(path: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RouteClientError("JSON input could not be read") from exc
    if not isinstance(value, dict):
        raise RouteClientError("JSON input must contain an object")
    return value


def _ordered_review_command(args: argparse.Namespace) -> int:
    try:
        if args.review_command == "prepare":
            payload = _route_client(args).review_prepare(args.submission_id)
            if args.output:
                Path(args.output).write_text(
                    json.dumps(
                        _redact_capabilities(payload),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    encoding="utf-8",
                )
            _print(_redact_capabilities(payload))
            return 0
        if args.review_command == "result":
            _print(
                _redact_capabilities(
                    _route_client(args).review_report(args.submission_id, cursor=args.cursor)
                )
            )
            return 0
        if args.review_command == "history":
            _print(
                _redact_capabilities(
                    _route_client(args).review_history(args.submission_id, cursor=args.cursor)
                )
            )
            return 0
        if args.review_command == "deployed":
            _print(
                _redact_capabilities(
                    _route_client(args).review_deployed(
                        args.submission_id,
                        _safe_json_file(args.acknowledgement),
                    )
                )
            )
            return 0
        if args.review_command in {"cancel", "retry"}:
            client = _route_client(args)
            if args.review_command == "cancel":
                payload = client.review_cancel(args.submission_id, args.assignment_id)
            else:
                approval_id = getattr(args, "approval_id", None)
                payload = client.review_retry(
                    args.submission_id,
                    args.assignment_id,
                    approval_id=approval_id,
                )
            _print(_redact_capabilities(payload))
            return 0
        if args.review_command == "teardown":
            payload, code = _teardown_payload(args.cvm_id, default_phala_teardown(args.cvm_id))
            _print(payload)
            return code
        if args.review_command == "deploy":
            # Live review deploy requires a signable identity. Prefer
            # --auto-sign when acknowledgement bodies are produced by the
            # same process; accept documented explicit signature headers
            # when the caller supplies a fresh signature over that request.
            if (
                not args.dry_run
                and not args.auto_sign
                and (not getattr(args, "signature", None) or not getattr(args, "nonce", None))
            ):
                raise RouteClientError(
                    "review deploy requires --auto-sign or explicit --signature and --nonce "
                    "so the created-CVM acknowledgement is signed over its returned receipt "
                    "and CVM identity"
                )
            if args.prepare_response:
                raise RouteClientError(
                    "review deploy does not accept persisted prepare capabilities; "
                    "run it with signed production route credentials"
                )
            client = _route_client(args)
            response = _obtain_review_prepare_with_token(client, args.submission_id)
            plan = review_deploy.build_review_deployment_plan(response)
            if plan.instance_type != args.review_instance_type:
                raise RouteClientError(
                    "review deployment shape differs from the validator-issued assignment"
                )
            lifecycle.validate_lifecycle_budget(
                review_instance_type=plan.instance_type,
                eval_instance_type=args.eval_instance_type,
                review_runtime_hours=args.review_runtime_hours,
                eval_runtime_hours=args.eval_runtime_hours,
                money_cap_usd=args.money_cap_usd,
            )
            key = os.environ.get(args.openrouter_key_env, "")
            if not args.dry_run and not key:
                raise RouteClientError(
                    f"{args.openrouter_key_env} is not set; review deployment cannot continue"
                )
            # REVIEW_API_BASE_URL is bound from the same production challenge
            # base URL the miner already uses (joinbase agent-challenge path).
            # Measured compose allows.
            # Without it, older review images default to chain.platform.network
            # (502) and never POST /report.
            api_base = str(args.base_url).rstrip("/")
            encrypted = (
                review_deploy.encrypt_review_secrets(
                    plan,
                    {
                        "OPENROUTER_API_KEY": key,
                        "REVIEW_API_BASE_URL": api_base,
                        "REVIEW_SESSION_TOKEN": plan.review_session_token,
                    },
                )
                if not args.dry_run
                else None
            )
            if not args.dry_run:
                assert encrypted is not None
                acknowledgement: dict[str, Any] | None = None
                try:
                    acknowledgement = review_deploy.HttpReviewPhalaDeployment(
                        PhalaCloudClient(base_url=args.phala_api or DEFAULT_PHALA_API)
                    ).deploy(plan, encrypted)
                    signed_ack = client.review_deployed(
                        args.submission_id,
                        acknowledgement,
                    )
                except Exception:
                    cvm_id = None
                    if isinstance(acknowledgement, Mapping):
                        cvm_id = acknowledgement.get("cvm_id")
                    _delete_attributable_cvm(cvm_id if isinstance(cvm_id, str) else None)
                    raise
                _print(
                    {
                        "stage": "review_deployed",
                        "acknowledgement": acknowledgement,
                        "signed_acknowledgement": signed_ack,
                        "encrypted_env_names": list(encrypted.env_keys),
                    }
                )
                return 0
            _print(
                {
                    "stage": "review_deploy_ready",
                    "assignment_id": plan.assignment["assignment_core"]["assignment_id"],
                    "image_ref": plan.image_ref,
                    "compose_hash": plan.compose_hash,
                    "measurement": plan.measurement,
                    "allowlist_verdict": _review_allowlist_verdict(plan),
                    "encrypted_env_names": list(review_deploy.REVIEW_ALLOWED_ENVS),
                    "encrypted_env_transmitted": False,
                    "dry_run": True,
                }
            )
            return 0
    except (
        RouteClientError,
        CredentialError,
        lifecycle.LifecycleBudgetError,
        review_deploy.ReviewDeploymentError,
        PhalaApiError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise RouteClientError("unknown review stage")


def _ordered_eval_command(args: argparse.Namespace) -> int:
    try:
        if args.eval_command == "prepare":
            payload = _route_client(args).eval_prepare(args.submission_id)
            if args.output:
                Path(args.output).write_text(
                    json.dumps(
                        _redact_capabilities(payload),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    encoding="utf-8",
                )
            _print(_redact_capabilities(payload))
            return 0
        if args.eval_command == "status":
            _print(
                _redact_capabilities(
                    _route_client(args).eval_status(args.submission_id, cursor=args.cursor)
                )
            )
            return 0
        if args.eval_command in {"cancel", "retry"}:
            client = _route_client(args)
            payload = (
                client.eval_cancel(args.submission_id, args.run_id)
                if args.eval_command == "cancel"
                else client.eval_retry(args.submission_id, args.run_id)
            )
            _print(_redact_capabilities(payload))
            return 0
        if args.eval_command == "failure":
            _print(
                _redact_capabilities(
                    _route_client(args).eval_failure(
                        args.submission_id,
                        args.run_id,
                        args.reason_code,
                    )
                )
            )
            return 0
        if args.eval_command == "result":
            token = os.environ.get(args.token_env, "")
            if not token:
                raise RouteClientError(f"{args.token_env} is not set")
            try:
                raw_result = Path(args.result).read_bytes()
            except OSError as exc:
                raise RouteClientError("Eval result JSON could not be read") from exc
            if len(raw_result) > 16 * 1024 * 1024:
                raise RouteClientError("Eval result exceeds the bounded request size")
            try:
                parsed_result = json.loads(raw_result)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RouteClientError("Eval result JSON is malformed") from exc
            if not isinstance(parsed_result, dict):
                raise RouteClientError("Eval result JSON must contain an object")
            _print(
                _redact_capabilities(
                    _route_client(args).eval_result_bytes(
                        args.run_id,
                        raw_result,
                        token,
                    )
                )
            )
            return 0
        if args.eval_command == "teardown":
            payload, code = _teardown_payload(args.cvm_id, default_phala_teardown(args.cvm_id))
            _print(payload)
            return code
        if args.eval_command == "deploy":
            if args.prepare_response:
                raise RouteClientError(
                    "Eval deploy does not accept persisted prepare capabilities; "
                    "run it with signed production route credentials"
                )
            raw = _route_client(args).eval_prepare(args.submission_id)
            plan = eval_deploy.build_eval_deployment_plan(raw)
            if plan.instance_type != args.eval_instance_type:
                raise RouteClientError(
                    "Eval deployment shape differs from the validator-issued plan"
                )
            lifecycle.validate_lifecycle_budget(
                review_instance_type=args.review_instance_type,
                eval_instance_type=plan.instance_type,
                review_runtime_hours=args.review_runtime_hours,
                eval_runtime_hours=args.eval_runtime_hours,
                money_cap_usd=args.money_cap_usd,
            )
            values = {
                "EVAL_RUN_TOKEN": plan.eval_run_token,
                "LLM_COST_LIMIT": os.environ.get(args.llm_cost_limit_env, "") or "0",
            }
            # VAL-ACAT-013: Base LLM gateway env vars are never injected into eval
            # encrypted_env. Residual --gateway-*-env flags are ignored if present.
            if not args.dry_run and any(not value for value in values.values()):
                raise RouteClientError(
                    "Eval run token and LLM_COST_LIMIT are required before deployment "
                    "(Base LLM gateway secrets are not used)"
                )
            values["CHALLENGE_PHALA_ATTESTATION_ENABLED"] = "1"
            values["CHALLENGE_PHALA_EVAL_PLAN"] = json.dumps(
                plan.plan,
                sort_keys=True,
                separators=(",", ":"),
            )
            values["CHALLENGE_PHALA_AGENT_HASH"] = plan.plan["agent_hash"]
            values["CHALLENGE_PHALA_CANONICAL_MEASUREMENT"] = json.dumps(
                {
                    "mrtd": plan.measurement["mrtd"],
                    "rtmr0": plan.measurement["rtmr0"],
                    "rtmr1": plan.measurement["rtmr1"],
                    "rtmr2": plan.measurement["rtmr2"],
                    "compose_hash": plan.compose_hash,
                    "os_image_hash": plan.measurement["os_image_hash"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            values["CHALLENGE_PHALA_VALIDATOR_NONCE"] = plan.plan["key_release_nonce"]
            # Production endpoint is provisioned via KEY_RELEASE_RA_TLS_HOST/PORT in
            # the measured compose; keep the plan authority only as a legacy accessor
            # for non-raw offline helpers, never as an HTTP fallback URL on the wire.
            # RTMR3 is produced by the in-CVM dstack quote/event-log replay.
            # The miner must not fabricate a runtime measurement in ordinary
            # encrypted_env; the measured image derives it from its quote.
            #
            # Inject the validator raw-listener trust CA so the guest verifies the
            # host RA-TLS cert (never the dstack guest intermediate). Production
            # requires CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM or _FILE (or
            # KEY_RELEASE_SERVER_CA_FILE pointing at the CA that signed server.crt).
            server_ca_pem = (os.environ.get("CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM") or "").strip()
            if not server_ca_pem:
                for ca_env in (
                    "CHALLENGE_PHALA_RA_TLS_SERVER_CA_FILE",
                    "KEY_RELEASE_SERVER_CA_FILE",
                ):
                    ca_path = (os.environ.get(ca_env) or "").strip()
                    if ca_path and Path(ca_path).is_file():
                        server_ca_pem = Path(ca_path).read_text(encoding="utf-8").strip()
                        break
            if server_ca_pem and "BEGIN CERTIFICATE" in server_ca_pem:
                # Normalize/unescape (encrypted_env or file may carry literal \\n)
                # and OpenSSL-preload before inject so the guest never gets junk.
                try:
                    from agent_challenge.keyrelease.client import normalize_server_ca_pem

                    values["CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM"] = normalize_server_ca_pem(
                        server_ca_pem
                    )
                except ValueError as exc:
                    raise RouteClientError(
                        f"raw RA-TLS eval deploy server CA is not OpenSSL-loadable: {exc}"
                    ) from exc
            elif not args.dry_run:
                raise RouteClientError(
                    "raw RA-TLS eval deploy requires the validator server CA "
                    "(set CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM or "
                    "CHALLENGE_PHALA_RA_TLS_SERVER_CA_FILE / KEY_RELEASE_SERVER_CA_FILE)"
                )
            encrypted = eval_deploy.encrypt_eval_secrets(plan, values) if not args.dry_run else None
            if not args.dry_run:
                assert encrypted is not None
                acknowledgement: dict[str, Any] | None = None
                try:
                    acknowledgement = eval_deploy.HttpEvalPhalaDeployment(
                        PhalaCloudClient(base_url=args.phala_api or DEFAULT_PHALA_API)
                    ).deploy(plan, encrypted)
                except Exception as exc:
                    attributable = getattr(exc, "attributable_cvm_id", None)
                    if not attributable and isinstance(acknowledgement, Mapping):
                        attributable = acknowledgement.get("cvm_id")
                    _delete_attributable_cvm(
                        attributable if isinstance(attributable, str) else None
                    )
                    raise
                _print(
                    {
                        "stage": "eval_deployed",
                        "acknowledgement": acknowledgement,
                        "encrypted_env_names": list(encrypted.env_keys),
                    }
                )
                return 0
            _print(
                {
                    "stage": "eval_deploy_ready",
                    "eval_run_id": plan.eval_run_id,
                    "image_ref": plan.image_ref,
                    "compose_hash": plan.compose_hash,
                    "key_release_endpoint": plan.plan["key_release_endpoint"],
                    "measurement": plan.measurement,
                    "allowlist_verdict": _eval_allowlist_verdict(plan),
                    "encrypted_env_names": list(eval_deploy.EVAL_ALLOWED_ENVS),
                    "encrypted_env_transmitted": False,
                    "dry_run": True,
                }
            )
            return 0
    except (
        RouteClientError,
        CredentialError,
        lifecycle.LifecycleBudgetError,
        eval_deploy.EvalDeploymentError,
        PhalaApiError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise RouteClientError("unknown Eval stage")


_REDACTED_CAPABILITY_KEYS = frozenset(
    {
        "review_session_token",
        "token",
        "OPENROUTER_API_KEY",
        "EVAL_RUN_TOKEN",
        "REVIEW_SESSION_TOKEN",
        "BASE_GATEWAY_TOKEN",  # residual key name only; not product eval secret
        "golden_plaintext",
        "golden_key",
        "raw_response",
        "raw_request",
        "unrestricted_source",
    }
)


def _redact_capabilities(value: Any) -> Any:
    """Remove one-time capability bytes from CLI output and persisted plans."""

    if isinstance(value, dict):
        result = {
            key: _redact_capabilities(item)
            for key, item in value.items()
            if key not in _REDACTED_CAPABILITY_KEYS
        }
        if "secret_delivery" in result and result["secret_delivery"] is not None:
            delivery = result["secret_delivery"]
            if isinstance(delivery, dict):
                result["secret_delivery"] = {"env_key": delivery.get("env_key")}
        return result
    if isinstance(value, list):
        return [_redact_capabilities(item) for item in value]
    return value


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Miner self-deploy flow for the canonical Phala TEE eval image: prepare, "
            "reproduce measurements, deploy a CPU-only CVM, run the eval against the "
            "validator key-release endpoint, surface the attested result, and tear down."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser(
        "prepare",
        help="fetch/prepare the canonical image + generated Phala compose",
        description="Resolve the digest-pinned canonical image and write the deployable compose.",
    )
    prep.add_argument("--image", required=True, help="canonical image ref (repo@sha256:<64hex>)")
    prep.add_argument("--key-release-url", required=True, help="validator key-release endpoint URL")
    prep.add_argument("--out", default=".", help="output directory for app-compose.json")
    prep.add_argument("--name", default=None, help="app name (default: canonical app name)")

    meas = sub.add_parser(
        "measurements",
        help="publish/reproduce the canonical measurement record",
        description="Deterministically recompute the pinnable {mrtd,rtmr0-2,compose_hash,"
        "os_image_hash} record.",
    )
    meas.add_argument("--metadata", required=True, help="dstack image metadata.json path")
    meas.add_argument("--cpu", type=int, required=True, help="vCPU count of the pinned VM shape")
    meas.add_argument("--memory", required=True, help="memory of the pinned VM shape, e.g. 4G")
    meas.add_argument("--compose", required=True, help="app-compose.json path to pin")
    meas.add_argument("--dstack-mr", default=None, help="override the dstack-mr binary")

    verd = sub.add_parser(
        "verdict",
        help="report a measurement and its validator-allowlist verdict",
        description="Report a measurement's canonical fields and whether it is in the allowlist.",
    )
    verd.add_argument("--measurement", default=None, help="measurement JSON string or file path")
    verd.add_argument("--from-result", default=None, help="a captured run output to read it from")
    verd.add_argument("--allowlist", required=True, help="validator allowlist JSON string/file")
    verd.add_argument("--domain", choices=("review", "eval"), default="eval")

    dep = sub.add_parser(
        "deploy",
        help="deploy a CPU-only Phala CVM (miner-funded); use --dry-run to plan",
        description="Build a validated CPU-only deploy plan and (unless --dry-run) deploy it.",
    )
    dep.add_argument("--image", required=True, help="canonical image ref (repo@sha256:<64hex>)")
    dep.add_argument("--key-release-url", required=True, help="validator key-release endpoint URL")
    dep.add_argument(
        "--instance-type",
        default=None,
        help="CPU Intel TDX shape (default: smallest, tdx.small)",
    )
    dep.add_argument("--os-image", default=DEFAULT_OS_IMAGE, help="dstack CPU OS image")
    dep.add_argument(
        "--region",
        default=None,
        help="Phala region (default: us-west-1; bare us-west remaps to us-west-1)",
    )
    dep.add_argument("--name", default=None, help="app/CVM name")
    dep.add_argument("--out", default=".", help="output directory for app-compose.json")
    dep.add_argument(
        "--max-runtime-hours",
        type=float,
        default=DEFAULT_MAX_RUNTIME_HOURS,
        help="projected max runtime used for the cost-cap guard",
    )
    dep.add_argument(
        "--money-cap-usd",
        type=float,
        default=DEFAULT_MONEY_CAP_USD,
        help="hard spend cap; a shape whose projected cost exceeds it is refused",
    )
    dep.add_argument(
        "--dry-run",
        action="store_true",
        help="print the full deploy plan and make zero CVM-creating calls",
    )

    runp = sub.add_parser(
        "run",
        help="run the eval against the validator key-release endpoint",
        description="Run the canonical eval; fails closed with no result if key-release fails.",
    )
    runp.add_argument("--job-dir", required=True, help="orchestrator job directory")
    runp.add_argument(
        "--task",
        dest="task_ids",
        action="append",
        required=True,
        metavar="TASK_ID",
        help="task id to evaluate (repeatable)",
    )
    runp.add_argument("--key-release-url", required=True, help="validator key-release endpoint URL")

    res = sub.add_parser(
        "result",
        help="surface + verify the attested-result envelope",
        description="Parse a captured run output, surface the envelope, and verify its binding.",
    )
    res.add_argument(
        "--from", dest="from_path", default=None, help="captured run output (else stdin)"
    )
    res.add_argument("--allowlist", default=None, help="also report the allowlist verdict")
    res.add_argument(
        "--quote-verified",
        choices=["true", "false"],
        default=None,
        help="fold a quote-verification verdict (Phala verify / dcap-qvl) into the acceptance",
    )
    res.add_argument(
        "--nonce-state",
        choices=["ok", "stale", "consumed", "unknown"],
        default=None,
        help="fold the validator nonce-ledger verdict into the acceptance",
    )
    res.add_argument(
        "--key-grant",
        choices=["true", "false"],
        default=None,
        help="fold the matching validator key-grant outcome into the acceptance",
    )

    tear = sub.add_parser(
        "teardown",
        help="delete a deployed CVM (idempotent)",
        description="Delete the CVM so no resource is left running (phala cvms delete -f).",
    )
    tear.add_argument("--cvm-id", required=True, help="the CVM id to delete")

    # Ordered production lifecycle.  The older top-level helpers remain as
    # compatibility shims for offline callers, but all new spend-capable work
    # is organized under review/eval and requires the stage-specific identity.
    review = sub.add_parser(
        "review",
        help="ordered review prepare/deploy/deployed/result/cancel/retry/teardown stages",
    )
    review_sub = review.add_subparsers(dest="review_command", required=True)
    review_prepare = review_sub.add_parser("prepare", help="request one signed review assignment")
    _add_route_args(review_prepare, include_submission=True)
    review_prepare.add_argument("--output", default=None, help="safe assignment output path")
    review_deploy = review_sub.add_parser("deploy", help="deploy the signed review assignment")
    _add_route_args(review_deploy, include_submission=True)
    review_deploy.add_argument(
        "--prepare-response",
        default=None,
        help=argparse.SUPPRESS,
    )
    review_deploy.add_argument(
        "--openrouter-key-env",
        default="OPENROUTER_API_KEY",
        help="environment variable holding the user key",
    )
    review_deploy.add_argument("--phala-api", default=None, help="Phala Cloud API base URL")
    review_deploy.add_argument("--review-instance-type", default="tdx.small")
    review_deploy.add_argument("--eval-instance-type", default="tdx.small")
    review_deploy.add_argument("--review-runtime-hours", type=float, default=6.0)
    review_deploy.add_argument("--eval-runtime-hours", type=float, default=6.0)
    review_deploy.add_argument("--money-cap-usd", type=float, default=20.0)
    review_deploy.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print safe deployment metadata without provisioning",
    )
    review_ack = review_sub.add_parser("deployed", help="acknowledge the created review CVM")
    _add_route_args(review_ack, include_submission=True)
    review_ack.add_argument("--acknowledgement", required=True, help="acknowledgement JSON path")
    review_result = review_sub.add_parser("result", help="surface the signed review audit bundle")
    _add_route_args(review_result, include_submission=True)
    review_result.add_argument("--cursor", default=None)
    review_history = review_sub.add_parser("history", help="read safe review attempt history")
    _add_route_args(review_history, include_submission=True)
    review_history.add_argument("--cursor", default=None)
    for name, help_text in (
        ("cancel", "cancel the expected review assignment"),
        ("retry", "retry the expected review assignment"),
    ):
        item = review_sub.add_parser(name, help=help_text)
        _add_route_args(item, include_submission=True)
        item.add_argument("--assignment-id", required=True)
        if name == "retry":
            item.add_argument(
                "--approval-id",
                default=None,
                help=(
                    "one-use operator approval_id required to retry after "
                    "policy reject/escalate (atomically consumed by the server)"
                ),
            )
    review_tear = review_sub.add_parser("teardown", help="delete the review CVM")
    review_tear.add_argument("--cvm-id", required=True)

    evaluation = sub.add_parser(
        "eval",
        help="ordered eval prepare/deploy/result/status/cancel/retry/failure/teardown stages",
    )
    eval_sub = evaluation.add_subparsers(dest="eval_command", required=True)
    eval_prepare = eval_sub.add_parser(
        "prepare",
        help="request the immutable Eval plan after allow",
    )
    _add_route_args(eval_prepare, include_submission=True)
    eval_prepare.add_argument("--output", default=None, help="safe plan output path")
    eval_deploy_parser = eval_sub.add_parser("deploy", help="deploy the signed Eval plan")
    _add_route_args(eval_deploy_parser, include_submission=True)
    eval_deploy_parser.add_argument(
        "--prepare-response",
        default=None,
        help=argparse.SUPPRESS,
    )
    # Residual CLI flags kept for operator scripts that still pass them; values
    # are ignored. Base gateway secrets are never required for eval deploy.
    eval_deploy_parser.add_argument(
        "--gateway-token-env",
        default="BASE_GATEWAY_TOKEN",
        help=argparse.SUPPRESS,
    )
    eval_deploy_parser.add_argument(
        "--gateway-url-env",
        default="BASE_LLM_GATEWAY_URL",
        help=argparse.SUPPRESS,
    )
    eval_deploy_parser.add_argument("--llm-cost-limit-env", default="LLM_COST_LIMIT")
    eval_deploy_parser.add_argument("--phala-api", default=None, help="Phala Cloud API base URL")
    eval_deploy_parser.add_argument("--review-instance-type", default="tdx.small")
    eval_deploy_parser.add_argument("--eval-instance-type", default="tdx.small")
    eval_deploy_parser.add_argument("--review-runtime-hours", type=float, default=6.0)
    eval_deploy_parser.add_argument("--eval-runtime-hours", type=float, default=6.0)
    eval_deploy_parser.add_argument("--money-cap-usd", type=float, default=20.0)
    eval_deploy_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print safe deployment metadata without provisioning",
    )
    eval_result_parser = eval_sub.add_parser(
        "result",
        help="post the exact result to the direct route",
    )
    _add_route_args(eval_result_parser, include_submission=False, signed=False)
    eval_result_parser.add_argument("--run-id", required=True)
    eval_result_parser.add_argument("--result", required=True, help="exact result JSON path")
    eval_result_parser.add_argument("--token-env", default="EVAL_RUN_TOKEN")
    eval_status = eval_sub.add_parser("status", help="read signed Eval receipt/history")
    _add_route_args(eval_status, include_submission=True)
    eval_status.add_argument("--cursor", default=None)
    for name, help_text in (
        ("cancel", "cancel the expected Eval run"),
        ("retry", "retry the expected Eval run"),
    ):
        item = eval_sub.add_parser(name, help=help_text)
        _add_route_args(item, include_submission=True)
        item.add_argument("--run-id", required=True)
    eval_failure = eval_sub.add_parser("failure", help="record a bounded pre-receipt failure")
    _add_route_args(eval_failure, include_submission=True)
    eval_failure.add_argument("--run-id", required=True)
    eval_failure.add_argument(
        "--reason-code",
        required=True,
        choices=(
            "eval_deploy_failed",
            "eval_tunnel_failed",
            "eval_key_release_unavailable",
            "eval_no_result",
        ),
    )
    eval_tear = eval_sub.add_parser("teardown", help="delete the Eval CVM")
    eval_tear.add_argument("--cvm-id", required=True)

    return parser


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _cmd_prepare(args: argparse.Namespace) -> int:
    kwargs: dict[str, Any] = {"image": args.image, "key_release_url": args.key_release_url}
    if args.name:
        kwargs["app_name"] = args.name
    try:
        prepared = prepare_deployment(**kwargs)
    except PrepareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    path = write_prepared(prepared, args.out)
    _print(
        {
            "image": prepared.image,
            "key_release_url": prepared.key_release_url,
            "compose_hash": prepared.compose_hash,
            "compose_path": str(path),
        }
    )
    return 0


def _cmd_measurements(args: argparse.Namespace) -> int:
    compose_text = Path(args.compose).read_text(encoding="utf-8")
    record = measure.reproduce_measurement(
        metadata_path=args.metadata,
        cpu=args.cpu,
        memory=args.memory,
        compose=compose_text,
        dstack_mr_bin=args.dstack_mr,
    )
    print(record.to_json())
    return 0


def _load_measurement_arg(args: argparse.Namespace) -> dict[str, Any]:
    if args.measurement:
        source = args.measurement.strip()
        if source.startswith("{"):
            return json.loads(source)
        return json.loads(Path(source).read_text(encoding="utf-8"))
    if args.from_result:
        stdout = Path(args.from_result).read_text(encoding="utf-8")
        surfaced = result_mod.surface_result(stdout)
        attestation = surfaced.attestation
        if attestation is None:
            raise measure.MeasurementError("captured result carries no attested measurement")
        return dict(attestation.get("measurement", {}))
    raise measure.MeasurementError("provide --measurement or --from-result")


def _cmd_verdict(args: argparse.Namespace) -> int:
    try:
        measurement = _load_measurement_arg(args)
        verdict = measure.domain_allowlist_verdict(
            domain=args.domain,
            measurement=measurement,
            review_allowlist=args.allowlist if args.domain == "review" else None,
            eval_allowlist=args.allowlist if args.domain == "eval" else None,
        )
    except (measure.MeasurementError, result_mod.ResultSurfaceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print(verdict.as_dict())
    return 0 if verdict.in_allowlist else 1


def _cmd_deploy(args: argparse.Namespace, *, deployer: Deployer) -> int:
    plan_kwargs: dict[str, Any] = {
        "image": args.image,
        "key_release_url": args.key_release_url,
        "instance_type": args.instance_type,
        "os_image": args.os_image,
        "money_cap_usd": args.money_cap_usd,
        "max_runtime_hours": args.max_runtime_hours,
    }
    if args.region:
        plan_kwargs["region"] = args.region
    if args.name:
        plan_kwargs["name"] = args.name
    try:
        plan = build_deploy_plan(**plan_kwargs)
    except (ShapeError, PrepareError) as exc:
        print(f"error: refusing to deploy: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        _print(render_plan(plan))
        return 0

    try:
        check_phala_credentials()
    except CredentialError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    outcome = deployer(plan, args.out)
    _print({"deployed": True, "instance_type": plan.instance_type, "result": outcome})
    return 0


def _cmd_run(args: argparse.Namespace, *, backend_main: run_mod.BackendMain | None) -> int:
    try:
        outcome = run_mod.run_eval(
            job_dir=args.job_dir,
            task_ids=args.task_ids,
            key_release_url=args.key_release_url,
            backend_main=backend_main,
        )
    except run_mod.RunError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if outcome.succeeded and outcome.surfaced is not None:
        _print(outcome.surfaced.summary())
        return 0

    # Fail closed: surface a clear error and NO attested result.
    print(f"error: {outcome.clear_error}", file=sys.stderr)
    if outcome.surfaced is not None:
        _print(
            {
                "attested": False,
                "status": outcome.surfaced.status,
                "reason_code": outcome.surfaced.reason_code,
            }
        )
    return outcome.exit_code or 1


def _cmd_result(args: argparse.Namespace) -> int:
    if args.from_path:
        stdout = Path(args.from_path).read_text(encoding="utf-8")
    else:
        stdout = sys.stdin.read()
    try:
        surfaced = result_mod.surface_result(stdout)
    except result_mod.ResultSurfaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    allowlist_verdict = None
    measurement_allowlisted: bool | None = None
    if args.allowlist and surfaced.attestation is not None:
        allowlist_verdict = measure.allowlist_verdict(
            surfaced.attestation.get("measurement", {}), args.allowlist
        )
        measurement_allowlisted = allowlist_verdict.in_allowlist

    quote_verified = None if args.quote_verified is None else args.quote_verified == "true"
    key_grant_value = getattr(args, "key_grant", None)
    key_grant_ok = None if key_grant_value is None else key_grant_value == "true"
    # When the miner explicitly supplies the complete positive-signal set expected
    # by the acceptance conjunction, require a known key-grant outcome. Without an
    # explicit CLI flag, treat allowlist+quote+nonce arguments as implying a
    # positive grant so the historical accepted-run control still works; callers
    # that only pass partial signals remain pending rather than false-accept.
    if (
        key_grant_ok is None
        and quote_verified is True
        and measurement_allowlisted is True
        and args.nonce_state == "ok"
    ):
        key_grant_ok = True
    verdict = result_mod.evaluate_acceptance(
        surfaced,
        quote_verified=quote_verified,
        measurement_allowlisted=measurement_allowlisted,
        nonce_state=args.nonce_state,
        key_grant_ok=key_grant_ok,
    )

    if verdict.accepted is False:
        # A rejected/unaccepted result: surface a clear, non-sensitive verdict
        # with NO fabricated score and no golden/key/quote material (VAL-DEPLOY-026).
        _print({"accepted": False, "reason": verdict.reason, "attested": surfaced.attested})
        return 1

    summary = surfaced.summary()
    if allowlist_verdict is not None:
        summary["allowlist_verdict"] = allowlist_verdict.as_dict()
    summary["acceptance"] = verdict.as_dict()
    _print(summary)
    return 0


def _cmd_teardown(args: argparse.Namespace, *, teardowner: Teardowner) -> int:
    outcome = teardowner(args.cvm_id)
    payload, code = _teardown_payload(args.cvm_id, outcome)
    _print(payload)
    return code


def main(
    argv: Sequence[str] | None = None,
    *,
    deployer: Deployer | None = None,
    teardowner: Teardowner | None = None,
    backend_main: run_mod.BackendMain | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        return _cmd_prepare(args)
    if args.command == "review":
        return _ordered_review_command(args)
    if args.command == "eval":
        return _ordered_eval_command(args)
    if args.command == "measurements":
        return _cmd_measurements(args)
    if args.command == "verdict":
        return _cmd_verdict(args)
    if args.command == "deploy":
        return _cmd_deploy(args, deployer=deployer or default_phala_deployer)
    if args.command == "run":
        return _cmd_run(args, backend_main=backend_main)
    if args.command == "result":
        return _cmd_result(args)
    if args.command == "teardown":
        return _cmd_teardown(args, teardowner=teardowner or default_phala_teardown)
    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2  # pragma: no cover


__all__ = [
    "PROG",
    "SPEND_CAPABLE_SUBCOMMANDS",
    "SUBCOMMANDS",
    "Deployer",
    "Teardowner",
    "build_parser",
    "default_phala_deployer",
    "default_phala_teardown",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(main())
