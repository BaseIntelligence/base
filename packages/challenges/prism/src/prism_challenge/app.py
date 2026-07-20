from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable, Coroutine, Mapping
from typing import Annotated, Any

from base.challenge_sdk.app_factory import create_challenge_app
from base.challenge_sdk.roles import Capability, Role, role_contract
from base.challenge_sdk.schemas import ExternalResultEnvelope
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from .admission import enforce_admission
from .audit import audit_sampler_from_config, resolve_audit_unit
from .auth import authenticate_internal, authenticate_validator
from .config import PrismSettings, configure_logging, get_settings
from .coordination import (
    audit_work_unit_to_payload,
    list_pending_prism_work_units,
    work_unit_to_payload,
)
from .db import Database
from .evaluator.checkpoint_intake import CheckpointIntakeError, CheckpointIntakeService
from .evaluator.checkpoint_publisher import (
    CheckpointPublisher,
    HuggingFaceCheckpointPublisher,
)
from .evaluator.interface import PrismContext
from .ingestion import ResultIngestionError, ingest_work_unit_result
from .models import (
    SubmissionCreate,
    SubmissionResponse,
)
from .plausibility import PlausibilityError
from .queue import PrismWorker
from .repository import PrismRepository
from .routes import router
from .weights import get_weights


def create_app(
    app_settings: PrismSettings | None = None,
    *,
    checkpoint_publisher: CheckpointPublisher | None = None,
) -> FastAPI:
    if app_settings is None:
        app_settings = get_settings()
    # Deploy entrypoint (uvicorn ``prism_challenge.app:app``, incl. combined mode which drains the
    # eval queue in-process) runs with no root logging config, so configure it here at import time
    # to surface application + worker-loop INFO under uvicorn (idempotent; see configure_logging).
    configure_logging(app_settings)
    database = Database(app_settings.resolved_database_path)
    repository = PrismRepository(
        database,
        app_settings.epoch_seconds,
        worker_claim_timeout_seconds=app_settings.worker_claim_timeout_seconds,
    )
    if app_settings.worker_plane.cpu_reexec_test_mode:
        # Explicit CPU re-exec test mode: install the repo's own CPU seam (no GPU/Docker/broker)
        # and re-execute with the tiny deterministic context (architecture.md 3.4; VAL-PRISM-013).
        from .evaluator.cpu_test_mode import configure_cpu_reexec_test_mode, cpu_test_context

        configure_cpu_reexec_test_mode(app_settings)
        ctx = cpu_test_context(app_settings.worker_plane)
    else:
        # Seq/token_budget pass through settings → PrismContext (VAL-SCALE-006).
        # Defaults remain short-ctx; operators raise sequence_length / token_budget for scale cups.
        ctx = PrismContext(**app_settings.prism_context_kwargs())
    worker = PrismWorker(
        repository,
        ctx,
        execution_backend=app_settings.execution_backend,
        settings=app_settings,
    )
    # Tests inject a MockCheckpointPublisher (no network); deploy uses the real lazy HF client.
    # Constructing the real client never imports huggingface_hub, so this stays offline-safe.
    publisher = checkpoint_publisher or HuggingFaceCheckpointPublisher(
        repo_id=app_settings.checkpoint_repo_id, token=app_settings.hf_token_value()
    )
    checkpoint_intake = CheckpointIntakeService(publisher=publisher, repository=repository)

    async def get_weights_fn() -> dict[str, float]:
        runtime_config = await repository.runtime_config(app_settings, official=True)
        return await get_weights(
            repository,
            app_settings.epoch_seconds,
            architecture_weight=runtime_config.reward_pools.architecture,
            training_weight=runtime_config.reward_pools.training,
        )

    # Combined mode (single-service deploy): the API process also drains the eval queue. The
    # worker loop is launched by the app-factory lifespan AFTER database.init() and cancelled +
    # awaited before database.close(), reusing this SAME PrismWorker via app.state.worker (no
    # second app/DB).
    background_tasks: list[Callable[[FastAPI], Coroutine[Any, Any, None]]] = []
    if app_settings.combined_mode:

        async def _run_combined_worker(app: FastAPI) -> None:
            from .worker import run_worker_loop

            await run_worker_loop(
                app.state.worker,
                interval_seconds=app_settings.combined_worker_interval_seconds,
                resilient=True,
            )

        background_tasks.append(_run_combined_worker)

    # Wire authenticated raw-weight push when master_base_url + token enable it.
    from .raw_weight_push import (
        maybe_build_push_client_from_settings,
        run_raw_weight_push_loop,
    )

    push_client = maybe_build_push_client_from_settings(
        settings=app_settings,
        database=database,
        repository=repository,
    )
    if push_client is not None:
        interval = float(getattr(push_client, "push_interval_seconds", 30.0))

        async def _run_raw_weight_push(app: FastAPI) -> None:
            client = getattr(app.state, "raw_weight_push_client", push_client)
            await run_raw_weight_push_loop(
                client,
                interval_seconds=interval,
                resilient=True,
            )

        background_tasks.append(_run_raw_weight_push)

    app = create_challenge_app(
        settings=app_settings,
        database=database,
        public_router=router,
        get_weights_fn=get_weights_fn,
        background_tasks=tuple(background_tasks),
    )
    if push_client is not None:
        app.state.raw_weight_push_client = push_client
    app.state.settings = app_settings
    app.state.database = database
    app.state.repository = repository
    app.state.worker = worker
    app.state.checkpoint_publisher = publisher
    app.state.checkpoint_intake = checkpoint_intake

    @app.post("/internal/v1/worker/process-next", dependencies=[Depends(authenticate_internal)])
    async def process_next() -> dict[str, str | None]:
        return {"submission_id": await worker.process_next()}

    @app.post("/internal/v1/checkpoints")
    async def publish_checkpoint(
        request: Request,
        validator_hotkey: Annotated[str, Depends(authenticate_validator)],
    ) -> dict[str, object]:
        """Receive a validator's pushed checkpoint and publish it to HuggingFace (mocked in tests).

        Hotkey-signed + validator-permit gated via ``authenticate_validator`` (a rejected caller
        never reaches this body, so no ``checkpoint_ref`` is recorded on rejection). On success the
        checkpoint is published through the publisher interface and its public ``checkpoint_ref`` is
        recorded on the submission's assignment for resume-on-reassignment (VAL-PRISM-022/038).
        """
        intake: CheckpointIntakeService = request.app.state.checkpoint_intake
        body = await request.body()
        submission_id, attempt, files, revision = _parse_checkpoint_upload(body)
        try:
            published = await intake.publish(
                submission_id=submission_id,
                attempt=attempt,
                validator_hotkey=validator_hotkey,
                files=files,
                revision=revision,
            )
        except CheckpointIntakeError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        return {
            "checkpoint_ref": published.checkpoint_ref,
            "repo_id": published.repo_id,
            "revision": published.revision,
            "files": list(published.files),
        }

    @app.get("/internal/v1/work_units", dependencies=[Depends(authenticate_internal)])
    async def work_units() -> dict[str, object]:
        """Expose pending prism work units (one gpu unit per submission) to the master plane.

        The master coordination plane reads this to create exactly one assignable work unit per
        submission and assign it - with concurrency 1 - to a single online gpu validator. This
        endpoint is execution-free: enumerating work units never invokes the broker/executor.

        With the worker plane ON it ALSO exposes pending validator AUDIT units (distinct ids,
        validator executor kind) sampled from finalized worker results so they run on the existing
        validator_dispatch path (VAL-PRISM-012); resolved/exhausted audits are not listed
        (pending-only listing semantics).
        """
        units = await list_pending_prism_work_units(repository)
        work_unit_payloads: list[Mapping[str, object]] = [
            work_unit_to_payload(unit) for unit in units
        ]
        if app_settings.worker_plane.enabled:
            for row in await repository.list_pending_audit_units():
                work_unit_payloads.append(audit_work_unit_to_payload(row))
        return {
            "challenge_slug": app_settings.slug,
            "work_units": work_unit_payloads,
        }

    @app.post(
        "/internal/v1/audit_units/{audit_unit_id}/result",
        dependencies=[Depends(authenticate_internal)],
    )
    async def audit_unit_result(audit_unit_id: str, request: Request) -> dict[str, object]:
        """Resolve a validator audit replay for a sampled result (architecture.md 3.5).

        Body: ``{manifest_sha256?: str, success?: bool, error?: str}``. The validator replay is
        authoritative: a hash EQUAL to the audited worker hash passes (score untouched); a DIFFERENT
        hash invalidates the audited submission's score and propagates to crown/weights
        (VAL-PRISM-013/023); a replay failure/timeout (``success: false`` or no ``manifest_sha256``)
        NEVER confirms the result -- it is re-audited within bounds, then reaches a terminal
        ``failed`` audit state with the submission left unresolved (VAL-PRISM-024). Disabled with
        the worker plane off (404).
        """
        if not app_settings.worker_plane.enabled:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "worker plane disabled")
        try:
            payload = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid JSON result body") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "result body must be an object")
        replay_hash = payload.get("manifest_sha256")
        if replay_hash is not None and not isinstance(replay_hash, str):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "manifest_sha256 must be a string")
        failed = payload.get("success") is False or bool(payload.get("failed"))
        error = payload.get("error") if isinstance(payload.get("error"), str) else None
        try:
            resolution = await resolve_audit_unit(
                repository,
                audit_unit_id=audit_unit_id,
                replay_manifest_sha256=replay_hash,
                failed=failed,
                error=error,
            )
        except KeyError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "audit unit not found") from exc
        return resolution.to_response()

    @app.post("/internal/v1/work_units/result", dependencies=[Depends(authenticate_internal)])
    @role_contract(role=Role.CHALLENGE, capability=Capability.CHALLENGE_ORDINARY_PROOF)
    async def work_unit_result(request: Request) -> dict[str, object]:
        """Accept a base-reconciled worker result as ExternalResultEnvelope only.

        The base master forwards exactly one accepted (R=2-reconciled) result here after reconciling
        the replicas' manifest hashes (architecture.md 3.3). The body must be the canonical
        :class:`ExternalResultEnvelope` (api_version, assignment/challenge bindings, and proof)
        -- dual/legacy reduced bodies without those fields are rejected 422 before scoring or
        persistence. The ExecutionProof is verified BEFORE anything is scored: a missing/malformed
        proof (VAL-PRISM-018) or a tampered manifest / a forged signature (VAL-PRISM-007) is
        rejected 422 with a distinguishable reason and never finalized (the unit stays eligible for
        retry). A verified result is then run through the plausibility gate (architecture.md 3.5;
        VAL-PRISM-009): an implausible manifest is rejected 422 with a distinct ``plausibility_*``
        reason and never scored, while a plausible manifest passes through UNCHANGED.
        A verified, plausible result is finalized idempotently: a duplicate delivery is
        a no-op and a conflicting delivery for an already-accepted unit is refused 409 so
        the stored score/leaderboard is never mutated (VAL-PRISM-017). The claimed tier is
        downgraded to its verified effective tier for audit sampling (VAL-PRISM-019).
        Disabled with the worker plane off (404) so it is inert in legacy deployments.
        """
        if not app_settings.worker_plane.enabled:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "worker plane disabled")
        try:
            payload = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid JSON result body") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "result body must be an object")
        try:
            envelope = ExternalResultEnvelope.model_validate(payload)
        except Exception as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                {
                    "code": "result_envelope_invalid",
                    "detail": (
                        "external result envelope does not match the canonical SDK contract"
                    ),
                },
            ) from exc
        if envelope.challenge_slug != app_settings.slug:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                {
                    "code": "result_challenge_mismatch",
                    "detail": "challenge binding is invalid",
                },
            )
        if envelope.assignment_id != envelope.work_unit_id:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                {
                    "code": "result_assignment_mismatch",
                    "detail": "assignment binding is invalid",
                },
            )
        work_unit_id = envelope.work_unit_id
        submission_ref = envelope.submission_ref
        result_payload = {
            **envelope.result,
            "execution_proof": envelope.proof.model_dump(mode="json"),
        }
        sampler = audit_sampler_from_config(app_settings.worker_plane)
        try:
            outcome = await ingest_work_unit_result(
                worker=worker,
                work_unit_id=work_unit_id,
                submission_ref=submission_ref,
                result=result_payload,
                pinned_image_digest=app_settings.worker_plane.pinned_image_digest,
                audit_sampler=sampler,
            )
        except ResultIngestionError as exc:
            # A transient finalization failure is retryable -> 503 so the forwarder retries; the
            # permanent rejections (bad proof / tampered / implausible manifest) stay 422.
            code = (
                status.HTTP_503_SERVICE_UNAVAILABLE
                if exc.reason == "finalization_failed"
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            )
            raise HTTPException(
                code,
                {"code": exc.reason, "detail": str(exc)},
            ) from exc
        except PlausibilityError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                {"code": exc.reason, "detail": str(exc)},
            ) from exc
        if outcome.status == "conflict":
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                {"code": outcome.reason, "detail": "conflicting result for finalized unit"},
            )
        return outcome.to_response()

    @app.post(
        "/internal/v1/bridge/submissions",
        response_model=SubmissionResponse,
        dependencies=[Depends(authenticate_internal)],
    )
    async def bridge_submission(
        request: Request,
        x_base_verified_hotkey: Annotated[str, Header(min_length=1, max_length=128)],
        x_submission_filename: Annotated[str | None, Header()] = None,
    ) -> SubmissionResponse:
        body = await request.body()
        submission = _bridge_submission_create(
            body=body,
            content_type=request.headers.get("content-type", ""),
            filename=x_submission_filename,
        )
        if len(submission.code.encode()) > app_settings.max_code_bytes:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "submission too large")
        await enforce_admission(app_settings, x_base_verified_hotkey)
        return await repository.create_submission(x_base_verified_hotkey, submission)

    return app


def _parse_checkpoint_upload(body: bytes) -> tuple[str, int, dict[str, bytes], str | None]:
    """Parse + validate a validator checkpoint-upload payload into (submission_id, attempt, files).

    The payload is JSON ``{submission_id, attempt, files:{name: base64}, revision?}``. A malformed
    body / non-integer attempt / non-base64 file bytes is a 400 (never a publish).
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid JSON checkpoint upload") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "checkpoint upload must be an object")
    submission_id = payload.get("submission_id")
    if not isinstance(submission_id, str) or not submission_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "submission_id is required")
    attempt_raw = payload.get("attempt", 1)
    try:
        attempt = int(attempt_raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "attempt must be an integer") from exc
    if attempt < 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "attempt must be >= 1")
    raw_files = payload.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "files must be a non-empty object")
    files: dict[str, bytes] = {}
    for name, encoded in raw_files.items():
        if not isinstance(name, str) or not isinstance(encoded, str):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "files entries must be base64 strings")
        try:
            files[name] = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"file {name} is not valid base64"
            ) from exc
    revision = payload.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "revision must be a string")
    return submission_id, attempt, files, revision


def _bridge_submission_create(
    *, body: bytes, content_type: str, filename: str | None
) -> SubmissionCreate:
    if "application/json" in content_type.lower():
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid JSON submission") from exc
        return SubmissionCreate.model_validate(payload)
    safe_filename = filename or "submission.zip"
    if not safe_filename.endswith((".py", ".zip")):
        safe_filename = "submission.zip"
    return SubmissionCreate(
        code=base64.b64encode(body).decode("ascii"),
        filename=safe_filename,
        metadata={"content_type": content_type or "application/octet-stream", "bridge": True},
    )


_app: FastAPI | None = None


def get_app() -> FastAPI:
    """Return the process app, creating production settings only on first use."""
    global _app
    if _app is None:
        _app = create_app()
    return _app


def __getattr__(name: str) -> Any:
    # Support ``uvicorn prism_challenge.app:app`` without instantiating production
    # settings (broker token file path / shared token) when the package is merely
    # imported for pytest collection or tooling.
    if name == "app":
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
