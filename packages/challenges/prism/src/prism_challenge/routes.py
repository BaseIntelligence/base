from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Any, SupportsFloat, SupportsInt, cast

from base.challenge_sdk.roles import public_route
from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    status,
)
from pydantic import ValidationError

from .admission import enforce_admission
from .attestation_routes import build_attestation_public_router
from .auth import (
    authenticate_internal,
    authenticate_miner,
    canonical_submission_message,
    verify_dev_signature,
    verify_hotkey_signature,
)
from .evaluator.train_series import downsample_train_series_for_api
from .models import (
    ArchitectureDetailResponse,
    ArchitectureListResponse,
    ArchitectureSummary,
    ArchitectureVariantsResponse,
    CurveBpb,
    CurveCompute,
    EpochResponse,
    EvalJobHealthEntry,
    GpuStatusSummary,
    LeaderboardEntry,
    LeaderboardResponse,
    LossCurveSeries,
    SubmissionCurveResponse,
    SubmissionHistoryBucket,
    SubmissionResponse,
    SubmissionStatusResponse,
    TrainingVariantEntry,
    TrainSeriesV1Response,
)
from .repository import PrismRepository, epoch_id_for

logger = logging.getLogger(__name__)

CURVE_MAX_POINTS = 500

# First-class score fields must never be accepted as event attributes (trust boundary).
_SCORE_FIELD_NAMES = frozenset(
    {
        "score",
        "final_score",
        "q_arch",
        "q_recipe",
        "anti_cheat_multiplier",
        "diversity_bonus",
        "penalty",
        "effective_tier",
    }
)

router = APIRouter(prefix="/v1")

# Public attestation challenge/answer (published via BASE proxy as
# /challenges/prism/v1/attestation/*). Lives on the challenge app, not master.
router.include_router(build_attestation_public_router())


def _optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(cast(SupportsFloat, value))
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN → None


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def repo_from_request(request: Request) -> PrismRepository:
    return request.app.state.repository


@public_route(tags=["submissions"], auth_required=True)
@router.post("/submissions", response_model=SubmissionResponse)
async def submit_model(
    request: Request,
    hotkey: str = Depends(authenticate_miner),
    repository: PrismRepository = Depends(repo_from_request),
) -> SubmissionResponse:
    from .app import _bridge_submission_create

    body = await request.body()
    try:
        request_body = _bridge_submission_create(
            body=body,
            content_type=request.headers.get("content-type", ""),
            filename=request.headers.get("x-submission-filename"),
        )
    except ValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, exc.errors()) from exc
    if len(request_body.code.encode()) > request.app.state.settings.max_code_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "submission too large")
    await enforce_admission(request.app.state.settings, hotkey)
    return await repository.create_submission(hotkey, request_body)


@public_route(tags=["submissions"])
@router.get("/submissions/history", response_model=list[SubmissionHistoryBucket])
async def submission_history(
    days: int = Query(default=90, ge=1, le=366),
    repository: PrismRepository = Depends(repo_from_request),
) -> list[SubmissionHistoryBucket]:
    return [
        SubmissionHistoryBucket(
            date=str(row["day"]),
            count=int(cast(SupportsInt, row["count"])),
        )
        for row in await repository.submission_history(days=days)
    ]


@public_route(tags=["submissions"])
@router.get("/submissions/{submission_id}", response_model=SubmissionStatusResponse)
async def submission_status(
    submission_id: str, repository: PrismRepository = Depends(repo_from_request)
) -> SubmissionStatusResponse:
    submission = await repository.get_submission(submission_id)
    if submission is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "submission not found")
    return submission


@public_route(tags=["leaderboard"])
@router.get("/leaderboard", response_model=LeaderboardResponse)
async def leaderboard(
    request: Request,
    epoch_id: int | None = Query(default=None, ge=0),
    repository: PrismRepository = Depends(repo_from_request),
) -> LeaderboardResponse:
    resolved_epoch_id = (
        epoch_id
        if epoch_id is not None
        else epoch_id_for(datetime.now(UTC), request.app.state.settings.epoch_seconds)
    )
    rows = await repository.leaderboard(resolved_epoch_id)
    entries = [
        LeaderboardEntry(
            rank=index + 1,
            hotkey=str(row["hotkey"]),
            score=float(cast(SupportsFloat, row["final_score"])),
            submission_id=str(row["id"]),
        )
        for index, row in enumerate(rows)
    ]
    return LeaderboardResponse(epoch_id=resolved_epoch_id, entries=entries)


@public_route(tags=["epochs"])
@router.get("/epochs/current")
async def current_epoch(request: Request) -> dict[str, int]:
    epoch_id = epoch_id_for(datetime.now(UTC), request.app.state.settings.epoch_seconds)
    return {"epoch_id": epoch_id, "epoch_seconds": request.app.state.settings.epoch_seconds}


@public_route(tags=["epochs"])
@router.get("/epochs", response_model=list[EpochResponse])
async def list_epochs(
    limit: int = Query(default=50, ge=1, le=200),
    repository: PrismRepository = Depends(repo_from_request),
) -> list[EpochResponse]:
    return [
        EpochResponse(
            id=int(cast(SupportsInt, row["id"])),
            starts_at=datetime.fromisoformat(str(row["starts_at"])),
            ends_at=datetime.fromisoformat(str(row["ends_at"])),
            status=str(row["status"]),
        )
        for row in await repository.list_epochs(limit=limit)
    ]


@public_route(tags=["health"])
@router.get("/health/eval-jobs", response_model=list[EvalJobHealthEntry])
async def eval_job_health(
    limit: int = Query(default=50, ge=1, le=200),
    repository: PrismRepository = Depends(repo_from_request),
) -> list[EvalJobHealthEntry]:
    return [
        EvalJobHealthEntry(
            id=str(row["id"]),
            submission_id=str(row["submission_id"]),
            level=str(row["level"]),
            status=str(row["status"]),
            attempts=int(cast(SupportsInt, row["attempts"])),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )
        for row in await repository.list_eval_job_health(limit=limit)
    ]


@public_route(tags=["gpu"])
@router.get("/gpu/status", response_model=GpuStatusSummary)
async def gpu_status(
    repository: PrismRepository = Depends(repo_from_request),
) -> GpuStatusSummary:
    status_rows, tier_rows = await repository.gpu_status_summary()
    by_status: dict[str, int] = {}
    total_gpus = 0
    for row in status_rows:
        status_value = str(row["status"])
        by_status[status_value] = int(cast(SupportsInt, row["lease_count"]))
        if status_value == "active":
            total_gpus = int(cast(SupportsInt, row["gpu_total"]))
    by_tier = {str(row["tier"]): int(cast(SupportsInt, row["lease_count"])) for row in tier_rows}
    return GpuStatusSummary(
        total_gpus=total_gpus,
        active_leases=by_status.get("active", 0),
        by_status=by_status,
        by_tier=by_tier,
    )


@public_route(tags=["architectures"])
@router.get("/architectures", response_model=ArchitectureListResponse)
async def list_architectures(
    epoch_id: int | None = Query(default=None, ge=0),
    repository: PrismRepository = Depends(repo_from_request),
) -> ArchitectureListResponse:
    resolved_epoch_id, rows = await repository.list_architectures(epoch_id)
    architectures = [
        ArchitectureSummary(
            rank=index + 1,
            architecture_id=str(row["architecture_id"]),
            arch_hash=str(row["arch_hash"]),
            name=str(row["name"]) if row["name"] is not None else None,
            owner_hotkey=str(row["owner_hotkey"]),
            best_final_score=float(cast(SupportsFloat, row["best_final_score"])),
            best_submission_id=str(row["best_submission_id"]),
            inventory_best_score=_optional_float(row.get("inventory_best_score")),
            inventory_best_submission_id=_optional_str(row.get("inventory_best_submission_id")),
            variant_count=int(cast(SupportsInt, row["variant_count"])),
            submission_count=int(cast(SupportsInt, row["submission_count"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )
        for index, row in enumerate(rows)
    ]
    return ArchitectureListResponse(epoch_id=resolved_epoch_id, architectures=architectures)


@public_route(tags=["architectures"])
@router.get("/architectures/{architecture_id}", response_model=ArchitectureDetailResponse)
async def get_architecture(
    architecture_id: str, repository: PrismRepository = Depends(repo_from_request)
) -> ArchitectureDetailResponse:
    row = await repository.get_architecture(architecture_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "architecture not found")
    return ArchitectureDetailResponse(
        architecture_id=str(row["architecture_id"]),
        arch_hash=str(row["arch_hash"]),
        name=str(row["name"]) if row["name"] is not None else None,
        owner_hotkey=str(row["owner_hotkey"]),
        best_final_score=float(cast(SupportsFloat, row["best_final_score"])),
        best_submission_id=str(row["best_submission_id"]),
        inventory_best_score=_optional_float(row.get("inventory_best_score")),
        inventory_best_submission_id=_optional_str(row.get("inventory_best_submission_id")),
        variant_count=int(cast(SupportsInt, row["variant_count"])),
        submission_count=int(cast(SupportsInt, row["submission_count"])),
        first_seen_at=datetime.fromisoformat(str(row["first_seen_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
    )


@public_route(tags=["architectures"])
@router.get(
    "/architectures/{architecture_id}/variants", response_model=ArchitectureVariantsResponse
)
async def list_architecture_variants(
    architecture_id: str, repository: PrismRepository = Depends(repo_from_request)
) -> ArchitectureVariantsResponse:
    if await repository.get_architecture(architecture_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "architecture not found")
    variants = [
        TrainingVariantEntry(
            variant_id=str(row["variant_id"]),
            training_hash=str(row["training_hash"]),
            owner_hotkey=str(row["owner_hotkey"]),
            submission_id=str(row["submission_id"]),
            final_score=float(cast(SupportsFloat, row["final_score"])),
            inventory_final_score=_optional_float(row.get("inventory_final_score")),
            metric_mean=float(cast(SupportsFloat, row["metric_mean"])),
            metric_std=float(cast(SupportsFloat, row["metric_std"])),
            is_current_best=bool(row["is_current_best"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )
        for row in await repository.list_training_variants(architecture_id)
    ]
    return ArchitectureVariantsResponse(architecture_id=architecture_id, variants=variants)


@public_route(tags=["submissions"])
@router.get("/submissions/{submission_id}/curve", response_model=SubmissionCurveResponse)
async def submission_curve(
    submission_id: str, repository: PrismRepository = Depends(repo_from_request)
) -> SubmissionCurveResponse:
    """Loss curve + optional challenge-owned ``prism_train_series.v1`` time-flow.

    Auth: same public internal auth path as other challenge routes (miner headers / Base
    proxy). Series is challenge-owned only; miner-planted payloads never appear here.
    Response never includes secret tokens, wallets, or proof private material.
    """
    curve = await repository.get_submission_curve(submission_id)
    if curve is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "submission curve not found")
    online_loss = [_opt_float(value) or 0.0 for value in curve["online_loss"]]
    covered_bytes = [_opt_float(value) or 0.0 for value in curve["covered_bytes_cumulative"]]
    length = min(len(online_loss), len(covered_bytes))
    indices = _downsample_indices(length, CURVE_MAX_POINTS)
    sampled_loss = [online_loss[i] for i in indices]
    sampled_bytes = [covered_bytes[i] for i in indices]
    compute = curve["compute"] if isinstance(curve["compute"], dict) else {}
    model_params = _opt_int(compute.get("model_params"))
    tokens_consumed = _opt_int(curve.get("tokens_consumed"))
    gpu_count = _opt_int(compute.get("gpu_count"))
    wall_clock = _opt_float(compute.get("wall_clock_seconds"))
    estimated_flops = _opt_float(compute.get("estimated_flops"))
    if estimated_flops is None and model_params is not None and tokens_consumed is not None:
        estimated_flops = 6.0 * float(model_params) * float(tokens_consumed)
    gpu_hours = _opt_float(compute.get("gpu_hours"))
    if gpu_hours is None and gpu_count is not None and wall_clock is not None:
        gpu_hours = float(gpu_count) * wall_clock / 3600.0
    train_series_payload = downsample_train_series_for_api(
        curve.get("train_series") if isinstance(curve.get("train_series"), dict) else None,
        max_points=CURVE_MAX_POINTS,
    )
    train_series: TrainSeriesV1Response | None = None
    if train_series_payload is not None:
        train_series = TrainSeriesV1Response.model_validate(train_series_payload)
    return SubmissionCurveResponse(
        submission_id=submission_id,
        loss_curve=LossCurveSeries(
            online_loss=sampled_loss,
            covered_bytes_cumulative=sampled_bytes,
            step0_loss=_opt_float(curve.get("step0_loss")),
            baseline_nats=_opt_float(curve.get("baseline_nats")),
            points=len(indices),
            downsampled=length > CURVE_MAX_POINTS,
        ),
        bpb=CurveBpb(
            prequential_bpb=_opt_float(curve.get("prequential_bpb")),
            bits_per_byte=_opt_float(curve.get("bits_per_byte")),
        ),
        compute=CurveCompute(
            gpu_count=gpu_count,
            device=str(compute["device"]) if isinstance(compute.get("device"), str) else None,
            gpu_tier=str(compute["gpu_tier"]) if isinstance(compute.get("gpu_tier"), str) else None,
            model_params=model_params,
            tokens_consumed=tokens_consumed,
            estimated_flops=estimated_flops,
            wall_clock_seconds=wall_clock,
            gpu_hours=gpu_hours,
            peak_vram_bytes=_opt_int(compute.get("peak_vram_bytes")),
            peak_rss_bytes=_opt_int(compute.get("peak_rss_bytes")),
        ),
        train_series=train_series,
    )


def _downsample_indices(n: int, cap: int) -> list[int]:
    """Even-stride indices that keep the first and last sample; identity when ``n <= cap``."""
    if n <= cap:
        return list(range(n))
    return [round(i * (n - 1) / (cap - 1)) for i in range(cap)]


def _opt_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _opt_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _verify_telemetry_hotkey_signature(
    request: Request,
    *,
    hotkey: str,
    nonce: str,
    timestamp: str,
    signature: str,
    body: bytes,
) -> None:
    app_settings = request.app.state.settings
    try:
        ts = int(timestamp)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid timestamp") from exc
    if abs(int(datetime.now(UTC).timestamp()) - ts) > app_settings.signature_ttl_seconds:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "stale signature")
    message = canonical_submission_message(
        hotkey=hotkey, nonce=nonce, timestamp=timestamp, body=body
    )
    valid = verify_hotkey_signature(hotkey, message, signature)
    if not valid and app_settings.allow_insecure_signatures:
        valid = verify_dev_signature(app_settings.internal_token(), message, signature)
    if not valid:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid signature")


@public_route(tags=["execution"])
@router.post("/execution/telemetry-session")
async def open_telemetry_session(
    request: Request,
    _: None = Depends(authenticate_internal),
    repository: PrismRepository = Depends(repo_from_request),
    x_hotkey: Annotated[str | None, Header()] = None,
    x_signature: Annotated[str | None, Header()] = None,
    x_nonce: Annotated[str | None, Header()] = None,
    x_timestamp: Annotated[str | None, Header()] = None,
) -> dict[str, str]:
    """Open a hotkey-signed telemetry session. Mnemonic is never accepted or returned."""

    body = await request.body()
    try:
        import json

        payload = json.loads(body.decode("utf-8") if body else "{}")
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid json") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "body must be object")

    # Reject secret material if present as first-class fields (optional hard reject).
    for banned in ("mnemonic", "wallet_seed", "private_key", "seed"):
        if banned in payload:
            # Strip path: ignore banned keys rather than echo; still open session.
            payload = {
                k: v
                for k, v in payload.items()
                if k not in {"mnemonic", "wallet_seed", "private_key", "seed"}
            }
            break

    hotkey = str(payload.get("hotkey_ss58") or x_hotkey or "").strip()
    nonce = str(payload.get("nonce") or x_nonce or "").strip()
    timestamp = str(x_timestamp or payload.get("timestamp") or "").strip()
    signature = str(x_signature or "").strip()
    if not hotkey or not nonce or not timestamp or not signature:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "hotkey signature required")
    if x_hotkey is not None and x_hotkey != hotkey:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "hotkey mismatch")
    _verify_telemetry_hotkey_signature(
        request,
        hotkey=hotkey,
        nonce=nonce,
        timestamp=timestamp,
        signature=signature,
        body=body,
    )

    eval_job_id = payload.get("eval_job_id")
    work_unit_id = payload.get("work_unit_id")
    if eval_job_id is not None:
        eval_job_id = str(eval_job_id)
    if work_unit_id is not None:
        work_unit_id = str(work_unit_id)
    if not eval_job_id and not work_unit_id:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "eval_job_id or work_unit_id required",
        )
    instance_id = str(payload.get("instance_id") or "").strip()
    if not instance_id:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "instance_id required")

    session = await repository.create_telemetry_session(
        eval_job_id=eval_job_id,
        work_unit_id=work_unit_id,
        instance_id=instance_id,
        hotkey_ss58=hotkey,
        nonce=nonce,
    )
    return session


@public_route(tags=["execution"])
@router.post("/execution/events")
async def ingest_execution_events(
    request: Request,
    _: None = Depends(authenticate_internal),
    repository: PrismRepository = Depends(repo_from_request),
    x_telemetry_session: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Ingest session-gated execution events. Never scores; never elevates tier."""

    body = await request.body()
    try:
        import json

        payload = json.loads(body.decode("utf-8") if body else "{}")
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid json") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "body must be object")

    session_id = str(x_telemetry_session or payload.get("session_id") or "").strip() or None
    if not session_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "telemetry session required")
    session = await repository.get_telemetry_session(session_id)
    if session is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid telemetry session")

    events = payload.get("events")
    if not isinstance(events, list) or not events:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "events required")

    inserted = 0
    duplicates = 0
    for raw_event in events:
        if not isinstance(raw_event, dict):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "event must be object")
        # Reject first-class score fields (trust boundary).
        if _SCORE_FIELD_NAMES.intersection(raw_event):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "score fields are not allowed on execution events",
            )
        try:
            sequence = int(raw_event["sequence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "sequence required") from exc
        task_id = str(raw_event.get("task_id") or "").strip()
        event_type = str(raw_event.get("event_type") or "").strip()
        if not task_id or not event_type:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, "task_id and event_type required"
            )
        eval_job_id = raw_event.get("eval_job_id")
        work_unit_id = raw_event.get("work_unit_id")
        if eval_job_id is not None:
            eval_job_id = str(eval_job_id)
        else:
            eval_job_id = (
                str(session["eval_job_id"]) if session.get("eval_job_id") is not None else None
            )
        if work_unit_id is not None:
            work_unit_id = str(work_unit_id)
        else:
            work_unit_id = (
                str(session["work_unit_id"]) if session.get("work_unit_id") is not None else None
            )
        event_payload: dict[str, Any] = {}
        if "message" in raw_event:
            event_payload["message"] = raw_event["message"]
        if "progress" in raw_event:
            event_payload["progress"] = raw_event["progress"]
        # metadata is observability-only; never promoted to score columns
        if isinstance(raw_event.get("metadata"), dict):
            event_payload["metadata"] = raw_event["metadata"]
        try:
            result = await repository.append_execution_event(
                session_id=session_id,
                hotkey_ss58=str(session["hotkey_ss58"]),
                eval_job_id=eval_job_id,
                work_unit_id=work_unit_id,
                task_id=task_id,
                sequence=sequence,
                event_type=event_type,
                payload=event_payload,
            )
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        if result == "inserted":
            inserted += 1
        else:
            duplicates += 1
    return {"accepted": inserted + duplicates, "inserted": inserted, "duplicates": duplicates}


@public_route(tags=["execution"])
@router.get("/execution-pool/live")
async def execution_pool_live(
    repository: PrismRepository = Depends(repo_from_request),
) -> dict[str, list[dict[str, Any]]]:
    """In-flight RUNNING jobs with latest event. Empty pool is honest empty list."""

    jobs = await repository.list_live_execution_pool()
    return {"jobs": jobs}
