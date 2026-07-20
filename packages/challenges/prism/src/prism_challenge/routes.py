from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, SupportsFloat, SupportsInt, cast

from base.challenge_sdk.roles import public_route
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from pydantic import ValidationError

from .admission import enforce_admission
from .auth import authenticate_miner
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

router = APIRouter(prefix="/v1")


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
