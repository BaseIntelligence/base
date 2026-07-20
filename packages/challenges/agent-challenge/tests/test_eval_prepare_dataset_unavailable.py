"""Closed 503 mapping when eval prepare plan-build residuals (dataset digest, etc.).

Live residual observed after verified_allow: missing ``/app/golden/dataset-digest.json``
raised bare FileNotFoundError → FastAPI text/plain 500. Product must convert path IO
and other plan-build unavailable failures into EvalAuthorizationUnavailable with a
closed ``detail.code`` (prefer ``eval_dataset_unavailable``), and never residual bare
500 for known prepare failures or unexpected Exception on prepare/retry routes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from agent_challenge.api import routes as api_routes
from agent_challenge.evaluation import authorization as auth
from agent_challenge.evaluation.authorization import EvalAuthorizationUnavailable


def test_dataset_digest_tasks_missing_file_raises_closed_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "dataset-digest.json"
    monkeypatch.setattr(
        auth,
        "_dataset_digest_manifest_path",
        lambda: missing,
    )
    monkeypatch.setattr(auth, "_CACHED_DATASET_DIGEST", None)

    with pytest.raises(EvalAuthorizationUnavailable) as excinfo:
        auth._dataset_digest_tasks()

    assert excinfo.value.code == "eval_dataset_unavailable"
    assert "dataset" in str(excinfo.value).lower()
    # Never leak absolute host paths in the closed message family.
    assert str(missing) not in str(excinfo.value)


def test_dataset_digest_tasks_oserror_raises_closed_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocked = tmp_path / "dataset-digest.json"
    blocked.write_text('{"tasks": {}}\n', encoding="utf-8")
    monkeypatch.setattr(auth, "_dataset_digest_manifest_path", lambda: blocked)
    monkeypatch.setattr(auth, "_CACHED_DATASET_DIGEST", None)

    def _raise_oserror(self, *args, **kwargs):  # noqa: ANN001
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", _raise_oserror)
    with pytest.raises(EvalAuthorizationUnavailable) as excinfo:
        auth._dataset_digest_tasks()
    assert excinfo.value.code == "eval_dataset_unavailable"


def test_dataset_digest_tasks_malformed_json_raises_closed_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "dataset-digest.json"
    path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(auth, "_dataset_digest_manifest_path", lambda: path)
    monkeypatch.setattr(auth, "_CACHED_DATASET_DIGEST", None)

    with pytest.raises(EvalAuthorizationUnavailable) as excinfo:
        auth._dataset_digest_tasks()
    assert excinfo.value.code == "eval_dataset_unavailable"


def test_task_config_digest_missing_entry_uses_dataset_unavailable_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth, "_CACHED_DATASET_DIGEST", {})
    task = SimpleNamespace(
        task_id="terminal-bench/not-in-manifest",
        docker_image="registry.example/task@sha256:" + "d" * 64,
        prompt="p",
        benchmark="terminal_bench",
        metadata={},
    )
    with pytest.raises(EvalAuthorizationUnavailable) as excinfo:
        auth._task_config_digest(task)
    assert excinfo.value.code == "eval_dataset_unavailable"


def test_task_config_digest_propagates_missing_file_as_dataset_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "nope-dataset-digest.json"
    monkeypatch.setattr(auth, "_dataset_digest_manifest_path", lambda: missing)
    monkeypatch.setattr(auth, "_CACHED_DATASET_DIGEST", None)
    task = SimpleNamespace(
        task_id="terminal-bench/adaptive-rejection-sampler",
        docker_image="registry.example/task@sha256:" + "a" * 64,
        prompt="p",
        benchmark="terminal_bench",
        metadata={},
    )
    with pytest.raises(EvalAuthorizationUnavailable) as excinfo:
        auth._task_config_digest(task)
    assert excinfo.value.code == "eval_dataset_unavailable"


def test_build_plan_empty_key_release_endpoint_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import UTC, datetime

    from agent_challenge.sdk.config import ChallengeSettings

    measurement = {
        "mrtd": "01" * 48,
        "rtmr0": "02" * 48,
        "rtmr1": "03" * 48,
        "rtmr2": "04" * 48,
        "os_image_hash": "05" * 32,
        "key_provider": "validator-kms",
        "vm_shape": "tdx-small",
    }
    settings = ChallengeSettings(
        attested_review_enabled=True,
        phala_attestation_enabled=True,
        eval_app_image_ref="registry.example/eval@sha256:" + "a" * 64,
        eval_app_compose_hash="06" * 32,
        eval_app_identity="agent-challenge-eval-v1",
        eval_app_kms_public_key_hex="07" * 32,
        eval_app_measurement=measurement,
        eval_app_measurement_allowlist=(
            {
                "mrtd": measurement["mrtd"],
                "rtmr0": measurement["rtmr0"],
                "rtmr1": measurement["rtmr1"],
                "rtmr2": measurement["rtmr2"],
                "compose_hash": "06" * 32,
                "os_image_hash": measurement["os_image_hash"],
            },
        ),
        eval_key_release_endpoint="",
        eval_k=1,
        evaluation_task_count=1,
    )
    monkeypatch.setattr(
        auth,
        "load_benchmark_tasks",
        lambda: [
            SimpleNamespace(
                task_id="task-a",
                docker_image="registry.example/task@sha256:" + "b" * 64,
                prompt="",
                benchmark="terminal_bench",
                metadata={"content_digest_sha256": "ab" * 32},
            )
        ],
    )
    submission = SimpleNamespace(id=1, agent_hash="11" * 32, version_number=1)
    with pytest.raises(EvalAuthorizationUnavailable) as excinfo:
        auth._build_plan(
            submission=submission,  # type: ignore[arg-type]
            review_digest="12" * 32,
            settings=settings,
            eval_run_id="eval_test",
            key_release_nonce="kr-nonce",
            score_nonce="score-nonce",
            token_sha256="13" * 32,
            now=datetime.now(UTC),
        )
    assert excinfo.value.code == "eval_key_release_endpoint_unavailable"


def test_unavailable_detail_code_prefers_explicit_code() -> None:
    exc = EvalAuthorizationUnavailable(
        "frozen dataset digest is unavailable",
        code="eval_dataset_unavailable",
    )
    assert api_routes._eval_authorization_unavailable_detail_code(exc) == "eval_dataset_unavailable"
    default = EvalAuthorizationUnavailable("identity missing")
    assert (
        api_routes._eval_authorization_unavailable_detail_code(default)
        == "eval_deployment_identity_unavailable"
    )


@pytest.mark.asyncio
async def test_prepare_route_maps_dataset_unavailable_to_503_closed_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(*_a, **_k):
        raise EvalAuthorizationUnavailable(
            "frozen dataset digest is unavailable",
            code="eval_dataset_unavailable",
        )

    async def _commit() -> None:
        return None

    async def _rollback() -> None:
        return None

    session = SimpleNamespace(commit=_commit, rollback=_rollback)
    submission = SimpleNamespace(id=30)

    async def _get_sub(*_a, **_k):
        return submission

    monkeypatch.setattr(api_routes, "_get_miner_eval_submission", _get_sub)
    monkeypatch.setattr(api_routes, "create_eval_run", _boom)

    with pytest.raises(HTTPException) as excinfo:
        await api_routes.prepare_submission_eval(
            submission_id=30,
            session=session,  # type: ignore[arg-type]
            auth=SimpleNamespace(),  # type: ignore[arg-type]
        )
    assert excinfo.value.status_code == 503
    detail = excinfo.value.detail
    assert isinstance(detail, dict)
    assert detail == {"code": "eval_dataset_unavailable"}
    body = json.dumps(detail)
    assert "traceback" not in body.lower()
    assert "FileNotFound" not in body
    assert "dataset-digest.json" not in body


@pytest.mark.asyncio
async def test_prepare_route_maps_unexpected_exception_to_internal_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(*_a, **_k):
        raise RuntimeError("secret-path=/tmp/leaked and stack")

    async def _commit() -> None:
        return None

    async def _rollback() -> None:
        return None

    session = SimpleNamespace(commit=_commit, rollback=_rollback)

    async def _get_sub(*_a, **_k):
        return SimpleNamespace(id=7)

    monkeypatch.setattr(api_routes, "_get_miner_eval_submission", _get_sub)
    monkeypatch.setattr(api_routes, "create_eval_run", _boom)

    with pytest.raises(HTTPException) as excinfo:
        await api_routes.prepare_submission_eval(
            submission_id=7,
            session=session,  # type: ignore[arg-type]
            auth=SimpleNamespace(),  # type: ignore[arg-type]
        )
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == {"code": "eval_prepare_internal_unavailable"}
    assert "secret-path" not in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_retry_route_maps_unexpected_exception_to_internal_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(*_a, **_k):
        raise ValueError("unexpected plan residual")

    async def _commit() -> None:
        return None

    async def _rollback() -> None:
        return None

    session = SimpleNamespace(commit=_commit, rollback=_rollback)

    async def _get_sub(*_a, **_k):
        return SimpleNamespace(id=8)

    monkeypatch.setattr(api_routes, "_get_miner_eval_submission", _get_sub)
    monkeypatch.setattr(api_routes, "retry_eval_run", _boom)

    with pytest.raises(HTTPException) as excinfo:
        await api_routes.retry_submission_eval(
            submission_id=8,
            request=SimpleNamespace(eval_run_id="eval_x"),  # type: ignore[arg-type]
            session=session,  # type: ignore[arg-type]
            auth=SimpleNamespace(),  # type: ignore[arg-type]
        )
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == {"code": "eval_prepare_internal_unavailable"}


@pytest.mark.asyncio
async def test_prepare_route_auth_matrix_required_stays_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_challenge.evaluation.authorization import EvalAuthorizationRequired

    async def _boom(*_a, **_k):
        raise EvalAuthorizationRequired("no allow")

    async def _commit() -> None:
        return None

    async def _rollback() -> None:
        return None

    session = SimpleNamespace(commit=_commit, rollback=_rollback)

    async def _get_sub(*_a, **_k):
        return SimpleNamespace(id=1)

    monkeypatch.setattr(api_routes, "_get_miner_eval_submission", _get_sub)
    monkeypatch.setattr(api_routes, "create_eval_run", _boom)

    with pytest.raises(HTTPException) as excinfo:
        await api_routes.prepare_submission_eval(
            submission_id=1,
            session=session,  # type: ignore[arg-type]
            auth=SimpleNamespace(),  # type: ignore[arg-type]
        )
    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == {"code": "review_allow_required"}


@pytest.mark.asyncio
async def test_prepare_route_auth_matrix_conflict_stays_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_challenge.evaluation.authorization import EvalAuthorizationConflict

    async def _boom(*_a, **_k):
        raise EvalAuthorizationConflict("retry required", code="eval_prepare_conflict")

    async def _commit() -> None:
        return None

    async def _rollback() -> None:
        return None

    session = SimpleNamespace(commit=_commit, rollback=_rollback)

    async def _get_sub(*_a, **_k):
        return SimpleNamespace(id=1)

    monkeypatch.setattr(api_routes, "_get_miner_eval_submission", _get_sub)
    monkeypatch.setattr(api_routes, "create_eval_run", _boom)

    with pytest.raises(HTTPException) as excinfo:
        await api_routes.prepare_submission_eval(
            submission_id=1,
            session=session,  # type: ignore[arg-type]
            auth=SimpleNamespace(),  # type: ignore[arg-type]
        )
    assert excinfo.value.status_code == 409
    assert excinfo.value.detail == {"code": "eval_prepare_conflict"}


@pytest.mark.asyncio
async def test_prepare_route_default_identity_unavailable_stays_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(*_a, **_k):
        raise EvalAuthorizationUnavailable("validator Eval measurement is unavailable")

    async def _commit() -> None:
        return None

    async def _rollback() -> None:
        return None

    session = SimpleNamespace(commit=_commit, rollback=_rollback)

    async def _get_sub(*_a, **_k):
        return SimpleNamespace(id=1)

    monkeypatch.setattr(api_routes, "_get_miner_eval_submission", _get_sub)
    monkeypatch.setattr(api_routes, "create_eval_run", _boom)

    with pytest.raises(HTTPException) as excinfo:
        await api_routes.prepare_submission_eval(
            submission_id=1,
            session=session,  # type: ignore[arg-type]
            auth=SimpleNamespace(),  # type: ignore[arg-type]
        )
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == {"code": "eval_deployment_identity_unavailable"}
