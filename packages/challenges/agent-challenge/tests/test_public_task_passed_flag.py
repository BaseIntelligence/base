"""Public task result / event ``passed`` flag derived from score >= 1.0."""

from __future__ import annotations

from types import SimpleNamespace

from agent_challenge.api.routes import (
    _public_task_event_metadata,
    _task_result_response,
)


def test_task_result_response_passed_false_when_score_zero_status_completed() -> None:
    result = SimpleNamespace(
        task_id="terminal-bench/count-dataset-tokens",
        docker_image="img:tag",
        status="completed",
        score=0.0,
        returncode=0,
        duration_seconds=139.76,
        stderr="",
        stdout="",
    )
    payload = _task_result_response(result).model_dump()
    assert payload["status"] == "completed"
    assert payload["score"] == 0.0
    assert payload["passed"] is False


def test_task_result_response_passed_true_when_score_one() -> None:
    result = SimpleNamespace(
        task_id="terminal-bench/fix-git",
        docker_image="img:tag",
        status="completed",
        score=1.0,
        returncode=0,
        duration_seconds=10.0,
        stderr="",
        stdout="",
    )
    payload = _task_result_response(result).model_dump()
    assert payload["status"] == "completed"
    assert payload["score"] == 1.0
    assert payload["passed"] is True


def test_public_task_event_metadata_exposes_passed_from_score() -> None:
    meta = _public_task_event_metadata({"duration_seconds": 139.76, "returncode": 0, "score": 0.0})
    assert meta["score"] == 0.0
    assert meta["passed"] is False

    meta_pass = _public_task_event_metadata({"score": 1.0, "returncode": 0})
    assert meta_pass["passed"] is True
