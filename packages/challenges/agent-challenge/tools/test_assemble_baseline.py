"""Tests for tools/assemble_baseline.py (Task 22, harbor-independence full-set gate).

Locks the own-runner baseline-assembler contract:
- reads an own-runner job dir's per-trial ``result.json`` files (TrialOutcome shape);
- groups by bare task key and derives ``{reward, status, reason_code, resolved}``
  via the own-runner's OWN reward path (no reimplemented reward/round math);
- fails loudly on an errored trial (an oracle baseline must be clean);
- emits a document whose ``results`` map is parity-diffable against the frozen
  golden at epsilon=0 for the assembled task subset.
"""

from __future__ import annotations

import json
from pathlib import Path

import assemble_baseline
import parity_diff
import pytest

_ARS = "adaptive-rejection-sampler"
_CGW = "configure-git-webserver"


def _write_trial(
    job_dir: Path,
    *,
    task_name: str,
    trial_name: str,
    reward: float | None,
    errored: bool = False,
) -> None:
    trial_dir = job_dir / "trials" / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    rewards = None if reward is None else {"reward": reward}
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": task_name,
                "trial_name": trial_name,
                "status": "failed" if errored else "completed",
                "rewards": rewards,
                "reason_code": None,
                "errored": errored,
                "agent_name": "oracle",
                "model_name": None,
                "source": "terminal-bench/terminal-bench-2-1",
            },
            sort_keys=True,
        )
    )


def _oracle_job(job_dir: Path) -> Path:
    _write_trial(job_dir, task_name=_ARS, trial_name=f"{_ARS}__attempt-0", reward=1.0)
    _write_trial(job_dir, task_name=_CGW, trial_name=f"{_CGW}__attempt-0", reward=0.0)
    return job_dir


# --------------------------------------------------------------------------- #
# build_results: own-runner trials -> golden-shaped per-task records
# --------------------------------------------------------------------------- #
def test_build_results_maps_oracle_trials_to_records(tmp_path: Path) -> None:
    results = assemble_baseline.build_results(_oracle_job(tmp_path))

    assert results == {
        _ARS: {"reward": 1.0, "status": "completed", "reason_code": None, "resolved": 1},
        _CGW: {"reward": 0.0, "status": "completed", "reason_code": None, "resolved": 0},
    }


def test_build_results_strips_task_name_prefix(tmp_path: Path) -> None:
    _write_trial(
        tmp_path,
        task_name=f"terminal-bench/{_ARS}",
        trial_name=f"{_ARS}__attempt-0",
        reward=1.0,
    )
    results = assemble_baseline.build_results(tmp_path)
    assert set(results) == {_ARS}


def test_build_results_fails_loudly_on_errored_trial(tmp_path: Path) -> None:
    _write_trial(
        tmp_path, task_name=_ARS, trial_name=f"{_ARS}__attempt-0", reward=None, errored=True
    )
    with pytest.raises(ValueError, match="errored"):
        assemble_baseline.build_results(tmp_path)


# --------------------------------------------------------------------------- #
# assemble: writes a document carrying the results map
# --------------------------------------------------------------------------- #
def test_assemble_writes_document_with_results(tmp_path: Path) -> None:
    job_dir = _oracle_job(tmp_path / "job")
    out = tmp_path / "out" / "tbench-2.1-oracle.json"
    out.parent.mkdir(parents=True)

    document = assemble_baseline.assemble(job_dir=job_dir, out_path=out)

    assert out.name == "tbench-2.1-oracle.json"
    on_disk = json.loads(out.read_text())
    assert on_disk == document
    assert on_disk["task_count"] == 2
    assert set(on_disk["results"]) == {_ARS, _CGW}


# --------------------------------------------------------------------------- #
# parity: assembled subset matches the frozen golden at epsilon=0
# --------------------------------------------------------------------------- #
def test_assembled_subset_matches_golden_zero_delta(tmp_path: Path) -> None:
    # The golden expected-output is encrypted at rest (feature
    # golden-encrypted-at-rest); this parity check needs the released key.
    from agent_challenge.golden import crypto, package

    try:
        key = crypto.load_golden_key()
    except crypto.GoldenKeyError:
        pytest.skip("golden key not available (set CHALLENGE_GOLDEN_KEY_FILE)")
    golden = package.load_encrypted_oracle_golden(key)

    assembled = assemble_baseline.build_results(_oracle_job(tmp_path))
    golden_subset = {k: golden["results"][k] for k in assembled}

    deltas = parity_diff.compare_records(assembled, golden_subset)
    assert deltas == []
