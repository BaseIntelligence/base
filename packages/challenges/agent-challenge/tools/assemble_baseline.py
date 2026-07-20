#!/usr/bin/env python3
"""assemble_baseline.py — assemble an own-runner oracle baseline (Task 22, full-set gate).

Builds a parity-diffable baseline (``tbench-2.1-oracle.json`` shape) from a
finished OWN-RUNNER job dir, so the own-runner's output can be compared at
epsilon=0 against the frozen harbor golden (``golden/tbench-2.1-oracle.json``)
via ``tools/parity_diff.py``.

This is the own-runner counterpart of ``tools/freeze_golden.py`` (which sources
HARBOR ``*/result.json`` records). Here the source is the own-runner orchestrator
layout ``<job-dir>/trials/<trial>/result.json`` (each a ``TrialOutcome`` dict).

Normalization authority (DO NOT reimplement the reward/round math here):
each task's single clean trial reward dict is fed through the own-runner's OWN
reward path (:func:`agent_challenge.evaluation.own_runner.verifier_runner.map_rewards_to_outcome`,
whose ``resolved`` uses Python's banker's rounding) -- identical to
``freeze_golden.py`` -- so the assembled record matches the golden by
construction for a clean (errored=0) oracle run.

Per-task record shape (what ``parity_diff.py`` consumes):
``{reward, status, reason_code, resolved}`` where ``reward`` is the raw observed
reward (``rewards["reward"]``, 1.0/0.0) and ``status``/``reason_code``/``resolved``
come from the own-runner mapper. An oracle baseline has EXACTLY one trial per
task (k=1); a missing/duplicate or errored trial STOPS the assembly.

Usage::

    uv run python tools/assemble_baseline.py \
        --job-dir /path/to/own-runner/job \
        --out out/tbench-2.1-oracle.json \
        [--digest golden/dataset-digest.json]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

from agent_challenge.evaluation.own_runner.orchestrator import (
    TRIAL_RESULT_FILENAME,
    TRIALS_DIRNAME,
    TrialOutcome,
)
from agent_challenge.evaluation.own_runner.verifier_runner import map_rewards_to_outcome

#: Provenance schema id for an own-runner assembled baseline.
BASELINE_SCHEMA = "harbor-independence/own-runner-baseline@1"
#: own-runner ``task_name`` prefix stripped to reach the bare digest key form.
TASK_NAME_PREFIX = "terminal-bench/"


def _digest_key(task_name: str) -> str:
    """Normalize an own-runner ``task_name`` to the bare ``dataset-digest`` key form."""
    if task_name.startswith(TASK_NAME_PREFIX):
        return task_name[len(TASK_NAME_PREFIX) :]
    return task_name


def _load_trials(job_dir: Path) -> list[TrialOutcome]:
    """Load every ``<job-dir>/trials/<trial>/result.json`` as a ``TrialOutcome``."""
    trials_dir = job_dir / TRIALS_DIRNAME
    outcomes: list[TrialOutcome] = []
    for result_path in sorted(trials_dir.glob(f"*/{TRIAL_RESULT_FILENAME}")):
        outcomes.append(TrialOutcome.from_dict(json.loads(result_path.read_text())))
    return outcomes


def build_results(job_dir: Path) -> dict[str, dict[str, Any]]:
    """Build the ``task -> {reward, status, reason_code, resolved}`` results map.

    Trials are grouped by bare task key; an oracle baseline has EXACTLY one clean
    trial per task (k=1), so each group's single reward dict is mapped via the
    own-runner reward path (:func:`map_rewards_to_outcome`, n_total_trials=1) --
    identical to ``freeze_golden.py`` -- and the record matches the golden by
    construction. A missing/duplicate trial or an errored trial fails loudly.
    """
    by_task: dict[str, list[TrialOutcome]] = {}
    for outcome in _load_trials(job_dir):
        by_task.setdefault(_digest_key(outcome.task_name), []).append(outcome)

    results: dict[str, dict[str, Any]] = {}
    for key, trials in sorted(by_task.items()):
        if len(trials) != 1:
            raise ValueError(
                f"{key}: expected exactly 1 trial for an oracle baseline, got {len(trials)}"
            )
        trial = trials[0]
        if trial.errored or trial.rewards is None:
            raise ValueError(
                f"{key}: trial {trial.trial_name!r} errored "
                f"(reason_code={trial.reason_code!r}), refusing to assemble baseline"
            )

        summary = map_rewards_to_outcome(trial.rewards, n_total_trials=1)
        results[key] = {
            "reward": trial.rewards["reward"],
            "status": summary["status"],
            "reason_code": summary["reason_code"],
            "resolved": summary["resolved"],
        }

    return results


def _assert_keys_within_digest(results: dict[str, dict[str, Any]], digest: dict[str, Any]) -> None:
    """Fail loudly unless every assembled task is a known ``dataset-digest`` task."""
    digest_tasks = digest.get("tasks")
    if not isinstance(digest_tasks, dict):
        raise ValueError("dataset-digest.json: missing 'tasks' map")
    unknown = sorted(set(results) - set(digest_tasks))
    if unknown:
        raise ValueError(f"assembled tasks absent from dataset-digest: {unknown}")


def assemble(
    *,
    job_dir: Path,
    out_path: Path,
    digest_path: Path | None = None,
    frozen_at: str | None = None,
) -> dict[str, Any]:
    """Build, optionally validate against the digest, and write the baseline."""
    results = build_results(job_dir)

    dataset: str | None = None
    if digest_path is not None:
        digest = json.loads(digest_path.read_text())
        _assert_keys_within_digest(results, digest)
        dataset = digest.get("dataset")

    assembled_at_utc = frozen_at or (_dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))

    document: dict[str, Any] = {
        "schema": BASELINE_SCHEMA,
        "assembled_at_utc": assembled_at_utc,
        "dataset": dataset,
        "source_job": str(job_dir),
        "task_count": len(results),
        "results": results,
    }

    out_path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, help="finished own-runner job dir")
    parser.add_argument("--out", required=True, help="output baseline JSON path")
    parser.add_argument("--digest", default=None, help="golden/dataset-digest.json path (optional)")
    parser.add_argument(
        "--frozen-at",
        default=None,
        help="pin provenance assembled_at_utc (default: now, UTC)",
    )
    args = parser.parse_args(argv)

    document = assemble(
        job_dir=Path(args.job_dir),
        out_path=Path(args.out),
        digest_path=Path(args.digest) if args.digest else None,
        frozen_at=args.frozen_at,
    )

    results = document["results"]
    resolved_1 = sum(1 for rec in results.values() if rec["resolved"] == 1)
    resolved_0 = sum(1 for rec in results.values() if rec["resolved"] == 0)
    print(
        f"ASSEMBLED {args.out}: {len(results)} records "
        f"(resolved=1: {resolved_1}, resolved=0: {resolved_0})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
