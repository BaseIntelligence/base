#!/usr/bin/env python3
"""agent_parity_harness.py — agent-driving parity on a subset (Task 23).

Scores the SAME recorded-deterministic agent under two execution paths and diffs
the per-task outcomes at epsilon=0:

* **PATH A (stock harbor)** -- ``harbor run --agent-import-path
  recorded_harbor_agent:Agent`` over a task subset, then parse each task's
  ``result.json`` into a ``{reward, status, reason_code, resolved}`` record using
  the own-runner reward authority (:func:`map_rewards_to_outcome`).
* **PATH B (own_runner)** -- :func:`run_own_runner_job` with
  ``agent_class=RecordedAgent`` over the same subset, then assemble the same
  record shape from the returned trial outcomes (the
  :mod:`tools.assemble_baseline` authority).

Both records are keyed by the bare task id and compared with
:func:`tools.parity_diff.compare_records` (canonical fields exact, reward at
``eps=0``). Identical outcomes => zero deltas => parity holds. ``resolved=0`` on
both sides is EXPECTED for the harmless ``probe`` transcript; the gate proves the
two paths AGREE, not that the task is solved.

The pure functions (record building, argv building, diffing) are import-safe and
docker-free so they can be unit-tested; the ``run-harbor`` / ``run-own`` / ``diff``
/ ``parity`` CLI subcommands perform the heavy real-docker work.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Make the in-repo tools + src importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent_challenge.evaluation.own_runner.verifier_runner import (  # noqa: E402
    map_rewards_to_outcome,
)
from tools.assemble_baseline import build_results as _own_build_results  # noqa: E402
from tools.parity_diff import compare_records  # noqa: E402

#: ``task_name`` prefix stripped to reach the bare ``dataset-digest`` key form.
TASK_NAME_PREFIX = "terminal-bench/"
#: Default harbor import path for the recorded agent (module:Class).
DEFAULT_HARBOR_IMPORT_PATH = "recorded_harbor_agent:Agent"
#: The directory holding the recorded agent modules (added to PYTHONPATH so
#: harbor's ``importlib.import_module`` can resolve the agent by bare name).
DEFAULT_AGENT_MODULE_DIR = str(_REPO_ROOT / "tests")


def digest_key(task_name: str) -> str:
    """Normalize a ``task_name`` to the bare ``dataset-digest`` key form."""
    if task_name.startswith(TASK_NAME_PREFIX):
        return task_name[len(TASK_NAME_PREFIX) :]
    return task_name


def record_from_rewards(rewards: Mapping[str, float | int]) -> dict[str, Any]:
    """Map a single clean trial's reward dict to a parity record.

    Uses the own-runner reward authority (:func:`map_rewards_to_outcome`,
    n_total_trials=1) so PATH A and PATH B normalize through the SAME math --
    banker's-rounded ``resolved`` included -- making the record shape identical by
    construction for a clean (non-errored) run.
    """
    summary = map_rewards_to_outcome(dict(rewards), n_total_trials=1)
    return {
        "reward": rewards["reward"],
        "status": summary["status"],
        "reason_code": summary["reason_code"],
        "resolved": summary["resolved"],
    }


# ===========================================================================
# PATH A: stock-harbor result.json -> records
# ===========================================================================
def _harbor_rewards_from_result(result: Mapping[str, Any]) -> dict[str, float | int]:
    """Extract the ``{"reward": ...}`` dict from a harbor ``result.json`` mapping.

    Faithful to the harbor 0.13.1 ``result.json`` shape:
    ``verifier_result.rewards["reward"]``. A trial that recorded an
    ``exception_info`` or carries no rewards is a hard error here (the recorded
    agent never errors on the subset; surfacing it loudly avoids silent parity
    passes on a degenerate run).
    """
    if result.get("exception_info") is not None:
        raise ValueError(
            f"harbor trial {result.get('task_name')!r} recorded an exception: "
            f"{result['exception_info']!r}"
        )
    verifier_result = result.get("verifier_result")
    if not isinstance(verifier_result, Mapping):
        raise ValueError(f"harbor trial {result.get('task_name')!r} has no verifier_result")
    rewards = verifier_result.get("rewards")
    if not isinstance(rewards, Mapping) or "reward" not in rewards:
        raise ValueError(f"harbor trial {result.get('task_name')!r} has no rewards['reward']")
    return dict(rewards)


def harbor_records_from_run_dir(run_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Build the ``task -> record`` map from a harbor job timestamp dir.

    Scans ``<run_dir>/<task>__<suffix>/result.json`` (harbor's per-trial layout)
    and keys each record by the bare task id. A duplicated task id (k>1) is a hard
    error -- this gate runs exactly one attempt per task.
    """
    run_dir = Path(run_dir)
    records: dict[str, dict[str, Any]] = {}
    for result_path in sorted(run_dir.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        task_name = result.get("task_name")
        if not task_name:
            raise ValueError(f"harbor result missing task_name: {result_path}")
        key = digest_key(task_name)
        if key in records:
            raise ValueError(f"duplicate harbor trial for task {key!r} in {run_dir}")
        records[key] = record_from_rewards(_harbor_rewards_from_result(result))
    if not records:
        raise ValueError(f"no harbor result.json files found under {run_dir}")
    return records


def harbor_records_from_run_dirs(
    run_dirs: Iterable[Path | str],
) -> dict[str, dict[str, Any]]:
    """Merge per-task harbor run dirs into one ``task -> record`` map.

    Offline harbor only accepts a single ``-p`` task per ``run``, so a subset is
    produced as one job dir per task; this merges them. A task appearing in two
    run dirs is a hard error (the subset must be a set).
    """
    merged: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        for key, record in harbor_records_from_run_dir(run_dir).items():
            if key in merged:
                raise ValueError(f"task {key!r} appears in more than one harbor run dir")
            merged[key] = record
    return merged


# ===========================================================================
# PATH B: own_runner -> records
# ===========================================================================
def own_records_from_job_dir(job_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Build the ``task -> record`` map from a finished own_runner job dir.

    Delegates to the :mod:`tools.assemble_baseline` authority so the own-runner
    side uses the exact same normalization as the frozen golden.
    """
    return _own_build_results(Path(job_dir))


def own_records_from_outcomes(outcomes: Iterable[Any]) -> dict[str, dict[str, Any]]:
    """Build the ``task -> record`` map from in-process ``TrialOutcome`` objects.

    The in-memory counterpart of :func:`own_records_from_job_dir` for callers that
    already hold a :class:`JobResult`. Mirrors ``assemble_baseline.build_results``:
    exactly one clean trial per task, mapped via :func:`map_rewards_to_outcome`.
    """
    by_task: dict[str, list[Any]] = {}
    for outcome in outcomes:
        by_task.setdefault(digest_key(outcome.task_name), []).append(outcome)

    records: dict[str, dict[str, Any]] = {}
    for key, trials in sorted(by_task.items()):
        if len(trials) != 1:
            raise ValueError(f"{key}: expected exactly 1 trial, got {len(trials)}")
        trial = trials[0]
        if trial.errored or trial.rewards is None:
            raise ValueError(
                f"{key}: trial {trial.trial_name!r} errored "
                f"(reason_code={trial.reason_code!r}), refusing to build records"
            )
        records[key] = record_from_rewards(trial.rewards)
    return records


# ===========================================================================
# Diffing
# ===========================================================================
def diff_records(
    harbor: dict[str, dict[str, Any]],
    own: dict[str, dict[str, Any]],
    *,
    reward_eps: float = 0.0,
) -> list[dict[str, Any]]:
    """Return the list of deltas between the harbor (left) and own (right) maps."""
    return compare_records(harbor, own, reward_eps=reward_eps)


# ===========================================================================
# PATH A invocation (harbor subprocess in its own venv)
# ===========================================================================
def build_harbor_argv(
    *,
    harbor_bin: str,
    jobs_dir: Path | str,
    task_paths: Sequence[str] = (),
    dataset: str | None = None,
    include_task_names: Sequence[str] = (),
    agent_import_path: str = DEFAULT_HARBOR_IMPORT_PATH,
    agent_env: Mapping[str, str] | None = None,
    job_name: str | None = None,
    n_attempts: int = 1,
    n_concurrent: int = 1,
) -> list[str]:
    """Build the ``harbor run`` argv for the recorded agent over a subset.

    Pure (no side effects) so it can be asserted in a unit test. Either
    ``task_paths`` (repeatable ``-p``) or ``dataset`` + ``include_task_names``
    (``-d`` + repeatable ``-i``) selects the subset; both may be combined.
    """
    argv: list[str] = [harbor_bin, "run"]
    for path in task_paths:
        argv += ["-p", str(path)]
    if dataset is not None:
        argv += ["-d", dataset]
    for name in include_task_names:
        argv += ["-i", name]
    argv += ["--agent-import-path", agent_import_path]
    argv += ["-o", str(jobs_dir)]
    argv += ["-k", str(n_attempts)]
    argv += ["--n-concurrent", str(n_concurrent)]
    argv += ["-e", "docker"]
    if job_name is not None:
        argv += ["--job-name", job_name]
    for key, value in (agent_env or {}).items():
        argv += ["--ae", f"{key}={value}"]
    argv += ["--yes"]
    return argv


def latest_run_dir(jobs_dir: Path | str) -> Path:
    """Return the most recently modified timestamp dir under ``jobs_dir``."""
    jobs_dir = Path(jobs_dir)
    candidates = [p for p in jobs_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise ValueError(f"no run dirs under {jobs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_harbor_subprocess(
    *,
    harbor_bin: str,
    jobs_dir: Path | str,
    agent_module_dir: str = DEFAULT_AGENT_MODULE_DIR,
    task_paths: Sequence[str] = (),
    dataset: str | None = None,
    include_task_names: Sequence[str] = (),
    agent_env: Mapping[str, str] | None = None,
    job_name: str | None = None,
) -> Path:
    """Invoke ``harbor run`` (in its own venv) and return the produced run dir.

    ``agent_module_dir`` is prepended to ``PYTHONPATH`` so harbor's
    ``importlib.import_module`` resolves ``recorded_harbor_agent`` by bare name.
    """
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    argv = build_harbor_argv(
        harbor_bin=harbor_bin,
        jobs_dir=jobs_dir,
        task_paths=task_paths,
        dataset=dataset,
        include_task_names=include_task_names,
        agent_env=agent_env,
        job_name=job_name,
    )
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{agent_module_dir}{os.pathsep}{existing}" if existing else agent_module_dir
    )
    print(f"[harness] PATH A argv: {' '.join(argv)}", flush=True)
    subprocess.run(argv, check=True, env=env)
    return latest_run_dir(jobs_dir)


# ===========================================================================
# CLI
# ===========================================================================
def _parse_kv(items: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        key, _, value = item.partition("=")
        if not key or "=" not in item:
            raise SystemExit(f"--agent-env expects KEY=VALUE, got {item!r}")
        out[key] = value
    return out


def _emit_parity(
    harbor: dict[str, dict[str, Any]],
    own: dict[str, dict[str, Any]],
    *,
    reward_eps: float,
    evidence_path: Path | None,
) -> int:
    deltas = diff_records(harbor, own, reward_eps=reward_eps)
    lines: list[str] = []
    lines.append("# Task 23 — agent-driving parity on subset")
    lines.append(f"subset ({len(harbor)}): {', '.join(sorted(harbor))}")
    lines.append("")
    lines.append("PATH A (stock harbor) outcomes:")
    for key in sorted(harbor):
        lines.append(f"  {key}: {json.dumps(harbor[key], sort_keys=True)}")
    lines.append("")
    lines.append("PATH B (own_runner) outcomes:")
    for key in sorted(own):
        lines.append(f"  {key}: {json.dumps(own[key], sort_keys=True)}")
    lines.append("")
    if deltas:
        lines.append(f"PARITY FAIL: {len(deltas)} delta(s)")
        for d in deltas:
            lines.append(f"  {json.dumps(d, sort_keys=True)}")
        verdict = 1
    else:
        lines.append(f"PARITY OK: {len(harbor)} records, 0 deltas")
        verdict = 0
    report = "\n".join(lines) + "\n"
    print(report, end="")
    if evidence_path is not None:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(report)
        print(f"[harness] wrote evidence -> {evidence_path}", flush=True)
    return verdict


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_harbor = sub.add_parser("run-harbor", help="run PATH A (harbor) and print run dir")
    p_harbor.add_argument("--harbor-bin", required=True)
    p_harbor.add_argument("--jobs-dir", required=True)
    p_harbor.add_argument("--agent-module-dir", default=DEFAULT_AGENT_MODULE_DIR)
    p_harbor.add_argument("--task-path", action="append", default=[])
    p_harbor.add_argument("--dataset", default=None)
    p_harbor.add_argument("--include-task-name", action="append", default=[])
    p_harbor.add_argument("--agent-env", action="append", default=[])
    p_harbor.add_argument("--job-name", default=None)

    p_parse = sub.add_parser("parse-harbor", help="parse a harbor run dir -> records JSON")
    p_parse.add_argument("--run-dir", required=True)
    p_parse.add_argument("--out", default=None)

    p_own = sub.add_parser("parse-own", help="parse an own_runner job dir -> records JSON")
    p_own.add_argument("--job-dir", required=True)
    p_own.add_argument("--out", default=None)

    p_diff = sub.add_parser("diff", help="diff two records JSON files (harbor LEFT, own RIGHT)")
    p_diff.add_argument("--harbor", required=True)
    p_diff.add_argument("--own", required=True)
    p_diff.add_argument("--reward-eps", type=float, default=0.0)
    p_diff.add_argument("--evidence", default=None)

    args = parser.parse_args(argv)

    if args.command == "run-harbor":
        run_dir = run_harbor_subprocess(
            harbor_bin=args.harbor_bin,
            jobs_dir=args.jobs_dir,
            agent_module_dir=args.agent_module_dir,
            task_paths=args.task_path,
            dataset=args.dataset,
            include_task_names=args.include_task_name,
            agent_env=_parse_kv(args.agent_env),
            job_name=args.job_name,
        )
        print(str(run_dir))
        return 0

    if args.command == "parse-harbor":
        records = harbor_records_from_run_dir(args.run_dir)
        payload = json.dumps(records, indent=2, sort_keys=True)
        if args.out:
            Path(args.out).write_text(payload + "\n")
        print(payload)
        return 0

    if args.command == "parse-own":
        records = own_records_from_job_dir(args.job_dir)
        payload = json.dumps(records, indent=2, sort_keys=True)
        if args.out:
            Path(args.out).write_text(payload + "\n")
        print(payload)
        return 0

    if args.command == "diff":
        harbor = json.loads(Path(args.harbor).read_text())
        own = json.loads(Path(args.own).read_text())
        return _emit_parity(
            harbor,
            own,
            reward_eps=args.reward_eps,
            evidence_path=Path(args.evidence) if args.evidence else None,
        )

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
