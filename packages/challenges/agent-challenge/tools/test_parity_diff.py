"""Tests for tools/parity_diff.py (Task 4, harbor-independence-a3).

Locks the parity-diff contract:
- ε=0 EXACT comparison on {status, reason_code, resolved}.
- reward precision rule: placeholder ε=0 string/decimal compare (Task 9 finalizes).
- extra metadata fields (provenance, blocker, ...) are ignored.
- CLI accepts two paths (file or directory); exit 0 iff zero task deltas.
"""

from __future__ import annotations

import json
from pathlib import Path

import parity_diff
import pytest


def _rec(reward=1.0, status="completed", reason_code=None, resolved=1, **extra):
    rec = {
        "reward": reward,
        "status": status,
        "reason_code": reason_code,
        "resolved": resolved,
    }
    rec.update(extra)
    return rec


# --------------------------------------------------------------------------- #
# reward precision rule (ε=0 string/decimal)
# --------------------------------------------------------------------------- #
def test_rewards_equal_exact_integerish_float():
    assert parity_diff.rewards_equal(1.0, 1.0)
    assert parity_diff.rewards_equal(0.0, 0.0)
    assert parity_diff.rewards_equal(1, 1.0)


def test_rewards_equal_decimal_string_no_float_drift():
    # 0.1+0.2 != 0.3 in binary float; the ε=0 rule compares decimal strings,
    # so two values that serialize to the same decimal string are equal,
    # while genuinely different decimals are not.
    assert parity_diff.rewards_equal(0.3, 0.3)
    assert not parity_diff.rewards_equal(0.30000000000000004, 0.3)


def test_rewards_equal_nan_treated_equal():
    assert parity_diff.rewards_equal(float("nan"), float("nan"))


def test_rewards_unequal():
    assert not parity_diff.rewards_equal(1.0, 0.0)
    assert not parity_diff.rewards_equal(0.5, 0.0)


def test_rewards_eps_band_optional():
    # default ε=0 means exact; a non-zero eps widens tolerance (forward-compat).
    assert not parity_diff.rewards_equal(1.0, 0.9)
    assert parity_diff.rewards_equal(1.0, 0.9, reward_eps=0.2)


# --------------------------------------------------------------------------- #
# compare_records
# --------------------------------------------------------------------------- #
def test_compare_identical_no_delta():
    left = {"t1": _rec(), "t2": _rec(reward=0.0, resolved=0)}
    right = {"t1": _rec(), "t2": _rec(reward=0.0, resolved=0)}
    assert parity_diff.compare_records(left, right) == []


def test_compare_flip_one_resolved_yields_exactly_one_task_delta():
    # S1: the RED anchor. Mutating one `resolved` -> exactly 1 task delta.
    left = {"t1": _rec(resolved=1), "t2": _rec(resolved=1)}
    right = {"t1": _rec(resolved=1), "t2": _rec(resolved=0)}
    deltas = parity_diff.compare_records(left, right)
    tasks = {d["task"] for d in deltas}
    assert tasks == {"t2"}
    assert any(d["field"] == "resolved" for d in deltas)


def test_compare_flip_status():
    left = {"t1": _rec(status="completed")}
    right = {"t1": _rec(status="failed")}
    deltas = parity_diff.compare_records(left, right)
    assert {d["task"] for d in deltas} == {"t1"}
    assert any(d["field"] == "status" for d in deltas)


def test_compare_flip_reason_code():
    left = {"t1": _rec(reason_code=None)}
    right = {"t1": _rec(reason_code="harbor_reward_empty")}
    deltas = parity_diff.compare_records(left, right)
    assert any(d["field"] == "reason_code" for d in deltas)


def test_compare_reward_mismatch_detected():
    left = {"t1": _rec(reward=1.0)}
    right = {"t1": _rec(reward=0.0, resolved=1)}  # resolved kept same to isolate reward
    deltas = parity_diff.compare_records(left, right)
    assert any(d["field"] == "reward" for d in deltas)


def test_compare_ignores_extra_metadata_fields():
    # provenance / blocker / observed_local must NOT count as deltas.
    left = {"t1": _rec(provenance="executed", blocker=None)}
    right = {
        "t1": _rec(provenance="contract-derived", blocker="gpu", observed_local={"reward": 0.0})
    }
    assert parity_diff.compare_records(left, right) == []


def test_compare_missing_task_in_right():
    left = {"t1": _rec(), "t2": _rec()}
    right = {"t1": _rec()}
    deltas = parity_diff.compare_records(left, right)
    assert {d["task"] for d in deltas} == {"t2"}
    assert deltas[0]["kind"] == "missing_in_right"


def test_compare_extra_task_in_right():
    left = {"t1": _rec()}
    right = {"t1": _rec(), "t2": _rec()}
    deltas = parity_diff.compare_records(left, right)
    assert {d["task"] for d in deltas} == {"t2"}
    assert deltas[0]["kind"] == "missing_in_left"


# --------------------------------------------------------------------------- #
# load_baseline
# --------------------------------------------------------------------------- #
def _write_baseline(path: Path, results: dict):
    path.write_text(json.dumps({"schema": "x", "results": results}, indent=2))


def test_load_baseline_file_results_key(tmp_path):
    f = tmp_path / "tbench-2.1-oracle.json"
    _write_baseline(f, {"t1": _rec(), "t2": _rec(reward=0.0, resolved=0)})
    loaded = parity_diff.load_baseline(f)
    assert set(loaded) == {"tbench-2.1-oracle.json::t1", "tbench-2.1-oracle.json::t2"}


def test_load_baseline_dir_skips_non_baseline(tmp_path):
    _write_baseline(tmp_path / "tbench-2.1-oracle.json", {"t1": _rec()})
    # a file without a "results" key (e.g. dataset-digest.json) is ignored
    (tmp_path / "dataset-digest.json").write_text(json.dumps({"task_count": 89}))
    loaded = parity_diff.load_baseline(tmp_path)
    assert set(loaded) == {"tbench-2.1-oracle.json::t1"}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_dir_compared_to_itself_exits_zero(tmp_path, capsys):
    _write_baseline(
        tmp_path / "tbench-2.1-oracle.json", {"t1": _rec(), "t2": _rec(reward=0.0, resolved=0)}
    )
    rc = parity_diff.main([str(tmp_path), str(tmp_path)])
    assert rc == 0


def test_cli_mutated_resolved_exits_nonzero_one_delta(tmp_path, capsys):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    _write_baseline(
        left / "tbench-2.1-oracle.json", {"t1": _rec(resolved=1), "t2": _rec(resolved=1)}
    )
    _write_baseline(
        right / "tbench-2.1-oracle.json", {"t1": _rec(resolved=1), "t2": _rec(resolved=0)}
    )
    rc = parity_diff.main([str(left), str(right)])
    out = capsys.readouterr().out
    assert rc != 0
    assert "1" in out  # one task delta reported
    assert "t2" in out


def test_cli_real_golden_self_parity_when_present(capsys):
    # Guards the DoD: `parity_diff golden golden` exits 0 against the frozen file.
    golden_dir = Path(__file__).resolve().parent.parent / "golden"
    oracle = golden_dir / "tbench-2.1-oracle.json"
    if not oracle.exists():
        pytest.skip("golden not frozen yet")
    rc = parity_diff.main([str(golden_dir), str(golden_dir)])
    assert rc == 0
