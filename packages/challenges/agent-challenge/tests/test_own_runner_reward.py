"""Tests for the independent harbor reward parser + scorer (Task 9, A3).

Every expected value below is anchored to the G2 ground-truth fixtures captured
by executing the real ``harbor==0.13.1`` code:

  - ``.omo/evidence/task-2-reward-semantics.txt``
  - ``.omo/evidence/task-2-reward-edgecases.txt``

and to the live agent-challenge outcome mapping at
``runner.py:1399-1438``. The independent runner MUST reproduce harbor
byte-for-byte (ε=0); these tests are the parity contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from agent_challenge.evaluation.own_runner.reward import (
    Max,
    Mean,
    Min,
    RewardFileEmptyError,
    RewardFileNotFoundError,
    Sum,
    Trial,
    VerifierOutputParseError,
    aggregate_reward_dicts,
    compute_pass_at_k_by_evals,
    compute_pass_at_k_for_trials,
    default_metrics,
    derive_outcome,
    derive_outcome_from_metrics,
    derive_outcome_from_result_data,
    eligible_k_values,
    floats_bit_identical,
    format_agent_evals_key,
    parse_reward_files,
    parse_reward_json,
    parse_reward_text,
    parse_verifier_dir,
    pass_at_k_for_task,
    reason_code_for_error,
    reward_parity_equal,
    reward_values_equal,
)

# ---------------------------------------------------------------------------
# 1. Reward file parsing — reward.txt (verifier.py:61-74)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1", 1.0),
        ("0", 0.0),
        ("1.0", 1.0),
        ("0.0", 0.0),
        ("0.5", 0.5),
        (" 1 \n", 1.0),
        ("1\n", 1.0),
        ("1e0", 1.0),
        ("-1", -1.0),
        ("0.333333333333", 0.333333333333),
    ],
)
def test_parse_reward_text_accepts_floats(tmp_path: Path, raw: str, expected: float) -> None:
    path = tmp_path / "reward.txt"
    path.write_text(raw)
    assert parse_reward_text(path) == {"reward": expected}


def test_parse_reward_text_accepts_nan(tmp_path: Path) -> None:
    path = tmp_path / "reward.txt"
    path.write_text("nan")
    result = parse_reward_text(path)
    assert set(result) == {"reward"}
    assert math.isnan(result["reward"])


def test_parse_reward_text_accepts_inf(tmp_path: Path) -> None:
    path = tmp_path / "reward.txt"
    path.write_text("inf")
    assert parse_reward_text(path) == {"reward": math.inf}


def test_parse_reward_text_empty_is_byte_size_zero(tmp_path: Path) -> None:
    path = tmp_path / "reward.txt"
    path.write_text("")
    with pytest.raises(RewardFileEmptyError) as exc:
        parse_reward_text(path)
    # harbor's exact message + canonical reason code (NOT message substring).
    assert "Reward file is empty at" in str(exc.value)
    assert exc.value.reason_code == "harbor_reward_empty"


def test_parse_reward_text_whitespace_only_is_parse_error_not_empty(tmp_path: Path) -> None:
    # st_size of "  " is 2 bytes (NOT 0), so it is NOT empty; float("  ") raises.
    path = tmp_path / "reward.txt"
    path.write_text("  ")
    with pytest.raises(VerifierOutputParseError) as exc:
        parse_reward_text(path)
    assert exc.value.reason_code == "harbor_reward_parse_error"


@pytest.mark.parametrize("raw", ["pass", "True", "1,0"])
def test_parse_reward_text_malformed_is_parse_error(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "reward.txt"
    path.write_text(raw)
    with pytest.raises(VerifierOutputParseError) as exc:
        parse_reward_text(path)
    assert "Failed to parse rewards from text file" in str(exc.value)
    assert exc.value.reason_code == "harbor_reward_parse_error"


# ---------------------------------------------------------------------------
# 2. Reward file parsing — reward.json (verifier.py:76-87)
# ---------------------------------------------------------------------------


def test_parse_reward_json_single(tmp_path: Path) -> None:
    path = tmp_path / "reward.json"
    path.write_text(json.dumps({"reward": 1}))
    assert parse_reward_json(path) == {"reward": 1}


def test_parse_reward_json_multi_metric_verbatim(tmp_path: Path) -> None:
    path = tmp_path / "reward.json"
    path.write_text(json.dumps({"correctness": 1, "speed": 0.5}))
    assert parse_reward_json(path) == {"correctness": 1, "speed": 0.5}


def test_parse_reward_json_empty_is_empty_error(tmp_path: Path) -> None:
    path = tmp_path / "reward.json"
    path.write_text("")
    with pytest.raises(RewardFileEmptyError) as exc:
        parse_reward_json(path)
    assert exc.value.reason_code == "harbor_reward_empty"


def test_parse_reward_json_malformed_is_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "reward.json"
    path.write_text("{not valid json")
    with pytest.raises(VerifierOutputParseError) as exc:
        parse_reward_json(path)
    assert "Failed to parse rewards from JSON file" in str(exc.value)
    assert exc.value.reason_code == "harbor_reward_parse_error"


# ---------------------------------------------------------------------------
# 3. Precedence + missing (verifier.py:209-218)
# ---------------------------------------------------------------------------


def test_json_beats_txt(tmp_path: Path) -> None:
    (tmp_path / "reward.txt").write_text("0")
    (tmp_path / "reward.json").write_text(json.dumps({"reward": 1}))
    assert parse_reward_files(
        reward_text_path=tmp_path / "reward.txt",
        reward_json_path=tmp_path / "reward.json",
    ) == {"reward": 1}


def test_txt_used_when_no_json(tmp_path: Path) -> None:
    (tmp_path / "reward.txt").write_text("1")
    assert parse_reward_files(
        reward_text_path=tmp_path / "reward.txt",
        reward_json_path=tmp_path / "reward.json",
    ) == {"reward": 1.0}


def test_missing_both_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(RewardFileNotFoundError) as exc:
        parse_reward_files(
            reward_text_path=tmp_path / "reward.txt",
            reward_json_path=tmp_path / "reward.json",
        )
    assert "No reward file found at" in str(exc.value)
    assert exc.value.reason_code == "harbor_reward_missing"


def test_parse_verifier_dir_uses_standard_filenames(tmp_path: Path) -> None:
    (tmp_path / "reward.txt").write_text("0.5")
    assert parse_verifier_dir(tmp_path) == {"reward": 0.5}


def test_parse_verifier_dir_json_precedence(tmp_path: Path) -> None:
    (tmp_path / "reward.txt").write_text("0")
    (tmp_path / "reward.json").write_text(json.dumps({"correctness": 1, "speed": 0.5}))
    assert parse_verifier_dir(tmp_path) == {"correctness": 1, "speed": 0.5}


# ---------------------------------------------------------------------------
# 4. reason_code mapping (the three reward reason codes)
# ---------------------------------------------------------------------------


def test_reason_code_for_each_error() -> None:
    assert reason_code_for_error(RewardFileNotFoundError("x")) == "harbor_reward_missing"
    assert reason_code_for_error(RewardFileEmptyError("x")) == "harbor_reward_empty"
    assert reason_code_for_error(VerifierOutputParseError("x")) == "harbor_reward_parse_error"


def test_reason_codes_are_in_valid_set() -> None:
    from agent_challenge.evaluation.terminal_bench import TERMINAL_BENCH_FINAL_REASON_CODES

    for code in ("harbor_reward_missing", "harbor_reward_empty", "harbor_reward_parse_error"):
        assert code in TERMINAL_BENCH_FINAL_REASON_CODES


# ---------------------------------------------------------------------------
# 5. Metric aggregation — single reward key keyed by metric name
#    (metrics/base.py:16-37, task-2-reward-semantics.txt:52-58)
# ---------------------------------------------------------------------------


def test_mean_single_key_keyed_by_metric_name() -> None:
    assert Mean().compute([{"reward": 1}, {"reward": 0}, {"reward": 1}]) == {
        "mean": 0.6666666666666666
    }
    assert Mean().compute([{"reward": 1}]) == {"mean": 1.0}
    assert Mean().compute([{"reward": 0}, {"reward": 0}]) == {"mean": 0.0}


def test_mean_none_trials_count_as_zero() -> None:
    assert Mean().compute([None, {"reward": 1}]) == {"mean": 0.5}
    assert Mean().compute([{"reward": 1}, None, None]) == {"mean": 0.3333333333333333}


def test_mean_empty_list_raises_zero_division() -> None:
    with pytest.raises(ZeroDivisionError):
        Mean().compute([])


def test_default_metrics_is_exactly_one_mean() -> None:
    metrics = default_metrics()
    assert len(metrics) == 1
    assert isinstance(metrics[0], Mean)


# ---------------------------------------------------------------------------
# 6. Metric aggregation — multiple reward keys keyed by reward key
#    (task-2-reward-semantics.txt:60-64)
# ---------------------------------------------------------------------------


MULTI = [{"correctness": 1, "speed": 0.5}, {"correctness": 0, "speed": 1.0}]


def test_multi_metric_mean_keyed_by_reward_key() -> None:
    assert Mean().compute(MULTI) == {"correctness": 0.5, "speed": 0.75}


def test_multi_metric_max() -> None:
    assert Max().compute(MULTI) == {"correctness": 1, "speed": 1.0}


def test_multi_metric_min() -> None:
    assert Min().compute(MULTI) == {"correctness": 0, "speed": 0.5}


def test_multi_metric_sum() -> None:
    assert Sum().compute(MULTI) == {"correctness": 1, "speed": 1.5}


def test_multi_metric_missing_key_counts_as_zero() -> None:
    rewards = [{"correctness": 1, "speed": 1}, {"correctness": 1}]
    assert Mean().compute(rewards) == {"correctness": 1.0, "speed": 0.5}


def test_aggregate_reward_dicts_direct() -> None:
    assert aggregate_reward_dicts(
        [{"reward": 1}, {"reward": 0}], "mean", lambda v: sum(v) / len(v)
    ) == {"mean": 0.5}


# ---------------------------------------------------------------------------
# 7. pass@k (pass_at_k.py, task-2-reward-semantics.txt:69-86)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "min_trials, expected",
    [
        (1, []),
        (2, [2]),
        (3, [2]),
        (4, [2, 4]),
        (5, [2, 4, 5]),
        (8, [2, 4, 5, 8]),
        (10, [2, 4, 5, 8, 10]),
        (16, [2, 4, 5, 8, 10, 15, 16]),
        (20, [2, 4, 5, 8, 10, 15, 16, 20]),
    ],
)
def test_eligible_k_values(min_trials: int, expected: list[int]) -> None:
    assert eligible_k_values(min_trials) == expected


@pytest.mark.parametrize(
    "n, c, k, expected",
    [
        # Exact CPython doubles (verified against real harbor 0.13.1). The G2
        # fixture printed these with %f (6 dp); the true ε=0 values differ in
        # the last ULP, e.g. 0.4 -> 0.3999999999999999.
        (5, 0, 2, 0.0),
        (5, 1, 2, 0.3999999999999999),
        (5, 5, 2, 1.0),
        (10, 3, 5, 0.9166666666666667),
        (4, 2, 2, 0.8333333333333334),
        (2, 1, 2, 1.0),
    ],
)
def test_pass_at_k_for_task(n: int, c: int, k: int, expected: float) -> None:
    assert pass_at_k_for_task(n, c, k) == expected


def _trials(task: str, values: list, *, agent: str = "agent", source: str = "suite") -> list[Trial]:
    out: list[Trial] = []
    for value in values:
        rewards = None if value is None else {"reward": value}
        out.append(Trial(task_name=task, rewards=rewards, agent_name=agent, source=source))
    return out


def test_pass_at_k_default_single_trial_is_empty() -> None:
    # --n-attempts 1 => min_trials_per_task == 1 => k_values == [] => {}
    assert compute_pass_at_k_for_trials(_trials("t", [1])) == {}


def test_pass_at_k_binary_group() -> None:
    # one task, 5 trials, 1 success => {2: 0.4, 4: ..., 5: 0.0}
    trials = _trials("t", [1, 0, 0, 0, 0])
    result = compute_pass_at_k_for_trials(trials)
    assert result[2] == 0.3999999999999999
    assert set(result) == {2, 4, 5}


def test_pass_at_k_none_counts_as_failure() -> None:
    trials = _trials("t", [1, None])
    # 2 trials, 1 success, k=2 => n-c=1 < 2 => 1.0
    assert compute_pass_at_k_for_trials(trials) == {2: 1.0}


def test_pass_at_k_disabled_for_multi_metric() -> None:
    trials = [
        Trial(task_name="t", rewards={"correctness": 1, "speed": 0}, agent_name="a", source="s"),
        Trial(task_name="t", rewards={"correctness": 1, "speed": 1}, agent_name="a", source="s"),
    ]
    assert compute_pass_at_k_for_trials(trials) == {}


def test_pass_at_k_disabled_for_fractional() -> None:
    trials = _trials("t", [0.5, 1])
    assert compute_pass_at_k_for_trials(trials) == {}


def test_pass_at_k_by_evals_groups_and_filters_empty() -> None:
    # single-trial group => empty pass@k => filtered out entirely
    trials = _trials("t", [1], agent="agent", source="suite")
    assert compute_pass_at_k_by_evals(trials) == {}


def test_pass_at_k_by_evals_emits_for_binary_multi_trial() -> None:
    trials = _trials("t", [1, 0], agent="agent", source="suite")
    out = compute_pass_at_k_by_evals(trials)
    assert out == {"agent__suite": {2: 1.0}}


def test_format_agent_evals_key() -> None:
    assert format_agent_evals_key("agent", None, "suite") == "agent__suite"
    assert format_agent_evals_key("agent", "gpt", "suite") == "agent__gpt__suite"


# ---------------------------------------------------------------------------
# 8. Outcome mapping — end-to-end (task-2-reward-edgecases.txt:5-12)
#    Mirrors runner.py:1399-1438 exactly.
# ---------------------------------------------------------------------------


def _result_data(metrics: list[dict], *, n_total: int, n_completed: int, n_errored: int) -> dict:
    return {
        "n_total_trials": n_total,
        "stats": {
            "n_completed_trials": n_completed,
            "n_errored_trials": n_errored,
            "evals": {"agent__suite": {"metrics": metrics}},
        },
    }


@pytest.mark.parametrize(
    "metrics, n_total, n_completed, n_errored, status, score, resolved, total",
    [
        # all pass (3/3)
        ([{"mean": 1.0}], 3, 3, 0, "completed", 1.0, 3, 3),
        # 2 of 3 pass
        ([{"mean": 0.6666666666666666}], 3, 3, 0, "completed", 0.6666666666666666, 2, 3),
        # all fail (0/3) clean run => still completed
        ([{"mean": 0.0}], 3, 3, 0, "completed", 0.0, 0, 3),
        # clean run score 0 single trial => completed
        ([{"mean": 0.0}], 1, 1, 0, "completed", 0.0, 0, 1),
        # 1 errored trial => failed
        ([{"mean": 1.0}], 2, 1, 1, "failed", 1.0, 2, 2),
        # None reward counts as 0
        ([{"mean": 0.3333333333333333}], 3, 3, 0, "completed", 0.3333333333333333, 1, 3),
        # fractional 0.5
        ([{"mean": 0.5}], 2, 2, 0, "completed", 0.5, 1, 2),
    ],
)
def test_outcome_mapping_end_to_end(
    metrics, n_total, n_completed, n_errored, status, score, resolved, total
) -> None:
    data = _result_data(metrics, n_total=n_total, n_completed=n_completed, n_errored=n_errored)
    outcome = derive_outcome_from_result_data(data)
    assert outcome["status"] == status
    assert reward_values_equal(outcome["score"], score)
    assert outcome["resolved"] == resolved
    assert outcome["total"] == total
    assert outcome["reason_code"] is None


# ---------------------------------------------------------------------------
# 9. Banker's rounding in resolved (task-2-reward-edgecases.txt:14-21)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score, total, resolved",
    [
        (0.5, 1, 0),
        (1.5, 1, 2),
        (2.5, 1, 2),
        (0.5, 3, 2),
        (0.166666666, 3, 0),
        (0.833333, 6, 5),
        (0.5, 2, 1),
    ],
)
def test_bankers_rounding(score: float, total: int, resolved: int) -> None:
    data = {
        "n_total_trials": total,
        "stats": {
            "n_completed_trials": total,
            "n_errored_trials": 0,
            "evals": {"agent__suite": {"metrics": [{"mean": score}]}},
        },
    }
    assert derive_outcome_from_result_data(data)["resolved"] == resolved


# ---------------------------------------------------------------------------
# 10. Multi-metric flatten (task-2-reward-edgecases.txt:23-25)
# ---------------------------------------------------------------------------


def test_multi_metric_runner_flattens_each_value_as_sample() -> None:
    # Mean.compute(MULTI) -> {'correctness': 0.5, 'speed': 0.75}; no 'mean' key
    metric = Mean().compute(MULTI)
    data = _result_data([metric], n_total=2, n_completed=2, n_errored=0)
    outcome = derive_outcome_from_result_data(data)
    assert reward_values_equal(outcome["score"], 0.625)
    assert outcome["resolved"] == 1
    assert outcome["status"] == "completed"


# ---------------------------------------------------------------------------
# 11. reason_code: missing file & malformed (task-2-reward-edgecases.txt:27-29)
# ---------------------------------------------------------------------------


def test_outcome_result_missing(tmp_path: Path) -> None:
    outcome = derive_outcome(tmp_path / "does_not_exist.json")
    assert outcome == {
        "status": "failed",
        "score": 0.0,
        "resolved": 0,
        "total": 0,
        "reason_code": "harbor_result_missing",
    }


def test_outcome_result_malformed_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("{not json")
    assert derive_outcome(path)["reason_code"] == "harbor_result_malformed"


def test_outcome_result_malformed_non_floatable_mean(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {
                    "n_completed_trials": 1,
                    "n_errored_trials": 0,
                    "evals": {"agent__suite": {"metrics": [{"mean": "not-a-number"}]}},
                },
            }
        )
    )
    assert derive_outcome(path)["reason_code"] == "harbor_result_malformed"


def test_outcome_from_file_clean(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(_result_data([{"mean": 1.0}], n_total=3, n_completed=3, n_errored=0))
    )
    outcome = derive_outcome(path)
    assert outcome == {
        "status": "completed",
        "score": 1.0,
        "resolved": 3,
        "total": 3,
        "reason_code": None,
    }


def test_outcome_total_falls_back_to_completed_plus_errored() -> None:
    data = {
        "n_total_trials": 0,
        "stats": {
            "n_completed_trials": 2,
            "n_errored_trials": 1,
            "evals": {"agent__suite": {"metrics": [{"mean": 1.0}]}},
        },
    }
    assert derive_outcome_from_result_data(data)["total"] == 3


def test_outcome_empty_metric_values_keeps_score_zero() -> None:
    data = {
        "n_total_trials": 2,
        "stats": {"n_completed_trials": 2, "n_errored_trials": 0, "evals": {}},
    }
    outcome = derive_outcome_from_result_data(data)
    assert outcome["score"] == 0.0
    assert outcome["status"] == "completed"


# ---------------------------------------------------------------------------
# 12. derive_outcome_from_metrics (in-memory entry for Task 14/22)
# ---------------------------------------------------------------------------


def test_derive_outcome_from_metrics_matches_result_data() -> None:
    evals_metrics = {"agent__suite": [{"mean": 0.6666666666666666}]}
    outcome = derive_outcome_from_metrics(
        evals_metrics, n_total_trials=3, n_completed_trials=3, n_errored_trials=0
    )
    assert outcome["status"] == "completed"
    assert reward_values_equal(outcome["score"], 0.6666666666666666)
    assert outcome["resolved"] == 2
    assert outcome["total"] == 3
    assert outcome["reason_code"] is None


# ---------------------------------------------------------------------------
# 13. Precision spec / comparator (docs/reward-semantics.md §6)
# ---------------------------------------------------------------------------


def test_reward_values_equal_exact() -> None:
    assert reward_values_equal(0.6666666666666666, 0.6666666666666666)
    assert not reward_values_equal(0.6666666666666666, 0.6666666666666667)
    assert reward_values_equal(1, 1.0)  # int/float by value
    assert reward_values_equal(0.0, 0.0)


def test_reward_values_equal_nan_is_special_cased() -> None:
    nan = float("nan")
    assert reward_values_equal(nan, nan)  # nan == nan must be True here
    assert not reward_values_equal(nan, 0.0)
    assert not reward_values_equal(0.0, nan)


def test_reward_values_equal_inf() -> None:
    assert reward_values_equal(math.inf, math.inf)
    assert not reward_values_equal(math.inf, -math.inf)


def test_reward_parity_equal_nested() -> None:
    a = {"correctness": 0.5, "speed": float("nan")}
    b = {"correctness": 0.5, "speed": float("nan")}
    assert reward_parity_equal(a, b)
    assert reward_parity_equal([{"mean": 1.0}], [{"mean": 1.0}])
    assert not reward_parity_equal({"mean": 1.0}, {"mean": 0.0})
    assert not reward_parity_equal({"mean": 1.0}, {"other": 1.0})


def test_floats_bit_identical() -> None:
    # 1/3 reproduced via CPython float division is bit-identical.
    assert floats_bit_identical(1 / 3, 0.3333333333333333)
    assert floats_bit_identical(float("nan"), float("nan"))
    assert not floats_bit_identical(0.3333333333333333, 0.33333333333333337)


def test_mean_reproduces_harbor_bit_exact() -> None:
    # task-2-reward-edgecases.txt:32-33 — non-terminating binary, must be ε=0.
    assert Mean().compute([{"reward": 1}, {"reward": 0}, {"reward": 0}]) == {
        "mean": 0.3333333333333333
    }
    assert Mean().compute([{"reward": 1}, {"reward": 1}, {"reward": 0}]) == {
        "mean": 0.6666666666666666
    }
    assert (1 / 3).hex() == "0x1.5555555555555p-2"


def test_trial_order_preserved_in_aggregation() -> None:
    # sum is left-to-right; reordering would diverge in the last ULP. The
    # harbor-exact double (verified vs harbor 0.13.1) is 0.19999999999999998.
    forward = Mean().compute([{"reward": 0.1}, {"reward": 0.2}, {"reward": 0.3}])
    assert forward == {"mean": 0.19999999999999998}
