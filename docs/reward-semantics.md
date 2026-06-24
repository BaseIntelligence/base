# Harbor 0.13.1 Reward Semantics Reference

Authoritative specification of **exactly** how stock `harbor==0.13.1` turns the
float in `/logs/verifier/reward.txt` into a reward, aggregates it across
metrics and trials (pass@k), and how agent-challenge maps that into
`{status, reason_code, resolved}`.

This is a **reference spec**. It documents existing harbor + agent-challenge
behavior so an independent runner can reproduce it **byte-for-byte**. It does
**not** propose changes and must not be used to alter thresholds or harbor
behavior.

- Ground truth = the `harbor==0.13.1` PyPI wheel. The runner image
  `ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1` ships prebuilt
  Harbor tooling (a `python:3.12-slim` base with `harbor==0.13.1`), so the wheel
  is the authority; every claim below was executed against the real installed
  harbor code.
- Consumer = agent-challenge `src/agent_challenge/evaluation/runner.py` and
  `terminal_bench.py`.

---

## 0. End-to-end pipeline (one trial)

```
reward.txt / reward.json   (written inside the task container by the verifier)
        │   harbor.verifier.verifier.Verifier.verify()            (per trial)
        ▼
VerifierResult.rewards : dict[str, float|int] | None
        │   harbor.job.Job  (per (agent,model,dataset) "evals" group)
        ▼
JobStats.evals[evals_key].metrics : list[dict]      ← metric aggregation
JobStats.evals[evals_key].pass_at_k : dict[int,float]  ← pass@k (optional)
        │   serialized to job result JSON (model_dump_json)
        ▼
agent-challenge runner.py inline python  → BASE_BENCHMARK_RESULT={...}
        ▼
{status, score, resolved, total, reason_code}
```

For terminal-bench tasks the in-container verifier writes a **binary** reward:
`harbor/mappers/terminal_bench.py:36-44` appends a shell suffix that does
`echo 1 > /logs/verifier/reward.txt` on test exit 0, else `echo 0`. So in
practice tbench `reward.txt` is exactly `1` or `0` — but the parser accepts any
float, and the runner MUST handle the general case.

---

## 1. Reward file parsing — `harbor/verifier/verifier.py`

### 1.1 File precedence (verifier.py:209-218)

```python
if self.trial_paths.reward_json_path.exists():     # reward.json
    rewards = self._parse_reward_json()
elif self.trial_paths.reward_text_path.exists():   # reward.txt
    rewards = self._parse_reward_text()
else:
    raise RewardFileNotFoundError(...)
return VerifierResult(rewards=rewards)
```

- **`reward.json` WINS over `reward.txt`** when both exist (checked first by
  `.exists()`). Verified: both present → json value returned, txt ignored.
- Paths come from `trial_paths`: `reward_json_path` = `.../reward.json`,
  `reward_text_path` = `.../reward.txt` (under the trial's verifier logs dir,
  i.e. the container's `/logs/verifier/`).

### 1.2 Text parse — `_parse_reward_text` (verifier.py:61-74)

```python
if reward_text_path.stat().st_size == 0:
    raise RewardFileEmptyError(...)
try:
    return {"reward": float(reward_text_path.read_text())}
except ValueError:
    raise VerifierOutputParseError(...)
```

- Empty key is the literal string **`"reward"`**. Value = `float(read_text())`.
- **Emptiness test is `st_size == 0`** — a byte-size check, NOT a
  `.strip()` check. A file containing only whitespace (e.g. `"  "`) is
  NOT empty; it goes to `float("  ")` which raises `ValueError`
  → `VerifierOutputParseError`.
- `float()` is Python's builtin and tolerates surrounding whitespace/newlines.

Observed `float(read_text())` behavior (the runner MUST match exactly):

| reward.txt bytes | result | rewards dict / error |
|---|---|---|
| `1` | 1.0 | `{"reward": 1.0}` |
| `0` | 0.0 | `{"reward": 0.0}` |
| `1.0` / `1\n` / ` 1 \n` | 1.0 | `{"reward": 1.0}` |
| `0.5` | 0.5 | `{"reward": 0.5}` |
| `1e0` | 1.0 | `{"reward": 1.0}` |
| `-1` | -1.0 | `{"reward": -1.0}` (negative IS accepted) |
| `nan` | nan | `{"reward": nan}` (accepted; poisons aggregation) |
| `inf` | inf | `{"reward": inf}` (accepted) |
| `` (0 bytes) | — | **`RewardFileEmptyError`** |
| `  ` (whitespace only) | — | **`VerifierOutputParseError`** |
| `pass`, `True`, `1,0` | — | **`VerifierOutputParseError`** |

### 1.3 JSON parse — `_parse_reward_json` (verifier.py:76-87)

```python
if reward_json_path.stat().st_size == 0:
    raise RewardFileEmptyError(...)
try:
    return json.loads(reward_json_path.read_text())   # returned VERBATIM
except (json.JSONDecodeError, ...):
    raise VerifierOutputParseError(...)
```

- The parsed JSON object is the rewards dict **verbatim** — keys are arbitrary
  metric names. e.g. `{"correctness": 1, "speed": 0.5}` → that exact dict.
  This is the multi-metric entry point.
- Same `st_size == 0` → `RewardFileEmptyError`; bad JSON → `VerifierOutputParseError`.
- harbor does NOT validate JSON value types here; downstream metric/pass@k code
  is what enforces numeric/binary constraints.

### 1.4 Exception → reason_code (agent-challenge `terminal_bench.py`)

harbor raises the exceptions; agent-challenge classifies them by **lowercased
substring** in `normalize_terminal_bench_reason_code` (terminal_bench.py:596-601):

```python
if "reward" in lowered and "missing" in lowered:                       → "harbor_reward_missing"
if "reward" in lowered and "empty"   in lowered:                       → "harbor_reward_empty"
if "reward" in lowered and ("parse" in lowered or "malformed"):        → "harbor_reward_parse_error"
```

| harbor exception | message contains | normalized reason_code |
|---|---|---|
| `RewardFileNotFoundError` | "No reward file found" | `harbor_reward_missing` |
| `RewardFileEmptyError` | "Reward file is empty" | `harbor_reward_empty` |
| `VerifierOutputParseError` | "...parse..." | `harbor_reward_parse_error` |

Valid reason-code set: terminal_bench.py:55-68 (incl. `harbor_reward_empty`,
`harbor_reward_missing`, `harbor_reward_parse_error`, `harbor_result_missing`,
`harbor_result_malformed`, ...). An independent runner MUST emit codes from this set.

---

## 2. VerifierResult model

`harbor/models/verifier/result.py` (entire file):

```python
class VerifierResult(BaseModel):
    rewards: dict[str, float | int] | None = None
```

- `rewards is None` ⇒ no verifier result (trial errored before/within verify).
  This is distinct from `{"reward": 0.0}` (a real failing reward).

---

## 3. Metric aggregation — `harbor/metrics/`

### 3.1 Default metric = `Mean` (job.py:441-460)

`Job._resolve_metrics` builds `metrics: dict[dataset_name → list[BaseMetric]]`.
Any dataset with no explicitly-configured metric gets **exactly one `Mean()`**
appended (job.py:456-458). Adhoc runs use key `"adhoc"`. So for terminal-bench
the active metric list is `[Mean()]` unless the dataset ships a `metric.py`
(package/registry datasets can add `UvScript` + configured metrics; tbench-2-1
does not — default Mean applies).

### 3.2 `aggregate_reward_dicts` — the core function (metrics/base.py:16-37)

```python
reward_keys = sorted({k for r in rewards if r is not None for k in r})

if len(reward_keys) <= 1:                          # ≤1 distinct reward key
    values = [0 if r is None else next(iter(r.values()), 0) for r in rewards]
    return {metric_name: aggregate(values)}        # OUTPUT KEYED BY METRIC NAME

return {                                            # >1 distinct reward key
    key: aggregate([0 if r is None else r.get(key, 0) for r in rewards])
    for key in reward_keys                          # OUTPUT KEYED BY REWARD KEY
}
```

`metric_name` / `aggregate` per metric:

| metric | metric_name | aggregate |
|---|---|---|
| `Mean` | `"mean"` | `lambda vs: sum(vs)/len(vs)` |
| `Max`  | `"max"`  | `max` |
| `Min`  | `"min"`  | `min` |
| `Sum`  | `"sum"`  | `sum` |

**CRITICAL output-shape rule (the #1 interop gotcha):**

- **Single reward key** (the tbench norm — every trial has only `"reward"`):
  output dict is keyed by the **metric name**, and the reward key is discarded.
  - `Mean.compute([{"reward":1},{"reward":0},{"reward":1}])` → `{"mean": 0.666…}`
  - `Mean.compute([{"reward":1}])` → `{"mean": 1.0}`
- **Multiple reward keys** (multi-metric, via reward.json):
  output dict is keyed by the **reward keys**, and the metric name is **LOST**.
  - `Mean.compute([{"correctness":1,"speed":0.5},{"correctness":0,"speed":1.0}])`
    → `{"correctness": 0.5, "speed": 0.75}`  (NO `"mean"` key!)
  - `Max`  → `{"correctness": 1, "speed": 1.0}`
  - `Min`  → `{"correctness": 0, "speed": 0.5}`
  - `Sum`  → `{"correctness": 1, "speed": 1.5}`

**None / missing handling inside aggregation:**
- A trial whose `rewards is None` contributes **0** to every value list (both
  branches). `Mean([{"reward":1}, None, None])` → `{"mean": 0.333…}`.
- In the multi-key branch a trial missing a particular key contributes
  `r.get(key, 0)` = **0** for that key.
- `next(iter(r.values()), 0)` in the single-key branch takes the trial's one
  value (or 0 if the dict is empty `{}`).
- **Empty trial list** (`[]`) → `Mean` does `sum([])/len([])` →
  **`ZeroDivisionError`**. (Reachable only if an evals group has metrics
  computed over zero rewards; harbor's live path guards this with
  `if not rewards_list: metrics = []` at job.py:402-403, but the final
  assembly at job.py:758-759 calls `metric.compute(rewards)` unconditionally —
  a group always has ≥1 entry there since it was created from a trial.)

### 3.3 Where metrics land in the result — `harbor/job.py`

Final assembly (job.py:748-766, the authoritative path written to result JSON):

```python
final_stats = JobStats.from_trial_results(combined_trial_results,
                                          n_total_trials=len(self._trial_configs),
                                          n_retries=self._n_retries)
for evals_key, rewards in final_rewards.items():     # rewards: list[dict|None] per trial
    dataset_name = evals_key.split("__")[-1]
    for metric in self._metrics[dataset_name]:
        final_stats.evals[evals_key].metrics.append(metric.compute(rewards))
for evals_key, pass_at_k in compute_pass_at_k_by_evals(combined_trial_results).items():
    final_stats.evals[evals_key].pass_at_k = pass_at_k
```

- `evals_key` = `format_agent_evals_key(agent, model, dataset)`
  (`models/job/result.py:59-66`): `"{agent}__{model}__{dataset}"` if model set,
  else `"{agent}__{dataset}"`. `dataset_name = evals_key.split("__")[-1]`.
  `source or "adhoc"` is the dataset name when no source.
- `metrics` is a **list, one entry per configured metric** (default → exactly
  one dict from `Mean`). Each entry is the dict from §3.2.
- `JobStats.increment` (result.py:129-169) separately builds, per evals group:
  `n_trials` (trials with non-None rewards), `n_errors`/`n_errored_trials`
  (trials with `exception_info`; `CancelledError` also bumps
  `n_cancelled_trials`), `reward_stats[key][value] = [trial_name,…]`, and token
  totals. These do NOT feed `score`; the agent-challenge consumer reads only
  `n_total_trials`, `n_completed_trials`, `n_errored_trials`, and
  `evals[*].metrics`.

---

## 4. pass@k — `harbor/utils/pass_at_k.py`

Computed by `compute_pass_at_k_by_evals(trial_results)` and stored at
`evals[evals_key].pass_at_k : dict[int, float]`.

### 4.1 Eligibility gate (pass_at_k.py:32-53) — STRICT binary 0/1

For each trial, the reward source is **`trial_result.verifier_result.rewards`**
(NOT JobStats). Per trial:
- `rewards is None` → that task gets a success of `0` (counts as a trial).
- `len(rewards) != 1` → **return `{}`** (pass@k disabled for the WHOLE group;
  i.e. multi-metric tasks never produce pass@k).
- the single value not `int|float` → **return `{}`**.
- the single value **not in `(0, 1)`** → **return `{}`** (fractional/partial
  rewards like 0.5 disable pass@k entirely).
- else success = `int(reward_value)`.

So pass@k is emitted **only** when every trial in the group has exactly one
reward key whose value is strictly 0 or 1.

### 4.2 k selection (pass_at_k.py:71-84)

`min_trials_per_task = min(len(successes) per task)`, then `_eligible_k_values`:
powers of two (2,4,8,16,…) **and** multiples of five (5,10,15,20,…), each ≤
`min_trials`, sorted/deduped. **k starts at 2 — pass@1 is NEVER computed.**

| min trials/task | k values |
|---|---|
| 1 | `[]` (empty — with `--n-attempts 1`, pass_at_k is `{}`) |
| 2 | `[2]` |
| 3 | `[2]` |
| 4 | `[2, 4]` |
| 5 | `[2, 4, 5]` |
| 8 | `[2, 4, 5, 8]` |
| 10 | `[2, 4, 5, 8, 10]` |
| 16 | `[2, 4, 5, 8, 10, 15, 16]` |
| 20 | `[2, 4, 5, 8, 10, 15, 16, 20]` |

### 4.3 Estimator (pass_at_k.py:61-94) — standard unbiased pass@k

Per task with `n` trials and `c` successes:
```python
def _pass_at_k_for_task(n, c, k):
    if n - c < k:            # enough successes that any k-subset hits one
        return 1.0
    product = 1.0
    for i in range(k):
        product *= (n - c - i) / (n - i)
    return 1.0 - product      # 1 - C(n-c,k)/C(n,k)
```
Group pass@k = **mean over tasks** of `_pass_at_k_for_task` (sum/len(tasks)).

Spot checks (the runner MUST match): `(n=5,c=0,k=2)=0.0`,
`(n=5,c=1,k=2)=0.4`, `(n=5,c=5,k=2)=1.0`, `(n=10,c=3,k=5)=0.9166̄`,
`(n=4,c=2,k=2)=0.8333̄`, `(n=2,c=1,k=2)=1.0`.

### 4.4 Multi-trial CLI flag

`--n-attempts` / `-k` (default **1**) = number of attempts per trial
(`harbor run --help`; `cli/jobs.py:337` → `JobConfig.n_attempts`). There is **no
`--n-trials` flag.** Concurrency is `--n-concurrent`/`-n` (default 4);
`--max-retries`/`-r` (default 0) is retry-on-exception, not extra attempts.
With the default `--n-attempts 1`, `min_trials_per_task == 1` ⇒ `k_values == []`
⇒ `pass_at_k == {}`. **pass@k only appears when the operator passes
`-k ≥ 2`.** agent-challenge's runner.py score path does NOT read `pass_at_k` at
all (it reads `metrics`), so pass@k is observability-only for the current
consumer; an independent runner must still reproduce it for result-JSON fidelity.

### 4.5 Multi-step single-trial reward (separate from pass@k)

`harbor/trial/multi_step.py:196-230`, `models/task/config.py:547-577`
(`MultiStepRewardStrategy`): for a multi-**step** task the trial-level
`VerifierResult` is derived from per-step results BEFORE any of the above:
- strategy `FINAL` → last step's `verifier_result` verbatim.
- strategy `MEAN` (default when unset on a multi-step task) → per-key mean
  across steps that have a verifier_result (missing key = 0, steps without a
  verifier_result excluded from the denominator).
This only affects how a single trial's `rewards` dict is formed; §1–4 then apply
unchanged. Single-step tbench tasks never hit this.

---

## 5. agent-challenge outcome mapping — `runner.py:1399-1438`

The runner injects an inline python block that reads the harbor job-result JSON
(`plan.result_path`) and prints `BASE_BENCHMARK_RESULT={json}`:

```python
summary = {"status":"failed","score":0.0,"resolved":0,"total":0,
           "reason_code":"harbor_result_missing"}
if result_path.exists():
    try:
        data = json.loads(result_path.read_text())
        stats = data.get("stats", {})
        total = int(data.get("n_total_trials") or 0)
        completed = int(stats.get("n_completed_trials") or 0)
        errored   = int(stats.get("n_errored_trials") or 0)
        score = 0.0
        metric_values = []
        for eval_stats in stats.get("evals", {}).values():
            for metric in eval_stats.get("metrics", []):
                if "mean" in metric:
                    metric_values.append(float(metric["mean"]))      # single-key path
                else:
                    metric_values.extend(float(v) for v in metric.values())  # multi-key path
        if metric_values:
            score = sum(metric_values) / len(metric_values)
        summary.update({
            "status": "completed" if errored == 0 else "failed",
            "score": score,
            "resolved": round(score * total),
            "total": total or completed + errored,
            "reason_code": None,
        })
    except Exception:
        summary["reason_code"] = "harbor_result_malformed"
```

Exact rules an independent runner's result JSON + any wrapper MUST satisfy:

1. **score** = arithmetic mean of a flat `metric_values` list gathered across
   ALL evals groups and ALL metric entries:
   - if a metric dict has a `"mean"` key → push `float(metric["mean"])` (the
     normal single-reward-key Mean case from §3.2).
   - else → push `float(v)` for **every** value in the dict (covers multi-metric
     `{"correctness":…,"speed":…}` AND non-Mean single-key metrics whose key is
     `"max"`/`"min"`/`"sum"`).
   - **Consequence:** a multi-metric task contributes each metric value as a
     SEPARATE sample to the average — they are NOT first combined per task.
   - Empty `metric_values` ⇒ `score` stays `0.0`.
2. **status** = `"completed"` iff `stats.n_errored_trials == 0`, else
   `"failed"`. NOTE: status is driven by error count, NOT by score — a clean run
   with score 0.0 is still `"completed"`.
3. **resolved** = `round(score * total)` where `total = n_total_trials`.
   Python `round()` is **banker's rounding** (round-half-to-even):
   `round(0.5)=0`, `round(1.5)=2`, `round(2.5)=2`. The runner MUST use the same.
4. **total** = `n_total_trials` if truthy else `n_completed_trials +
   n_errored_trials`.
5. **reason_code**: `None` on a clean parse; `"harbor_result_missing"` if the
   result JSON file is absent; `"harbor_result_malformed"` if any exception is
   raised while parsing/aggregating.
6. Output line is literally `BASE_BENCHMARK_RESULT=` + `json.dumps(summary,
   sort_keys=True)`, emitted to stdout; `exit $status` preserves the harbor
   command's exit code.

Downstream, `_normalize_terminal_bench_result` (runner.py:1443+) parses that
line and `normalize_terminal_bench_reason_code` canonicalizes any reason string.

---

## 6. Floating-point determinism (for an ε=0 / exact-match precision spec)

- **Reward read**: `float(text)`. For the tbench-relevant inputs `0` and `1`
  this is **exact** (0.0, 1.0 are exactly representable). General floats follow
  IEEE-754 round-to-nearest from CPython's `float()` and are deterministic for a
  given input string.
- **Mean**: `sum(values) / len(values)`. `sum` over a Python list is
  **left-to-right sequential** float addition (deterministic order given a
  deterministic trial order). Division can be non-terminating in binary
  (e.g. `2/3 = 0.6666666666666666`, `1/3 = 0.3333333333333333`) — these are the
  exact CPython double results and reproduce bit-for-bit **only if** the runner
  uses the same algorithm (Python `sum`/`/`, same operand order). A runner that
  reorders trials, uses `math.fsum`, NumPy pairwise summation, or `Decimal` will
  diverge in the last ULP → an ε=0 comparator would FAIL.
- **Trial order**: `combined_trial_results` order determines both `sum` operand
  order and (via `task_successes` insertion) pass@k task iteration. An
  independent runner MUST preserve harbor's trial ordering to be bit-identical.
- **Max/Min/Sum**: exact (no division).
- **pass@k**: `product *= (n-c-i)/(n-i)` sequential float mult/div, then
  `sum(...)/len(tasks)`. Deterministic but ULP-sensitive to operand order.
- **`nan`/`inf`**: a `nan` reward propagates through `sum`/`/` to a `nan` metric;
  `float("nan")` comparisons are always false, so an ε comparator must treat
  `nan==nan` specially. `inf` propagates to `inf`/`nan`.
- **Recommended precision spec for parity:** reproduce harbor's exact
  algorithm (CPython `float()`, list `sum`, `/`, same trial order) and assert
  **exact equality** on `0/1`-derived rewards and Mean results; for the general
  float case, since the algorithm is identical, exact (ε=0) equality is
  achievable and is the correct target. Only introduce a tiny ε (e.g. 1e-12) if
  the runner deliberately diverges from CPython's summation, which it should not.

---

## 7. Reproduction checklist for an independent runner

To be byte-compatible with stock harbor 0.13.1 + agent-challenge:

1. Parse reward with json-over-txt precedence; `st_size==0`→empty error;
   `float(read_text())` for txt (key `"reward"`); `json.loads` verbatim for json;
   raise the three error classes with messages containing
   missing/empty/parse so the reason-code substring matcher works (§1).
2. `rewards is None` semantics distinct from `{"reward":0.0}` (§2).
3. Default metric exactly `[Mean()]`; aggregate via the ≤1-key vs >1-key branch
   rule, None→0, missing-key→0; output keyed by metric-name (single) or
   reward-key (multi) (§3).
4. Emit `evals[key].metrics` as a list (one dict per metric) and the JobStats
   counters `n_total_trials`/`n_completed_trials`/`n_errored_trials` (§3.3).
5. pass@k only for strictly-binary single-key groups; k from
   powers-of-2 ∪ multiples-of-5, ≥2, ≤ min-trials; unbiased estimator; mean over
   tasks; `--n-attempts`/`-k` controls trial count (§4).
6. score = flat mean of metric values (`"mean"` key else all values);
   status by `n_errored_trials==0`; `resolved = round(score*total)` with
   banker's rounding; reason_code None/missing/malformed (§5).
7. Preserve trial order and use CPython `float`/`sum`/`/` for bit-exact rewards
   (§6).

---

## Source references (harbor 0.13.1 wheel + agent-challenge)

- `harbor/verifier/verifier.py:22-87, 198-220` — error classes, parse, precedence
- `harbor/models/verifier/result.py` — `VerifierResult`
- `harbor/metrics/base.py:16-37` — `aggregate_reward_dicts`
- `harbor/metrics/{mean,max,min,sum}.py` — metric names + aggregate fns
- `harbor/job.py:441-460` (default Mean), `:400-407` (live refresh),
  `:748-766` (final assembly + pass@k)
- `harbor/models/job/result.py:15-169` — `AgentDatasetStats`, `JobStats`,
  `format_agent_evals_key`, `increment`
- `harbor/utils/pass_at_k.py` — full pass@k
- `harbor/trial/multi_step.py:196-247`, `harbor/models/task/config.py:547-577`
  — multi-step reward strategy
- `harbor/mappers/terminal_bench.py:36-44` — binary reward.txt writer
- `cli/jobs.py:337` (`--n-attempts`), `:416` (`--n-concurrent`); `harbor run --help`
- agent-challenge `evaluation/runner.py:1399-1438` — outcome mapping
- agent-challenge `evaluation/terminal_bench.py:55-68` (reason-code set),
  `:566-604` (normalizer)
