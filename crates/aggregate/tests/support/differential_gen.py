"""Emit randomized aggregation cases plus the authoritative Python output as JSON.

Consumed by ``tests/differential.rs`` (an ``#[ignore]``d test, because it needs the
Python venv). Usage::

    /root/prism-compute-plane/base/.venv/bin/python differential_gen.py <count> <seed>

The BASE package must be importable; ``BASE_SRC`` may point at its ``src`` directory.
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.environ.get("BASE_SRC", "/root/prism-compute-plane/base/src"))

from base.master.aggregator import (  # noqa: E402
    CHAIN_U16_MAX,
    ZeroMinerWeightError,
    aggregate_challenge_weights,
)
from base.schemas.weights import ChallengeWeightsResult  # noqa: E402

WEIRD = [0.0, -0.0, -1.0, -1e-300, float("nan"), float("inf"), float("-inf"), 1e-320]


def random_weight(rng: random.Random) -> float:
    kind = rng.random()
    if kind < 0.12:
        return rng.choice(WEIRD)
    if kind < 0.3:
        return rng.random() * 10 ** rng.randint(-12, 12)
    if kind < 0.5:
        return float(rng.randint(0, 1000))
    return rng.random()


def random_case(rng: random.Random, index: int) -> dict:
    n_hotkeys = rng.randint(1, 12)
    hotkeys = [f"hk{i:02d}" for i in range(n_hotkeys)]
    rng.shuffle(hotkeys)

    mapping_keys = [h for h in hotkeys if rng.random() < 0.85]
    rng.shuffle(mapping_keys)
    hotkey_to_uid = {h: rng.randint(0, 40) for h in mapping_keys}

    n_challenges = rng.randint(0, 4)
    results = []
    for c in range(n_challenges):
        members = [h for h in hotkeys if rng.random() < 0.7]
        rng.shuffle(members)
        results.append(
            {
                "slug": f"ch{rng.randint(0, n_challenges)}" if rng.random() < 0.15 else f"ch{c}",
                "emission_percent": rng.choice(
                    [
                        rng.random() * 120.0,
                        float(rng.randint(-20, 120)),
                        0.0,
                        100.0 / max(1, n_challenges),
                    ]
                ),
                "weights": {h: random_weight(rng) for h in members},
                "ok": rng.random() < 0.85,
            }
        )

    kwargs = {}
    if rng.random() < 0.5:
        kwargs["min_allowed_weights"] = rng.randint(1, 6)
    if rng.random() < 0.35:
        kwargs["max_weight_limit"] = rng.choice([CHAIN_U16_MAX, 40000, 20000, 9000, 3000])

    doc = {
        "name": f"random_{index:04d}",
        "inputs": {
            "challenge_results": results,
            "hotkey_to_uid": hotkey_to_uid,
            "kwargs": kwargs,
        },
    }
    try:
        out = aggregate_challenge_weights(
            [ChallengeWeightsResult(**r) for r in results], hotkey_to_uid, **kwargs
        )
    except ZeroMinerWeightError as exc:
        doc["python_error"] = str(exc)
        return doc

    doc["python_float_output"] = {
        "uids": out.uids,
        "weights": out.weights,
        "hotkey_weights": out.hotkey_weights,
    }
    doc["expected_vector"] = [
        [uid, round(w * CHAIN_U16_MAX)] for uid, w in zip(out.uids, out.weights)
    ]
    doc["chain_u16_sum"] = sum(v for _, v in doc["expected_vector"])
    return doc


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260101
    rng = random.Random(seed)
    cases = [random_case(rng, i) for i in range(count)]
    json.dump(
        {"python_version": sys.version, "seed": seed, "cases": cases},
        sys.stdout,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()
