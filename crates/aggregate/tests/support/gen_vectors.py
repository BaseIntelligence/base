"""Regenerate Rust characterization vectors from the authoritative Python aggregator."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.environ.get("BASE_SRC", "/root/prism-compute-plane/base/src"))

from base.master.aggregator import (  # noqa: E402
    CHAIN_U16_MAX,
    ZeroMinerWeightError,
    aggregate_challenge_weights,
)
from base.schemas.weights import ChallengeWeightsResult  # noqa: E402

SHA = "8249563774ee2e71c41ae2cfac182ff32aa35dd1"
OUT = os.environ.get(
    "VECTOR_OUT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vectors", "python", SHA),
)

SPEC = [
    "Per-challenge weights are cleaned (drop non-finite and <= 0) then normalized by their float sum.",
    "emission_percent is an ABSOLUTE share of 100; sum(shares) > 1.0 is scaled back to exactly 1.0.",
    "Any share not landing on a real miner (unallocated remainder, uid 0, unknown hotkey, empty challenge) burns to uid 0.",
    "No miner mass at all (<= 1e-12) falls back to build_zero_miner_weights: {0: 1.0}, padded with metagraph uids when min_allowed_weights or max_weight_limit demand it, else ZeroMinerWeightError.",
    "All summation is CPython >= 3.12 sum(), i.e. Neumaier compensated - NOT a naive fold.",
    "Map iteration follows Python dict insertion order; only the final vector is sorted by uid.",
    "round(w * 65535) is Python round(), i.e. round-half-to-EVEN. The u16 vector is NOT renormalized and may sum to 65534/65535/65536.",
    "BUNDLE_SPEC section 6 (u128 FIXED + Hamilton, no burn, empty vector on all-zero) describes a DIFFERENT algorithm that is not what chain.joinbase.ai serves.",
]

HEADER = {
    "authority": (
        "Python base.master.aggregator.aggregate_challenge_weights IS the authority for "
        "the served weight vector (chain.joinbase.ai). Rust crates/aggregate::python is a "
        "bit-for-bit port; on disagreement Python wins."
    ),
    "upstream_base_sha": SHA,
    "upstream_module": "base.master.aggregator.aggregate_challenge_weights",
    "python_float_path": True,
    "served_algorithm_specification": SPEC,
    "expected_vector_encoding": (
        "list of [uid:u16, weight_u16] where weight_u16 = round(float_weight * 65535) "
        "using Python round() semantics (round-half-to-EVEN)"
    ),
}


def cr(slug, pct, weights, ok=True):
    return {"slug": slug, "emission_percent": pct, "weights": weights, "ok": ok}


def run(case, results, hotkey_to_uid, **kwargs):
    models = [ChallengeWeightsResult(**r) for r in results]
    doc = {
        "header": dict(HEADER, case=case),
        "inputs": {
            "challenge_results": results,
            "hotkey_to_uid": hotkey_to_uid,
            "kwargs": kwargs,
        },
    }
    try:
        out = aggregate_challenge_weights(models, hotkey_to_uid, **kwargs)
    except ZeroMinerWeightError as exc:
        doc["python_error"] = str(exc)
        doc["chain_u16_max"] = CHAIN_U16_MAX
        return doc

    vec = [[uid, round(w * CHAIN_U16_MAX)] for uid, w in zip(out.uids, out.weights)]
    doc["python_float_output"] = {
        "uids": out.uids,
        "weights": out.weights,
        "hotkey_weights": out.hotkey_weights,
    }
    doc["expected_vector"] = vec
    doc["chain_u16_sum"] = sum(w for _, w in vec)
    doc["chain_u16_max"] = CHAIN_U16_MAX
    return doc


CASES = []


def add(name, results, mapping, **kwargs):
    CASES.append((name, run(name, results, mapping, **kwargs)))


# --- zero-miner fallback across min_allowed_weights -------------------------------
for n in (1, 2, 3):
    add(
        f"06_zero_miner_min_allowed_{n}",
        [cr("empty", 100.0, {})],
        {"validator": 0, "hkA": 5, "hkB": 7, "hkC": 9},
        min_allowed_weights=n,
    )

# --- max_weight_limit below 65535 forces extra padding uids -----------------------
add(
    "07_max_weight_limit_20000_pads_to_four",
    [cr("empty", 100.0, {})],
    {"validator": 0, "hkA": 1, "hkB": 2, "hkC": 3, "hkD": 4},
    min_allowed_weights=1,
    max_weight_limit=20000,
)
add(
    "08_max_weight_limit_9000_pads_to_eight",
    [cr("empty", 100.0, {})],
    {"validator": 0, **{f"hk{i}": i for i in range(1, 12)}},
    min_allowed_weights=1,
    max_weight_limit=9000,
)

# --- ZeroMinerWeightError: not enough candidate uids ------------------------------
add(
    "09_zero_miner_error_not_enough_uids",
    [cr("empty", 100.0, {})],
    {"validator": 0, "hkA": 5},
    min_allowed_weights=5,
)
add(
    "10_zero_miner_error_max_weight_limit_too_low",
    [cr("empty", 100.0, {})],
    {"validator": 0, "hkA": 5, "hkB": 6},
    min_allowed_weights=1,
    max_weight_limit=1000,
)

# --- over-allocation: sum(emission_percent) > 100 scales back, no burn ------------
add(
    "11_over_allocation_scaled_back",
    [
        cr("a", 80.0, {"ha": 1.0}),
        cr("b", 80.0, {"hb": 1.0}),
        cr("c", 20.0, {"hc": 1.0}),
    ],
    {"ha": 1, "hb": 2, "hc": 3},
)
add(
    "12_over_allocation_three_way_uneven",
    [
        cr("a", 70.0, {"ha": 2.0, "hb": 1.0}),
        cr("b", 45.0, {"hc": 1.0}),
    ],
    {"ha": 11, "hb": 12, "hc": 13},
)

# --- hotkey mapped to uid 0 is dropped; its mass burns ----------------------------
add(
    "13_hotkey_on_uid_zero_burns",
    [cr("c", 100.0, {"validator": 1.0, "hkA": 1.0, "hkB": 2.0})],
    {"validator": 0, "hkA": 4, "hkB": 6},
)

# --- hotkey absent from hotkey_to_uid is dropped; its mass burns ------------------
add(
    "14_unknown_hotkey_burns",
    [cr("c", 100.0, {"ghost": 3.0, "hkA": 1.0})],
    {"hkA": 4},
)

# --- _clean_weights filters negative / zero / NaN / infinite ----------------------
add(
    "15_clean_weights_filters_bad_values",
    [
        cr(
            "c",
            100.0,
            {
                "neg": -5.0,
                "zero": 0.0,
                "nan": float("nan"),
                "inf": float("inf"),
                "neginf": float("-inf"),
                "hkA": 1.0,
                "hkB": 3.0,
            },
        )
    ],
    {"neg": 1, "zero": 2, "nan": 3, "inf": 4, "neginf": 5, "hkA": 6, "hkB": 7},
)
add(
    "16_all_weights_invalid_falls_back_to_zero_miner",
    [cr("c", 100.0, {"neg": -1.0, "nan": float("nan")})],
    {"neg": 1, "nan": 2},
    min_allowed_weights=1,
)

# --- input-order fidelity: same data, several insertion orders --------------------
ORDER_W = [
    ("hk01", 0.1),
    ("hk02", 0.2),
    ("hk03", 0.30000000000000004),
    ("hk04", 1.0 / 3.0),
    ("hk05", 0.7),
    ("hk06", 1e-8),
    ("hk07", 1e8),
    ("hk08", 3.14159265358979),
    ("hk09", 2.718281828459045),
    ("hk10", 1.0 / 7.0),
]
ORDER_MAP = [(f"hk{i:02d}", i) for i in range(1, 11)]
ORDERS = {
    "17_order_fidelity_natural": (list(range(10)), list(range(10))),
    "18_order_fidelity_reversed": (list(range(9, -1, -1)), list(range(9, -1, -1))),
    "19_order_fidelity_shuffled": (
        [4, 0, 7, 2, 9, 1, 6, 3, 8, 5],
        [9, 3, 1, 8, 0, 6, 2, 7, 5, 4],
    ),
    "20_order_fidelity_weights_only_shuffled": (
        [7, 1, 9, 0, 5, 3, 8, 2, 6, 4],
        list(range(10)),
    ),
}
for name, (worder, morder) in ORDERS.items():
    add(
        name,
        [cr("c", 100.0, {ORDER_W[i][0]: ORDER_W[i][1] for i in worder})],
        {ORDER_MAP[i][0]: ORDER_MAP[i][1] for i in morder},
    )

# --- duplicate slug collapses in `frac` (last emission_percent wins) --------------
add(
    "21_duplicate_slug_last_emission_wins",
    [
        cr("dup", 10.0, {"ha": 1.0}),
        cr("dup", 40.0, {"hb": 1.0}),
    ],
    {"ha": 1, "hb": 2},
)

# --- negative / zero emission_percent ---------------------------------------------
add(
    "22_negative_and_zero_emission_percent",
    [
        cr("neg", -50.0, {"ha": 1.0}),
        cr("zero", 0.0, {"hb": 1.0}),
        cr("pos", 40.0, {"hc": 1.0}),
    ],
    {"ha": 1, "hb": 2, "hc": 3},
)

# --- not-ok challenge is ignored entirely -----------------------------------------
add(
    "23_not_ok_challenge_ignored",
    [
        cr("good", 25.0, {"ha": 1.0}),
        cr("bad", 75.0, {"hb": 1.0}, ok=False),
    ],
    {"ha": 1, "hb": 2},
)

# --- many miners, uneven shares: exercises u16 rounding sum -----------------------
add(
    "24_many_miners_uneven_shares",
    [
        cr("alpha", 33.0, {f"a{i}": float(i) for i in range(1, 8)}),
        cr("beta", 33.0, {f"b{i}": float(i * i) for i in range(1, 5)}),
        cr("gamma", 34.0, {"a3": 1.0, "b2": 2.0, "c1": 3.0}),
    ],
    {
        **{f"a{i}": i for i in range(1, 8)},
        **{f"b{i}": 10 + i for i in range(1, 5)},
        "c1": 21,
    },
)

# --- exact-tie rounding pins (banker's rounding) ----------------------------------
add(
    "25_half_even_rounding_pins",
    [cr("c", 50.0, {"ha": 1.0})],
    {"ha": 7},
)

# --- residual burn just above / below EPS -----------------------------------------
add(
    "26_burn_below_eps_not_added",
    [cr("c", 100.0, {"ha": 1.0, "hb": 1.0, "hc": 1.0})],
    {"ha": 1, "hb": 2, "hc": 3},
)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, doc in CASES:
        with open(f"{OUT}/{name}.json", "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        summary = doc.get("python_error") or (
            f"uids={doc['python_float_output']['uids']} "
            f"u16sum={doc['chain_u16_sum']}"
        )
        print(f"{name}: {summary}")
