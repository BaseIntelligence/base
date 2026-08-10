#!/usr/bin/env python3
"""Unit smoke: ItemRecorder inference traces carry prompt + choices.

No torch required — exercises the additive observation channel only.
"""

import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT))

from eval import common  # noqa: E402


def test_trace_includes_prompt_and_choices():
    rec = common.ItemRecorder()
    ctx = {"items": rec}
    common.record_trace(
        ctx,
        {
            "kind": "mc",
            "group": "g2",
            "task": "hellaswag",
            "metric": "g2.hellaswag.acc",
            "cluster": "g2/hellaswag",
            "prompt": "The dog",
            "choices": [" barked", " flew"],
            "gold": 0,
            "selected": 0,
            "value": 1.0,
            "gold_nll": 0.2,
            "choice_logprobs": [
                {"i": 0, "sum_lp": -0.1, "n_tok": 1, "norm_lp": -0.014},
                {"i": 1, "sum_lp": -3.0, "n_tok": 1, "norm_lp": -0.5},
            ],
        },
    )
    dump = rec.dump_traces()
    assert dump["version"] == 1
    assert dump["truncated"] is False
    assert dump["n_items"] == 1
    assert dump["caps"]["global"] == common.TRACE_CAP_GLOBAL
    item = dump["items"][0]
    assert item["prompt"] == "The dog"
    assert item["choices"] == [" barked", " flew"]
    assert item["gold"] == 0
    assert item["selected"] == 0
    assert item["choice_logprobs"][0]["i"] == 0
    # Bootstrap channel stays separate and empty here.
    assert rec.dump() == {}


def test_prompt_truncation_flag():
    rec = common.ItemRecorder()
    long = "x" * (common.TRACE_PROMPT_CHARS + 50)
    rec.add_trace({"kind": "mc", "group": "g3", "prompt": long, "choices": [" a"], "gold": 0})
    item = rec.dump_traces()["items"][0]
    assert len(item["prompt"]) == common.TRACE_PROMPT_CHARS
    assert item["prompt_truncated"] is True


def test_per_group_cap_sets_truncated():
    rec = common.ItemRecorder()
    for i in range(common.TRACE_CAP_PER_GROUP + 3):
        rec.add_trace(
            {
                "kind": "mc",
                "group": "g4",
                "cluster": f"c{i}",
                "prompt": "p",
                "choices": [" a", " b"],
                "gold": 0,
                "selected": 0,
                "value": 1.0,
            }
        )
    dump = rec.dump_traces()
    assert dump["truncated"] is True
    assert dump["n_items"] == common.TRACE_CAP_PER_GROUP


if __name__ == "__main__":
    test_trace_includes_prompt_and_choices()
    test_prompt_truncation_flag()
    test_per_group_cap_sets_truncated()
    print("OK: inference_traces unit smoke")
