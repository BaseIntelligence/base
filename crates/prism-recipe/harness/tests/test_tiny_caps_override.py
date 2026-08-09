"""PRISM_TEST_EVAL_CAPS=0 keeps full battery under short-train knobs."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval import common  # noqa: E402


def _clear():
    for k in (
        "PRISM_TEST_EVAL_CAPS",
        "PRISM_TEST_TRAIN_MINUTES",
        "PRISM_TEST_MAX_PARAMS",
        "PRISM_EVAL_N_ITEMS",
        "PRISM_EVAL_G5_N_ITEMS",
        "PRISM_EVAL_G2_CAP",
    ):
        os.environ.pop(k, None)


def main():
    _clear()
    assert common.tiny_caps() is False

    os.environ["PRISM_TEST_TRAIN_MINUTES"] = "15"
    assert common.tiny_caps() is True, "legacy: train minutes imply tiny"

    os.environ["PRISM_TEST_EVAL_CAPS"] = "0"
    assert common.tiny_caps() is False, "explicit 0 forces full battery"
    assert common.eval_n_items() == 4
    assert common.eval_g5_n_items() == 2
    assert common.eval_asset_cap(200, 8, env_key="PRISM_EVAL_G2_CAP") == 200

    os.environ["PRISM_EVAL_N_ITEMS"] = "1"
    os.environ["PRISM_EVAL_G5_N_ITEMS"] = "1"
    os.environ["PRISM_EVAL_G2_CAP"] = "40"
    assert common.eval_n_items() == 1
    assert common.eval_g5_n_items() == 1
    assert common.eval_asset_cap(200, 8, env_key="PRISM_EVAL_G2_CAP") == 40

    os.environ["PRISM_TEST_EVAL_CAPS"] = "1"
    assert common.tiny_caps() is True
    assert common.eval_n_items() == 2
    assert common.eval_g5_n_items() == 1

    _clear()
    print("tiny_caps override OK")


if __name__ == "__main__":
    main()
