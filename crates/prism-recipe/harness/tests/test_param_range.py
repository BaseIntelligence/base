"""Recipe 2.1 total-param floor/cap: 849M fail, 850M–1B pass, 1.01B fail."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prismlib.params import ParamRangeError, enforce_param_range

FLOOR = 850_000_000
CAP = 1_000_000_000


def test_floor_and_cap_inclusive():
    assert enforce_param_range(850_000_000, FLOOR, CAP) == 850_000_000
    assert enforce_param_range(900_000_000, FLOOR, CAP) == 900_000_000
    assert enforce_param_range(1_000_000_000, FLOOR, CAP) == 1_000_000_000
    try:
        enforce_param_range(849_000_000, FLOOR, CAP)
        raise AssertionError("849M must fail")
    except ParamRangeError as exc:
        assert exc.under is True
        assert exc.n_params == 849_000_000
    try:
        enforce_param_range(1_010_000_000, FLOOR, CAP)
        raise AssertionError("1.01B must fail")
    except ParamRangeError as exc:
        assert exc.under is False
        assert exc.n_params == 1_010_000_000


def test_zero_floor_disables_min():
    assert enforce_param_range(100_000, 0, 2_000_000) == 100_000


if __name__ == "__main__":
    test_floor_and_cap_inclusive()
    test_zero_floor_disables_min()
    print("param range OK")
