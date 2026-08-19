"""Inclusive total-param floor/cap (tied embeddings counted once)."""


class ParamRangeError(Exception):
    """Miner-attributable size breach (below floor or over cap)."""

    def __init__(self, n_params, min_params, max_params, under):
        self.n_params = int(n_params)
        self.min_params = int(min_params or 0)
        self.max_params = int(max_params)
        self.under = bool(under)
        if self.under:
            msg = f"model below parameter floor: {self.n_params} < {self.min_params}"
        else:
            msg = f"model exceeds parameter cap: {self.n_params} > {self.max_params}"
        super().__init__(msg)


def enforce_param_range(n_params, min_params, max_params):
    """Raise ParamRangeError unless min_params ≤ n_params ≤ max_params.

    `min_params <= 0` disables the floor (tiny-cap / staging profile).
    """
    n = int(n_params)
    lo = int(min_params or 0)
    hi = int(max_params)
    if lo > 0 and n < lo:
        raise ParamRangeError(n, lo, hi, under=True)
    if n > hi:
        raise ParamRangeError(n, lo, hi, under=False)
    return n
