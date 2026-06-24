from __future__ import annotations

import pytest

from base.bittensor.metagraph_cache import MetagraphCache
from base.bittensor.weight_setter import WeightSetter


def test_metagraph_cache_from_hotkeys() -> None:
    cache = MetagraphCache(netuid=1)
    assert cache.update_from_hotkeys(["a", "b"]) == {"a": 0, "b": 1}


def test_weight_setter_requires_subtensor() -> None:
    with pytest.raises(RuntimeError, match="Subtensor is required"):
        WeightSetter(subtensor=None, wallet=None, netuid=1).set_weights([1], [1.0])


def test_weight_setter_allows_uid_zero_fallback() -> None:
    class Subtensor:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def set_weights(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(kwargs)
            return {"ok": True, **kwargs}

    subtensor = Subtensor()

    result = WeightSetter(subtensor=subtensor, wallet="wallet", netuid=1).set_weights(
        [0], [1.0]
    )

    assert result["ok"] is True
    assert subtensor.calls == [
        {
            "wallet": "wallet",
            "netuid": 1,
            "uids": [0],
            "weights": [1.0],
            "wait_for_inclusion": False,
            "wait_for_finalization": False,
        }
    ]


@pytest.mark.parametrize(
    ("result", "match"),
    [
        (False, "subtensor rejected weight submission"),
        ((False, "chain rejected"), "chain rejected"),
        ([False, "chain rejected"], "chain rejected"),
    ],
)
def test_weight_setter_raises_on_rejected_set_weights_result(
    result: object,
    match: str,
) -> None:
    class Subtensor:
        def set_weights(self, **kwargs: object) -> object:
            return result

    with pytest.raises(RuntimeError, match=match):
        WeightSetter(subtensor=Subtensor(), wallet="wallet", netuid=1).set_weights(
            [1], [1.0]
        )
