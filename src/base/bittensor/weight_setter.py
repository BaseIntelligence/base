from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def is_rejected_set_weights_result(result: Any) -> bool:
    if result is False:
        return True
    if isinstance(result, (tuple, list)) and result and result[0] is False:
        return True
    return False


def set_weights_rejection_message(result: Any) -> str:
    if isinstance(result, (tuple, list)) and len(result) > 1:
        return f"subtensor rejected weight submission: {result[1]}"
    return "subtensor rejected weight submission"


@dataclass
class WeightSetter:
    subtensor: Any | None
    wallet: Any | None
    netuid: int

    def set_weights(self, uids: list[int], weights: list[float]) -> Any:
        if not uids:
            raise ValueError("Cannot submit empty weights")
        if self.subtensor is None:
            raise RuntimeError("Subtensor is required to submit weights")
        if self.wallet is None:
            raise RuntimeError("Wallet is required to submit weights")
        result = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=False,
            wait_for_finalization=False,
        )
        if is_rejected_set_weights_result(result):
            raise RuntimeError(set_weights_rejection_message(result))
        return result
