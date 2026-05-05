from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WeightSetter:
    subtensor: Any | None
    wallet: Any | None
    netuid: int

    def set_weights(self, uids: list[int], weights: list[float]) -> Any:
        if self.subtensor is None:
            return {"dry_run": True, "uids": uids, "weights": weights}
        return self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=False,
            wait_for_finalization=False,
        )
