"""Self-deploy Phala CVM surface removed (T40).

Host-trust product path does not provision Phala CVMs. Importing this package
is allowed for residual scripts; mutating APIs raise.
"""

from __future__ import annotations


class SelfDeployRemovedError(RuntimeError):
    """Raised when a deleted Phala self-deploy API is invoked."""


def _removed(name: str) -> None:
    raise SelfDeployRemovedError(
        f"agent_challenge.selfdeploy.{name} was removed with Phala TEE (T40). "
        "Use host-trust unattested execution (no_phala / CHALLENGE_UNATTESTED_EXECUTION)."
    )


__all__ = ["SelfDeployRemovedError"]
