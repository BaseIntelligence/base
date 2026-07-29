"""Self-deploy lifecycle removed with Phala (T40)."""

from __future__ import annotations

from agent_challenge.selfdeploy import SelfDeployRemovedError


def __getattr__(name: str):  # noqa: ANN001
    raise SelfDeployRemovedError(f"selfdeploy.lifecycle.{name} removed with Phala TEE (T40)")
