"""``python -m agent_challenge.selfdeploy`` entrypoint."""

from __future__ import annotations

from agent_challenge.selfdeploy.cli import main

if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(main())
