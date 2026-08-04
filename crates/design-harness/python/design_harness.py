"""Operator-owned harness: loads miner agent.py and writes pages under /out."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from base_design import LlmClient, OutWriter, Task


REQUIRED_PAGES = ("index.html", "pricing.html", "components.html")


def _load_agent(path: Path):
    spec = importlib.util.spec_from_file_location("miner_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run"):
        raise RuntimeError("agent.py must define run(task, llm, out)")
    return mod


def main() -> int:
    work = Path(os.environ.get("DESIGN_WORK_ROOT", "/work"))
    out_root = Path(os.environ.get("DESIGN_OUT_ROOT", "/out"))
    proxy = os.environ.get("DESIGN_LLM_PROXY", "http://design-egress-proxy:8094")
    run_id = os.environ.get("DESIGN_RUN_ID", "unknown")
    round_id = os.environ.get("DESIGN_ROUND_ID", "0")
    prompt = os.environ.get("DESIGN_PROMPT", "")
    budget = int(os.environ.get("DESIGN_TOKEN_BUDGET", "8000"))

    pages_dir = out_root / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    task = Task(
        prompt=prompt,
        round_id=round_id,
        pages_required=list(REQUIRED_PAGES),
        budget=budget,
        run_id=run_id,
    )
    llm = LlmClient(proxy_base=proxy, run_id=run_id)
    out = OutWriter(pages_dir=pages_dir, assets_dir=out_root / "assets")

    agent_path = work / "agent.py"
    mod = _load_agent(agent_path)
    mod.run(task, llm, out)

    missing = [p for p in REQUIRED_PAGES if not (pages_dir / p).is_file()]
    if missing:
        raise RuntimeError(f"missing required pages: {missing}")

    manifest = {
        "run_id": run_id,
        "round_id": round_id,
        "pages": list(REQUIRED_PAGES),
        "assets": sorted(p.name for p in (out_root / "assets").glob("*") if p.is_file()),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — surface to orchestrator logs
        print(f"design_harness error: {exc}", file=sys.stderr)
        raise SystemExit(1)
