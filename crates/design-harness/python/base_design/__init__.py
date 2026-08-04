"""Injected SDK for design-challenge miner harnesses (not miner-modifiable)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Task:
    prompt: str
    round_id: str
    pages_required: list[str]
    budget: int
    run_id: str = ""


class LlmClient:
    def __init__(self, proxy_base: str, run_id: str) -> None:
        self.proxy_base = proxy_base.rstrip("/")
        self.run_id = run_id

    def chat(self, messages: list[dict[str, str]], model: str = "openrouter/auto") -> str:
        body = json.dumps(
            {"run_id": self.run_id, "model": model, "messages": messages}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.proxy_base}/v1/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"llm proxy error: {exc}") from exc
        return str(data.get("content", ""))


class OutWriter:
    MAX_PAGE_BYTES = 512 * 1024
    MAX_ASSET_BYTES = 256 * 1024

    def __init__(self, pages_dir: Path, assets_dir: Path) -> None:
        self.pages_dir = pages_dir
        self.assets_dir = assets_dir
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def write_page(self, name: str, html: str) -> None:
        if "/" in name or name.startswith("."):
            raise ValueError(f"invalid page name: {name}")
        raw = html.encode("utf-8")
        if len(raw) > self.MAX_PAGE_BYTES:
            raise ValueError(f"page too large: {name}")
        (self.pages_dir / name).write_bytes(raw)

    def write_asset(self, name: str, data: bytes) -> None:
        if "/" in name or name.startswith("."):
            raise ValueError(f"invalid asset name: {name}")
        if len(data) > self.MAX_ASSET_BYTES:
            raise ValueError(f"asset too large: {name}")
        (self.assets_dir / name).write_bytes(data)


__all__ = ["Task", "LlmClient", "OutWriter"]
