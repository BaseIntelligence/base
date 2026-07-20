"""Offline reference-tokenizer staging + loading (architecture.md sections 3, 9).

The eval container runs with ``network=none``; the miner's ``training.py`` may obtain a
pre-staged reference tokenizer via ``ctx.reference_tokenizer("gpt2"|"llama")``. Both must load
with NO network:

* **gpt2** -- a tiktoken BPE cache: the two BPE blobs are baked under ``<root>/gpt2`` and read
  back through ``TIKTOKEN_CACHE_DIR`` (tiktoken's own offline cache mechanism).
* **llama** -- a NON-GATED sentencepiece ``.model`` baked at ``<root>/llama/tokenizer.model``.

``stage_reference_tokenizers`` runs in a NETWORK-ENABLED step (the evaluator image build or a
one-time GPU-node prep) and writes both to a fixed path. ``load_reference_tokenizer`` reads them
back offline. The source for llama is pinned to a non-gated mirror and verified by sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

REFERENCE_TOKENIZER_NAMES = ("gpt2", "llama")

# gpt2 reference (tiktoken). 50257 real tokens; LM heads typically pad to 50304.
GPT2_ENCODING = "gpt2"
GPT2_VOCAB_SIZE = 50257
GPT2_PADDED_VOCAB_SIZE = 50304

# llama reference: a NON-GATED sentencepiece tokenizer mirror (no auth / no gate). Pinned to an
# immutable commit and verified by sha256 so the staged ``.model`` is reproducible + tamper-evident.
LLAMA_SOURCE_REPO = "hf-internal-testing/llama-tokenizer"
LLAMA_SOURCE_REVISION = "d02ad6cb9dd2c2296a6332199fa2fdca5938fef0"
LLAMA_MODEL_FILENAME = "tokenizer.model"
LLAMA_SOURCE_URL = (
    f"https://huggingface.co/{LLAMA_SOURCE_REPO}/resolve/"
    f"{LLAMA_SOURCE_REVISION}/{LLAMA_MODEL_FILENAME}"
)
LLAMA_EXPECTED_SHA256 = "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"

REFERENCE_MANIFEST_FILENAME = "reference_manifest.json"

_DOWNLOAD_TIMEOUT_SECONDS = 120


class ReferenceTokenizerError(RuntimeError):
    """Raised when a reference tokenizer cannot be staged or loaded offline."""


def gpt2_cache_dir(root: Path | str) -> Path:
    """The fixed gpt2 tiktoken cache directory under ``root`` (used as TIKTOKEN_CACHE_DIR)."""
    return Path(root) / "gpt2"


def llama_model_path(root: Path | str) -> Path:
    """The fixed llama sentencepiece ``.model`` path under ``root``."""
    return Path(root) / "llama" / LLAMA_MODEL_FILENAME


def reference_manifest_path(root: Path | str) -> Path:
    return Path(root) / REFERENCE_MANIFEST_FILENAME


# --- staging (network-enabled prep / image build) -------------------------------------


def stage_gpt2(root: Path | str) -> Path:
    """Bake the gpt2 tiktoken BPE cache under ``<root>/gpt2`` (downloads the BPE blobs once)."""
    cache_dir = gpt2_cache_dir(root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prev = os.environ.get("TIKTOKEN_CACHE_DIR")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)
    try:
        import tiktoken
        import tiktoken.registry

        # Drop any in-process memoization so the BPE blobs are materialized into ``cache_dir``.
        tiktoken.registry.ENCODINGS.pop(GPT2_ENCODING, None)
        encoding = tiktoken.get_encoding(GPT2_ENCODING)
        encoding.encode("warm the offline cache")
    except Exception as exc:  # pragma: no cover - network/prep failure
        raise ReferenceTokenizerError(f"failed to stage gpt2 tiktoken cache: {exc}") from exc
    finally:
        if prev is None:
            os.environ.pop("TIKTOKEN_CACHE_DIR", None)
        else:
            os.environ["TIKTOKEN_CACHE_DIR"] = prev
    blobs = [path for path in cache_dir.iterdir() if path.is_file()]
    if not blobs:
        raise ReferenceTokenizerError(f"gpt2 tiktoken cache is empty after staging: {cache_dir}")
    return cache_dir


def stage_llama(root: Path | str, *, url: str = LLAMA_SOURCE_URL) -> Path:
    """Download the non-gated llama sentencepiece ``.model`` (sha256-verified) under ``root``."""
    model_path = llama_model_path(root)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest == LLAMA_EXPECTED_SHA256:
            return model_path
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "prism-reference-staging"})
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            data = response.read()
    except Exception as exc:  # pragma: no cover - network/prep failure
        raise ReferenceTokenizerError(
            f"failed to download llama sentencepiece model from {url}: {exc}"
        ) from exc
    if not data:
        raise ReferenceTokenizerError(f"llama sentencepiece model download was empty: {url}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != LLAMA_EXPECTED_SHA256:
        raise ReferenceTokenizerError(
            f"llama sentencepiece sha256 mismatch: expected {LLAMA_EXPECTED_SHA256}, got {digest}"
        )
    model_path.write_bytes(data)
    return model_path


def stage_reference_tokenizers(root: Path | str) -> dict[str, Any]:
    """Stage gpt2 + llama under ``root`` (idempotent) and write a provenance manifest."""
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    gpt2_dir = stage_gpt2(base)
    llama_path = stage_llama(base)
    manifest: dict[str, Any] = {
        "schema_version": "prism_reference_tokenizers.v1",
        "names": list(REFERENCE_TOKENIZER_NAMES),
        "gpt2": {
            "kind": "tiktoken",
            "encoding": GPT2_ENCODING,
            "vocab_size": GPT2_VOCAB_SIZE,
            "padded_vocab_size": GPT2_PADDED_VOCAB_SIZE,
            "cache_dir": gpt2_dir.name,
            "blobs": sorted(p.name for p in gpt2_dir.iterdir() if p.is_file()),
        },
        "llama": {
            "kind": "sentencepiece",
            "source_repo": LLAMA_SOURCE_REPO,
            "source_revision": LLAMA_SOURCE_REVISION,
            "model": f"{llama_path.parent.name}/{llama_path.name}",
            "sha256": LLAMA_EXPECTED_SHA256,
            "license": "non-gated",
        },
    }
    reference_manifest_path(base).write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


# --- offline loading ------------------------------------------------------------------


def _resolve_root(root: Path | str | None) -> Path | None:
    if root:
        return Path(root)
    env_root = os.environ.get("PRISM_REFERENCE_TOKENIZER_DIR")
    return Path(env_root) if env_root else None


def _load_gpt2(root: Path | None) -> Any:
    if root is not None:
        cache_dir = gpt2_cache_dir(root)
        if cache_dir.is_dir() and any(cache_dir.iterdir()):
            # tiktoken reads its BPE blobs from TIKTOKEN_CACHE_DIR (offline). A baked image env
            # already pointing at a populated cache is preserved.
            os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache_dir))
    try:
        import tiktoken

        return tiktoken.get_encoding(GPT2_ENCODING)
    except Exception as exc:
        raise ReferenceTokenizerError(
            f"failed to load gpt2 reference tokenizer offline: {exc}"
        ) from exc


def _load_llama(root: Path | None) -> Any:
    if root is None:
        raise ReferenceTokenizerError(
            "llama reference tokenizer dir is not configured "
            "(set ctx.reference_tokenizer_dir or PRISM_REFERENCE_TOKENIZER_DIR)"
        )
    model_path = llama_model_path(root)
    if not model_path.exists():
        raise ReferenceTokenizerError(f"llama reference tokenizer missing at {model_path}")
    try:
        import sentencepiece

        processor = sentencepiece.SentencePieceProcessor()
        processor.Load(str(model_path))
    except Exception as exc:
        raise ReferenceTokenizerError(
            f"failed to load llama reference tokenizer offline: {exc}"
        ) from exc
    return processor


def load_reference_tokenizer(name: str, root: Path | str | None = None) -> Any:
    """Load a pre-staged reference tokenizer offline (no network).

    ``root`` is the fixed staging directory (gpt2 cache + llama ``.model``). When ``root`` is
    falsy it is resolved from ``PRISM_REFERENCE_TOKENIZER_DIR`` so a baked image works without an
    explicit context path. gpt2 additionally falls back to a baked ``TIKTOKEN_CACHE_DIR``.
    """
    key = name.lower()
    if key not in REFERENCE_TOKENIZER_NAMES:
        raise ValueError(
            f"unknown reference tokenizer {name!r}; expected one of {REFERENCE_TOKENIZER_NAMES}"
        )
    resolved = _resolve_root(root)
    if key == "gpt2":
        return _load_gpt2(resolved)
    return _load_llama(resolved)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage the offline reference tokenizers (gpt2 tiktoken cache + llama .model).",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = stage_reference_tokenizers(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
