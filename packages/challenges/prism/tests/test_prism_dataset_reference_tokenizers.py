"""Offline reference-tokenizer staging + loading (architecture.md sections 3, 9).

Covers VAL-DATA-012/013/014: the eval image enforces Hugging Face offline mode and the
gpt2 (tiktoken) + llama (sentencepiece) reference tokenizers load with NO network.
"""

import re
import socket
from pathlib import Path

import pytest
import tiktoken
import tiktoken.registry

from prism_challenge.evaluator import reference_tokenizers as rt
from prism_challenge.evaluator.interface import PrismContext

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile"


def _network_available() -> bool:
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=8):
            return True
    except OSError:
        return False


requires_network = pytest.mark.skipif(
    not _network_available(),
    reason="reference-tokenizer staging needs network (prep/build step)",
)


class _BlockNetwork:
    """Context manager that makes ANY socket creation raise (proves offline loading)."""

    def __init__(self) -> None:
        self._orig_socket = socket.socket
        self._orig_create = socket.create_connection

    def __enter__(self) -> "_BlockNetwork":
        def _blocked(*_a: object, **_k: object) -> object:
            raise OSError("network access is blocked (offline assertion)")

        socket.socket = _blocked  # type: ignore[assignment, misc]
        socket.create_connection = _blocked  # type: ignore[assignment]
        return self

    def __exit__(self, *_exc: object) -> None:
        socket.socket = self._orig_socket  # type: ignore[misc]
        socket.create_connection = self._orig_create


def _clear_tiktoken_cache() -> None:
    """Drop tiktoken's in-process encoding memoization so a load re-reads the disk cache."""
    tiktoken.registry.ENCODINGS.clear()


@pytest.fixture(scope="module")
def staged_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Stage both reference tokenizers once (network/prep step) for the offline load tests."""
    if not _network_available():
        pytest.skip("reference-tokenizer staging needs network (prep/build step)")
    root = tmp_path_factory.mktemp("reference-tokenizers")
    rt.stage_reference_tokenizers(root)
    return root


# --- staging --------------------------------------------------------------------------


@requires_network
def test_staging_creates_gpt2_cache_and_llama_model(tmp_path: Path) -> None:
    root = tmp_path / "ref"
    manifest = rt.stage_reference_tokenizers(root)

    gpt2_dir = rt.gpt2_cache_dir(root)
    assert gpt2_dir.is_dir()
    blobs = [p for p in gpt2_dir.iterdir() if p.is_file()]
    assert blobs, "gpt2 tiktoken cache should contain BPE blob files"

    llama_model = rt.llama_model_path(root)
    assert llama_model.is_file()
    assert llama_model.stat().st_size > 0

    assert manifest["names"] == list(rt.REFERENCE_TOKENIZER_NAMES)
    assert manifest["gpt2"]["vocab_size"] == rt.GPT2_VOCAB_SIZE
    assert manifest["llama"]["sha256"] == rt.LLAMA_EXPECTED_SHA256


@requires_network
def test_staging_llama_sha256_matches_pinned_non_gated_source(tmp_path: Path) -> None:
    root = tmp_path / "ref"
    rt.stage_reference_tokenizers(root)
    import hashlib

    digest = hashlib.sha256(rt.llama_model_path(root).read_bytes()).hexdigest()
    assert digest == rt.LLAMA_EXPECTED_SHA256


@requires_network
def test_staging_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "ref"
    rt.stage_reference_tokenizers(root)
    first = rt.llama_model_path(root).read_bytes()
    # second run must not corrupt or duplicate the staged assets
    rt.stage_reference_tokenizers(root)
    assert rt.llama_model_path(root).read_bytes() == first


# --- offline loading (VAL-DATA-013 / VAL-DATA-014) ------------------------------------


def test_gpt2_reference_tokenizer_loads_offline(
    staged_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(rt.gpt2_cache_dir(staged_dir)))
    monkeypatch.delenv("PRISM_REFERENCE_TOKENIZER_DIR", raising=False)
    _clear_tiktoken_cache()

    with _BlockNetwork():
        enc = rt.load_reference_tokenizer("gpt2", staged_dir)
        assert enc.n_vocab == rt.GPT2_VOCAB_SIZE
        assert enc.decode(enc.encode("hello world")) == "hello world"


def test_llama_reference_tokenizer_loads_offline(staged_dir: Path) -> None:
    with _BlockNetwork():
        sp = rt.load_reference_tokenizer("llama", staged_dir)
        ids = sp.encode("hello world")
        assert isinstance(ids, list)
        assert ids
        assert sp.decode(ids).strip() == "hello world".strip() or "hello" in sp.decode(ids)
        assert sp.vocab_size() > 0


def test_prism_context_reference_tokenizer_gpt2_offline(
    staged_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(rt.gpt2_cache_dir(staged_dir)))
    _clear_tiktoken_cache()
    ctx = PrismContext(reference_tokenizer_dir=str(staged_dir))
    with _BlockNetwork():
        enc = ctx.reference_tokenizer("gpt2")
        assert enc.decode(enc.encode("prism")) == "prism"


def test_prism_context_reference_tokenizer_llama_offline(staged_dir: Path) -> None:
    ctx = PrismContext(reference_tokenizer_dir=str(staged_dir))
    with _BlockNetwork():
        sp = ctx.reference_tokenizer("llama")
        assert sp.encode("prism")


def test_reference_tokenizer_dir_falls_back_to_env(
    staged_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRISM_REFERENCE_TOKENIZER_DIR", str(staged_dir))
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(rt.gpt2_cache_dir(staged_dir)))
    _clear_tiktoken_cache()
    ctx = PrismContext()  # reference_tokenizer_dir unset -> resolve from env
    with _BlockNetwork():
        sp = ctx.reference_tokenizer("llama")
        assert sp.encode("prism")


# --- error handling -------------------------------------------------------------------


def test_unknown_reference_tokenizer_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        rt.load_reference_tokenizer("bert", tmp_path)


def test_llama_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(rt.ReferenceTokenizerError):
        rt.load_reference_tokenizer("llama", tmp_path)


def test_llama_without_dir_or_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRISM_REFERENCE_TOKENIZER_DIR", raising=False)
    with pytest.raises(rt.ReferenceTokenizerError):
        rt.load_reference_tokenizer("llama", None)


# --- eval image enforces HF offline + baked tokenizers (VAL-DATA-012) -----------------


def test_dockerfile_evaluator_enforces_offline_and_bakes_tokenizers() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    evaluator = text.split("AS service")[0]  # restrict to the evaluator stage

    for var in ("HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"):
        assert var in evaluator, f"evaluator image must set {var}"

    assert re.search(r"TIKTOKEN_CACHE_DIR=", evaluator), "evaluator must set TIKTOKEN_CACHE_DIR"
    assert "sentencepiece" in evaluator, "evaluator image must include sentencepiece"
    assert "reference_tokenizers" in evaluator, "evaluator must bake the reference tokenizers"
