"""Miner tokenizer contract — `ctx["tokenizer"]` is submitted, not imposed.

Recipe <= 1.3.0 hardcoded `GPT2TokenizerFast.from_pretrained("gpt2")` in
every phase entry. The tokenizer is now part of the submission and this
module is the single resolution path, used identically by the v1 miner
entry, the v3 TRAIN child and the v3 EVAL child — so the eval phase (a
fresh subprocess) reconstructs exactly the tokenizer training used.

Resolution order (first match wins, deterministic, offline):

1. `<workdir>/submission/tokenizer/` (or `<workdir>/tokenizer/` for the
   legacy two-script layout) — tokenizer files shipped inside the
   submission source tree, loaded with `AutoTokenizer.from_pretrained(dir,
   local_files_only=True)`.
2. `build_tokenizer(ctx)` exported by the miner's `architecture.py` — a
   code hook (train a tokenizer on the pinned shard, wrap a vendored
   implementation, ...). It MUST live in the module that defines
   `build_model`: the EVAL child imports that module only, so a hook
   hiding in `training.py` would make train and eval disagree — that is a
   hard error, not a silent fallback.
3. The pinned fallback [`DEFAULT_TOKENIZER`] from the pod's warm HF cache —
   what submissions written against recipe <= 1.3.0 already got. The pin
   is a *default*, never a challenge rule.

Every resolved tokenizer is validated before miner code sees it: duck-typed
interface, bounded vocab, every probe id inside that vocab, exact
encode/decode roundtrip on the ASCII probe. Validation also derives a
`fingerprint` — sha256 over the token ids of a fixed probe corpus — which
the TRAIN child stores in the checkpoint meta and the EVAL child re-checks
([`assert_matches`]): a tokenizer that does not reconstruct identically
fails the run instead of silently scoring a mismatched model.

Minimal duck-typed interface (also what `harness/tests/smoke_battery.py`
fakes with a byte-level tokenizer):

    tok(text, add_special_tokens=False)["input_ids"] -> list[int]
    tok(text, return_tensors="pt")["input_ids"]      -> (1, T) tensor or list
    tok.decode(ids) -> str             # roundtrips plain ASCII
    len(tok) or tok.vocab_size -> int  # within MIN_VOCAB..MAX_VOCAB
    tok.eos_token_id -> int | None     # document separator in the stream
    tok.convert_ids_to_tokens(ids)     # optional (G1 key-token heuristic)

Network: never. The miner child runs under `unshare --net`, so a tokenizer
that would need a download fails closed here with a clear error instead of
deep inside `transformers`.
"""

import hashlib
import json
import os

from . import DEFAULT_TOKENIZER

#: Source-tree directory carrying miner tokenizer files.
TOKENIZER_DIRNAME = "tokenizer"
#: Miner hook name (must sit beside `build_model` in `architecture.py`).
HOOK_NAME = "build_tokenizer"
#: Vocab bounds. Floor: a byte-level tokenizer is the smallest sane vocab.
#: Ceiling: the embedding/head dominates the 350M parameter cap past this.
#: Intake cannot see a vocab (it never runs miner code), so this pair is
#: harness-side only.
MIN_VOCAB = 256
MAX_VOCAB = 1 << 18
#: `tokenizer/` caps — mirrored by `prism_tree` / `prism_recipe::zip_submit`
#: at intake so a malformed directory fails identically there and in-pod.
#: Sized to admit a real HF `tokenizer.json` (~1.4 MiB) with headroom.
MAX_TOKENIZER_FILES = 12
MAX_TOKENIZER_BYTES = 8 * 1024 * 1024
ALLOWED_EXTENSIONS = (".json", ".txt", ".model", ".vocab", ".merges", ".bpe")
#: Fixed probe corpus: the roundtrip smoke AND the cross-phase fingerprint.
#: The first probe is the ASCII roundtrip contract; the rest only add
#: entropy to the fingerprint.
PROBE_TEXTS = (
    "the quick brown fox jumps over the lazy dog 0123456789",
    "bits per byte stays the tokenizer-neutral unit of G1 .",
    "unicode probe: caf\u00e9 na\u00efve \u4e2d\u6587 \u2713",
    " leading and trailing whitespace \t",
)
#: Spec fields that must be identical in the train and eval phases.
CHECKED_SPEC_KEYS = ("source", "id", "vocab_size", "fingerprint")
# Harness-internal ctx keys never handed to miner code (mirrors the
# `_HARNESS_CTX_KEYS` tuple of the phase entries).
_HOOK_CTX_DROP = ("arch_path", "train_path", "workdir")


class TokenizerContractError(RuntimeError):
    """Fail-closed tokenizer contract breach (miner-attributable)."""


# ---------------------------------------------------------------- duck typing


def encode(tok, text):
    """`tok(text, add_special_tokens=False)["input_ids"]` as a flat list."""
    try:
        out = tok(text, add_special_tokens=False)
        ids = out["input_ids"]
    except Exception as exc:  # noqa: BLE001 — miner-supplied callable
        raise TokenizerContractError(
            f"tokenizer must support tok(text, add_special_tokens=False)['input_ids']: {exc}"
        ) from exc
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(i) for i in ids]


def encode_tensor(tok, text, device=None, truncation=False, max_length=None):
    """`(1, T)` int64 tensor of `text` ids for the harness forward passes.

    Accepts dict-style and attribute-style tokenizer outputs, and wraps a
    plain id list when the tokenizer ignores `return_tensors` — so the
    contract only ever demands `tok(text)["input_ids"]`.
    """
    import torch

    kwargs = {"return_tensors": "pt"}
    if truncation:
        kwargs.update({"truncation": True, "max_length": max_length})
    try:
        out = tok(text, **kwargs)
        ids = out["input_ids"] if hasattr(out, "__getitem__") else out.input_ids
    except (KeyError, TypeError, AttributeError) as exc:
        raise TokenizerContractError(
            f"tokenizer must support tok(text, return_tensors='pt')['input_ids']: {exc}"
        ) from exc
    if not hasattr(ids, "dim"):
        ids = torch.tensor([list(ids)], dtype=torch.long)
    elif ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids.to(device) if device is not None else ids


def decode(tok, ids):
    """`tok.decode(ids)` — required by the contract (exact token budgets)."""
    fn = getattr(tok, "decode", None)
    if not callable(fn):
        raise TokenizerContractError("tokenizer must implement decode(ids) -> str")
    try:
        return str(fn(list(ids)))
    except Exception as exc:  # noqa: BLE001
        raise TokenizerContractError(f"tokenizer decode() failed: {exc}") from exc


def vocab_size(tok):
    """Largest of `len(tok)` / `tok.vocab_size` (embedding-sizing number)."""
    cands = []
    for get in (lambda: len(tok), lambda: getattr(tok, "vocab_size", None)):
        try:
            v = get()
        except Exception:  # noqa: BLE001
            v = None
        if isinstance(v, int) and v > 0:
            cands.append(v)
    if not cands:
        raise TokenizerContractError("tokenizer must expose len(tok) or tok.vocab_size")
    return max(cands)


def validate(tok, source, tok_id=None):
    """Contract checks + the cross-phase spec dict. Fail-closed on breach."""
    if not callable(tok):
        raise TokenizerContractError("tokenizer must be callable: tok(text)['input_ids']")
    n_vocab = vocab_size(tok)
    if not MIN_VOCAB <= n_vocab <= MAX_VOCAB:
        raise TokenizerContractError(
            f"tokenizer vocab {n_vocab} outside [{MIN_VOCAB}, {MAX_VOCAB}]"
        )
    h = hashlib.sha256()
    n_ids, max_id = 0, -1
    for i, text in enumerate(PROBE_TEXTS):
        ids = encode(tok, text)
        if not ids:
            raise TokenizerContractError(f"tokenizer encoded probe {i} to zero tokens")
        n_ids += len(ids)
        max_id = max(max_id, max(ids))
        h.update(json.dumps(ids, separators=(",", ":")).encode("utf-8"))
        h.update(b"\x00")
        if i == 0 and decode(tok, ids).strip() != text.strip():
            raise TokenizerContractError(
                "tokenizer failed the encode/decode roundtrip on the ASCII probe"
            )
    if max_id >= n_vocab:
        raise TokenizerContractError(
            f"tokenizer emitted id {max_id} outside its own vocab ({n_vocab})"
        )
    # Historical harness courtesy (GPT-2 ships pad_token=None): only fill it
    # in for tokenizers that declare the attribute, never invent one on a
    # minimal duck-typed object.
    if getattr(tok, "pad_token", "") is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    return {
        "source": source,
        "id": str(tok_id) if tok_id else None,
        "class": type(tok).__name__,
        "vocab_size": int(n_vocab),
        "probe_tokens": int(n_ids),
        "fingerprint": h.hexdigest(),
    }


# ---------------------------------------------------------------- resolution


def tokenizer_dir(workdir):
    """Validated tokenizer directory, or None when the miner ships none.

    Prefers `<workdir>/submission/tokenizer/` (full source-tree delivery),
    then falls back to `<workdir>/tokenizer/` (legacy two-script layout).

    Raises when the directory exists but breaks the caps mirrored from
    `prism_recipe::zip_submit`, so a bad tree fails the same way in-pod as
    at intake.
    """
    if not workdir:
        return None
    candidates = (
        os.path.join(str(workdir), "submission", TOKENIZER_DIRNAME),
        os.path.join(str(workdir), TOKENIZER_DIRNAME),
    )
    path = next((p for p in candidates if os.path.isdir(p)), None)
    if path is None:
        return None
    names = sorted(n for n in os.listdir(path) if not n.startswith("."))
    files = [n for n in names if os.path.isfile(os.path.join(path, n))]
    if not files:
        raise TokenizerContractError(f"{TOKENIZER_DIRNAME}/ is present but carries no files")
    if len(files) > MAX_TOKENIZER_FILES:
        raise TokenizerContractError(
            f"{TOKENIZER_DIRNAME}/ has {len(files)} files (max {MAX_TOKENIZER_FILES})"
        )
    total = 0
    for name in files:
        if os.path.splitext(name)[1].lower() not in ALLOWED_EXTENSIONS:
            raise TokenizerContractError(
                f"{TOKENIZER_DIRNAME}/{name}: extension not in {ALLOWED_EXTENSIONS}"
            )
        total += os.path.getsize(os.path.join(path, name))
    if total > MAX_TOKENIZER_BYTES:
        raise TokenizerContractError(
            f"{TOKENIZER_DIRNAME}/ is {total} bytes (max {MAX_TOKENIZER_BYTES})"
        )
    return path


def declared(workdir, arch_path=None, train_path=None):
    """Import-free declaration probe for the PARENT (never imports miner code).

    True when the submission ships `tokenizer/` files or defines the
    `build_tokenizer` hook (string scan, same style as the Rust intake
    contract screen).
    """
    try:
        if tokenizer_dir(workdir):
            return True
    except TokenizerContractError:
        # A malformed tokenizer/ dir IS a declaration; the child reports it.
        return True
    for path in (arch_path, train_path):
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if f"def {HOOK_NAME}(" in f.read():
                    return True
        except OSError:
            continue
    return False


def warm_default_cache():
    """Parent-side (networked) warm of the pinned fallback tokenizer so the
    netns child resolves it offline. Never imports miner code."""
    return load_default()


def load_default():
    """The pinned fallback tokenizer (fast GPT-2 pin, Auto as a fallback).

    Public so a submission that *wants* the pin (both v3 baselines do) can
    take it from the contract instead of re-hardcoding `from_pretrained`.
    """
    from transformers import AutoTokenizer, GPT2TokenizerFast

    try:
        return GPT2TokenizerFast.from_pretrained(DEFAULT_TOKENIZER)
    except Exception:  # noqa: BLE001 — non-GPT-2 pin or no fast files
        return AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)


def _load_dir(path):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        raise TokenizerContractError(
            f"{TOKENIZER_DIRNAME}/ is not loadable offline ({type(exc).__name__}: "
            f"{str(exc)[:200]}); ship the full tokenizer file set"
        ) from exc


def _hook_ctx(cfg, device, tok_dir, telemetry=None):
    ctx = {k: v for k, v in cfg.items() if k not in _HOOK_CTX_DROP}
    ctx["device"] = device
    ctx["tokenizer_dir"] = tok_dir
    if telemetry is not None:
        ctx["telemetry"] = telemetry
    return ctx


def resolve(cfg, device, arch_mod=None, train_mod=None, telemetry=None, log=None):
    """The single tokenizer resolution path. Returns `(tok, spec)`.

    `arch_mod` is the miner architecture module (already imported by the
    caller — this module never imports miner code itself). `train_mod`, when
    given, is only inspected to reject a misplaced `build_tokenizer` hook.
    """
    def _say(msg):
        if log:
            log(msg)

    tok_dir = tokenizer_dir(cfg.get("workdir"))
    if tok_dir:
        _say(f"tokenizer: loading submitted files from {tok_dir}")
        tok = _load_dir(tok_dir)
        return tok, validate(tok, "tree", TOKENIZER_DIRNAME)
    hook = getattr(arch_mod, HOOK_NAME, None) if arch_mod is not None else None
    if hook is None and train_mod is not None and hasattr(train_mod, HOOK_NAME):
        raise TokenizerContractError(
            f"{HOOK_NAME}(ctx) found in training.py but not in architecture.py; "
            "define it beside build_model (the eval phase imports that module only)"
        )
    if hook is not None:
        if not callable(hook):
            raise TokenizerContractError(f"{HOOK_NAME} must be callable")
        _say(f"tokenizer: calling miner {HOOK_NAME}(ctx)")
        try:
            tok = hook(_hook_ctx(cfg, device, tok_dir, telemetry))
        except TokenizerContractError:
            raise
        except Exception as exc:  # noqa: BLE001 — miner code
            raise TokenizerContractError(
                f"{HOOK_NAME}(ctx) raised {type(exc).__name__}: {str(exc)[:200]}"
            ) from exc
        if tok is None:
            raise TokenizerContractError(f"{HOOK_NAME}(ctx) returned None")
        return tok, validate(tok, "hook", getattr(tok, "name_or_path", None))
    _say(f"tokenizer: no submitted tokenizer — pinned default {DEFAULT_TOKENIZER}")
    try:
        tok = load_default()
    except Exception as exc:  # noqa: BLE001
        raise TokenizerContractError(
            f"pinned default tokenizer {DEFAULT_TOKENIZER} unavailable offline "
            f"({type(exc).__name__}: {str(exc)[:160]}); declare a tokenizer or "
            "warm the pod cache"
        ) from exc
    return tok, validate(tok, "default", DEFAULT_TOKENIZER)


def assert_matches(spec, expected):
    """Cross-phase equality: the EVAL tokenizer must be the TRAIN tokenizer.

    `expected` is the spec the TRAIN child stored in the checkpoint meta.
    Returns a small audit dict; raises on any divergence of
    [`CHECKED_SPEC_KEYS`] so a mismatched tokenizer can never be scored.
    """
    if not isinstance(expected, dict) or not expected.get("fingerprint"):
        return {"checked": False, "reason": "train phase recorded no tokenizer spec"}
    diff = {
        k: {"train": expected.get(k), "eval": spec.get(k)}
        for k in CHECKED_SPEC_KEYS
        if expected.get(k) != spec.get(k)
    }
    if diff:
        raise TokenizerContractError(
            "tokenizer is not reproducible across phases: "
            + json.dumps(diff, separators=(",", ":"), default=str)[:300]
        )
    return {"checked": True, "fingerprint": spec.get("fingerprint")}
