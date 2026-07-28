"""Prism Lium pod boot contract — pure plan builders (no real git/pip execution).

Boot flow (IMAGE ENTRYPOINT, not Lium ``startup_commands``)
----------------------------------------------------------
Lium rejects shell metacharacters in ``startup_commands`` (deploy keeps a
metachar-free hold like ``tail -f /dev/null``). The real boot sequence therefore
runs via the **digest-pinned image ENTRYPOINT**, which:

1. Clones the miner repo at a validated commit SHA.
2. Installs the miner ``pyproject.toml`` (local project) via the argv from
   :func:`build_install_plan`.
3. Runs training.
4. Pushes crash-recovery checkpoints to master over the **existing signed HTTP
   path** (:mod:`prism_challenge.evaluator.checkpoint_push`). The validator/pod
   holds **no** HuggingFace token; master publishes via
   ``HuggingFaceCheckpointPublisher``.

Zero secrets on the pod
-----------------------
:func:`build_boot_env` and :func:`assert_no_forbidden_env` refuse
``HF_TOKEN``, ``PRISM_HF_TOKEN``, ``LIUM_API_KEY``, ``LIUM_API_KEY_FILE``, and
other obvious secret names. Only non-secret ``PRISM_*`` coordination keys are
emitted.

Supply-chain residual risk
--------------------------
Installing an arbitrary miner ``pyproject`` on a BASE-owned pod is intentional
but residual risk. Containment:

* short pod TTL
* **no secrets** on the pod (this module)
* digest-pinned base image
* outbound network limited to the git host + master checkpoint URL

This module is offline-importable and unit-testable: pure validation and plan
builders only — no network, no subprocess.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

__all__ = [
    "FORBIDDEN_ENV_KEYS",
    "PodBootError",
    "assert_no_forbidden_env",
    "build_boot_env",
    "build_install_plan",
    "validate_commit_sha",
    "validate_repo_url",
]

# Exact env keys that must never appear on a Lium training pod.
FORBIDDEN_ENV_KEYS: Final[frozenset[str]] = frozenset(
    {
        "HF_TOKEN",
        "PRISM_HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_API_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "LIUM_API_KEY",
        "LIUM_API_KEY_FILE",
        "LIUM_TOKEN",
        "LIUM_API_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SESSION_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "NPM_TOKEN",
        "PYPI_TOKEN",
    }
)

# Suffixes that mark obvious secret-shaped names (case-insensitive).
_SECRET_SUFFIXES: Final[tuple[str, ...]] = (
    "_PASSWORD",
    "_PASSWD",
    "_SECRET",
    "_SECRET_KEY",
    "_API_KEY",
    "_ACCESS_TOKEN",
    "_PRIVATE_KEY",
)

# Full 40-char hex or unambiguous short (7–39) hex; no path/shell characters.
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

# Shell / injection metacharacters forbidden in SHA and URL strings.
_SHELL_METACHAR_RE = re.compile(r"""[\s;|&$`<>(){}[\]!*?\\'"]""")

# https git host path: owner/repo with optional .git, no query/fragment/userinfo.
_HTTPS_GIT_PATH_RE = re.compile(r"^/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(?:\.git)?/?$")

_PYPROJECT_NAME = "pyproject.toml"

# Boot env keys owned by this contract (values are non-secret coordination only).
_BOOT_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "PRISM_REPO_URL",
    "PRISM_COMMIT_SHA",
    "PRISM_MASTER_CHECKPOINT_URL",
    "PRISM_SUBMISSION_ID",
    "PRISM_ATTEMPT",
)


class PodBootError(ValueError):
    """Fail-closed rejection of an unsafe pod boot input or env plan."""


def validate_commit_sha(sha: str) -> str:
    """Return a normalized commit SHA, or raise :class:`PodBootError`.

    Accepts a full 40-char hex SHA or an unambiguous short SHA (7–40 hex digits).
    Rejects path traversal, whitespace, and shell metacharacters.
    """
    if not isinstance(sha, str) or not sha:
        raise PodBootError("commit SHA must be a non-empty string")
    if sha != sha.strip():
        raise PodBootError("commit SHA must not have leading/trailing whitespace")
    if _SHELL_METACHAR_RE.search(sha) or ".." in sha or "/" in sha or "\\" in sha:
        raise PodBootError("commit SHA contains invalid or injection characters")
    normalized = sha.lower()
    if not _COMMIT_SHA_RE.fullmatch(normalized):
        raise PodBootError("commit SHA must be 7–40 lowercase hex digits (full 40-char preferred)")
    return normalized


def validate_repo_url(url: str) -> str:
    """Return a validated https git URL, or raise :class:`PodBootError`.

    Allowlist shape: ``https://<host>/<owner>/<repo>[.git]`` with no userinfo,
    query, fragment, or shell metacharacters. Rejects ``file://``,
    ``javascript:``, ``http://``, and scp-style git URLs.
    """
    if not isinstance(url, str) or not url:
        raise PodBootError("repo URL must be a non-empty string")
    if url != url.strip():
        raise PodBootError("repo URL must not have leading/trailing whitespace")
    if _SHELL_METACHAR_RE.search(url):
        raise PodBootError("repo URL contains shell metacharacters")
    lower = url.lower()
    if lower.startswith("file:") or lower.startswith("javascript:"):
        raise PodBootError("repo URL scheme is not allowed")
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        raise PodBootError("repo URL must use https")
    if parsed.username is not None or parsed.password is not None:
        raise PodBootError("repo URL must not embed credentials")
    if parsed.query or parsed.fragment:
        raise PodBootError("repo URL must not include query or fragment")
    if not parsed.hostname or not parsed.path:
        raise PodBootError("repo URL must include host and path")
    if not _HTTPS_GIT_PATH_RE.fullmatch(parsed.path):
        raise PodBootError("repo URL path must be /owner/repo[.git]")
    # Rebuild a canonical form without trailing slash noise beyond optional /
    path = parsed.path.rstrip("/")
    host = parsed.hostname.lower()
    return f"https://{host}{path}"


def build_install_plan(pyproject_path: Path) -> list[str]:
    """Return argv to install the local miner project (no execution).

    Example::

        ["uv", "pip", "install", "--no-cache", "/workspace/miner"]

    The path must name ``pyproject.toml`` and must not contain ``..`` components
    (path-traversal reject). Callers run this argv inside the image ENTRYPOINT.
    """
    path = Path(pyproject_path)
    if path.name != _PYPROJECT_NAME:
        raise PodBootError("install plan requires a pyproject.toml path")
    parts = path.parts
    if any(part == ".." for part in parts):
        raise PodBootError("pyproject path must not contain '..' components")
    if "\x00" in str(path):
        raise PodBootError("pyproject path contains NUL")
    # Prefer the unresolved parent so the plan stays under the given work tree
    # without following symlinks that could escape (pure string/path plan).
    project_dir = path.parent
    if any(part == ".." for part in project_dir.parts):
        raise PodBootError("project directory path must not contain '..' components")
    return ["uv", "pip", "install", "--no-cache", str(project_dir)]


def _is_obvious_secret_name(name: str) -> bool:
    upper = name.upper()
    if upper in FORBIDDEN_ENV_KEYS or name in FORBIDDEN_ENV_KEYS:
        return True
    # Case-insensitive exact match against the forbidden set.
    forbidden_upper = {k.upper() for k in FORBIDDEN_ENV_KEYS}
    if upper in forbidden_upper:
        return True
    if any(upper.endswith(suffix) for suffix in _SECRET_SUFFIXES):
        return True
    # Bare TOKEN / SECRET / PASSWORD keys.
    if upper in {"TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY"}:
        return True
    # HF / LIUM family prefixes even when not exact.
    if upper.startswith("HF_") and (
        upper.endswith("_TOKEN") or upper.endswith("_KEY") or "TOKEN" in upper
    ):
        return True
    if upper.startswith("LIUM_") and ("KEY" in upper or "TOKEN" in upper or "SECRET" in upper):
        return True
    return False


def assert_no_forbidden_env(env: Mapping[str, str]) -> None:
    """Raise :class:`PodBootError` if any forbidden or secret-shaped key is present.

    Reasons never echo values (secrets hygiene).
    """
    for key in env:
        if not isinstance(key, str):
            raise PodBootError("env keys must be strings")
        if _is_obvious_secret_name(key):
            raise PodBootError(f"forbidden secret env key on pod: {key}")


def build_boot_env(
    *,
    repo_url: str,
    commit_sha: str,
    master_checkpoint_url: str,
    submission_id: str,
    attempt: int,
    **non_secret: str,
) -> dict[str, str]:
    """Build the non-secret env map injected into the pod ENTRYPOINT process.

    Always includes::

        PRISM_REPO_URL, PRISM_COMMIT_SHA, PRISM_MASTER_CHECKPOINT_URL,
        PRISM_SUBMISSION_ID, PRISM_ATTEMPT

    Extra ``non_secret`` kwargs are admitted only when they are not secret-shaped.
    Never emits HF/LIUM tokens. Checkpoint egress uses signed HTTP to master
    (see :mod:`prism_challenge.evaluator.checkpoint_push`); master holds the HF token.
    """
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise PodBootError("attempt must be an integer >= 1")
    if not isinstance(submission_id, str) or not submission_id.strip():
        raise PodBootError("submission_id must be a non-empty string")
    if _SHELL_METACHAR_RE.search(submission_id):
        raise PodBootError("submission_id contains invalid characters")

    safe_repo = validate_repo_url(repo_url)
    safe_sha = validate_commit_sha(commit_sha)
    safe_master = _validate_master_checkpoint_url(master_checkpoint_url)

    env: dict[str, str] = {
        "PRISM_REPO_URL": safe_repo,
        "PRISM_COMMIT_SHA": safe_sha,
        "PRISM_MASTER_CHECKPOINT_URL": safe_master,
        "PRISM_SUBMISSION_ID": submission_id.strip(),
        "PRISM_ATTEMPT": str(attempt),
    }

    for key, value in non_secret.items():
        if not isinstance(key, str) or not key:
            raise PodBootError("extra env key must be a non-empty string")
        if _is_obvious_secret_name(key):
            raise PodBootError(f"forbidden secret env key on pod: {key}")
        if key in _BOOT_REQUIRED_KEYS:
            raise PodBootError(f"cannot override required boot key via kwargs: {key}")
        if not isinstance(value, str):
            raise PodBootError(f"env value for {key} must be a string")
        env[key] = value

    assert_no_forbidden_env(env)
    return env


def _validate_master_checkpoint_url(url: str) -> str:
    """Master checkpoint push URL: https only, no secrets/metachar, no file://."""
    if not isinstance(url, str) or not url:
        raise PodBootError("master checkpoint URL must be a non-empty string")
    if url != url.strip():
        raise PodBootError("master checkpoint URL must not have leading/trailing whitespace")
    if _SHELL_METACHAR_RE.search(url):
        raise PodBootError("master checkpoint URL contains shell metacharacters")
    lower = url.lower()
    if lower.startswith("file:") or lower.startswith("javascript:"):
        raise PodBootError("master checkpoint URL scheme is not allowed")
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        raise PodBootError("master checkpoint URL must use https")
    if parsed.username is not None or parsed.password is not None:
        raise PodBootError("master checkpoint URL must not embed credentials")
    if not parsed.hostname or not parsed.path:
        raise PodBootError("master checkpoint URL must include host and path")
    return url
