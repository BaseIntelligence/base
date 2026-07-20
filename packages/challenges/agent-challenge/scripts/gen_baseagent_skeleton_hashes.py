#!/usr/bin/env python3
"""Regenerate the canonical baseagent skeleton fingerprint manifest.

Every miner forks the same ``baseagent`` (github ``BaseIntelligence/baseagent``).
Its Python skeleton (``agent.py`` + ``src/**``) is shared boilerplate that must be
subtracted before similarity scoring so the score reflects each miner's DELTA
rather than the common base. This script fingerprints that skeleton with the
*same* extraction the analyzer uses for submissions
(``ast_features.extract_python_ast_features`` over a ``ZipArtifactManifest``), so
the stored ``ast_hash`` / ``file_hash`` values are identical to how submissions are
fingerprinted, and writes them to
``src/agent_challenge/analyzer/baseagent-skeleton-hashes.json`` (packaged next to
the analyzer so the manifest ships in every image/install).

Usage:
    uv run python scripts/gen_baseagent_skeleton_hashes.py \
        --baseagent /path/to/baseagent [--git-ref <sha>] [--output <path>]

The default baseagent path is a sibling ``baseagent`` checkout next to this repo.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_challenge.analyzer.ast_features import (  # noqa: E402
    AST_STATUS_OK,
    extract_python_ast_features,
)
from agent_challenge.analyzer.similarity import ALGORITHM_VERSION  # noqa: E402
from agent_challenge.submissions.artifacts import (  # noqa: E402
    ArtifactReadSession,
    build_zip_manifest,
)

DEFAULT_BASEAGENT_PATH = REPO_ROOT.parent / "baseagent"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "src" / "agent_challenge" / "analyzer" / "baseagent-skeleton-hashes.json"
)
BASEAGENT_REPO = "https://github.com/BaseIntelligence/baseagent.git"
#: The skeleton miners fork: the root entrypoint plus the ``src`` package tree.
SKELETON_GLOBS = ("agent.py", "src/**/*.py")


def _collect_skeleton_files(baseagent_path: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for pattern in SKELETON_GLOBS:
        for path in sorted(baseagent_path.glob(pattern)):
            if not path.is_file():
                continue
            files[path.relative_to(baseagent_path).as_posix()] = path.read_bytes()
    return files


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(files.items()):
            archive.writestr(name, payload)
    return buffer.getvalue()


def _fingerprint(files: dict[str, bytes]) -> tuple[list[str], list[str], int]:
    zip_bytes = _zip_bytes(files)
    # The submission read path is deliberately budget-capped, but the base
    # fingerprint must cover EVERY skeleton file so any base file that a
    # submission happens to fingerprint is recognized and subtracted. Read the
    # whole skeleton (budgets >= its total size); the content-derived ast_hash /
    # file_hash are identical to how a submission fingerprints the same file.
    budget = sum(len(payload) for payload in files.values()) + 1
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "baseagent.zip"
        zip_path.write_bytes(zip_bytes)
        manifest = build_zip_manifest(zip_bytes=zip_bytes, artifact_reference=str(zip_path))
        report = extract_python_ast_features(
            manifest=manifest,
            read_session=ArtifactReadSession(
                zip_path=zip_path,
                manifest=manifest,
                per_read_max_bytes=budget,
                total_read_budget=budget,
            ),
        )
    ast_hashes: set[str] = set()
    file_hashes: set[str] = set()
    fingerprinted = 0
    for file_result in report.files:
        if file_result.status != AST_STATUS_OK or not file_result.ast_hash:
            continue
        fingerprinted += 1
        ast_hashes.add(file_result.ast_hash)
        file_hashes.add(file_result.file_hash)
    return sorted(ast_hashes), sorted(file_hashes), fingerprinted


def _git_ref(baseagent_path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(baseagent_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the baseagent skeleton fingerprint.")
    parser.add_argument("--baseagent", type=Path, default=DEFAULT_BASEAGENT_PATH)
    parser.add_argument("--git-ref", default=None, help="override the recorded baseagent git ref")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    baseagent_path = args.baseagent.expanduser().resolve()
    if not baseagent_path.is_dir():
        print(f"baseagent path not found: {baseagent_path}", file=sys.stderr)
        return 1

    files = _collect_skeleton_files(baseagent_path)
    if "agent.py" not in files:
        print(f"agent.py not found under {baseagent_path}", file=sys.stderr)
        return 1

    ast_hashes, file_hashes, fingerprinted = _fingerprint(files)
    git_ref = args.git_ref or _git_ref(baseagent_path)
    manifest = {
        "algorithm_version": ALGORITHM_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "generated_from": {
            "repo": BASEAGENT_REPO,
            "git_ref": git_ref,
            "path_globs": list(SKELETON_GLOBS),
        },
        "skeleton_file_count": len(files),
        "fingerprinted_file_count": fingerprinted,
        "ast_hashes": ast_hashes,
        "file_hashes": file_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} "
        f"({fingerprinted} files fingerprinted, "
        f"{len(ast_hashes)} ast hashes, {len(file_hashes)} file hashes, git_ref={git_ref})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
