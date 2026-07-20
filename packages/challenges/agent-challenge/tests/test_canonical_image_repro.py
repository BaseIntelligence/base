"""Behavioral tests for the canonical, reproducibly-built eval image (M1).

Fulfils VAL-IMG-001..005:
  * VAL-IMG-001 canonical image builds reproducibly to an identical digest
  * VAL-IMG-002 image reference + build inputs are digest-pinned (no floating tags)
  * VAL-IMG-003 a non-reproducible build input is detected by the repro guard
  * VAL-IMG-004 image wraps the existing own_runner eval unchanged
  * VAL-IMG-005 image contains no secrets (golden plaintext / phala / provider keys)

The offline/static assertions run everywhere; the assertions that need a real
image build are guarded on ``docker buildx`` availability.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from agent_challenge.canonical import build as cbuild
from agent_challenge.canonical import entrypoint, secrets_scan

REPO_ROOT = cbuild.REPO_ROOT
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

_BUILDX = cbuild.buildx_available()
docker_required = pytest.mark.skipif(not _BUILDX, reason="docker buildx not available")


# --------------------------------------------------------------------------- #
# VAL-IMG-002 (static): digest-pinned base, no floating tags, locked deps
# --------------------------------------------------------------------------- #


def test_canonical_dockerfile_and_requirements_exist():
    assert cbuild.CANONICAL_DOCKERFILE.is_file()
    assert cbuild.CANONICAL_REQUIREMENTS.is_file()


def test_base_image_is_digest_pinned_no_floating_tag():
    report = cbuild.validate_build_definition(cbuild.CANONICAL_DOCKERFILE.read_text())
    assert report.resolved_bases, "no FROM base image found"
    assert report.floating_tags == [], f"floating tags present: {report.floating_tags}"
    assert report.digest_pinned
    assert all(cbuild.DIGEST_PIN_RE.search(b) for b in report.resolved_bases)


def test_python_dependencies_are_locked_and_hashed():
    text = cbuild.CANONICAL_REQUIREMENTS.read_text()
    assert cbuild.requirements_are_hash_pinned(text)
    parsed = cbuild.parse_requirements(text)
    assert parsed, "no requirements parsed"
    for req in parsed:
        assert req.version, f"{req.name} is not pinned to an exact version"
        assert req.hashes, f"{req.name} has no --hash"


def test_uv_lock_present_as_lockfile():
    assert (REPO_ROOT / "uv.lock").is_file()


def test_validate_build_definition_flags_unpinned_base():
    bad = "FROM python:3.12-slim\nRUN echo hi\n"
    report = cbuild.validate_build_definition(bad)
    assert not report.digest_pinned
    assert "python:3.12-slim" in report.floating_tags


def test_validate_build_definition_resolves_arg_default_base():
    text = (
        "ARG BASE_IMAGE=python:3.12-slim@sha256:" + ("a" * 64) + "\n"
        "FROM ${BASE_IMAGE}\n"
        "RUN echo ok\n"
    )
    report = cbuild.validate_build_definition(text)
    assert report.digest_pinned
    assert report.floating_tags == []


def test_requirements_not_hash_pinned_is_detected():
    assert not cbuild.requirements_are_hash_pinned("pydantic==2.13.4\n")
    assert not cbuild.requirements_are_hash_pinned("pydantic>=2\n")


# --------------------------------------------------------------------------- #
# VAL-IMG-004 (source presence + entrypoint): own_runner wrap
# --------------------------------------------------------------------------- #


def test_own_runner_modules_present_in_source_tree():
    own = REPO_ROOT / "src" / "agent_challenge" / "evaluation" / "own_runner"
    for module in (
        "orchestrator.py",
        "container_builder.py",
        "result_schema.py",
        "taskdefs.py",
        "reward.py",
        "verifier_runner.py",
    ):
        assert (own / module).is_file(), module
    assert (
        REPO_ROOT / "src" / "agent_challenge" / "evaluation" / "own_runner_backend.py"
    ).is_file()


def test_entrypoint_help_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        entrypoint.main(["--help"])
    assert excinfo.value.code == 0


def test_entrypoint_check_verifies_own_runner_modules(capsys):
    rc = entrypoint.main(["check"])
    assert rc == 0
    assert "own_runner" in capsys.readouterr().out


def test_dockerfile_entrypoint_targets_canonical_module():
    text = cbuild.CANONICAL_DOCKERFILE.read_text()
    assert "agent_challenge.canonical.entrypoint" in text


def test_entrypoint_run_delegates_to_own_runner_backend(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args
        return 7

    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend.main", fake_main, raising=True
    )
    rc = entrypoint.main(["run", "run", "--job-dir", "/tmp/job"])
    assert rc == 7
    assert captured["args"] == ["run", "--job-dir", "/tmp/job"]


def test_entrypoint_normalizes_compose_single_run_without_double_token(monkeypatch):
    """Measured compose command is one ``run``; backend still receives the subcommand."""

    captured = {}

    def fake_main(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(
        "agent_challenge.evaluation.own_runner_backend.main", fake_main, raising=True
    )
    rc = entrypoint.main(["run", "--job-dir", "/tmp/job", "--cache-root", "/tmp/cache"])
    assert rc == 0
    assert captured["args"] == ["run", "--job-dir", "/tmp/job", "--cache-root", "/tmp/cache"]


def test_entrypoint_check_fails_when_module_missing(monkeypatch, tmp_path):
    fake_pkg = tmp_path / "agent_challenge" / "__init__.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("")
    import agent_challenge

    monkeypatch.setattr(agent_challenge, "__file__", str(fake_pkg), raising=True)
    with pytest.raises(RuntimeError, match="own_runner modules missing"):
        entrypoint.main(["check"])


# --------------------------------------------------------------------------- #
# VAL-IMG-005 (scanner unit): secret detection is a real discriminator
# --------------------------------------------------------------------------- #


def test_secret_scanner_detects_each_class(tmp_path):
    samples = {
        "phala.txt": "PHALA_CLOUD_API_KEY=phak_0123456789abcdef0123456789abcdef",
        "anthropic.txt": "key=sk-ant-0123456789abcdef0123456789",
        "aws.txt": "AKIAABCDEFGHIJKLMNOP",
        # Marker assembled from fragments so this test source never itself trips
        # a golden-plaintext repo scan; the written sample still carries it.
        "golden.json": '{"schema": "harbor-independence/' + 'oracle-golden@1"}',
        "id.pem": "-----BEGIN PRIVATE KEY-----\nMIIB\n-----END PRIVATE KEY-----\n",
    }
    for name, body in samples.items():
        (tmp_path / name).write_text(body)
    hits = secrets_scan.scan_path(tmp_path)
    found = {h.pattern for h in hits}
    assert {
        "phala_api_key",
        "anthropic_key",
        "aws_access_key",
        "golden_oracle_plaintext",
        "pem_private_key",
    } <= found


def test_secret_scanner_detects_legacy_encrypted_pem(tmp_path):
    # A legacy ("traditional" OpenSSL) encrypted PEM carries Proc-Type / DEK-Info
    # header lines between the BEGIN marker and the base64 body. Those headers must
    # not blind the scanner: real legacy private keys are still caught, not only
    # modern PKCS#8. (Discriminator: the pre-broadening body class rejected the
    # ':'/','/'-' header chars, so this legacy key slipped through undetected.)
    legacy_pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "Proc-Type: 4,ENCRYPTED\n"
        "DEK-Info: DES-EDE3-CBC,9F1E2D3C4B5A6978\n"
        "\n"
        "MIIBOwIBAAJBAKj34Gkxu2F3l8p1w0k0g6t5n2K2xVQZ0m8dQ1c3n5r7s9t0u1v2\n"
        "w3x4y5z6A7B8C9D0E1F2G3H4I5J6K7L8M9N0O1P2Q3R4S5T6U7V8W9X0Y1Z2a3b4\n"
        "-----END RSA PRIVATE KEY-----\n"
    )
    (tmp_path / "legacy.pem").write_text(legacy_pem)
    hits = {h.pattern for h in secrets_scan.scan_path(tmp_path)}
    assert "pem_private_key" in hits


def test_secret_scanner_no_false_positive_on_cryptography_bare_headers():
    # cryptography embeds bare PEM header/footer strings as module constants (e.g.
    # ssh.py's _SK_START/_SK_END) with no newline+base64 body. Broadening the
    # legacy-PEM coverage must NOT reintroduce a false positive on such bare
    # constants (the M3 tightening that removed them must be preserved).
    bare_constants = (
        '_SK_START = b"-----BEGIN OPENSSH PRIVATE KEY-----"\n'
        '_SK_END = b"-----END OPENSSH PRIVATE KEY-----"\n'
        '_PEM_RC = re.compile(_SK_START + b"(.*?)" + _SK_END, re.DOTALL)\n'
    )
    assert "pem_private_key" not in {h.pattern for h in secrets_scan.scan_text(bare_constants)}

    # The real installed cryptography ssh module source must not self-match either.
    from cryptography.hazmat.primitives.serialization import ssh as _ssh

    ssh_hits = [h for h in secrets_scan.scan_path(_ssh.__file__) if h.pattern == "pem_private_key"]
    assert ssh_hits == []


def test_secret_scanner_clean_tree_has_no_hits(tmp_path):
    (tmp_path / "readme.txt").write_text("just some ordinary text, nothing secret here")
    assert secrets_scan.scan_path(tmp_path) == []


def test_secret_scanner_does_not_report_secret_values(tmp_path):
    (tmp_path / "phala.txt").write_text("phak_0123456789abcdef0123456789abcdef")
    hits = secrets_scan.scan_path(tmp_path)
    assert hits
    for hit in hits:
        assert "0123456789abcdef" not in repr(hit)


# --------------------------------------------------------------------------- #
# Docker-backed assertions (real build)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def repro_digests(tmp_path_factory) -> list[str]:
    dest = tmp_path_factory.mktemp("repro")
    result = cbuild.check_reproducible(builds=2, dest_dir=dest)
    return result.digests


@pytest.fixture(scope="module")
def loaded_image() -> str:
    tag = "agent-challenge-canonical:pytest"
    cbuild.build_image(load_tag=tag, no_cache=False)
    yield tag
    subprocess.run(["docker", "image", "rm", "-f", tag], capture_output=True, text=True)


@docker_required
def test_canonical_build_reproducible_digest(repro_digests):
    # VAL-IMG-001
    assert len(repro_digests) == 2
    for digest in repro_digests:
        assert DIGEST_RE.match(digest), digest
    assert repro_digests[0] == repro_digests[1], repro_digests


@docker_required
def test_published_reference_is_sha256_digest(repro_digests):
    # VAL-IMG-002 (dynamic): the canonical reference is a sha256 digest
    assert DIGEST_RE.match(repro_digests[0])


@docker_required
def test_nonreproducible_input_is_detected(tmp_path):
    # VAL-IMG-003: inject a build-time timestamp -> two builds diverge
    perturbed = (
        cbuild.CANONICAL_DOCKERFILE.read_text() + "\nRUN date +%s%N > /nondeterministic_marker\n"
    )
    dockerfile = tmp_path / "Dockerfile.perturbed"
    dockerfile.write_text(perturbed)
    result = cbuild.check_reproducible(builds=2, dockerfile=dockerfile, dest_dir=tmp_path)
    assert not result.reproducible, result.digests
    assert result.digests[0] != result.digests[1]


@docker_required
def test_image_entrypoint_help_runs_inside_image(loaded_image):
    # VAL-IMG-004: --help dry invocation exits 0 inside the image
    proc = subprocess.run(
        ["docker", "run", "--rm", loaded_image, "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


@docker_required
def test_image_contains_own_runner_modules(loaded_image):
    # VAL-IMG-004: the entrypoint's dry `check` confirms own_runner modules are
    # present at the expected locations inside the image.
    proc = subprocess.run(
        ["docker", "run", "--rm", loaded_image, "check"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "own_runner" in proc.stdout


@docker_required
def test_image_has_no_secrets(loaded_image, tmp_path):
    # VAL-IMG-005: exported image filesystem has zero secret hits
    export_tar = tmp_path / "image-fs.tar"
    create = subprocess.run(
        ["docker", "create", loaded_image],
        capture_output=True,
        text=True,
    )
    assert create.returncode == 0, create.stderr
    container_id = create.stdout.strip()
    try:
        with export_tar.open("wb") as handle:
            export = subprocess.run(
                ["docker", "export", container_id],
                stdout=handle,
                stderr=subprocess.PIPE,
                text=False,
            )
        assert export.returncode == 0, export.stderr
    finally:
        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, text=True)

    hits = secrets_scan.scan_tar(export_tar)
    assert hits == [], [f"{h.member}:{h.pattern}" for h in hits]


@docker_required
def test_image_filesystem_has_no_golden_plaintext(loaded_image):
    # VAL-IMG-005: golden may be absent OR present only encrypted-at-rest.
    # Live-prep bakes ciphertext (.enc) + digests under /app/golden; plaintext
    # task tests (*.py / tests/ dirs / unencrypted oracle JSON) must never land.
    proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            loaded_image,
            "-c",
            (
                "if [ ! -e /app/golden ]; then echo no-golden; exit 0; fi; "
                # ciphertext present (required when the mount exists)
                "ls /app/golden/*.enc >/dev/null 2>&1 || { echo missing-enc; exit 1; }; "
                # no plaintext test suites / source
                "find /app/golden -type d -name tests | grep -q . "
                "&& { echo plain-tests; exit 1; }; "
                "find /app/golden -type f -name '*.py' | grep -q . "
                "&& { echo plain-py; exit 1; }; "
                # no unencrypted oracle/task JSON (permit only digest/registry pin files)
                "find /app/golden -type f -name '*.json' ! -name '*digest*' "
                "! -name '*registry-refs*' | grep -q . && { echo plain-json; exit 1; }; "
                "echo encrypted-only-ok"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "no-golden" in proc.stdout or "encrypted-only-ok" in proc.stdout
