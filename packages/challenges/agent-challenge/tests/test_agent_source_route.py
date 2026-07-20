from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from _routing import public_route_paths

from agent_challenge.app import app
from agent_challenge.evaluation.task_events import redact_secrets
from agent_challenge.models import AgentSubmission, SubmissionArtifact
from agent_challenge.submissions.artifacts import store_zip_bytes

ENTRYPOINT_SOURCE = "class Agent:\n    pass\n"


def _mk(*parts: str) -> str:
    """Assemble a secret-shaped test fixture from fragments so it is not a
    contiguous literal (avoids GitHub secret-scanning push protection). The
    runtime value is identical to the concatenation of ``parts``, so redaction
    still sees the same input and the assertions below keep passing."""
    return "".join(parts)


# Fabricated, non-functional credentials used only to exercise each redaction
# shape. Every provider-format token is assembled from fragments via ``_mk`` so
# the raw contiguous secret literal never appears in the committed source (GitHub
# secret-scanning push protection rejects such literals). The assembled runtime
# values are unchanged, so they remain valid inputs for the redaction regexes.
_FAKE_GITHUB_PAT = _mk("ghp", "_", "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8")
_FAKE_GITHUB_FINE = _mk(
    "github", "_pat_", "11ABCDEF0aBcDeFgHiJkL_", "MnOpQrStUvWx0123456789AbCdEfGh"
)
_FAKE_GITLAB_PAT = _mk("glpat", "-", "ABCDEFGHIJ1234567890xy")
_FAKE_SLACK_TOKEN = _mk(
    "xox", "b-", "2222222222", "-", "3333333333", "-", "abcdEFGHijklMNOPqrstUVWX"
)
_FAKE_GOOGLE_KEY = _mk("AIz", "a", "B1234567890C1234567890D1234567890EF")
_FAKE_STRIPE_SECRET = _mk("sk", "_live_", "0123456789abcdefghijKLMN")
_FAKE_STRIPE_PUBLISHABLE = _mk("pk", "_live_", "0123456789abcdefghijKLMN")
_FAKE_AWS_ACCESS_KEY = _mk("AKIA", "1234567890ABCDEF")
_FAKE_PEM_EC_KEY = _mk(
    "-----BEGIN EC ",
    "PRIVATE KEY-----\n",
    "BODYbody123\n",
    "-----END EC ",
    "PRIVATE KEY-----",
)


def _agent_source(contents: str | bytes) -> str | bytes:
    if isinstance(contents, bytes):
        return contents
    if "class Agent" in contents:
        return contents
    return f"{ENTRYPOINT_SOURCE}\n{contents}"


def _zip_bytes(entries: dict[str, str | bytes]) -> bytes:
    buffer = io.BytesIO()
    archive_entries: dict[str, str | bytes] = {"agent.py": ENTRYPOINT_SOURCE, **entries}
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, contents in archive_entries.items():
            payload = _agent_source(contents) if filename == "agent.py" else contents
            data = payload.encode("utf-8") if isinstance(payload, str) else payload
            archive.writestr(filename, data)
    return buffer.getvalue()


async def _seed_submission(
    session,
    tmp_path: Path,
    agent_hash: str,
    entries: dict[str, str | bytes],
    *,
    store: bool = True,
    miner_hotkey: str = "miner-src",
    name: str = "src-agent",
) -> int:
    metadata = store_zip_bytes(zip_bytes=_zip_bytes(entries), artifact_root=str(tmp_path))
    submission = AgentSubmission(
        miner_hotkey=miner_hotkey,
        name=name,
        agent_name=name,
        agent_hash=agent_hash,
        artifact_uri=metadata.artifact_path,
        artifact_path=metadata.artifact_path,
        zip_sha256=metadata.zip_sha256,
        zip_size_bytes=metadata.zip_size_bytes,
        raw_status="received",
        effective_status="received",
    )
    session.add(submission)
    await session.flush()
    session.add(
        SubmissionArtifact(
            submission_id=submission.id,
            artifact_kind="source_zip",
            uri=metadata.artifact_path,
            sha256=metadata.zip_sha256,
            size_bytes=metadata.zip_size_bytes,
            metadata_json=json.dumps(
                {
                    "content_type": "application/zip",
                    "manifest_path": metadata.manifest_path,
                    "manifest": metadata.manifest.to_dict(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    )
    submission_id = submission.id
    await session.commit()
    if not store:
        Path(metadata.artifact_path).unlink()
    return submission_id


async def test_agent_source_returns_files_and_contents(client, database_session, tmp_path):
    async with database_session() as session:
        submission_id = await _seed_submission(
            session,
            tmp_path,
            "hash-basic",
            {"helper.py": "print('hi')\n"},
        )

    response = await client.get("/agents/hash-basic/source")

    assert response.status_code == 200
    assert response.json() == {
        "agent_hash": "hash-basic",
        "submission_id": submission_id,
        "agent_name": "src-agent",
        "miner_hotkey": "miner-src",
        "available": True,
        "total_files": 2,
        "total_bytes": 34,
        "truncated": False,
        "files": [
            {
                "path": "agent.py",
                "size_bytes": 22,
                "content": "class Agent:\n    pass\n",
                "truncated": False,
                "redacted": False,
                "binary": False,
            },
            {
                "path": "helper.py",
                "size_bytes": 12,
                "content": "print('hi')\n",
                "truncated": False,
                "redacted": False,
                "binary": False,
            },
        ],
    }


async def test_agent_source_unknown_hash_returns_404(client):
    response = await client.get("/agents/does-not-exist/source")

    assert response.status_code == 404


async def test_agent_source_available_false_when_zip_missing(client, database_session, tmp_path):
    async with database_session() as session:
        await _seed_submission(
            session,
            tmp_path,
            "hash-missing",
            {"helper.py": "x = 1\n"},
            store=False,
        )

    response = await client.get("/agents/hash-missing/source")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["files"] == []
    assert payload["total_files"] == 0
    assert payload["total_bytes"] == 0
    assert payload["truncated"] is False
    assert payload["agent_hash"] == "hash-missing"


async def test_agent_source_available_false_when_no_artifact_row(client, database_session):
    async with database_session() as session:
        session.add(
            AgentSubmission(
                miner_hotkey="miner-src",
                name="no-artifact",
                agent_name="no-artifact",
                agent_hash="hash-no-artifact",
                artifact_uri="/tmp/does-not-exist.zip",
                raw_status="received",
                effective_status="received",
            )
        )
        await session.commit()

    response = await client.get("/agents/hash-no-artifact/source")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["total_files"] == 0
    assert payload["files"] == []


async def test_agent_source_redacts_every_secret_pattern(client, database_session, tmp_path):
    openai_key = _mk("sk", "-", "abcdefghijklmnopqrstuvwxyz012345")
    anthropic_key = _mk("sk", "-ant-", "api03-", "ABCDEFabcdefghijklmnopqrstuvwxyz0123")
    pem_block = _mk(
        "-----BEGIN RSA ",
        "PRIVATE KEY-----\n",
        "FAKEKEYBODY\n",
        "-----END RSA ",
        "PRIVATE KEY-----",
    )
    leak = (
        "# openai style\n"
        f"{openai_key}\n"
        "# anthropic style\n"
        f"{anthropic_key}\n"
        "# bearer token\n"
        "Bearer abcdefghijklmnopqrstuv.wxyz1234\n"
        "# aws access key\n"
        f"{_FAKE_AWS_ACCESS_KEY}\n"
        "# pem block\n"
        f"{pem_block}\n"
        "# quoted assignment\n"
        'DATABASE_PASSWORD = "s3cr3tP@ssw0rd"\n'
        "# unquoted assignment\n"
        "MY_SECRET_TOKEN=deadbeef12345678\n"
    )
    async with database_session() as session:
        await _seed_submission(session, tmp_path, "hash-secrets", {"leak.py": leak})

    response = await client.get("/agents/hash-secrets/source")

    assert response.status_code == 200
    payload = response.json()
    file = next(item for item in payload["files"] if item["path"] == "leak.py")
    content = file["content"]

    assert file["redacted"] is True
    assert file["binary"] is False
    # OpenAI / Anthropic style keys.
    assert "sk-[REDACTED]" in content
    assert "abcdefghijklmnopqrstuvwxyz012345" not in content
    assert "ABCDEFabcdefghijklmnopqrstuvwxyz0123" not in content
    # Bearer token.
    assert "Bearer [REDACTED]" in content
    assert "abcdefghijklmnopqrstuv.wxyz1234" not in content
    # AWS access key id.
    assert _FAKE_AWS_ACCESS_KEY not in content
    # PEM private-key block.
    assert "PRIVATE KEY" not in content
    assert "FAKEKEYBODY" not in content
    # key-like assignments keep the name, drop the value.
    assert "DATABASE_PASSWORD" in content
    assert "s3cr3tP@ssw0rd" not in content
    assert "MY_SECRET_TOKEN" in content
    assert "deadbeef12345678" not in content


async def test_agent_source_enforces_per_file_and_total_caps(
    client,
    database_session,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("agent_challenge.api.routes.PUBLIC_SOURCE_MAX_FILE_BYTES", 16)
    monkeypatch.setattr("agent_challenge.api.routes.PUBLIC_SOURCE_MAX_TOTAL_BYTES", 32)
    big = "abcdefghijklmnopqrstuvwxyz0123456789\n"
    async with database_session() as session:
        await _seed_submission(
            session,
            tmp_path,
            "hash-caps",
            {"b_file.py": big, "c_file.py": big},
        )

    response = await client.get("/agents/hash-caps/source")

    assert response.status_code == 200
    payload = response.json()
    assert payload["truncated"] is True
    paths = [item["path"] for item in payload["files"]]
    # agent.py + b_file.py fill the 32-byte total budget; c_file.py is omitted.
    assert "c_file.py" not in paths
    assert any(item["truncated"] for item in payload["files"])
    for item in payload["files"]:
        if item["content"] is not None:
            assert len(item["content"].encode("utf-8")) <= 16


async def test_agent_source_handles_binary_files(client, database_session, tmp_path):
    async with database_session() as session:
        await _seed_submission(
            session,
            tmp_path,
            "hash-binary",
            {"blob.bin": b"\x00\x01\x02\x03BINARY\xff\xfe"},
        )

    response = await client.get("/agents/hash-binary/source")

    assert response.status_code == 200
    payload = response.json()
    blob = next(item for item in payload["files"] if item["path"] == "blob.bin")
    assert blob["binary"] is True
    assert blob["content"] is None
    assert blob["truncated"] is False
    assert blob["redacted"] is False
    agent = next(item for item in payload["files"] if item["path"] == "agent.py")
    assert agent["binary"] is False
    assert agent["content"] is not None


def test_agent_source_route_is_public():
    assert "/agents/{agent_hash}/source" in public_route_paths(app)


def test_agent_source_download_route_is_public():
    assert "/agents/{agent_hash}/source/download" in public_route_paths(app)


async def test_agent_source_download_returns_zip_with_headers(client, database_session, tmp_path):
    async with database_session() as session:
        await _seed_submission(
            session,
            tmp_path,
            "hash-download",
            {"pkg/helper.py": "print('hi')\n"},
        )

    response = await client.get("/agents/hash-download/source/download")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    # agent_hash[:8] == "hash-dow"
    assert response.headers["content-disposition"] == 'attachment; filename="agent-hash-dow.zip"'
    assert response.headers["cache-control"] == (
        "public, max-age=300, s-maxage=300, stale-while-revalidate=86400"
    )

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert archive.testzip() is None
    names = set(archive.namelist())
    assert "agent.py" in names
    assert "pkg/helper.py" in names
    assert archive.read("pkg/helper.py").decode("utf-8") == "print('hi')\n"


async def test_agent_source_download_redacts_secret_in_zip(client, database_session, tmp_path):
    openai_key = _mk("sk", "-", "abcdefghijklmnopqrstuvwxyz012345")
    leak = f"# leaked key\n{openai_key}\n"
    async with database_session() as session:
        await _seed_submission(session, tmp_path, "hash-dl-secret", {"leak.py": leak})

    response = await client.get("/agents/hash-dl-secret/source/download")

    assert response.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    content = archive.read("leak.py").decode("utf-8")
    assert "abcdefghijklmnopqrstuvwxyz012345" not in content
    assert "sk-[REDACTED]" in content


async def test_agent_source_download_includes_binary_unchanged(client, database_session, tmp_path):
    blob = b"\x00\x01\x02\x03BINARY\xff\xfe"
    async with database_session() as session:
        await _seed_submission(session, tmp_path, "hash-dl-binary", {"blob.bin": blob})

    response = await client.get("/agents/hash-dl-binary/source/download")

    assert response.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert archive.read("blob.bin") == blob


async def test_agent_source_download_truncates_over_total_cap(
    client,
    database_session,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("agent_challenge.api.routes.PUBLIC_SOURCE_MAX_FILE_BYTES", 16)
    monkeypatch.setattr("agent_challenge.api.routes.PUBLIC_SOURCE_MAX_TOTAL_BYTES", 32)
    big = "abcdefghijklmnopqrstuvwxyz0123456789\n"
    async with database_session() as session:
        await _seed_submission(
            session,
            tmp_path,
            "hash-dl-caps",
            {"b_file.py": big, "c_file.py": big},
        )

    response = await client.get("/agents/hash-dl-caps/source/download")

    assert response.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = set(archive.namelist())
    # agent.py + b_file.py fill the 32-byte total budget; c_file.py is omitted.
    assert "TRUNCATED.txt" in names
    assert "c_file.py" not in names


async def test_agent_source_download_unknown_hash_returns_404(client):
    response = await client.get("/agents/does-not-exist/source/download")

    assert response.status_code == 404


async def test_agent_source_download_missing_zip_returns_404(client, database_session, tmp_path):
    async with database_session() as session:
        await _seed_submission(
            session,
            tmp_path,
            "hash-dl-missing",
            {"helper.py": "x = 1\n"},
            store=False,
        )

    response = await client.get("/agents/hash-dl-missing/source/download")

    assert response.status_code == 404


@pytest.mark.parametrize(
    ("raw", "must_not_contain", "must_contain"),
    [
        (
            "use " + _mk("sk", "-", "abcdefghijklmnopqrst012345") + " now",
            ["abcdefghijklmnopqrst012345"],
            ["sk-[REDACTED]"],
        ),
        (
            "key " + _mk("sk", "-ant-", "0123456789abcdefghijABCDEFGHIJ") + " end",
            ["0123456789abcdefghijABCDEFGHIJ"],
            ["sk-[REDACTED]"],
        ),
        (
            "Bearer abcdefghijklmnop1234.QRST",
            ["abcdefghijklmnop1234.QRST"],
            ["Bearer [REDACTED]"],
        ),
        (
            "cred " + _FAKE_AWS_ACCESS_KEY + " here",
            [_FAKE_AWS_ACCESS_KEY],
            ["[REDACTED]"],
        ),
        (
            _FAKE_PEM_EC_KEY,
            ["PRIVATE KEY", "BODYbody123"],
            ["[REDACTED]"],
        ),
        (
            'api_key = "s3cr3t-value-here"',
            ["s3cr3t-value-here"],
            ["api_key", "[REDACTED]"],
        ),
        (
            "AUTH_TOKEN=abcdef123456",
            ["abcdef123456"],
            ["AUTH_TOKEN", "[REDACTED]"],
        ),
        (
            '{"password": "hunter2pass"}',
            ["hunter2pass"],
            ["password", "[REDACTED]"],
        ),
    ],
)
def test_redact_secrets_masks_each_pattern(raw, must_not_contain, must_contain):
    redacted, changed = redact_secrets(raw)

    assert changed is True
    for fragment in must_not_contain:
        assert fragment not in redacted
    for fragment in must_contain:
        assert fragment in redacted


def test_redact_secrets_leaves_plain_source_untouched():
    text = "def get_token():\n    return compute()\n\nvalue = total + 1\n"

    redacted, changed = redact_secrets(text)

    assert redacted == text
    assert changed is False


@pytest.mark.parametrize(
    ("raw", "must_not_contain", "must_contain"),
    [
        # URL-embedded credentials keep the scheme + host, mask user:password.
        (
            "postgres://svc_user:s3cr3t_pw@db.internal:5432/appdb",
            ["svc_user", "s3cr3t_pw"],
            ["postgres://[REDACTED]@db.internal:5432/appdb"],
        ),
        (
            f"git clone https://gituser:{_FAKE_GITHUB_PAT}@github.com/org/repo.git",
            [_FAKE_GITHUB_PAT, "gituser"],
            ["https://[REDACTED]@github.com"],
        ),
        # Bare provider tokens (distinctive prefixes -> always masked).
        (f"# ref {_FAKE_GITHUB_PAT} end", [_FAKE_GITHUB_PAT], ["[REDACTED]"]),
        (f"# ref {_FAKE_GITHUB_FINE} end", [_FAKE_GITHUB_FINE], ["[REDACTED]"]),
        (f"# ref {_FAKE_GITLAB_PAT} end", [_FAKE_GITLAB_PAT], ["[REDACTED]"]),
        (f"# ref {_FAKE_SLACK_TOKEN} end", [_FAKE_SLACK_TOKEN], ["[REDACTED]"]),
        (f"# ref {_FAKE_GOOGLE_KEY} end", [_FAKE_GOOGLE_KEY], ["[REDACTED]"]),
        (f"# ref {_FAKE_STRIPE_SECRET} end", [_FAKE_STRIPE_SECRET], ["[REDACTED]"]),
        # Unquoted .env/config assignments, incl. all-alpha and lowercase values.
        ("PASSWORD=SuperSecret", ["SuperSecret"], ["PASSWORD=[REDACTED]"]),
        ("api_key=abcdefgh", ["abcdefgh"], ["api_key=[REDACTED]"]),
        # Quoted assignments (JSON + Python), masked regardless of digits.
        ('{"access_key": "ABCdef"}', ["ABCdef"], ["access_key", "[REDACTED]"]),
        ('signing_key = "xyzvalue"', ["xyzvalue"], ["signing_key", "[REDACTED]"]),
        # Triple-quoted value body is masked (delimiters kept).
        ('SECRET = """topsecretbody"""', ["topsecretbody"], ["SECRET", "[REDACTED]", '"""']),
    ],
)
def test_redact_secrets_masks_new_secret_shapes(raw, must_not_contain, must_contain):
    redacted, changed = redact_secrets(raw)

    assert changed is True
    for fragment in must_not_contain:
        assert fragment not in redacted
    for fragment in must_contain:
        assert fragment in redacted


@pytest.mark.parametrize(
    "text",
    [
        "token = get_token()",
        "api_key = config.api_key",
        'password = os.environ["PW"]',
        "self.secret = None",
        'AUTHOR = "Jane"',
        'MONKEY = "banana"',
        'KEYBOARD_LAYOUT = "qwerty"',
        "def get_secret():\n    return 1\n",
        "import secrets\n",
        "from os import environ\n",
        'password = ""',
        "token = None",
        "password = get_password()",
        'secret = os.environ["X"]',
        # A publishable Stripe key must never be masked.
        f"stripe_publishable = {_FAKE_STRIPE_PUBLISHABLE}\n",
    ],
)
def test_redact_secrets_leaves_code_untouched(text):
    redacted, changed = redact_secrets(text)

    assert redacted == text
    assert changed is False
