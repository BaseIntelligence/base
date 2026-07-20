#!/usr/bin/env python3
"""Agent Challenge — A→Z miner submission CLI.

End-to-end helper that takes a miner agent from source to a scored leaderboard
entry against a live Agent Challenge validator (directly or through the BASE
proxy):

  1. package an agent directory into a valid submission ZIP (agent.py at root);
  2. sign the upload with a Bittensor/substrate hotkey using the validator's
     canonical signing scheme;
  3. POST the submission and verify the receipt (zip_sha256 match);
  4. poll public status through the analyzer/env/terminal-bench lifecycle;
  5. confirm-empty (or PUT) miner env so terminal-bench can launch;
  6. stream per-channel task logs (agent | harness | test_stdout | test_stderr);
  7. read the leaderboard.

The script depends only on the Python standard library plus ``bittensor``
(``bittensor.Keypair``, the sr25519 signer every Bittensor miner already has)
for signing. No requests/httpx needed.

Canonical signed-request contract (must match
``agent_challenge.auth.security.canonical_request_string``)::

    {METHOD}
    {PATH_WITH_SORTED_QUERY}
    {X-TIMESTAMP}
    {X-NONCE}
    {SHA256_HEX_OF_RAW_BODY}

Headers sent on every signed request: ``X-Hotkey``, ``X-Signature`` (hex,
``0x``-prefixed), ``X-Nonce`` (unique per request), ``X-Timestamp`` (ISO-8601
UTC, accepted within 300s).

Examples
--------
Generate a throwaway hotkey and submit the bundled example agent::

    python scripts/submit_agent.py submit \\
        --api-base http://localhost:8000 \\
        --agent-dir scripts/example_agent \\
        --name "my-first-agent" \\
        --generate-hotkey --confirm-empty --watch

Submit with an existing hotkey mnemonic and provide env vars::

    export MINER_HOTKEY_MNEMONIC="word1 word2 ... word12"
    python scripts/submit_agent.py submit \\
        --api-base https://base.example/challenges/agent-challenge \\
        --agent-dir ./my-agent --name "my-agent" \\
        --env EXAMPLE_API_TOKEN=<write-only> --watch

Just build a ZIP without submitting::

    python scripts/submit_agent.py build --agent-dir ./my-agent --out ./my-agent.zip

Verify signing locally (no network)::

    python scripts/submit_agent.py selfcheck
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit
from urllib.request import Request, urlopen

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

EMPTY_BODY_SHA256 = hashlib.sha256(b"").hexdigest()
MAX_ZIP_BYTES = 1_048_576  # 1 MiB, matches validator zip_too_large limit.
DEFAULT_TIMEOUT = 30.0
# Browser-like UA: the default "Python-urllib/3.x" is banned by Cloudflare's WAF
# (error 1010) in front of the BASE proxy, blocking every submission POST.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Public terminal raw statuses the lifecycle can settle on.
TERMINAL_RAW_STATUSES = {
    "tb_completed",
    "tb_failed_final",
    "rejected",
    "invalid",
    "error",
    "admin_paused",
    "suspicious",
    "overridden_valid",
    "overridden_invalid",
}
# Raw status at which the miner must provide env (or confirm empty) to proceed.
WAITING_ENV_RAW_STATUS = "waiting_miner_env"
# Log stream channels exposed by the separated-log task-events feature.
LOG_STREAM_CHANNELS = ("agent", "harness", "test_stdout", "test_stderr")


# --------------------------------------------------------------------------- #
# Signing                                                                      #
# --------------------------------------------------------------------------- #


def _load_keypair(args: argparse.Namespace) -> Any:
    """Resolve a substrate ``Keypair`` from CLI flags / environment."""
    try:
        from bittensor import Keypair  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "bittensor is required for signing. Install it with `pip install bittensor`."
        ) from exc

    import os

    if args.generate_hotkey:
        mnemonic = Keypair.generate_mnemonic()
        keypair = Keypair.create_from_mnemonic(mnemonic)
        _eprint(f"[hotkey] generated throwaway hotkey: {keypair.ss58_address}")
        _eprint(f"[hotkey] mnemonic (test only, not registered): {mnemonic}")
        return keypair

    mnemonic = args.hotkey_mnemonic or os.environ.get("MINER_HOTKEY_MNEMONIC")
    if mnemonic:
        return Keypair.create_from_mnemonic(mnemonic.strip())

    uri = args.hotkey_uri or os.environ.get("MINER_HOTKEY_URI")
    if uri:
        return Keypair.create_from_uri(uri.strip())

    if args.wallet_name:
        try:
            import bittensor  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional path
            raise SystemExit("--wallet-name needs the `bittensor` package installed.") from exc
        wallet = bittensor.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
        return wallet.hotkey

    raise SystemExit(
        "No hotkey provided. Use one of: --generate-hotkey, --hotkey-mnemonic, "
        "--hotkey-uri (or MINER_HOTKEY_MNEMONIC / MINER_HOTKEY_URI env), or "
        "--wallet-name/--wallet-hotkey."
    )


def _sorted_path_with_query(path: str, query: str) -> str:
    """Mirror ``auth.security.sorted_path_with_query``."""
    if not query:
        return path
    pairs = sorted(parse_qsl(query, keep_blank_values=True))
    return f"{path}?{urlencode(pairs)}"


def canonical_request_string(
    *, method: str, path: str, query: str, timestamp: str, nonce: str, raw_body: bytes
) -> str:
    """Mirror ``auth.security.canonical_request_string`` exactly."""
    return "\n".join(
        (
            method.upper(),
            _sorted_path_with_query(path, query),
            timestamp,
            nonce,
            hashlib.sha256(raw_body).hexdigest(),
        )
    )


def _sign(keypair: Any, message: str) -> str:
    signature = keypair.sign(message)
    if isinstance(signature, bytes):
        return "0x" + signature.hex()
    text = str(signature)
    return text if text.startswith("0x") else "0x" + text


# --------------------------------------------------------------------------- #
# HTTP                                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class HttpResponse:
    status: int
    body: bytes

    def json(self) -> Any:
        if not self.body:
            return None
        return json.loads(self.body.decode("utf-8"))


class SignedClient:
    """Minimal signed HTTP client for the Agent Challenge API."""

    def __init__(self, api_base: str, keypair: Any, *, timeout: float = DEFAULT_TIMEOUT):
        self.api_base = api_base.rstrip("/")
        self.keypair = keypair
        self.hotkey = keypair.ss58_address
        self.timeout = timeout
        # The signing path is the challenge-LOCAL path, even when the request is
        # routed through the BASE proxy under /challenges/agent-challenge.
        # We sign the path component after the api_base, defaulting to the
        # local route names ("/submissions", ...).
        self._base_path = urlsplit(self.api_base).path.rstrip("/")

    def _sign_path(self, route: str) -> str:
        """The validator signs the challenge-local route (e.g. /submissions)."""
        return route

    def request(
        self,
        method: str,
        route: str,
        *,
        query: dict[str, Any] | None = None,
        body: bytes | None = None,
        signed: bool = False,
        accept: str = "application/json",
    ) -> HttpResponse:
        query_string = urlencode(sorted((query or {}).items())) if query else ""
        url = f"{self.api_base}{route}"
        if query_string:
            url = f"{url}?{query_string}"

        headers = {"Accept": accept, "User-Agent": DEFAULT_USER_AGENT}
        raw_body = body or b""
        if body is not None:
            headers["Content-Type"] = "application/json"

        if signed:
            timestamp = datetime.now(UTC).isoformat()
            nonce = uuid.uuid4().hex
            canonical = canonical_request_string(
                method=method,
                path=self._sign_path(route),
                query=query_string,
                timestamp=timestamp,
                nonce=nonce,
                raw_body=raw_body,
            )
            headers.update(
                {
                    "X-Hotkey": self.hotkey,
                    "X-Signature": _sign(self.keypair, canonical),
                    "X-Nonce": nonce,
                    "X-Timestamp": timestamp,
                }
            )

        req = Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (trusted base url)
                return HttpResponse(status=resp.status, body=resp.read())
        except HTTPError as exc:
            return HttpResponse(status=exc.code, body=exc.read())
        except URLError as exc:  # pragma: no cover - network guard
            raise SystemExit(f"network error contacting {url}: {exc}") from exc


# --------------------------------------------------------------------------- #
# ZIP packaging                                                                #
# --------------------------------------------------------------------------- #


def build_agent_zip(agent_dir: Path) -> bytes:
    """Deterministically package an agent directory into a submission ZIP.

    Requires ``agent.py`` at the root defining a top-level ``class Agent``.
    Skips caches, VCS dirs, and compiled artefacts.
    """
    agent_dir = agent_dir.resolve()
    entrypoint = agent_dir / "agent.py"
    if not entrypoint.is_file():
        raise SystemExit(f"missing required entrypoint: {entrypoint}")
    if "class Agent" not in entrypoint.read_text(encoding="utf-8"):
        raise SystemExit(f"{entrypoint} must define a top-level `class Agent`")

    skip_dirs = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
    skip_suffixes = {".pyc", ".pyo"}

    files: list[Path] = []
    for path in sorted(agent_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(agent_dir).parts
        if any(part in skip_dirs for part in rel_parts):
            continue
        if path.suffix in skip_suffixes:
            continue
        files.append(path)

    buffer = BytesIO()
    # Fixed timestamp -> reproducible archives -> stable zip_sha256.
    fixed_date = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            arcname = path.relative_to(agent_dir).as_posix()
            info = zipfile.ZipInfo(arcname, date_time=fixed_date)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())

    data = buffer.getvalue()
    if len(data) > MAX_ZIP_BYTES:
        raise SystemExit(
            f"packaged ZIP is {len(data)} bytes, exceeds the {MAX_ZIP_BYTES} byte limit"
        )
    return data


# --------------------------------------------------------------------------- #
# Lifecycle helpers                                                            #
# --------------------------------------------------------------------------- #


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def _post_submission(client: SignedClient, *, name: str, zip_bytes: bytes) -> dict[str, Any]:
    payload = {
        "miner_hotkey": client.hotkey,  # informational; validator uses the signed hotkey
        "name": name,
        "artifact_zip_base64": base64.b64encode(zip_bytes).decode("ascii"),
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    resp = client.request("POST", "/submissions", body=body, signed=True)
    data = resp.json()
    if resp.status != 201:
        raise SystemExit(f"submission rejected (HTTP {resp.status}): {json.dumps(data, indent=2)}")

    local_sha = hashlib.sha256(zip_bytes).hexdigest()
    if data.get("zip_sha256") != local_sha:
        raise SystemExit(
            f"receipt zip_sha256 mismatch: server={data.get('zip_sha256')} local={local_sha}"
        )
    _eprint(
        f"[submit] HTTP 201 — submission_id={data['submission_id']} "
        f"version={data.get('version_label')} zip_sha256={local_sha}"
    )
    return data


def _get_status(client: SignedClient, submission_id: int) -> dict[str, Any]:
    resp = client.request("GET", f"/submissions/{submission_id}/status")
    if resp.status != 200:
        raise SystemExit(f"status fetch failed (HTTP {resp.status}): {resp.body!r}")
    return resp.json()


def _raw_status(status_payload: dict[str, Any]) -> str:
    # Public status omits raw_status; infer the gate from phase/status copy.
    phase = status_payload.get("phase")
    if phase == "waiting_environments" or status_payload.get("env_action_required"):
        return WAITING_ENV_RAW_STATUS
    if phase == "complete":
        return "tb_completed"
    if phase == "error":
        return "error"
    return str(status_payload.get("status") or phase or "unknown")


def _confirm_empty_env(client: SignedClient, submission_id: int) -> None:
    resp = client.request(
        "POST", f"/submissions/{submission_id}/env/confirm-empty", body=b"", signed=True
    )
    if resp.status in (200, 201):
        _eprint(f"[env] confirmed-empty accepted (HTTP {resp.status})")
        return
    if resp.status == 409:
        _eprint("[env] env already locked/confirmed (HTTP 409) — continuing")
        return
    raise SystemExit(f"confirm-empty failed (HTTP {resp.status}): {resp.body!r}")


def _put_env(client: SignedClient, submission_id: int, env: dict[str, str]) -> None:
    body = json.dumps({"env": env}, separators=(",", ":")).encode("utf-8")
    resp = client.request("PUT", f"/submissions/{submission_id}/env", body=body, signed=True)
    if resp.status in (200, 201):
        meta = resp.json()
        _eprint(f"[env] stored {meta.get('env_var_count')} var(s): {meta.get('env_keys')}")
        return
    if resp.status == 409:
        _eprint("[env] env already locked (HTTP 409) — continuing")
        return
    raise SystemExit(f"PUT env failed (HTTP {resp.status}): {resp.body!r}")


def _drain_task_events(client: SignedClient, submission_id: int, cursors: dict[str, int]) -> None:
    """Poll the per-channel task-event replay once and print new lines."""
    for stream in (None, *LOG_STREAM_CHANNELS):
        key = stream or "_status"
        cursor = cursors.get(key, 0)
        query: dict[str, Any] = {"cursor": cursor, "limit": 200}
        if stream is not None:
            query["stream"] = stream
        resp = client.request("GET", f"/submissions/{submission_id}/task-events", query=query)
        if resp.status != 200:
            continue
        page = resp.json() or {}
        events = page.get("events", [])
        for event in events:
            label = stream or event.get("event_type", "status")
            message = (event.get("message") or "").rstrip()
            if message:
                _eprint(f"  [{label}] {message[:240]}")
        if page.get("next_cursor") is not None:
            cursors[key] = page["next_cursor"]


def _watch(
    client: SignedClient, submission_id: int, *, poll: float, timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    cursors: dict[str, int] = {}
    last_status = None
    while True:
        status_payload = _get_status(client, submission_id)
        raw = _raw_status(status_payload)
        if status_payload.get("status") != last_status:
            _eprint(
                f"[status] {status_payload.get('status')} "
                f"(phase={status_payload.get('phase')}, "
                f"effective={status_payload.get('effective_status')})"
            )
            last_status = status_payload.get("status")
        _drain_task_events(client, submission_id, cursors)
        if raw in TERMINAL_RAW_STATUSES or status_payload.get("phase") == "complete":
            return status_payload
        if time.monotonic() > deadline:
            _eprint(f"[watch] timeout after {timeout:.0f}s — last status {last_status}")
            return status_payload
        time.sleep(poll)


def _leaderboard(client: SignedClient) -> None:
    resp = client.request("GET", "/leaderboard")
    if resp.status != 200:
        _eprint(f"[leaderboard] fetch failed (HTTP {resp.status})")
        return
    rows = resp.json() or []
    _eprint(f"[leaderboard] {len(rows)} row(s)")
    for row in rows[:20]:
        _eprint(
            f"  {row.get('miner_hotkey')}  score={row.get('score')}  "
            f"passed={row.get('passed_tasks')}/{row.get('total_tasks')}  "
            f"name={row.get('display_name') or row.get('name')}"
        )


# --------------------------------------------------------------------------- #
# Sub-commands                                                                 #
# --------------------------------------------------------------------------- #


def cmd_build(args: argparse.Namespace) -> int:
    zip_bytes = build_agent_zip(Path(args.agent_dir))
    out = Path(args.out)
    out.write_bytes(zip_bytes)
    print(f"wrote {out} ({len(zip_bytes)} bytes, sha256={hashlib.sha256(zip_bytes).hexdigest()})")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    if args.zip:
        zip_bytes = Path(args.zip).read_bytes()
        if len(zip_bytes) > MAX_ZIP_BYTES:
            raise SystemExit(f"ZIP exceeds {MAX_ZIP_BYTES} byte limit")
    else:
        zip_bytes = build_agent_zip(Path(args.agent_dir))
    _eprint(f"[zip] {len(zip_bytes)} bytes, sha256={hashlib.sha256(zip_bytes).hexdigest()}")

    keypair = _load_keypair(args)
    client = SignedClient(args.api_base, keypair, timeout=args.http_timeout)
    _eprint(f"[submit] api_base={client.api_base} hotkey={client.hotkey}")

    receipt = _post_submission(client, name=args.name, zip_bytes=zip_bytes)
    submission_id = receipt["submission_id"]

    env = _parse_env_pairs(args.env)
    if not args.watch:
        print(json.dumps(receipt, indent=2))
        return 0

    # Drive the lifecycle: wait until env gate, then confirm-empty / PUT env.
    deadline = time.monotonic() + args.env_wait
    handled_env = False
    cursors: dict[str, int] = {}
    last_status = None
    while not handled_env:
        status_payload = _get_status(client, submission_id)
        raw = _raw_status(status_payload)
        if status_payload.get("status") != last_status:
            _eprint(
                f"[status] {status_payload.get('status')} (phase={status_payload.get('phase')})"
            )
            last_status = status_payload.get("status")
        _drain_task_events(client, submission_id, cursors)
        if raw == WAITING_ENV_RAW_STATUS:
            if env:
                _put_env(client, submission_id, env)
            else:
                _confirm_empty_env(client, submission_id)
            handled_env = True
        elif raw in TERMINAL_RAW_STATUSES:
            _eprint(f"[lifecycle] reached terminal status before env gate: {raw}")
            _print_final(client, status_payload)
            return 0
        elif time.monotonic() > deadline:
            _eprint(
                f"[lifecycle] env gate not reached within {args.env_wait:.0f}s "
                f"(status={last_status}) — the analyzer may have escalated/rejected"
            )
            _print_final(client, status_payload)
            return 0
        else:
            time.sleep(args.poll)

    final = _watch(client, submission_id, poll=args.poll, timeout=args.watch_timeout)
    _print_final(client, final)
    return 0


def _print_final(client: SignedClient, status_payload: dict[str, Any]) -> None:
    _eprint(
        "[final] "
        + json.dumps(
            {
                "status": status_payload.get("status"),
                "phase": status_payload.get("phase"),
                "effective_status": status_payload.get("effective_status"),
                "score": status_payload.get("score"),
                "passed_tasks": status_payload.get("passed_tasks"),
                "total_tasks": status_payload.get("total_tasks"),
            },
            indent=2,
        )
    )
    _leaderboard(client)


def cmd_selfcheck(args: argparse.Namespace) -> int:
    """Offline verification that the signing scheme round-trips."""
    from bittensor import Keypair  # type: ignore[import-untyped]

    keypair = Keypair.create_from_uri("//Alice")
    body = json.dumps({"name": "x", "artifact_zip_base64": "QUJD"}, separators=(",", ":")).encode()
    canonical = canonical_request_string(
        method="POST",
        path="/submissions",
        query="",
        timestamp="2026-01-01T00:00:00+00:00",
        nonce="test-nonce",
        raw_body=body,
    )
    signature = _sign(keypair, canonical)
    ok = Keypair(ss58_address=keypair.ss58_address).verify(canonical, signature)
    expected = (
        "POST\n/submissions\n2026-01-01T00:00:00+00:00\ntest-nonce\n"
        + hashlib.sha256(body).hexdigest()
    )
    assert canonical == expected, "canonical string drift vs server contract"
    assert ok, "signature failed to verify"
    # Query sorting check.
    q = canonical_request_string(
        method="GET",
        path="/submissions/1/task-events",
        query="stream=agent&cursor=0",
        timestamp="t",
        nonce="n",
        raw_body=b"",
    )
    assert q.split("\n")[1] == "/submissions/1/task-events?cursor=0&stream=agent", q
    assert hashlib.sha256(b"").hexdigest() == EMPTY_BODY_SHA256
    print("selfcheck OK — canonical signing matches validator contract; signature verifies")
    return 0


def _parse_env_pairs(pairs: list[str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="submit_agent.py",
        description="Agent Challenge — A→Z miner submission CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = sub.add_parser("build", help="Package an agent directory into a submission ZIP.")
    p_build.add_argument("--agent-dir", required=True, help="Directory containing agent.py.")
    p_build.add_argument("--out", default="agent.zip", help="Output ZIP path (default: agent.zip).")
    p_build.set_defaults(func=cmd_build)

    # submit
    p_submit = sub.add_parser("submit", help="Sign and submit an agent, optionally watching it.")
    src = p_submit.add_mutually_exclusive_group(required=True)
    src.add_argument("--agent-dir", help="Directory containing agent.py (packaged on the fly).")
    src.add_argument("--zip", help="Pre-built submission ZIP.")
    p_submit.add_argument(
        "--api-base",
        default="https://chain.joinbase.ai/challenges/agent-challenge",
        help=(
            "Validator base URL or BASE proxy base (default: the BASE public API, "
            "https://chain.joinbase.ai/challenges/agent-challenge). Override for a "
            "specific validator or a different deployment."
        ),
    )
    p_submit.add_argument("--name", required=True, help="Human-readable agent name.")
    # hotkey sources
    p_submit.add_argument(
        "--generate-hotkey",
        action="store_true",
        help="Generate a throwaway (unregistered) test hotkey.",
    )
    p_submit.add_argument(
        "--hotkey-mnemonic", help="BIP39 mnemonic (or MINER_HOTKEY_MNEMONIC env)."
    )
    p_submit.add_argument("--hotkey-uri", help="Substrate URI/seed (or MINER_HOTKEY_URI env).")
    p_submit.add_argument(
        "--wallet-name", help="Bittensor wallet name (needs bittensor installed)."
    )
    p_submit.add_argument(
        "--wallet-hotkey", default="default", help="Bittensor wallet hotkey name."
    )
    # env / lifecycle
    p_submit.add_argument(
        "--env",
        action="append",
        metavar="KEY=VALUE",
        help="Miner env var (repeatable). If omitted, confirm-empty is used.",
    )
    p_submit.add_argument(
        "--confirm-empty",
        action="store_true",
        help="Explicitly confirm no env vars (default when --env is absent).",
    )
    p_submit.add_argument(
        "--watch",
        action="store_true",
        help="Drive env gate and stream lifecycle to terminal state.",
    )
    p_submit.add_argument("--poll", type=float, default=3.0, help="Status poll interval seconds.")
    p_submit.add_argument(
        "--env-wait", type=float, default=300.0, help="Max seconds to wait for the env gate."
    )
    p_submit.add_argument(
        "--watch-timeout",
        type=float,
        default=1800.0,
        help="Max seconds to watch terminal-bench to completion.",
    )
    p_submit.add_argument(
        "--http-timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Per-request HTTP timeout seconds.",
    )
    p_submit.set_defaults(func=cmd_submit)

    # selfcheck
    p_self = sub.add_parser("selfcheck", help="Offline check that signing matches the contract.")
    p_self.set_defaults(func=cmd_selfcheck)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
