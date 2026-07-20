"""Canonical eval image entrypoint.

Stable command the canonical image runs. ``--help`` and the default invocation
touch only the standard library so the image entrypoint is always invokable for
a dry check; the own_runner evaluation pipeline is imported lazily so an actual
``run`` delegates to the unchanged :mod:`agent_challenge.evaluation.own_runner_backend`.

Before a real ``run``, the production RA-TLS path materializes a dstack-issued
client certificate under ``/run/secrets/ra_tls`` so the key-release client can
present end-to-end attested mTLS credentials to the validator raw listener. The
validator *server* CA is never fabricated from the guest chain: it must be
supplied by the deploy (``CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM`` or a pre-written
``CHALLENGE_PHALA_RA_TLS_CA_FILE``). Measured compose ships a single leading
``run`` that is normalized to the backend subcommand before key acquisition.

When a production raw RA-TLS endpoint is configured (static
``KEY_RELEASE_RA_TLS_HOST``/``PORT``, or signed plan
``CHALLENGE_PHALA_EVAL_PLAN.key_release_endpoint`` on a measure-time HTTP(S)
placeholder pin), the bootstrap path always emits flushed, secret-free stage
markers and fails closed within a hard wallclock: missing server CA,
``GetTlsKey`` hang/failure, or an unreachable raw listener produce
``stage=fail reason=...`` and a non-zero exit instead of a silent multi-minute
CVM billed at eval_prepared with zero TCP dials. Never invents PEM roots.

After ``material_ready``, the entrypoint also exports the **public-only**
client fullchain (leaf + intermediates) via
:mod:`agent_challenge.canonical.public_client_fullchain` so operators can
harvest the chain for host ``KEY_RELEASE_RA_TLS_CA_FILE`` / client-trust install
when non-dev Phala disables SSH/SCP. Export surfaces (``phala logs`` + well-known
``/var/log/agent-challenge/client-fullchain.pem``) never contain private keys;
private keys remain only on the mTLS dial paths under ``/run/secrets/ra_tls``.
"""

from __future__ import annotations

import argparse
import os
import socket
import ssl
import sys
from collections.abc import Sequence
from pathlib import Path

from agent_challenge.canonical.public_client_fullchain import (
    PublicFullchainExportError,
    export_public_client_fullchain,
)
from agent_challenge.canonical.wallclock import WallclockTimeout, call_with_wallclock

PROG = "agent-challenge-canonical"

#: Production paths baked by :mod:`agent_challenge.canonical.compose`.
DEFAULT_RA_TLS_DIR = Path("/run/secrets/ra_tls")
DEFAULT_RA_TLS_CERT = DEFAULT_RA_TLS_DIR / "client.crt"
DEFAULT_RA_TLS_KEY = DEFAULT_RA_TLS_DIR / "client.key"
DEFAULT_RA_TLS_CA = DEFAULT_RA_TLS_DIR / "ca.crt"

#: Env carrying the PEM of the CA that signed the validator raw RA-TLS listener
#: certificate (host-side ``KEY_RELEASE_RA_TLS_CERT_FILE``). Distinct from the
#: dstack KMS CA that issues the guest client certificate.
SERVER_CA_PEM_ENV = "CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM"
SERVER_CA_FILE_ENV = "CHALLENGE_PHALA_RA_TLS_SERVER_CA_FILE"

#: Hard wallclock for guest ``GetTlsKey`` (seconds). Live dstack can exceed the
#: SDK default; past this the guest exits with stage=fail rather than billing.
GET_TLS_KEY_WALLCLOCK_SECONDS = 90.0
#: TCP dial budget after materials are ready so the host log shows a dial (or
#: a fast closed failure) before framed RA-TLS begins.
TCP_CONNECT_TIMEOUT_SECONDS = 15.0
#: Marker prefix shared by host log scrapers (secret-free).
RA_TLS_BOOTSTRAP_MARKER = "ra_tls_bootstrap"


def build_parser() -> argparse.ArgumentParser:
    """Help/check-only parser. ``run`` is intentionally not registered here.

    Measured compose ships ``command: [run, --job-dir, ...]``. argparse's
    ``REMAINDER`` still refuses unknown ``--`` options on a subparser, so the
    production ``run`` path is handled manually in :func:`main`.
    """

    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Canonical Agent Challenge evaluation entrypoint (wraps own_runner).",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "check",
        help="verify the own_runner eval pipeline is importable inside the image and exit",
    )
    return parser


_OWN_RUNNER_MODULES = (
    "orchestrator.py",
    "container_builder.py",
    "result_schema.py",
    "taskdefs.py",
    "reward.py",
    "verifier_runner.py",
)


def _run_check() -> int:
    # Verify the own_runner eval modules are present at the expected locations
    # without importing the heavy evaluation package (which pulls the API/chain
    # stack via ``evaluation.__init__``), so the dry check works in the lean
    # canonical image too.
    import agent_challenge

    evaluation = Path(agent_challenge.__file__).resolve().parent / "evaluation"
    own_runner = evaluation / "own_runner"
    missing = [name for name in _OWN_RUNNER_MODULES if not (own_runner / name).is_file()]
    if not (evaluation / "own_runner_backend.py").is_file():
        missing.append("own_runner_backend.py")
    if missing:
        raise RuntimeError(f"own_runner modules missing from image: {', '.join(missing)}")
    print("canonical eval entrypoint OK: own_runner modules present")
    return 0


def _normalize_backend_argv(args: list[str]) -> list[str]:
    """Normalize measured compose argv to the own_runner_backend ``run`` form.

    The Phala docker-compose command is ``["run", "--job-dir", ...]`` because
    the image entrypoint already owns a top-level ``run`` subcommand. The
    remainder may therefore be either:

    * ``["run", "--task", ..., "--job-dir", ...]`` (legacy double-run), or
    * ``["--job-dir", ...]`` / ``["--task", ...]`` (compose shape).

    In the latter case, prepend the backend ``run`` token so argparse sees the
    required subcommand. Never invent ``--task`` values here: the backend pulls
    the immutable Eval plan's selected tasks when CLI tasks are omitted on the
    Phala path.
    """

    tokens = list(args)
    # argparse.REMAINDER keeps a leading "--"; strip a pure separator.
    if tokens and tokens[0] == "--":
        tokens = tokens[1:]
    if not tokens:
        return ["run"]
    if tokens[0] == "run":
        return tokens
    return ["run", *tokens]


def _emit_bootstrap_marker(stage: str, **fields: str | int | bool) -> None:
    """Print a flushed, secret-free bootstrap progress marker for host log scrapers.

    Never includes PEMs, keys, token values, or other secret material. Field
    values are already-sanitized present/missing flags or non-secret host/port.
    """

    parts = [f"{RA_TLS_BOOTSTRAP_MARKER} stage={stage}"]
    for key, value in fields.items():
        if isinstance(value, bool):
            text = "present" if value else "missing"
        else:
            text = str(value).replace("\n", " ").replace("\r", " ").strip()
            # Belt-and-suspenders: never echo obviously secret-looking blobs.
            if "BEGIN " in text.upper() or len(text) > 256:
                text = "redacted"
        parts.append(f"{key}={text}")
    print(" ".join(parts), flush=True)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - best-effort flush only
        pass


def normalize_server_ca_pem(raw: str) -> str:
    """Return an OpenSSL-loadable multi-line PEM, or raise ValueError.

    Live residual: Phala ``encrypted_env`` (or shelling) can collapse a multi-line
    certificate into a single line with *literal* ``\\n`` / ``\\r\\n`` escape
    sequences. Weak ``BEGIN CERTIFICATE`` substring checks still accept that
    shape, but OpenSSL reports ``NO_CERTIFICATE_OR_CRL_FOUND`` when building the
    trust store via ``create_default_context(cafile=...)``.

    Contract:
    * empty / whitespace-only → ``empty_server_ca``
    * missing PEM markers after unescape → ``malformed_server_ca``
    * PEM markers present but unloadable by OpenSSL → ``malformed_server_ca``
    * success → trailing-newline PEM that ``SSLContext.load_verify_locations
      (cadata=...)`` accepts
    """

    if raw is None:
        raise ValueError("empty_server_ca: server CA PEM is empty")
    text = str(raw).strip()
    if not text:
        raise ValueError("empty_server_ca: server CA PEM is empty")

    # Unescape only when the payload looks collapsed (one logical line) and still
    # carries PEM markers via literal backslash-n sequences. Do not rewrite
    # already multi-line PEMs that happen to include backslash characters.
    if "BEGIN CERTIFICATE" in text and "\n" not in text and "\\n" in text:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    elif "BEGIN CERTIFICATE" in text and "\\n" in text and text.count("\n") < 2:
        # Semi-collapsed: a few real newlines but body still escaped.
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n")

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError("empty_server_ca: server CA PEM is empty after normalize")
    if "BEGIN CERTIFICATE" not in text or "END CERTIFICATE" not in text:
        raise ValueError("malformed_server_ca: PEM markers missing after normalize")
    if not text.endswith("\n"):
        text = text + "\n"

    # Hard gate: OpenSSL must accept the CA bytes *before* we write ca.crt or
    # hand the path to create_default_context (prevents opaque pre-frame SSLError).
    try:
        probe = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        probe.load_verify_locations(cadata=text)
    except ssl.SSLError as exc:
        raise ValueError(f"malformed_server_ca: OpenSSL rejected server CA PEM ({exc})") from exc
    return text


def preload_server_ca_pem(pem: str) -> str:
    """Normalize + OpenSSL-preload a server CA PEM; alias of :func:`normalize_server_ca_pem`."""

    return normalize_server_ca_pem(pem)


def _server_ca_presence() -> str:
    """Return ``present`` or ``missing`` without reading secret PEM into logs.

    Marker only: a bare ``BEGIN CERTIFICATE`` substring counts as present so the
    start marker is informative before full OpenSSL preload. Unloadable PEMs are
    rejected later with a durable ``malformed_server_ca`` reason.
    """

    pem = (os.environ.get(SERVER_CA_PEM_ENV) or "").strip()
    if pem and "BEGIN CERTIFICATE" in pem:
        return "present"
    for env_name in (SERVER_CA_FILE_ENV, "CHALLENGE_PHALA_RA_TLS_CA_FILE"):
        raw = (os.environ.get(env_name) or "").strip()
        if not raw:
            continue
        path = Path(raw)
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if "BEGIN CERTIFICATE" in text:
                return "present"
    return "missing"


def _resolve_server_ca_pem() -> str:
    """Return the validator raw-listener CA PEM (OpenSSL-loadable), or fail closed."""

    pem = (os.environ.get(SERVER_CA_PEM_ENV) or "").strip()
    if pem and "BEGIN CERTIFICATE" in pem:
        try:
            return normalize_server_ca_pem(pem)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
    ca_file = (os.environ.get(SERVER_CA_FILE_ENV) or "").strip()
    if ca_file:
        path = Path(ca_file)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if "BEGIN CERTIFICATE" in text:
                try:
                    return normalize_server_ca_pem(text)
                except ValueError as exc:
                    raise RuntimeError(str(exc)) from exc
    # A pre-provisioned CA file path counts only when it already holds real PEM
    # (deploy mounts the validator server CA). Empty placeholder paths fail closed.
    configured = (os.environ.get("CHALLENGE_PHALA_RA_TLS_CA_FILE") or "").strip()
    if configured:
        path = Path(configured)
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if "BEGIN CERTIFICATE" in text:
                try:
                    return normalize_server_ca_pem(text)
                except ValueError as exc:
                    raise RuntimeError(str(exc)) from exc
    raise RuntimeError(
        "missing_server_ca: raw RA-TLS path requires the validator server CA "
        f"({SERVER_CA_PEM_ENV} or {SERVER_CA_FILE_ENV} or a non-empty "
        "CHALLENGE_PHALA_RA_TLS_CA_FILE); refusing to trust the guest dstack chain"
    )


def _call_get_tls_key_with_wallclock(client: object) -> object:
    """Invoke ``GetTlsKey`` with a hard wallclock; raise on hang/failure.

    Uses a daemon-thread wallclock (not ThreadPoolExecutor) so a hung dstack
    RPC never re-joins on the timeout path. Fail-closed stage=fail markers are
    still emitted by the provision caller.
    """

    get_tls_key = getattr(client, "get_tls_key", None)
    if not callable(get_tls_key):
        raise RuntimeError("gettlskey_unavailable: dstack client lacks get_tls_key")

    def _invoke() -> object:
        return get_tls_key(
            subject="agent-challenge-key-release",
            usage_ra_tls=True,
            usage_server_auth=False,
            usage_client_auth=True,
        )

    try:
        return call_with_wallclock(
            _invoke,
            timeout_seconds=GET_TLS_KEY_WALLCLOCK_SECONDS,
            label="GetTlsKey",
        )
    except WallclockTimeout as exc:
        raise RuntimeError(
            f"gettlskey_timeout: GetTlsKey exceeded {GET_TLS_KEY_WALLCLOCK_SECONDS:.0f}s"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - any dstack failure fails closed
        text = str(exc).lower()
        if "timeout" in text or "timed out" in text:
            raise RuntimeError(f"gettlskey_timeout: {exc}") from exc
        raise RuntimeError(f"gettlskey_failed: {exc}") from exc


def _probe_raw_tcp(host: str, port: int) -> None:
    """Force a TCP dial so the host log sees connect activity, or fail closed."""

    try:
        with socket.create_connection((host, port), timeout=TCP_CONNECT_TIMEOUT_SECONDS):
            pass
    except OSError as exc:
        _emit_bootstrap_marker(
            "tcp_connect_fail",
            host=host,
            port=port,
            server_ca=_server_ca_presence(),
            reason="tcp_connect_fail",
        )
        raise RuntimeError(
            f"tcp_connect_fail: cannot dial raw RA-TLS listener {host}:{port}: {exc}"
        ) from exc
    _emit_bootstrap_marker(
        "tcp_connect_ok",
        host=host,
        port=port,
        server_ca="present",
    )


def _parse_raw_host_port(endpoint: str) -> tuple[str, int] | None:
    """Parse a production raw RA-TLS authority (host:8701 / ratls://host:port).

    Mirrors :func:`agent_challenge.canonical.compose._parse_raw_key_release_endpoint`
    without importing compose (lean-image boundary). Returns ``None`` for HTTP(S)
    and non-raw authorities so the measure-time HTTPS placeholder does not invent
    a raw bake from the placeholder URL.
    """

    value = (endpoint or "").strip()
    if not value:
        return None
    scheme = ""
    authority = value
    if "://" in value:
        scheme, authority = value.split("://", 1)
        scheme = scheme.lower()
        if scheme in {"http", "https"}:
            return None
        if scheme not in {"ratls", "tls", "tcp"}:
            return None
    if "/" in authority:
        authority = authority.split("/", 1)[0]
    if ":" not in authority:
        return None
    host, port_text = authority.rsplit(":", 1)
    try:
        port = int(port_text)
    except ValueError:
        return None
    host = host.strip().strip("[]")
    if not host or not 1 <= port <= 65535:
        return None
    # Bare host:port without scheme is production only on the RA-TLS listener port.
    if not scheme and port != 8701:
        return None
    return host, port


def _resolve_raw_ra_tls_host_port() -> tuple[str, int] | None:
    """Return production raw RA-TLS ``(host, port)`` for GetTlsKey provisioning.

    Sources, in order (never invent PEMs / listener endpoints):

    1. Static compose env ``KEY_RELEASE_RA_TLS_HOST`` + ``PORT`` (raw bake path).
    2. Signed plan ``CHALLENGE_PHALA_EVAL_PLAN.key_release_endpoint`` when the
       measure-time pin used an HTTPS placeholder that left host/port unset
       (Path B residual: guest still dials raw ``host:8701`` from the plan, so
       client mTLS material must exist before that dial).
    3. Legacy ``CHALLENGE_PHALA_KEY_RELEASE_URL`` when it is a raw authority.

    Returns ``None`` for flag-off / non-raw legacy HTTP paths so provision is
    skipped entirely outside production raw key-release.
    """

    host = (os.environ.get("KEY_RELEASE_RA_TLS_HOST") or "").strip()
    port_text = (os.environ.get("KEY_RELEASE_RA_TLS_PORT") or "").strip()
    if host and port_text:
        try:
            port = int(port_text)
        except ValueError as exc:
            raise RuntimeError(
                f"invalid_port: KEY_RELEASE_RA_TLS_PORT is not an integer: {port_text!r}"
            ) from exc
        if not 1 <= port <= 65535:
            raise RuntimeError(f"invalid_port: KEY_RELEASE_RA_TLS_PORT out of range: {port}")
        return host, port

    # Plan-first fallback: measure-time HTTPS pin leaves static HOST/PORT empty.
    raw_plan = (os.environ.get("CHALLENGE_PHALA_EVAL_PLAN") or "").strip()
    if raw_plan:
        try:
            import json

            plan = json.loads(raw_plan)
        except (json.JSONDecodeError, TypeError, ValueError):
            plan = None
        if isinstance(plan, dict):
            endpoint = str(plan.get("key_release_endpoint") or "").strip()
            parsed = _parse_raw_host_port(endpoint) if endpoint else None
            if parsed is not None:
                return parsed

    url = (os.environ.get("CHALLENGE_PHALA_KEY_RELEASE_URL") or "").strip()
    if url:
        return _parse_raw_host_port(url)
    return None


def _provision_ra_tls_client_material() -> None:
    """Issue a dstack RA-TLS client cert when the production raw path is configured.

    The measured compose pins ``CHALLENGE_PHALA_RA_TLS_{CERT,KEY,CA}_FILE`` at
    ``/run/secrets/ra_tls/*`` when host/port were baked raw. When the pin used a
    measure-time HTTPS placeholder, those env names may be absent; resolve the
    raw production endpoint from the signed plan (never invent PEMs) and still
    materialize default paths under ``/run/secrets/ra_tls``. When cert/key files
    are not already present and the guest dstack socket is available, request a
    client-auth + RA-TLS certificate with ``GetTlsKey`` and write the chain +
    key in place. The CA file is *always* the validator server-trust CA (never
    the dstack guest intermediate). Fail closed (raise) for the raw RA-TLS
    production path; legacy HTTP key-release skips this entirely.

    Emits flushed, secret-free stage markers (start / material_ready /
    tcp_connect_ok|fail) and always dials the raw listener after materials are
    ready so a silent multi-minute hang at eval_prepared is impossible when
    a raw host+port is configured (compose env or signed plan endpoint).
    """

    host_port = _resolve_raw_ra_tls_host_port()
    if host_port is None:
        return  # not the production raw path

    host, port = host_port
    # Export host/port so own_runner KR dial + residual scrapers see the raw path
    # even when measure-time compose left them unset (HTTP placeholder pin).
    os.environ["KEY_RELEASE_RA_TLS_HOST"] = host
    os.environ["KEY_RELEASE_RA_TLS_PORT"] = str(port)

    server_ca_state = _server_ca_presence()
    _emit_bootstrap_marker(
        "start",
        host=host,
        port=port,
        server_ca=server_ca_state,
    )

    cert_path = Path(
        (os.environ.get("CHALLENGE_PHALA_RA_TLS_CERT_FILE") or "").strip() or DEFAULT_RA_TLS_CERT
    )
    key_path = Path(
        (os.environ.get("CHALLENGE_PHALA_RA_TLS_KEY_FILE") or "").strip() or DEFAULT_RA_TLS_KEY
    )
    ca_path = Path(
        (os.environ.get("CHALLENGE_PHALA_RA_TLS_CA_FILE") or "").strip() or DEFAULT_RA_TLS_CA
    )

    server_ca_pem = _resolve_server_ca_pem()
    need_client = not (cert_path.is_file() and key_path.is_file())
    if need_client:
        from dstack_sdk import DstackClient

        # SDK timeout is an advisory; the wallclock wrapper is the hard deadline.
        client = DstackClient(timeout=int(GET_TLS_KEY_WALLCLOCK_SECONDS))
        response = _call_get_tls_key_with_wallclock(client)
        key_pem = getattr(response, "key", None) or ""
        chain = list(getattr(response, "certificate_chain", None) or [])
        if not isinstance(key_pem, str) or not key_pem.strip():
            raise RuntimeError("gettlskey_failed: dstack GetTlsKey returned no client private key")
        if not chain or not all(isinstance(item, str) and item.strip() for item in chain):
            raise RuntimeError(
                "gettlskey_failed: dstack GetTlsKey returned no client certificate chain"
            )

        cert_path.parent.mkdir(parents=True, exist_ok=True)
        # Leaf first, then intermediates for a complete client chain file.
        cert_path.write_text("".join(chain), encoding="utf-8")
        key_path.write_text(key_pem if key_pem.endswith("\n") else key_pem + "\n", encoding="utf-8")
        os.chmod(key_path, 0o600)

    if not cert_path.is_file() or not key_path.is_file():
        raise RuntimeError("client_material_incomplete: raw RA-TLS client cert/key files missing")

    # Always materialize the *server* trust CA at the configured CA path so the
    # key-release client verifies the validator listener (not a guest issuer).
    ca_path.parent.mkdir(parents=True, exist_ok=True)
    ca_path.write_text(server_ca_pem, encoding="utf-8")

    cert_text = cert_path.read_text(encoding="utf-8")
    key_text = key_path.read_text(encoding="utf-8")
    if not cert_text.strip() or not key_text.strip():
        raise RuntimeError("client_material_empty: raw RA-TLS client cert/key must be non-empty")

    os.environ["CHALLENGE_PHALA_RA_TLS_CERT_FILE"] = str(cert_path)
    os.environ["CHALLENGE_PHALA_RA_TLS_KEY_FILE"] = str(key_path)
    os.environ["CHALLENGE_PHALA_RA_TLS_CA_FILE"] = str(ca_path)

    # Observability: materialize the leaf SPKI digest so the acquire path never
    # falls back to sha256(b"") when env SPKI/PUBKEY are unset (live residual).
    if not (os.environ.get("CHALLENGE_PHALA_RA_TLS_SPKI_SHA256") or "").strip():
        try:
            import hashlib

            from cryptography import x509
            from cryptography.hazmat.primitives import serialization

            certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
            spki = certificate.public_key().public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            os.environ["CHALLENGE_PHALA_RA_TLS_SPKI_SHA256"] = hashlib.sha256(spki).hexdigest()
        except (OSError, ValueError):
            # Non-fatal: acquire-time resolution still derives from the cert file.
            pass

    _emit_bootstrap_marker(
        "material_ready",
        host=host,
        port=port,
        server_ca="present",
        client_cert="present",
        client_key="present",
    )

    # Public-only fullchain export for operator client-trust install. Private
    # key never leaves key_path; fail closed when public chain cannot be exported
    # (missing/empty/shredded) so residual harvest does not silently skip.
    try:
        export = export_public_client_fullchain(cert_path=cert_path)
    except PublicFullchainExportError as export_exc:
        raise RuntimeError(str(export_exc)) from export_exc
    _emit_bootstrap_marker(
        "public_fullchain_exported",
        host=host,
        port=port,
        cert_count=export.cert_count,
        pem_len=export.pem_len,
        path_written="yes" if export.path_written else "no",
    )

    # Host must see a TCP hit (even before framed release) or the guest exits.
    _probe_raw_tcp(host, port)


def _run_eval(args: list[str]) -> int:
    try:
        _provision_ra_tls_client_material()
    except Exception as exc:  # noqa: BLE001 - every provision error is durable fail-closed
        host = (os.environ.get("KEY_RELEASE_RA_TLS_HOST") or "").strip() or "unset"
        port = (os.environ.get("KEY_RELEASE_RA_TLS_PORT") or "").strip() or "unset"
        message = str(exc)
        reason = message.split(":", 1)[0].strip() if message else "provision_failed"
        if not reason or " " in reason:
            reason = "provision_failed"
        # Collapse verbose RuntimeError prefixes to stable, secret-free reasons.
        for known in (
            "malformed_server_ca",
            "empty_server_ca",
            "missing_server_ca",
            "gettlskey_timeout",
            "gettlskey_failed",
            "gettlskey_unavailable",
            "tcp_connect_fail",
            "invalid_port",
            "client_material_incomplete",
            "client_material_empty",
            "public_chain_missing",
            "private_key_in_export",
        ):
            if known in message:
                reason = known
                break
        _emit_bootstrap_marker(
            "fail",
            host=host,
            port=port,
            server_ca=_server_ca_presence(),
            reason=reason,
        )
        print(f"canonical eval RA-TLS bootstrap failed: {reason}", file=sys.stderr, flush=True)
        return 1

    from agent_challenge.evaluation.own_runner_backend import main as backend_main

    return backend_main(_normalize_backend_argv(args))


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(argv) if argv is not None else None
    if tokens is None:
        tokens = list(sys.argv[1:])

    # Production path: first token is ``run`` and everything after (including bare
    # ``--job-dir`` flags from measured compose) is backend argv. Do not route
    # this through argparse subparsers: REMAINDER cannot capture leading options.
    if tokens and tokens[0] == "run":
        return _run_eval(tokens[1:])

    parser = build_parser()
    if not tokens:
        return _run_check()
    namespace = parser.parse_args(tokens)
    if namespace.command in (None, "check"):
        return _run_check()
    parser.error(f"unknown command: {namespace.command}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(main())
