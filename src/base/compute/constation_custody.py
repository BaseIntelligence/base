"""Encrypted at-rest custody for miner-supplied Lium API keys (todo 15).

Where constation runs
---------------------
``LiumKeyCustody`` + :class:`~base.compute.constation_runner.ConstationRunner`
are **master / worker-plane services**. The miner CLI is not the only caller:
control-plane code registers a miner key, then the runner builds short-lived
:class:`~base.compute.lium.LiumClient` instances for pod/template reads during
continuous constation. Miners never receive decrypted peer keys.

Custody threat model (M6) — full account power
----------------------------------------------
A miner-supplied ``LIUM_API_KEY`` authenticates as that Lium account. It can
list/destroy pods, mutate templates, and incur billing. BASE holds the key
solely to perform same-account **corroboration** reads (declared template
digest). Compromise of:

* the Fernet master key used for at-rest encryption, or
* process memory while a key is unlocked for a request,

yields **full account power** over the miner's Lium account — including pod
destruction. Mitigations (not eliminations): Fernet encryption at rest, never
log or ``repr`` the plaintext key, probe-on-registration, fail-closed on HTTP
401 / mid-run revocation (``lium_auth_revoked``). This is tamper-evidence
infrastructure, not a hardware trust boundary. No TEE claims.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from cryptography.fernet import Fernet, InvalidToken

from base.compute.constation_types import ConstationFailCode, ConstationVerdict
from base.compute.lium import LiumAuthError, LiumClient, LiumError

logger = logging.getLogger(__name__)

ProbeFn = Callable[[LiumClient], Awaitable[None]]
ClientFactory = Callable[[str], LiumClient]


class _SupportsBalance(Protocol):
    async def balance(self) -> float: ...


async def default_lium_probe(client: LiumClient) -> None:
    """Probe registration by reading account balance (cheap authenticated GET)."""
    await client.balance()


def generate_custody_master_key() -> bytes:
    """Return a new Fernet key suitable for :class:`LiumKeyCustody`."""
    return Fernet.generate_key()


@dataclass
class LiumKeyCustody:
    """In-memory encrypted store of miner Lium API keys.

    Plaintext keys exist only ephemerally inside :meth:`unlock_api_key` /
    :meth:`build_client`. Stored values are Fernet tokens. ``repr`` / logs never
    include plaintext or ciphertext blobs that embed the key material in a
    recoverable form beyond the opaque token length.
    """

    master_key: bytes
    client_factory: ClientFactory = field(default=LiumClient)
    probe_fn: ProbeFn = field(default=default_lium_probe)
    _fernet: Fernet = field(init=False, repr=False)
    _by_hotkey: dict[str, bytes] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._fernet = Fernet(self.master_key)

    def __repr__(self) -> str:
        return f"LiumKeyCustody(registered={len(self._by_hotkey)})"

    def registered_hotkeys(self) -> frozenset[str]:
        return frozenset(self._by_hotkey)

    def has_key(self, miner_hotkey: str) -> bool:
        return miner_hotkey.strip() in self._by_hotkey

    async def register(self, *, miner_hotkey: str, api_key: str) -> ConstationVerdict:
        """Probe ``api_key``, then encrypt-at-rest under ``miner_hotkey``.

        Fail closed on probe 401 (``lium_auth_revoked``) or other probe errors
        (``probe_failed``). Never logs ``api_key``.
        """
        hotkey = _require_nonblank("miner_hotkey", miner_hotkey)
        key = _require_nonblank("api_key", api_key)
        client = self.client_factory(key)
        try:
            await self.probe_fn(client)
        except LiumAuthError:
            logger.warning("lium key probe rejected (401) for miner_hotkey=%s", hotkey)
            return ConstationVerdict(
                ok=False,
                reason=ConstationFailCode.LIUM_AUTH_REVOKED,
                detail="probe_401",
            )
        except LiumError as exc:
            logger.warning(
                "lium key probe failed for miner_hotkey=%s status=%s",
                hotkey,
                getattr(exc, "status_code", None),
            )
            return ConstationVerdict(
                ok=False,
                reason=ConstationFailCode.PROBE_FAILED,
                detail=type(exc).__name__,
            )
        token = self._fernet.encrypt(key.encode("utf-8"))
        self._by_hotkey[hotkey] = token
        logger.info("lium key registered (encrypted) for miner_hotkey=%s", hotkey)
        return ConstationVerdict(ok=True, reason=ConstationFailCode.OK)

    def unlock_api_key(self, miner_hotkey: str) -> str:
        """Decrypt the stored key for in-process use. Raises if missing/corrupt."""
        hotkey = _require_nonblank("miner_hotkey", miner_hotkey)
        token = self._by_hotkey.get(hotkey)
        if token is None:
            raise KeyError(f"no Lium key registered for miner_hotkey={hotkey!r}")
        try:
            return self._fernet.decrypt(token).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError("custody token decrypt failed") from exc

    def build_client(self, miner_hotkey: str) -> LiumClient:
        """Return a :class:`LiumClient` using the unlocked key (not logged)."""
        return self.client_factory(self.unlock_api_key(miner_hotkey))

    def export_encrypted(self) -> Mapping[str, bytes]:
        """Snapshot ciphertext tokens for a durable adapter (no plaintext)."""
        return dict(self._by_hotkey)

    def import_encrypted(self, mapping: Mapping[str, bytes]) -> None:
        """Load ciphertext tokens previously produced by :meth:`export_encrypted`."""
        self._by_hotkey = {
            _require_nonblank("miner_hotkey", k): bytes(v) for k, v in mapping.items()
        }


def _require_nonblank(field_name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    return normalized


__all__ = [
    "LiumKeyCustody",
    "default_lium_probe",
    "generate_custody_master_key",
]
