"""Shared e2e helper: normalize `/v1/weights/latest` hotkey_weights keys.

The gateway serves `hotkey_weights` keyed by SS58 address while the e2e
scripts track miner hotkeys as 64-char hex. Pure-stdlib base58 decode
(SS58 = base58(prefix ‖ pubkey32 ‖ checksum2); 1-byte prefix when < 64).
"""

B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58decode(s: str) -> bytes:
    n = 0
    for ch in s:
        n = n * 58 + B58.index(ch)
    body = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    return b"\x00" * (len(s) - len(s.lstrip("1"))) + body


def hotkey_to_hex(key: str) -> str:
    """Return 64-char lowercase hex for a hex or SS58 hotkey ('' if unknown)."""
    k = key.strip()
    if k.lower().startswith("0x"):
        k = k[2:]
    if len(k) == 64 and all(c in "0123456789abcdefABCDEF" for c in k):
        return k.lower()
    try:
        raw = _b58decode(k)
    except ValueError:
        return ""
    if len(raw) == 35:  # 1-byte prefix + 32-byte pubkey + 2-byte checksum
        return raw[1:33].hex()
    if len(raw) == 36:  # 2-byte prefix
        return raw[2:34].hex()
    return ""


def hotkey_weight_map(latest: dict) -> dict:
    """`hotkey_weights` of a /v1/weights/latest body, keyed by hex hotkey."""
    out = {}
    for k, v in (latest.get("hotkey_weights") or {}).items():
        hk = hotkey_to_hex(k)
        if hk:
            out[hk] = float(v)
    return out
