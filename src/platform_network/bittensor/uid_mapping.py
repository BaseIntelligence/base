from __future__ import annotations


def hotkeys_to_uids(hotkeys: list[str]) -> dict[str, int]:
    return {hotkey: uid for uid, hotkey in enumerate(hotkeys)}
