"""Canonical CVM entrypoint removed with Phala (T40)."""

from __future__ import annotations


class CanonicalEntrypointRemoved(RuntimeError):
    pass


def main(*args, **kwargs):  # noqa: ANN001, ANN002
    raise CanonicalEntrypointRemoved("canonical entrypoint removed with Phala TEE (T40)")


__all__ = ["CanonicalEntrypointRemoved", "main"]
