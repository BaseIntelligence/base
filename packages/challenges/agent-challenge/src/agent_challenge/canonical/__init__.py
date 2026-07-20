"""Canonical, reproducibly-built eval image support for agent-challenge.

This package holds the build/measurement/entrypoint tooling for the canonical
Phala eval image. The scoring pipeline itself is the unchanged ``own_runner``
backend; this package only adds the reproducible, digest-pinned packaging around
it.
"""
