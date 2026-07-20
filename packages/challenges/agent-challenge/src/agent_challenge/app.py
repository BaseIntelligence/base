"""Compatibility wrapper for the FastAPI application entrypoint."""

from .api.app import app

__all__ = ["app"]
