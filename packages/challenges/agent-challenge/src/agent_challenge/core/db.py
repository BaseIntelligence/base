"""Challenge database exports."""

from __future__ import annotations

from ..sdk.db import Base, Database
from .config import settings

database = Database(settings.database_url)

__all__ = ["Base", "database"]
