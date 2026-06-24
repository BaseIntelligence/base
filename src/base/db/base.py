"""SQLAlchemy declarative base for the base master database."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all base network ORM models."""
