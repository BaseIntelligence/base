"""SQLAlchemy declarative base for the platform master database."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all platform network ORM models."""
