from __future__ import annotations

from pathlib import Path

from prism_challenge.db import Database


async def test_database_healthcheck_tracks_required_challenge_schema(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "prism.sqlite3")

    assert await database.healthcheck() is False
    await database.init()
    assert await database.healthcheck() is True
