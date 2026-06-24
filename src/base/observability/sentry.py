from __future__ import annotations


def init_sentry(dsn: str | None, environment: str | None = None) -> None:
    if not dsn:
        return
    import sentry_sdk

    sentry_sdk.init(dsn=dsn, environment=environment)
