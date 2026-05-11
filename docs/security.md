# Security Model

![Platform Banner](../assets/banner.jpg)

## Isolation rules

- PostgreSQL central is available only to the master.
- Challenges never receive PostgreSQL credentials.
- Normal validators never receive master DB credentials.
- Internal challenge calls require per-challenge shared tokens.
- Public proxy strips sensitive headers.
- Public proxy blocks internal challenge paths.

## Secrets

Admin and challenge tokens are loaded from files or environment variables. Tokens are never stored in clear text in registry metadata responses.

## Failure behavior

If a challenge fails health checks or `get_weights`, its contribution is zero for that epoch. The master does not auto-disable it.
