# Challenges

![Platform Banner](../assets/banner.jpg)

## Model

A challenge is an independent repository and Docker image. It owns its logic, public routes, submissions, scoring data, and SQLite schema.

## Required API

```text
GET /health
GET /version
GET /internal/v1/get_weights
```

The internal endpoint is authenticated with a per-challenge shared token mounted by the master.

## Create a challenge

```bash
uv run platform challenge create code-arena --out ../code-arena
cd ../code-arena
uv run --extra dev pytest
```

## Public routes

Public routes are exposed through:

```text
/challenges/{slug}/...
```

The master blocks `/internal/*`, `/health`, and `/version` from the public proxy.
