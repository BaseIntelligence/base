# Validator Operations

![Platform Banner](../../assets/banner.jpg)

## Local validation

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

## Compose validation

```bash
docker compose -f docker/compose.dev.yml config
```

## Master deployment checklist

1. Configure `config/master.example.yaml` or provide env overrides.
2. Provide an admin token file.
3. Run Alembic migrations.
4. Start the master API.
5. Start the proxy API.
6. Register and activate challenge images.
7. Monitor logs, Sentry, and OpenTelemetry.
