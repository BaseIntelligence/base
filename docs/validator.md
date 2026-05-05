# Validator Guide

![Platform Banner](https://raw.githubusercontent.com/PlatformNetwork/platform/main/assets/banner.jpg)

## Master mode

```bash
uv run platform master run --config config/master.example.yaml
uv run platform master proxy --config config/master.example.yaml
```

The master exposes:

- private admin/registry API
- public challenge proxy API
- Docker orchestration for active challenges
- weight aggregation and Bittensor wrappers

## Normal validator mode

```bash
uv run platform validator run --config config/validator.example.yaml
```

A normal validator fetches active challenges from `rpc.platform.network/v1/registry` and launches them locally.

## CLI

```bash
uv run platform challenge create demo --out ../demo
uv run platform challenge register demo ghcr.io/org/demo:latest 10
uv run platform challenge activate demo
uv run platform challenge pull demo
uv run platform challenge restart demo
uv run platform db migrate
```
