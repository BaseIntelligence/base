# API Reference

**API truth is OpenAPI**, not this file.

- Live production: `https://chain.joinbase.ai/challenges/prism/openapi.json`
- Interactive: `https://chain.joinbase.ai/challenges/prism/docs`
- Local / in-process challenge app: `GET /openapi.json` and `GET /docs`

Day-1 miners: repo-root [`docs/miner/getting-started.md`](../../../../docs/miner/getting-started.md).

## Live internal bridges (also in OpenAPI)

- `GET /internal/v1/get_weights`
- `POST /internal/v1/bridge/submissions`
- `POST /internal/v1/worker/process-next`

Serving layer uses `/v1/architectures` (variants re-homed there). Do not call
retired training-variants paths.

Internal master bridges (weights push, ZIP bridge) are described in the live
schema. Do not maintain parallel markdown route dumps here.
