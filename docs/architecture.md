# Architecture

![Platform Banner](https://raw.githubusercontent.com/PlatformNetwork/platform/main/assets/banner.jpg)

## Components

```mermaid
flowchart TB
    subgraph M[Master]
      A[Admin API]
      P[Proxy API]
      O[Docker Orchestrator]
      G[Weight Aggregator]
      DB[(Postgres)]
    end
    subgraph D[Docker Network]
      C1[Challenge]
      CDB[(SQLite)]
    end
    V[Normal Validator] --> A
    A --> DB
    A --> O
    P --> C1
    O --> C1
    C1 --> CDB
    G --> C1
    G --> BT[Bittensor]
```

## Master validator

The master owns registry metadata, admin operations, Docker lifecycle, challenge tokens, emission configuration, and final Bittensor weight submission.

## Normal validator

Normal validators read `/v1/registry`, launch all active challenge images locally, and keep retrying if the registry is unavailable.

## Challenge isolation

Each challenge runs in Docker with its own image, named SQLite volume, internal shared token, and public routes behind the Platform proxy.
