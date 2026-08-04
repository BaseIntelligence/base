# design-baseline

Reference miner harness for the design challenge.

| File | Role |
|------|------|
| `agent.py` | `run(task, llm, out)` — calls `llm.chat`, writes required pages |
| `pyproject.toml` | Empty deps; operator sandbox installs extras you add |

Pack into `POST /v1/harness` as in [`../../design.md`](../../design.md).
Required outputs: `index.html`, `pricing.html`, `components.html`.
