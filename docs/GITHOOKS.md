# Git hooks (Base)

Repo-managed hooks live in `.githooks/`. Install once per clone:

```bash
./scripts/install-githooks.sh
```

Sets `core.hooksPath=.githooks`.

| Hook | Enforces |
|------|----------|
| `pre-commit` | `cargo check --workspace --all-targets` |
| `commit-msg` | Conventional subject: `type(scope): summary` (≤72 chars) |
| `pre-push` | `cargo check --workspace --all-targets` + `cargo test --workspace --lib` |

No author/committer identity enforcement in hooks.
