# Git hooks (Base)

Repo-managed hooks live in `.githooks/`. Install once per clone:

```bash
./scripts/install-githooks.sh
```

This sets `core.hooksPath=.githooks`.

| Hook | Enforces |
|------|----------|
| `pre-commit` | `cargo fmt --check` + `cargo check --workspace --all-targets` |
| `commit-msg` | Conventional subject: `type(scope): summary` (≤72 chars) |
| `pre-push` | `cargo fmt --check` + `cargo clippy -D warnings` + `cargo test --workspace --lib` |

No author/committer identity checks in hooks (identity is a local/operator concern).
