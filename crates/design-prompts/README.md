# design-prompts

Repo-pinned Design challenge prompt bank. **No human prompt-approval API** —
bank evolution is a normal code PR.

| Item | Value |
|------|--------|
| Bank file | [`prompts/bank_v1.json`](prompts/bank_v1.json) |
| Digest | `bank_digest()` / alias `prompt_set_digest()` — SHA-256 hex of bank JSON bytes |
| Round length | 6h (`ROUND_SECS = 21_600` in `design-challenge-task`) |
| Selection | Weighted draw without replacement from `SHA256(domain ‖ round_id ‖ bank_digest)` |
| Per round | `PROMPTS_PER_ROUND = 3` |

Each bank entry: `id`, `category`, `title`, `weight`, `temperature`, `prompt`
(rich brief requiring `index.html` / `pricing.html` / `components.html`).
