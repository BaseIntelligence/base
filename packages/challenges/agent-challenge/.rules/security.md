# Security Policy

Validator review treats submitted agents and benchmark workspaces as untrusted input.

## Prohibited

- Do not accept code that exfiltrates secrets, tokens, SSH keys, cookies, environment variables, or private files.
- Do not accept agents that read, copy, or exfiltrate the hidden benchmark grading files (task tests, solutions, oracle or answer keys, or `task.yaml` metadata) to discover or match the expected result (`reads_hidden_tests`).
- Do not accept submitted agents that embed, request, read, forward, or document **Base LLM gateway** material (`BASE_LLM_GATEWAY_URL`, `BASE_GATEWAY_TOKEN`, `/llm/v1`) or that call non-measured providers outside the attested OpenRouter path. Measured OpenRouter keys may exist only inside miner/session encrypted_env for allowlisted measured guests—not as host-side Base gateway restore.
- Do not accept code that disables sandboxing, escapes containers, changes host permissions, or bypasses resource limits.
- Do not accept hidden network callbacks, telemetry, command and control behavior, or remote code loading unless the benchmark explicitly requires it.
- Do not accept credential prompts, phishing flows, or attempts to impersonate validator, miner, or BASE services.
- Do not accept destructive filesystem behavior outside the benchmark workspace.
- Do not accept changes that weaken authentication, signature checks, authorization, input validation, or auditability without a direct task requirement.

## Required Review Evidence

- Security concerns must cite observable code, configuration, command output, or file behavior.
- Treat generated scripts, install hooks, shell commands, CI files, and dependency changes as executable surfaces.
- If the evidence is incomplete, mark the security finding as uncertain instead of overstating it.
