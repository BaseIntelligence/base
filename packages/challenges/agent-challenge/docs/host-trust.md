# Host-trust execution (product path)

After Phala TEE removal (T40), Agent Challenge production scoring is **host-trust only**:

- Enable via `CHALLENGE_UNATTESTED_EXECUTION` / `CHALLENGE_NO_PHALA` / `NO_PHALA`
- Results are marked `attested=false` / `attestation_status=unattested`
- Integrity still uses `package_tree_sha` + AGATE residual (host residual kinds)
- **No** TDX quotes, **no** Phala CVM, **no** independent hardware attestation

Do not describe this mode as TEE, tamper-proof, or independently verified.
