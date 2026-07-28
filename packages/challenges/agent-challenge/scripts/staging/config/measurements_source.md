# Measurement allowlist provenance

## Shared TDX.small register core

Obtained from live Phala TDX quotes on `tdx.small` / `us-west-1`:

- Review TEE evidence: `/work/baseintelligence/.omo/evidence/ac-attested-review-20260727/review-tee.json`
- T8 eval KR allowlist dumps: `/work/baseintelligence/.omo/start-work/T8-e2e/kr-meas-diag-20260725T233258Z.txt`
- Prod KR file (same core): host `/var/lib/base/keyrelease/eval-allowlist.json` on the prod master

Fields `mrtd`, `rtmr0`, `rtmr1`, `rtmr2` are stable for the dstack OS + `tdx.small`
shape. `os_image_hash` is the **product formula** `sha256(MRTD||RTMR1||RTMR2)` =
`5c6d8f757e3adb0563efc809710076a631442db3b4de02ad32d33fe1994721e0`
(not the Phala catalog digest `bd369a8c…`).

## Review compose_hash

Offline:

```text
generate_review_app_compose(
  image=ghcr.io/baseintelligence/agent-challenge-review@sha256:25300418…
) → ade5a1cf9efe93c78e5840544877b5223912c06978e0bf1703d4dfefb5db774c
```

Matches live review TEE `compose_hash` in the evidence file above.

## Eval compose_hash

Offline:

```text
generate_app_compose(
  orchestrator_image=ghcr.io/baseintelligence/agent-challenge-eval@sha256:bf598fb8…
) default → 0647b4d9b1e3d458b7910638ee187c968835840fe2e65f1f332dbe69c518dfd9
```

`selfdeploy eval deploy` regenerates the same bytes to match the signed plan.

## Re-derive after image change

```bash
cd /work/baseintelligence/base
uv run --package agent-challenge python - <<'PY'
from agent_challenge.review.compose import generate_review_app_compose, review_app_compose_hash
from agent_challenge.canonical.compose import generate_app_compose, app_compose_hash
print(review_app_compose_hash(generate_review_app_compose(review_image="IMAGE")))
print(app_compose_hash(generate_app_compose(orchestrator_image="IMAGE")))
PY
```

If a live quote's six-field subset is NOT-IN-LIST, `run_staging.sh` can capture the
quote measurement into `config/*_allowlist.json` and restart AC
(`--capture-measurements`).
