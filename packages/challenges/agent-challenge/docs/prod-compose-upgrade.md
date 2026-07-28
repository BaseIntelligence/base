# Production eval compose upgrade path (artifact-aware pin)

> **Status:** documentation only. This file does **not** authorize or perform a
> production change. Do not SSH to the prod master, do not rewrite live
> allowlists, and do not deploy a new pin until an explicit ops authorization
> names this document and a measured evidence pack.
>
> **Blocking prerequisite for execution proof** (PR #5 / live residual): a
> matching `guest_artifact_proof` is structurally impossible on the live pin
> `daf0f209…` because Phala never injects
> `CHALLENGE_PHALA_EVAL_ARTIFACT_{URL,TOKEN}`.

## 0. Why this exists

Live runs (including submission 13) showed the production eval deploy's
`encrypted_env_names` **omit** both artifact env names. Root cause is the
**measured** `app-compose` pin, not the encrypt path alone:

| Pin (compose_hash) | Role today |
| --- | --- |
| `daf0f2090c02546c694bc7dc49516fd2629f4b8f9dd89e9bc2ed5c4156b662df` | **Live production** eval pin (joinbase / T8 residual). Measured **without** artifact envs. |
| `9a550b2dc0f06797976194bd4b53b8d7bfc8630f6390689f51b0bfebd36de622` | **Current generator** artifact-aware pin used by the repo's hash-determine tests (`LIVE_PIN_COMPOSE_HASH`). Includes artifact envs. |

Everything below that ceiling already works end-to-end on the old pin:
submit → review CVM → `review_allowed` → eval prepare → eval deploy
(`tdx.xlarge` observed). The ceiling is guest ZIP import +
`guest_artifact_proof`.

Code anchors (do not weaken):

- `src/agent_challenge/canonical/compose.py` — `DEFAULT_ALLOWED_ENVS`,
  `generate_app_compose`, `app_compose_hash`
- `src/agent_challenge/selfdeploy/eval.py` —
  `EVAL_ALLOWED_ENVS`, `MEASURE_TIME_EVAL_KEY_RELEASE_PLACEHOLDER`,
  pre-artifact hash-determine candidate, encrypt scoped to measured
  `allowed_envs`
- `tests/test_eval_compose_hash_determine.py` — locks both hashes offline
- `src/agent_challenge/evaluation/plan_scoring.py` —
  `require_host_guest_artifact_proof` (fail-closed)

---

## 1. Exact delta (`daf0f209…` → `9a550b2d…`)

### 1.1 How the hashes were derived (reproducible)

Run from the monorepo root (package on `PYTHONPATH` via uv):

```bash
cd /path/to/base
uv run --package agent-challenge python - <<'PY'
from agent_challenge.canonical.compose import generate_app_compose, app_compose_hash
from agent_challenge.selfdeploy import eval as E

NAME = E.DEFAULT_EVAL_COMPOSE_NAME  # "agent-challenge-eval-v1"
KR = E.MEASURE_TIME_EVAL_KEY_RELEASE_PLACEHOLDER
# https://validator-kr.example.invalid:8701

IMG_OLD = (
    "ghcr.io/baseintelligence/agent-challenge-eval@sha256:"
    "bf598fb8a3391fdbbef9b03184727a1615810a2cb31367e6d6d6b5c2a711d6e4"
)
IMG_NEW = (
    "ghcr.io/baseintelligence/agent-challenge-canonical@sha256:"
    "753e2296635bcd3a30703dc706509f0f8c0e7dd2f82bef730ad7f1cc9443933c"
)
pre = tuple(
    n for n in E.EVAL_ALLOWED_ENVS
    if n not in {E.EVAL_ARTIFACT_URL_ENV, E.EVAL_ARTIFACT_TOKEN_ENV}
)
old = generate_app_compose(
    orchestrator_image=IMG_OLD, name=NAME, key_release_url=KR, allowed_envs=pre,
)
new = generate_app_compose(
    orchestrator_image=IMG_NEW, name=NAME, key_release_url=KR,
    allowed_envs=E.EVAL_ALLOWED_ENVS,
)
assert app_compose_hash(old) == (
    "daf0f2090c02546c694bc7dc49516fd2629f4b8f9dd89e9bc2ed5c4156b662df"
)
assert app_compose_hash(new) == (
    "9a550b2dc0f06797976194bd4b53b8d7bfc8630f6390689f51b0bfebd36de622"
)
print("ok")
PY
```

Both asserts pass on this branch (see
`tests/test_eval_compose_hash_determine.py`).

### 1.2 Inputs that differ

| Factor | `daf0f209…` (live) | `9a550b2d…` (target) |
| --- | --- | --- |
| Orchestrator image | `ghcr.io/baseintelligence/agent-challenge-eval@sha256:bf598fb8a3391fdbbef9b03184727a1615810a2cb31367e6d6d6b5c2a711d6e4` | `ghcr.io/baseintelligence/agent-challenge-canonical@sha256:753e2296635bcd3a30703dc706509f0f8c0e7dd2f82bef730ad7f1cc9443933c` |
| Compose `name` | `agent-challenge-eval-v1` | same |
| Measure-time `key_release_url` bake | `https://validator-kr.example.invalid:8701` (placeholder; **not** plan trust root) | same |
| `allowed_envs` count | 22 | 24 |
| Artifact env names | **absent** | **present** |

**Unchanged** top-level envelope fields (verified equal in the generator
output): `manifest_version`, `runner`, `kms_enabled`, `gateway_enabled`,
`tproxy_enabled`, `local_key_provider_enabled`, `public_logs`,
`public_sysinfo`, `public_tcbinfo`, `no_instance_id`, `secure_time`,
`storage_fs`, `features`, `pre_launch_script`.

### 1.3 `allowed_envs` delta (only material name change)

**Added in `9a550b2d…` (sorted position):**

- `CHALLENGE_PHALA_EVAL_ARTIFACT_TOKEN`
- `CHALLENGE_PHALA_EVAL_ARTIFACT_URL`

Full NEW list (24 names) is exactly `sorted(EVAL_ALLOWED_ENVS)` /
`DEFAULT_ALLOWED_ENVS` as of this branch. Full OLD list is that set minus the
two artifact names.

### 1.4 `docker_compose_file` unified diff (derived)

```diff
--- daf0f209 (prod live pin)
+++ 9a550b2d (artifact-aware generator)
@@ environment passthrough names @@
       - "CHALLENGE_PHALA_AGENT_HASH"
       - "CHALLENGE_PHALA_ATTESTATION_ENABLED"
       - "CHALLENGE_PHALA_CANONICAL_MEASUREMENT"
+      - "CHALLENGE_PHALA_EVAL_ARTIFACT_TOKEN"
+      - "CHALLENGE_PHALA_EVAL_ARTIFACT_URL"
       - "CHALLENGE_PHALA_EVAL_PLAN"
       - "CHALLENGE_PHALA_KEY_RELEASE_URL=https://validator-kr.example.invalid:8701"
       …
-    "image": "ghcr.io/baseintelligence/agent-challenge-eval@sha256:bf598fb8…"
+    "image": "ghcr.io/baseintelligence/agent-challenge-canonical@sha256:753e2296…"
```

No other service keys change (`restart`, `command`, socket volumes).

### 1.5 Same-image alternative (not `9a550b2d…`)

If ops keep the **eval** image `bf598…` and only add artifact envs, the
generator yields a **different** hash:

```text
3a81feaf607d28aabd4e7705b3c5cbf6999b7fa4fa3f796247f9bf79fad95e38
```

That pin is **also** artifact-capable. It is **not** the
`LIVE_PIN_COMPOSE_HASH` / `9a550b2d…` value. Choose one target and measure it;
do not mix labels.

---

## 2. What must change in production (ordered)

Fail-closed rule: **empty allowlist accepts nothing**. Never clear an
allowlist “to unblock”; always replace with a measured entry set.

Prod topology reminder (from monorepo `AGENTS.md`): Agent Challenge is
**embedded** in the master container. Live residual notes that the running
master AC is largely older code with only `review/deployment.py` hotpatched
under:

```text
/var/lib/base/compose-master/base-master-prod/hotpatches/
docker-compose.override.yml   # bind-mounts hotpatches into the master container
```

Do **not** invent host paths beyond what ops already uses; confirm on the
authorized change window.

### Step A — freeze and announce

1. Record current live values (names/digests only) from the running AC env /
   prepare payload:
   - `CHALLENGE_EVAL_APP_IMAGE_REF`
   - `CHALLENGE_EVAL_APP_COMPOSE_HASH` (expect `daf0f209…`)
   - `CHALLENGE_EVAL_APP_MEASUREMENT` (JSON)
   - `CHALLENGE_EVAL_APP_MEASUREMENT_ALLOWLIST` (JSON array)
   - `CHALLENGE_EVAL_APP_KMS_PUBLIC_KEY_HEX` (public)
   - `CHALLENGE_EVAL_APP_IDENTITY`
   - KR allowlist file used by the RA-TLS listener (path is host-local; staging
     analogue is `scripts/staging/config/kr_allowlist.json` /
     `config/kr/eval-allowlist.json`)
2. Drain or wait out in-flight evals where possible (see §4). Submission 11 is
   wedged until ~`2026-07-28T15:03:54Z` — do not cancel/fail it (409).

### Step B — offline compose pin (no prod write yet)

1. Confirm target image is pullable on Phala (GHCR digest).
2. Recompute compose_hash with the snippet in §1.1 for the **chosen** target
   (`9a550b2d…` or same-image `3a81feaf…`).
3. Write the deployable `app-compose.json` bytes via
   `render_app_compose` / `write_app_compose` only (never a hand-pretty
   `json.dumps`).

### Step C — measure the new guest (required; values unknown until then)

See §3. Capture at least:

- `mrtd`, `rtmr0`, `rtmr1`, `rtmr2` (96 hex each)
- product `os_image_hash` = `sha256(MRTD || RTMR1 || RTMR2)` (64 hex)
- live `compose_hash` from provision / quote (must equal offline pin)
- Phala KMS app public key hex + sha256 (if the new app identity rotates)
- `vm_shape` actually used (`tdx.small` vs `tdx.xlarge` — shape can change
  RTMR/MRTD; pin the shape you will run in prod)

**Open input:** the exact MRTD/RTMR/os_image_hash/KMS pubkey for
`canonical@753e` on the production shape are **not** known from this repo
alone. Staging `pins.json` registers are for `eval@bf598` / `tdx.small` and
must not be copied onto a different image/shape without a fresh quote.

### Step D — update validator AC config (master embed env)

Settings class: `agent_challenge.sdk.config.ChallengeSettings`
(`env_prefix=CHALLENGE_`).

| Env key | Action |
| --- | --- |
| `CHALLENGE_EVAL_APP_IMAGE_REF` | Set to target image digest ref |
| `CHALLENGE_EVAL_APP_COMPOSE_HASH` | Set to target compose_hash (`9a550b2d…` or `3a81feaf…`) |
| `CHALLENGE_EVAL_APP_MEASUREMENT` | JSON of measured registers + `os_image_hash` + `key_provider` + `vm_shape` |
| `CHALLENGE_EVAL_APP_MEASUREMENT_ALLOWLIST` | JSON array of allowlist entries; each entry must include the new `compose_hash` and matching registers. Prefer **dual-entry** briefly (old + new) only if you intentionally accept both pins during a cutover window; otherwise replace with the single new entry. **Empty = admit nothing.** |
| `CHALLENGE_EVAL_APP_KMS_PUBLIC_KEY_HEX` | Update if provision identity rotates |
| `CHALLENGE_EVAL_APP_IDENTITY` | Keep moniker stable unless ops intentionally renames (`agent-challenge-eval-v1` is the measured compose `name` for both pins above) |
| `CHALLENGE_EVAL_KEY_RELEASE_ENDPOINT` | Unchanged (live RA-TLS `host:8701`); must remain the plan trust root, not the measure-time HTTPS placeholder |
| `CHALLENGE_PHALA_ATTESTATION_ENABLED` / `CHALLENGE_ATTESTED_REVIEW_ENABLED` | Stay `true` (production dual-on) |

Where these live on the host is an **open input** (confirm during the change
window). Candidates historically used on master-embed installs:

- compose project env / `embed.env` for the AC child
- `/var/lib/base/compose-master/base-master-prod/` (and override)
- hotpatch bind-mounts under `…/hotpatches/` (code only — **pins are env/config**,
  not Python hotpatches)

Review pins (`CHALLENGE_REVIEW_APP_*`) do **not** need to move for the artifact
fix unless ops is deliberately rebasing review in the same window.

### Step E — update key-release allowlist (host KR, not `keyrelease/` package edits)

The RA-TLS grant path is fail-closed on measurement allowlist match
(`agent_challenge.keyrelease.allowlist`). The host file must gain an entry
whose `compose_hash` (and registers) match the **new** guest.

Staging shape (illustrative only):

```json
{
  "entries": [
    {
      "mrtd": "<measured>",
      "rtmr0": "<measured>",
      "rtmr1": "<measured>",
      "rtmr2": "<measured>",
      "os_image_hash": "<product formula>",
      "compose_hash": "9a550b2dc0f06797976194bd4b53b8d7bfc8630f6390689f51b0bfebd36de622",
      "key_provider": "phala"
    }
  ]
}
```

**Open input:** absolute path of the prod KR allowlist on the validator host
(residual notes mention `/var/lib/base/keyrelease/eval-allowlist.json` as a
historical location — confirm before edit). Reload/restart the KR listener so
the new file is live.

Do **not** remove DCAP / quote verification, run-token binding, or digest
checks anywhere to “make it pass”.

### Step F — roll the AC process

1. Apply env + allowlist files.
2. Restart only the AC embed / master unit that loads them (ops runbook).
3. Health gate (§5) before admitting miner traffic on the new pin.

### Step G — miner / selfdeploy side

Miners on current selfdeploy already hash-determine both full and pre-artifact
`allowed_envs` (`selfdeploy/eval.py`). After the validator signs plans with the
new `compose_hash` + image_ref, deploy will match the full set and
`encrypt_eval_secrets` will include artifact URL/token **because those names
are in the measured allowlist**.

No miner-side weaken of compose_hash checks.

---

## 3. How new measurement values are obtained

**Never invent MRTD/RTMR/os_image_hash.** Only two legitimate sources:

### 3.1 Offline compose_hash (already known)

Use §1.1. Record the hex in the change ticket **before** any CVM is created.

### 3.2 Live TDX registers (must measure)

1. Deploy a **throwaway** eval CVM with the **exact** target `app-compose`
   bytes and target image, on the **exact** prod shape/region you will use.
2. From the provision response and/or guest quote / Phala attestation:
   - read `compose_hash` → must equal offline pin
   - read `mrtd`, `rtmr0`, `rtmr1`, `rtmr2`
   - compute product `os_image_hash = sha256(MRTD||RTMR1||RTMR2)` (binary
     concat of the 48-byte registers, then SHA-256) — see
     `canonical/measurement.py` and `scripts/staging/config/measurements_source.md`
3. Persist the six-field subset + `compose_hash` into:
   - `CHALLENGE_EVAL_APP_MEASUREMENT`
   - `CHALLENGE_EVAL_APP_MEASUREMENT_ALLOWLIST` entry
   - KR `eval-allowlist.json` entry
4. Tear down the throwaway CVM. Confirm account hygiene
   (`npx phala@latest cvms list --json` → only expected leftovers, ideally 0
   for a dedicated measure account).

Optional tooling: `dstack-mr` for OS-image replay when you have the dstack OS
bundle; still **cross-check** against a real quote before production pin.

### 3.3 What this repo already knows vs open inputs

| Value | Known offline? |
| --- | --- |
| `daf0f209…` compose bytes / hash | Yes (generator + tests) |
| `9a550b2d…` compose bytes / hash | Yes (generator + tests) |
| Artifact env name delta | Yes |
| Image digest delta (`bf598` → `753e`) | Yes |
| MRTD/RTMR/os_image_hash for **new** image@prod shape | **No — measure** |
| Prod KMS pubkey after new app provision | **No — capture from provision** |
| Exact prod env file paths / unit names | **No — confirm on host** |
| Whether prod eval shape is `tdx.small` or `tdx.xlarge` for the pin | **Confirm** (live residual saw `tdx.xlarge` deploys; staging pins use `tdx.small`) |

---

## 4. Ordering and compatibility

### 4.1 Must review and eval pins move together?

**No.** Artifact delivery is eval-only. Review compose/image/allowlist can stay
on the current review pin (`ade5a1cf…` / review image `25300418…` in staging
examples) unless ops chooses a broader rebase.

### 4.2 In-flight submissions

| State | Effect of flipping eval pin mid-flight |
| --- | --- |
| `review_*` only | Unaffected (review pin unchanged). |
| `eval_prepared` / plan signed under **old** `compose_hash` | Miner deploy still hash-determines pre-artifact compose. Guest still **cannot** receive artifact envs. Result admission still requires `guest_artifact_proof` → will fail closed. Prefer let TTL expire or keep dual allowlist only for KR/quote verify of old guests, not for new prepares. |
| `eval_running` with **old** guest | Guest continues on old measured compose. KR allowlist must still contain the **old** entry until that guest exits, or grants deny. |
| New `eval/prepare` after config flip | Plans carry **new** `compose_hash` + image. Miners must deploy the new compose. |

**Recommendation:** dual-entry allowlists (old+new) only for the KR + AC
measurement allowlists during a short overlap; set
`CHALLENGE_EVAL_APP_COMPOSE_HASH` / image / single measurement object to the
**new** pin so **new** prepares only issue the artifact-aware plan. Remove the
old allowlist entry after no old guests remain.

### 4.3 Submission 11 (wedged)

Observed: `eval_running`, `key_grant_state=granted`, `retryable=false`,
cancel/failure **409**, until ~`2026-07-28T15:03:54Z`.

- Do not attempt cancel/fail (API correctly refuses).
- Do not account-sweep CVMs (owned-only teardown policy).
- After expiry, history retains the attempt; a **fresh submission** is required
  for proof (see §6).
- Upgrading the pin does not unwedge 11.

### 4.4 Selfdeploy pre-artifact compatibility

`build_eval_deployment_plan` still searches the pre-artifact `allowed_envs`
candidate so old signed plans remain deployable. That is **hash-determine
only**. Encrypt deliberately **omits** names absent from the matched measured
allowlist — so old pins never get a fake artifact grant injected into a
compose that cannot list those envs. Do not “fix” that by forcing env names
into `encrypted_env` outside `allowed_envs` (Phala would drop them; and it
would be a measurement lie).

---

## 5. Rollback

### 5.1 Exact revert

1. Restore previous env values:
   - `CHALLENGE_EVAL_APP_IMAGE_REF` → `…/agent-challenge-eval@sha256:bf598fb8…`
   - `CHALLENGE_EVAL_APP_COMPOSE_HASH` → `daf0f2090c02546c694bc7dc49516fd2629f4b8f9dd89e9bc2ed5c4156b662df`
   - `CHALLENGE_EVAL_APP_MEASUREMENT` / `…_ALLOWLIST` → prior JSON (from Step A freeze)
   - `CHALLENGE_EVAL_APP_KMS_PUBLIC_KEY_HEX` → prior
2. Restore KR allowlist to the pre-change file (must still contain `daf0f209…`
   entry if old guests exist).
3. Restart AC embed + KR listener.
4. Run health gate (§5.2).
5. Confirm a new `eval/prepare` returns `eval_app.compose_hash == daf0f209…`.

### 5.2 Go / no-go health check

| Check | Pass condition |
| --- | --- |
| Validator / master health | `GET https://chain.joinbase.ai/health` → **200** |
| Challenge OpenAPI | `GET https://chain.joinbase.ai/challenges/agent-challenge/openapi.json` → **200** |
| AC process stability | Docker/compose `RestartCount` for master/AC container **stable** across ≥2–3 minutes (no crash loop) |
| KR health (host) | Local offline fixture `GET http://127.0.0.1:8700/health` → `{"status":"ok"}` if that listener is part of the install; RA-TLS :8701 accepts a known-good probe without process exit |
| Config load | Process logs show allowlist entry count **> 0** for eval; no startup fail-closed on empty allowlist |
| Pin smoke | Fresh `eval/prepare` (after review_allowed) shows expected `compose_hash` |

**No-go:** RestartCount climbing, `/health` non-200, empty allowlist, prepare
compose_hash neither old nor intended new, or KR crash on reload.

---

## 6. Verification plan (post-upgrade execution proof)

Goal: a **fresh** submission yields `guest_artifact_proof` with all three
hashes equal to the known miner ZIP pin:

```text
61cca9bc06c52644182a4de98b89207742369589859d84a00ac6494327413f68
```

(from `scripts/staging/run_staging.sh` `EXPECTED_AGENT_HASH` /
`scripts/miner_agent/dist/miner_agent.zip`).

### 6.1 Preconditions

- [ ] Prod eval pin is artifact-aware (`9a550b2d…` or chosen same-image hash)
- [ ] AC measurement allowlist + KR allowlist contain the measured entry
- [ ] Health gate green (§5.2)
- [ ] No reliance on submission 11
- [ ] Phala account CVM hygiene understood (owned-only teardown)

### 6.2 Run (fresh submission)

1. Submit the pinned miner ZIP (hash `61cca9bc…`).
2. `selfdeploy review deploy` → wait `review_allowed` → teardown review CVM.
3. `eval/prepare` → assert signed plan:
   - `eval_app.image_ref` == target image
   - `eval_app.compose_hash` == target hash
4. `selfdeploy eval deploy` → inspect deploy material (names only):
   - measured compose `allowed_envs` contains both
     `CHALLENGE_PHALA_EVAL_ARTIFACT_URL` and
     `CHALLENGE_PHALA_EVAL_ARTIFACT_TOKEN`
   - `encrypted_env` / env key list includes both names
5. Wait for result acceptance.
6. Assert `guest_artifact_proof` present and:

   ```text
   package_sha256 == zip_sha256 == agent_hash
     == 61cca9bc06c52644182a4de98b89207742369589859d84a00ac6494327413f68
   ```

7. Tear down **owned** eval CVM only; confirm no account sweep.

### 6.3 Failure signatures (do not weaken gates)

| Symptom | Likely cause |
| --- | --- |
| `encrypted_env_names` still missing artifact keys | Plan still on `daf0f209…` or encrypt scoped to pre-artifact match |
| Guest cannot download ZIP | Artifact grant mint/URL wrong, or names not in measured allowlist |
| `guest_artifact_proof_missing` | Guest never imported artifact / old image without importer |
| `guest_artifact_proof_hash_mismatch` | Wrong ZIP bytes inside guest |
| KR deny / measurement not in list | Allowlist missing new compose_hash or wrong registers/shape |
| Compose hash mismatch on deploy | Miner generator ≠ signed plan (image/name/KR bake/allowed_envs) |

---

## 7. Explicit non-goals / safety

- **Do not** execute this upgrade from an agent session without separate human
  authorization naming this document and a measurement evidence pack.
- **Do not** SSH to `86.38.238.235` or any prod host as part of “finishing”
  this doc.
- **Do not** weaken `compose_hash`, RTMR/MRTD allowlists, KMS digest checks,
  DCAP verification, run-token binding, or `guest_artifact_proof`.
- **Do not** print Phala tokens, OpenRouter keys, mnemonics, or private keys.
- **Do not** edit `keyrelease/` package code for this pin cut — host allowlist
  + AC env only, unless a separate authorized code change exists.

---

## 8. Related files

| Path | Role |
| --- | --- |
| `src/agent_challenge/canonical/compose.py` | Measured compose generator |
| `src/agent_challenge/selfdeploy/eval.py` | Plan → deploy + encrypt |
| `src/agent_challenge/sdk/config.py` | `CHALLENGE_EVAL_APP_*` settings |
| `src/agent_challenge/evaluation/plan_scoring.py` | Host `guest_artifact_proof` gate |
| `tests/test_eval_compose_hash_determine.py` | Offline pin locks |
| `tests/test_eval_artifact_encrypted_env.py` | Artifact env encrypt behavior |
| `scripts/staging/config/measurements_source.md` | Staging measurement provenance |
| `docs/staging.md` | Local real-Phala loop (not prod) |
| `docs/validator/self-deploy.md` | Validator surfaces |

---

## 9. Open inputs checklist (must close before execution)

- [ ] Target pin choice: `9a550b2d…` (canonical@753e) vs `3a81feaf…` (eval@bf598 + artifact envs)
- [ ] Measured MRTD/RTMR0-2/os_image_hash for that image@prod shape
- [ ] Prod eval `vm_shape` / region for the pin
- [ ] Prod KMS public key hex after provision (if rotated)
- [ ] Absolute paths of prod AC env + KR allowlist + restart unit
- [ ] Whether dual-entry allowlist overlap is required for in-flight guests
- [ ] Authorized change window and owner
