//! Operator-side held-out Harbor verifier.
//!
//! Runs the pack's `tests/` harness in a digest-pinned container through
//! [`docker_engine::AllowlistClient`] + [`docker_engine::Allowlist::verifier`].
//! Held-out bytes stay on the operator host (`HarborPack::held_out` /
//! staged workdir) and are never projected into agent-facing types.
//!
//! Resolve rule (deepagent): reward `1` iff every F2P and every P2P node passes;
//! otherwise reward `0`. Dual-truth: `solution.patch` → 1, empty patch → 0.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use agent_pack::HarborPack;
use docker_engine::{
    Allowlist, AllowlistClient, DockerApi, DockerError, RunSpec, OWNED_NAME_PREFIX,
};
use serde::Deserialize;
use thiserror::Error;

/// Default verifier wall-clock budget (seconds) when pack omits `verifier.timeout_sec`.
pub const DEFAULT_VERIFIER_TIMEOUT_SEC: u64 = 1800;

/// Binary reward from the held-out grader (`0` or `1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reward(u8);

impl Reward {
    /// Construct a binary reward; only `0` and `1` are valid.
    ///
    /// # Errors
    /// [`VerifyError::MalformedOutput`] when value is not 0/1.
    pub fn try_new(v: u8) -> Result<Self, VerifyError> {
        if v <= 1 {
            Ok(Self(v))
        } else {
            Err(VerifyError::MalformedOutput {
                message: format!("reward must be 0 or 1, got {v}"),
            })
        }
    }

    /// Raw binary value.
    #[must_use]
    pub const fn value(self) -> u8 {
        self.0
    }

    /// True when the pack resolved (all F2P + P2P passed).
    #[must_use]
    pub const fn is_resolve(self) -> bool {
        self.0 == 1
    }
}

/// Why a grade produced reward 0 (miner-attributable), distinct from operator faults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZeroReason {
    /// `model.patch` failed `git apply`.
    ApplyFailed {
        /// Grader / log excerpt.
        detail: String,
    },
    /// Tests ran; F2P and/or P2P failures (or missing nodes).
    TestsFailed {
        /// F2P failures counted by grader.
        f2p_failed: u32,
        /// P2P failures counted by grader.
        p2p_failed: u32,
        /// Optional grader summary line.
        detail: String,
    },
    /// Reward file said 0 without richer stats.
    Unspecified {
        /// Captured detail.
        detail: String,
    },
}

/// Operator-side verify failures and miner-attributable zero rewards.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// Wall-clock timeout — distinct from reward 0.
    #[error("verifier timed out after {timeout_sec}s")]
    Timeout {
        /// Configured timeout seconds.
        timeout_sec: u64,
    },
    /// Miner patch graded to reward 0 with a typed reason.
    #[error("reward 0: {reason:?}")]
    RewardZero {
        /// Structured zero reason.
        reason: ZeroReason,
    },
    /// Patch apply failed before tests (also reward 0 path).
    #[error("model.patch apply failed: {message}")]
    ApplyFailed {
        /// Detail.
        message: String,
    },
    /// Docker / allowlist / engine failure (operator fault).
    #[error("docker: {0}")]
    Docker(#[from] DockerError),
    /// reward.json / junit unreadable or invalid.
    #[error("malformed verifier output: {message}")]
    MalformedOutput {
        /// Parse detail.
        message: String,
    },
    /// Staging / I/O on the operator host.
    #[error("verify staging: {message}")]
    Staging {
        /// I/O detail.
        message: String,
    },
    /// Pack missing held-out tests material required to grade.
    #[error("pack missing held-out tests: {message}")]
    MissingHeldOut {
        /// What was missing.
        message: String,
    },
}

/// Grade a miner `model.patch` against a pack's held-out harness.
pub trait Verifier: Send + Sync {
    /// Run the held-out verifier; `Ok(Reward(1|0))` or typed error.
    ///
    /// # Errors
    /// [`VerifyError`] variants for timeout, docker, staging, or malformed output.
    /// Reward 0 is returned as [`VerifyError::RewardZero`] **or** as `Ok(Reward(0))`
    /// depending on [`HarborVerifier::reward_zero_as_err`].
    fn grade(&self, pack: &HarborPack, model_patch: &[u8]) -> Result<Reward, VerifyError>;
}

/// Configuration for [`HarborVerifier`].
#[derive(Debug, Clone)]
pub struct HarborVerifierConfig {
    /// Docker Engine HTTP base (socket-proxy), e.g. `http://127.0.0.1:2375`.
    pub docker_base: String,
    /// Digest-pinned environment image (`name@sha256:…` or `sha256:…` id).
    pub environment_image: String,
    /// Operator work root for staging binds (created if missing).
    pub work_root: PathBuf,
    /// Override timeout; default pack `verifier_timeout_sec` or [`DEFAULT_VERIFIER_TIMEOUT_SEC`].
    pub timeout_sec_override: Option<u64>,
    /// When true, reward 0 is `Err(RewardZero|ApplyFailed)`; when false, `Ok(Reward(0))`.
    pub reward_zero_as_err: bool,
}

/// Harbor pack verifier using allowlisted Docker one-shots.
pub struct HarborVerifier {
    client: AllowlistClient,
    cfg: HarborVerifierConfig,
}

impl HarborVerifier {
    /// Build a verifier client on the **verifier** allowlist profile.
    ///
    /// # Errors
    /// When the HTTP client cannot be constructed.
    pub fn new(cfg: HarborVerifierConfig) -> Result<Self, VerifyError> {
        let client = AllowlistClient::with_allowlist(&cfg.docker_base, Allowlist::verifier())?;
        Ok(Self { client, cfg })
    }

    /// Borrow the underlying allowlisted client (cleanup / QA).
    #[must_use]
    pub const fn client(&self) -> &AllowlistClient {
        &self.client
    }

    /// Resolve timeout for a pack.
    #[must_use]
    pub fn timeout_for(&self, pack: &HarborPack) -> u64 {
        self.cfg
            .timeout_sec_override
            .or(pack.verifier_timeout_sec)
            .unwrap_or(DEFAULT_VERIFIER_TIMEOUT_SEC)
    }

    /// Remove any leftover `base-verify-*` containers.
    ///
    /// # Errors
    /// Docker list failures.
    pub fn cleanup_owned(&self) -> Result<usize, VerifyError> {
        Ok(self.client.cleanup_owned()?)
    }

    /// Count surviving owned containers.
    ///
    /// # Errors
    /// Docker list failures.
    pub fn owned_count(&self) -> Result<usize, VerifyError> {
        Ok(self.client.list_owned()?.len())
    }
}

impl Verifier for HarborVerifier {
    fn grade(&self, pack: &HarborPack, model_patch: &[u8]) -> Result<Reward, VerifyError> {
        let timeout_sec = self.timeout_for(pack);
        let stage = stage_workdir(&self.cfg.work_root, pack, model_patch)?;
        let name = unique_container_name(&pack.task_id);
        let binds = vec![
            format!("{}:/tests:ro", stage.tests_dir.display()),
            format!("{}:/logs", stage.logs_dir.display()),
        ];
        // model.patch is under logs/artifacts via staging layout
        let spec = RunSpec {
            name: name.clone(),
            image: self.cfg.environment_image.clone(),
            cmd: vec![
                "bash".into(),
                "-lc".into(),
                // test.sh is the pack entrypoint; ensure executable + run.
                "chmod +x /tests/test.sh 2>/dev/null; bash /tests/test.sh; \
                 code=$?; \
                 if [ -f /logs/verifier/reward.json ]; then cat /logs/verifier/reward.json; fi; \
                 exit $code"
                    .into(),
            ],
            binds,
            env: vec![
                "TESTS_DIR=/tests".into(),
                "VERIFIER_DIR=/logs/verifier".into(),
                "APP_DIR=/app".into(),
                "ARTIFACTS_DIR=/logs/artifacts".into(),
            ],
            network_disabled: true,
            working_dir: Some("/app".into()),
            timeout_sec: Some(timeout_sec),
        };

        let run_result = self.client.run_owned(&spec);
        // Always attempt cleanup of *this* name in case run_owned failed mid-flight.
        // Do not cleanup_owned() here — parallel grades must not reap each other.
        let _ = self.client.remove_container(&name);

        let run_result = match run_result {
            Ok(r) => r,
            Err(DockerError::Timeout { timeout_sec }) => {
                let _ = fs::remove_dir_all(&stage.root);
                return Err(VerifyError::Timeout { timeout_sec });
            }
            Err(e) => {
                let _ = fs::remove_dir_all(&stage.root);
                return Err(VerifyError::Docker(e));
            }
        };

        let reward_path = stage.logs_dir.join("verifier/reward.json");
        let parsed = parse_reward_file(&reward_path).or_else(|e| {
            // Fall back to scanning container logs for reward.json dump.
            parse_reward_from_logs(&run_result.logs).map_err(|_| e)
        });

        let _ = fs::remove_dir_all(&stage.root);

        match parsed {
            Ok(outcome) => finalize_outcome(outcome, self.cfg.reward_zero_as_err),
            Err(e) => Err(e),
        }
    }
}

/// Parsed `reward.json` from the pack grader.
#[derive(Debug, Clone, Deserialize)]
struct RewardJson {
    reward: u8,
    #[serde(default)]
    apply_failed: Option<u8>,
    #[serde(default)]
    f2p_total: Option<u32>,
    #[serde(default)]
    f2p_passed: Option<u32>,
    #[serde(default)]
    p2p_total: Option<u32>,
    #[serde(default)]
    p2p_passed: Option<u32>,
}

#[derive(Debug, Clone)]
struct GradeOutcome {
    reward: Reward,
    apply_failed: bool,
    f2p_failed: u32,
    p2p_failed: u32,
    detail: String,
}

fn finalize_outcome(
    outcome: GradeOutcome,
    reward_zero_as_err: bool,
) -> Result<Reward, VerifyError> {
    if outcome.reward.is_resolve() {
        return Ok(outcome.reward);
    }
    if !reward_zero_as_err {
        return Ok(outcome.reward);
    }
    if outcome.apply_failed {
        return Err(VerifyError::ApplyFailed {
            message: outcome.detail,
        });
    }
    let reason = if outcome.f2p_failed > 0 || outcome.p2p_failed > 0 {
        ZeroReason::TestsFailed {
            f2p_failed: outcome.f2p_failed,
            p2p_failed: outcome.p2p_failed,
            detail: outcome.detail,
        }
    } else {
        ZeroReason::Unspecified {
            detail: outcome.detail,
        }
    };
    Err(VerifyError::RewardZero { reason })
}

fn parse_reward_file(path: &Path) -> Result<GradeOutcome, VerifyError> {
    let text = fs::read_to_string(path).map_err(|e| VerifyError::MalformedOutput {
        message: format!("read {}: {e}", path.display()),
    })?;
    parse_reward_json(&text)
}

fn parse_reward_from_logs(logs: &str) -> Result<GradeOutcome, VerifyError> {
    // Last JSON object containing "reward" wins.
    for line in logs.lines().rev() {
        let t = line.trim();
        if t.contains("\"reward\"") && t.starts_with('{') {
            if let Ok(o) = parse_reward_json(t) {
                return Ok(o);
            }
        }
    }
    Err(VerifyError::MalformedOutput {
        message: "no reward.json in logs".into(),
    })
}

fn parse_reward_json(text: &str) -> Result<GradeOutcome, VerifyError> {
    let rj: RewardJson =
        serde_json::from_str(text.trim()).map_err(|e| VerifyError::MalformedOutput {
            message: format!("reward.json: {e}"),
        })?;
    let reward = Reward::try_new(rj.reward)?;
    let apply_failed = rj.apply_failed.unwrap_or(0) != 0;
    let f2p_total = rj.f2p_total.unwrap_or(0);
    let f2p_passed = rj.f2p_passed.unwrap_or(0);
    let p2p_total = rj.p2p_total.unwrap_or(0);
    let p2p_passed = rj.p2p_passed.unwrap_or(0);
    let f2p_failed = f2p_total.saturating_sub(f2p_passed);
    let p2p_failed = p2p_total.saturating_sub(p2p_passed);
    Ok(GradeOutcome {
        reward,
        apply_failed,
        f2p_failed,
        p2p_failed,
        detail: text.trim().to_owned(),
    })
}

struct Stage {
    root: PathBuf,
    tests_dir: PathBuf,
    logs_dir: PathBuf,
}

fn stage_workdir(
    work_root: &Path,
    pack: &HarborPack,
    model_patch: &[u8],
) -> Result<Stage, VerifyError> {
    fs::create_dir_all(work_root).map_err(|e| VerifyError::Staging {
        message: format!("create work_root: {e}"),
    })?;
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let root = work_root.join(format!("stage-{}-{ns}", pack.task_id));
    let tests_dir = root.join("tests");
    let logs_dir = root.join("logs");
    let artifacts = logs_dir.join("artifacts");
    let verifier_dir = logs_dir.join("verifier");
    for d in [&tests_dir, &artifacts, &verifier_dir] {
        fs::create_dir_all(d).map_err(|e| VerifyError::Staging {
            message: format!("mkdir {}: {e}", d.display()),
        })?;
    }

    // Materialize tests/* from pack.files (held-out stays operator-side only).
    let mut wrote_test = false;
    for (rel, bytes) in &pack.files {
        if let Some(rest) = rel.strip_prefix("tests/") {
            if rest.is_empty() {
                continue;
            }
            let dest = tests_dir.join(rest);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent).map_err(|e| VerifyError::Staging {
                    message: format!("mkdir {}: {e}", parent.display()),
                })?;
            }
            fs::write(&dest, bytes).map_err(|e| VerifyError::Staging {
                message: format!("write {}: {e}", dest.display()),
            })?;
            wrote_test = true;
        }
    }
    // Ensure critical held-out files exist even if path layout differs.
    if let Some(g) = &pack.held_out.grader_py {
        fs::write(tests_dir.join("grader.py"), g).map_err(|e| VerifyError::Staging {
            message: format!("grader.py: {e}"),
        })?;
        wrote_test = true;
    }
    if let Some(tp) = &pack.held_out.test_patch {
        fs::write(tests_dir.join("test.patch"), tp).map_err(|e| VerifyError::Staging {
            message: format!("test.patch: {e}"),
        })?;
        wrote_test = true;
    }
    if !wrote_test {
        return Err(VerifyError::MissingHeldOut {
            message: "no tests/ files in pack".into(),
        });
    }

    fs::write(artifacts.join("model.patch"), model_patch).map_err(|e| VerifyError::Staging {
        message: format!("model.patch: {e}"),
    })?;

    Ok(Stage {
        root,
        tests_dir,
        logs_dir,
    })
}

fn unique_container_name(task_id: &str) -> String {
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let safe: String = task_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    format!("{OWNED_NAME_PREFIX}{safe}-{ns}")
}

/// Pure dual-truth helper used by unit tests without Docker: grade from reward.json bytes.
///
/// # Errors
/// Parse failures.
pub fn reward_from_json_bytes(bytes: &[u8]) -> Result<Reward, VerifyError> {
    let text = std::str::from_utf8(bytes).map_err(|e| VerifyError::MalformedOutput {
        message: e.to_string(),
    })?;
    Ok(parse_reward_json(text)?.reward)
}

/// Map docker timeout into verify timeout (identity helper for call sites / tests).
#[must_use]
pub fn map_docker_timeout(err: &DockerError) -> Option<VerifyError> {
    match err {
        DockerError::Timeout { timeout_sec } => Some(VerifyError::Timeout {
            timeout_sec: *timeout_sec,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_try_new_accepts_binary_only() {
        assert_eq!(Reward::try_new(0).expect("0").value(), 0);
        assert_eq!(Reward::try_new(1).expect("1").value(), 1);
        assert!(Reward::try_new(2).is_err());
    }

    #[test]
    fn parse_reward_json_resolve_and_zero() {
        let one = parse_reward_json(
            r#"{"reward":1,"f2p_total":4,"f2p_passed":4,"p2p_total":8,"p2p_passed":8}"#,
        )
        .expect("one");
        assert!(one.reward.is_resolve());
        assert_eq!(one.f2p_failed, 0);

        let zero = parse_reward_json(
            r#"{"reward":0,"f2p_total":4,"f2p_passed":0,"p2p_total":8,"p2p_passed":8}"#,
        )
        .expect("zero");
        assert!(!zero.reward.is_resolve());
        assert_eq!(zero.f2p_failed, 4);
    }

    #[test]
    fn parse_apply_failed_flag() {
        let o = parse_reward_json(r#"{"reward":0,"apply_failed":1,"f2p_total":0,"f2p_passed":0,"p2p_total":0,"p2p_passed":0}"#)
            .expect("af");
        assert!(o.apply_failed);
        let err = finalize_outcome(o, true).expect_err("err");
        assert!(matches!(err, VerifyError::ApplyFailed { .. }));
    }

    #[test]
    fn finalize_tests_failed_distinct_from_apply() {
        let o = GradeOutcome {
            reward: Reward(0),
            apply_failed: false,
            f2p_failed: 0,
            p2p_failed: 3,
            detail: "p2p broke".into(),
        };
        let err = finalize_outcome(o, true).expect_err("z");
        match err {
            VerifyError::RewardZero {
                reason: ZeroReason::TestsFailed { p2p_failed, .. },
            } => assert_eq!(p2p_failed, 3),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn timeout_variant_distinct_from_reward_zero() {
        let t = VerifyError::Timeout { timeout_sec: 5 };
        let z = VerifyError::RewardZero {
            reason: ZeroReason::Unspecified { detail: "x".into() },
        };
        assert_ne!(format!("{t}"), format!("{z}"));
        assert!(matches!(t, VerifyError::Timeout { .. }));
        assert!(!matches!(t, VerifyError::RewardZero { .. }));
    }

    #[test]
    fn map_docker_timeout_typed() {
        let e = DockerError::Timeout { timeout_sec: 12 };
        let v = map_docker_timeout(&e).expect("mapped");
        assert_eq!(v, VerifyError::Timeout { timeout_sec: 12 });
    }

    #[test]
    fn owned_name_uses_prefix() {
        let n = unique_container_name("realpr-x");
        assert!(n.starts_with(OWNED_NAME_PREFIX));
    }

    #[test]
    fn reward_from_json_bytes_ok() {
        let r = reward_from_json_bytes(br#"{"reward":1}"#).expect("r");
        assert_eq!(r.value(), 1);
    }
}
