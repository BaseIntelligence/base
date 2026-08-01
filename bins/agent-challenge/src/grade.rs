//! Grading of returned patches into real leaf values.
//!
//! Nothing here fabricates a score. A patch is either run through the pack's
//! held-out Harbor harness — producing `Score{SCORE_MAX}` or `Score{0}` — or the
//! miner gets a `NoScore` carrying the reason it could not be graded.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use agent_challenge::{
    grade_to_score_or_absence, ExpectedSet, HarborVerifier, HarborVerifierConfig,
    MinerEpochOutcome, Verifier,
};
use agent_dispatch::{patch_sha256, TaskResultV1, DISPATCH_PROTOCOL};
use agent_pack::{load_pack, HarborPack, PACKS_DIR_NAME};
use bundle::{NoScoreReasonCode, ScoreOrAbsence};

use crate::chainsnap::Hotkey;

/// Where a pack and its verifier come from for one grade.
pub trait GradeSource {
    /// Full pack (including held-out tests) for `pack_id`.
    ///
    /// # Errors
    /// Unknown pack id or unreadable pack directory.
    fn pack(&self, pack_id: &str) -> Result<HarborPack, String>;

    /// Verifier bound to `pack_id`'s digest-pinned environment image.
    ///
    /// # Errors
    /// No image pinned for the pack, or client construction failure.
    fn verifier(&self, pack_id: &str) -> Result<Box<dyn Verifier>, String>;
}

/// Production grade source: packs from the materialized cache, verifiers from
/// the operator's pinned image map.
#[derive(Debug, Clone)]
pub struct HarborGradeSource {
    /// Docker Engine HTTP base (socket-proxy).
    pub docker_base: String,
    /// Materialized pack cache root (`<cache>/packs/<pack_id>`).
    pub cache_dir: PathBuf,
    /// Staging root for verifier binds.
    pub work_root: PathBuf,
    /// `pack_id` → digest-pinned verifier image.
    pub images: BTreeMap<String, String>,
    /// Image used when a pack has no explicit entry.
    pub default_image: Option<String>,
}

impl HarborGradeSource {
    /// Read a `{"pack_id": "image@sha256:…"}` map from disk.
    ///
    /// # Errors
    /// Unreadable file or non-object / non-string JSON.
    pub fn load_image_map(path: &Path) -> Result<BTreeMap<String, String>, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("read verifier image map {}: {e}", path.display()))?;
        serde_json::from_str(&text)
            .map_err(|e| format!("parse verifier image map {}: {e}", path.display()))
    }
}

impl GradeSource for HarborGradeSource {
    fn pack(&self, pack_id: &str) -> Result<HarborPack, String> {
        let dir = self.cache_dir.join(PACKS_DIR_NAME).join(pack_id);
        load_pack(&dir).map_err(|e| format!("load pack {}: {e}", dir.display()))
    }

    fn verifier(&self, pack_id: &str) -> Result<Box<dyn Verifier>, String> {
        let image = self
            .images
            .get(pack_id)
            .or(self.default_image.as_ref())
            .ok_or_else(|| format!("no verifier image pinned for pack {pack_id}"))?;
        let v = HarborVerifier::new(HarborVerifierConfig {
            docker_base: self.docker_base.clone(),
            environment_image: image.clone(),
            work_root: self.work_root.clone(),
            timeout_sec_override: None,
            reward_zero_as_err: false,
        })
        .map_err(|e| format!("verifier for pack {pack_id}: {e}"))?;
        Ok(Box::new(v))
    }
}

/// Grade every outcome that can be graded, keyed by miner hotkey.
///
/// Entries returned here take precedence over the outcome→reason fallback in
/// `score_map_covering_expected`; outcomes left out of the map fall through to
/// that fallback. Miners with no endpoint are recorded as
/// [`NoScoreReasonCode::NotAttempted`] — the challenge genuinely never called
/// them, and that is not an operator fault.
#[must_use]
pub fn grade_outcomes<S: GradeSource + ?Sized>(
    source: &S,
    epoch: u64,
    challenge_id: &str,
    scoring_version: u16,
    expected: &ExpectedSet,
    endpoints: &BTreeMap<Hotkey, String>,
    outcomes: &BTreeMap<Hotkey, MinerEpochOutcome>,
) -> BTreeMap<Hotkey, ScoreOrAbsence> {
    let mut graded = BTreeMap::new();
    for p in &expected.participants {
        if !endpoints.contains_key(&p.hotkey) {
            graded.insert(
                p.hotkey,
                ScoreOrAbsence::NoScore {
                    reason: NoScoreReasonCode::NotAttempted,
                },
            );
            continue;
        }
        let Some(MinerEpochOutcome::Completed { pack_id, result }) = outcomes.get(&p.hotkey) else {
            continue;
        };
        let bind = Bind {
            challenge_id,
            scoring_version,
            epoch,
            hotkey: p.hotkey,
            pack_id,
        };
        graded.insert(
            p.hotkey,
            grade_one(source, &bind, result).unwrap_or_else(|(reason, detail)| {
                tracing::warn!(
                    event = "grade_unavailable",
                    hotkey = %hex::encode(p.hotkey),
                    pack_id,
                    detail,
                    "miner covered with NoScore"
                );
                ScoreOrAbsence::NoScore { reason }
            }),
        );
    }
    graded
}

/// Fields a result envelope must echo back before its patch is graded.
struct Bind<'a> {
    challenge_id: &'a str,
    scoring_version: u16,
    epoch: u64,
    hotkey: Hotkey,
    pack_id: &'a str,
}

fn grade_one<S: GradeSource + ?Sized>(
    source: &S,
    bind: &Bind<'_>,
    result: &TaskResultV1,
) -> Result<ScoreOrAbsence, (NoScoreReasonCode, String)> {
    let patch =
        bound_patch(bind, result).map_err(|why| (NoScoreReasonCode::InvalidResponse, why))?;
    let pack = source
        .pack(bind.pack_id)
        .map_err(|e| (NoScoreReasonCode::ChallengeInternal, e))?;
    let verifier = source
        .verifier(bind.pack_id)
        .map_err(|e| (NoScoreReasonCode::ChallengeInternal, e))?;
    Ok(grade_to_score_or_absence(verifier.as_ref(), &pack, &patch))
}

/// Check the result echoes the dispatch it answers, and that the patch matches
/// its own digest.
///
/// The work-receipt **signature** is not checked here: the receipt public key is
/// pinned by attestation and the daemon has no attestation lookup yet, so
/// binding stops at the envelope fields. Forging them buys nothing — the patch
/// is still graded by the held-out harness.
fn bound_patch(bind: &Bind<'_>, result: &TaskResultV1) -> Result<Vec<u8>, String> {
    if result.protocol != DISPATCH_PROTOCOL {
        return Err(format!("protocol {}", result.protocol));
    }
    if result.challenge_id != bind.challenge_id {
        return Err(format!("challenge_id {}", result.challenge_id));
    }
    if result.scoring_version != bind.scoring_version {
        return Err(format!("scoring_version {}", result.scoring_version));
    }
    if result.epoch != bind.epoch {
        return Err(format!("epoch {} != {}", result.epoch, bind.epoch));
    }
    if result.miner_hotkey_hex != hex::encode(bind.hotkey) {
        return Err("miner_hotkey_hex mismatch".to_owned());
    }
    if result.pack_id != bind.pack_id {
        return Err(format!("pack_id {}", result.pack_id));
    }
    let patch = result
        .model_patch
        .as_deref()
        .unwrap_or_default()
        .as_bytes()
        .to_vec();
    if hex::encode(patch_sha256(&patch)) != result.patch_sha256_hex {
        return Err("patch_sha256 mismatch".to_owned());
    }
    Ok(patch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_challenge::{
        score_map_covering_expected, ExpectedParticipant, Reward, VerifyError, SCORE_MAX,
    };
    use agent_dispatch::{TaskResultV1, TaskStatusV1};
    use agent_pack::HeldOutMaterials;

    const MINER: Hotkey = [0xB1; 32];
    const ABSENT: Hotkey = [0xC2; 32];
    const PACK: &str = "pack-a";

    fn pack() -> HarborPack {
        HarborPack {
            task_id: PACK.into(),
            schema_version: "1.1".into(),
            repository_url: "https://example.invalid/r".into(),
            base_commit_hash: "0".repeat(40),
            instruction: "fix it".into(),
            dockerfile: b"FROM scratch\n".to_vec(),
            agent_timeout_sec: 60,
            verifier_timeout_sec: Some(60),
            held_out: HeldOutMaterials {
                solution_patch: None,
                test_patch: None,
                grader_py: None,
            },
            files: vec![("tests/test.sh".into(), b"true\n".to_vec())],
        }
    }

    /// Verifier stub: `Ok(reward)` or a fixed error, no Docker.
    struct StubVerifier(Result<Reward, VerifyError>);

    impl Verifier for StubVerifier {
        fn grade(&self, _pack: &HarborPack, _patch: &[u8]) -> Result<Reward, VerifyError> {
            self.0.clone()
        }
    }

    enum Source {
        Grades(Result<Reward, VerifyError>),
        NoImage,
    }

    impl GradeSource for Source {
        fn pack(&self, _pack_id: &str) -> Result<HarborPack, String> {
            Ok(pack())
        }

        fn verifier(&self, pack_id: &str) -> Result<Box<dyn Verifier>, String> {
            match self {
                Self::Grades(r) => Ok(Box::new(StubVerifier(r.clone()))),
                Self::NoImage => Err(format!("no verifier image pinned for pack {pack_id}")),
            }
        }
    }

    fn expected() -> ExpectedSet {
        ExpectedSet {
            block_hash: [0x11; 32],
            participants: vec![
                ExpectedParticipant {
                    hotkey: MINER,
                    uid: 1,
                },
                ExpectedParticipant {
                    hotkey: ABSENT,
                    uid: 2,
                },
            ],
        }
    }

    fn completed(patch: &str) -> BTreeMap<Hotkey, MinerEpochOutcome> {
        let bytes = patch.as_bytes().to_vec();
        BTreeMap::from([(
            MINER,
            MinerEpochOutcome::Completed {
                pack_id: PACK.into(),
                result: TaskResultV1 {
                    protocol: DISPATCH_PROTOCOL.into(),
                    challenge_id: "agent-v1".into(),
                    scoring_version: 2,
                    epoch: 7,
                    miner_hotkey_hex: hex::encode(MINER),
                    pack_id: PACK.into(),
                    status: TaskStatusV1::Completed,
                    model_patch: Some(patch.to_owned()),
                    patch_sha256_hex: hex::encode(patch_sha256(&bytes)),
                    receipt_sig_hex: hex::encode([0_u8; 64]),
                },
            },
        )])
    }

    fn run(source: &Source, patch: &str) -> BTreeMap<Hotkey, ScoreOrAbsence> {
        let expected = expected();
        let endpoints = BTreeMap::from([(MINER, "http://miner.invalid".to_owned())]);
        let outcomes = completed(patch);
        let graded = grade_outcomes(source, 7, "agent-v1", 2, &expected, &endpoints, &outcomes);
        score_map_covering_expected(&expected.hotkeys(), &graded, &outcomes)
    }

    /// Regression: the graded map used to be unconditionally empty, so a
    /// resolving patch still scored `NoScore`.
    #[test]
    fn resolving_patch_scores_and_absent_miner_is_not_attempted() {
        let scores = run(&Source::Grades(Ok(Reward::try_new(1).expect("r"))), "diff");
        assert_eq!(scores[&MINER], ScoreOrAbsence::Score { value: SCORE_MAX });
        assert_eq!(
            scores[&ABSENT],
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::NotAttempted
            }
        );
    }

    #[test]
    fn failing_patch_scores_zero_not_no_score() {
        let scores = run(&Source::Grades(Ok(Reward::try_new(0).expect("r"))), "diff");
        assert_eq!(scores[&MINER], ScoreOrAbsence::Score { value: 0 });
    }

    /// An operator fault must never be charged to the miner as a zero.
    #[test]
    fn operator_fault_is_challenge_internal_no_score() {
        let scores = run(
            &Source::Grades(Err(VerifyError::Timeout { timeout_sec: 5 })),
            "diff",
        );
        assert_eq!(
            scores[&MINER],
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::ChallengeInternal
            }
        );
    }

    #[test]
    fn missing_verifier_image_is_challenge_internal_no_score() {
        let scores = run(&Source::NoImage, "diff");
        assert_eq!(
            scores[&MINER],
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::ChallengeInternal
            }
        );
    }

    #[test]
    fn envelope_that_does_not_echo_the_dispatch_is_invalid_response() {
        let bind = Bind {
            challenge_id: "agent-v1",
            scoring_version: 2,
            epoch: 7,
            hotkey: MINER,
            pack_id: PACK,
        };
        let mut outcomes = completed("diff");
        let Some(MinerEpochOutcome::Completed { result, .. }) = outcomes.get_mut(&MINER) else {
            panic!("completed");
        };
        result.epoch = 8;
        let err = bound_patch(&bind, result).expect_err("epoch mismatch");
        assert!(err.contains("epoch 8"), "{err}");

        let source = Source::Grades(Ok(Reward::try_new(1).expect("r")));
        let graded = grade_outcomes(
            &source,
            7,
            "agent-v1",
            2,
            &expected(),
            &BTreeMap::from([(MINER, "http://miner.invalid".to_owned())]),
            &outcomes,
        );
        assert_eq!(
            graded[&MINER],
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::InvalidResponse
            }
        );
    }

    #[test]
    fn tampered_patch_digest_is_invalid_response() {
        let bind = Bind {
            challenge_id: "agent-v1",
            scoring_version: 2,
            epoch: 7,
            hotkey: MINER,
            pack_id: PACK,
        };
        let mut outcomes = completed("diff");
        let Some(MinerEpochOutcome::Completed { result, .. }) = outcomes.get_mut(&MINER) else {
            panic!("completed");
        };
        result.model_patch = Some("other".into());
        let err = bound_patch(&bind, result).expect_err("digest");
        assert!(err.contains("patch_sha256"), "{err}");
    }
}
