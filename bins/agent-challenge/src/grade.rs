//! Grading of returned patches into real leaf values.
//!
//! Nothing here fabricates a score. A patch is either run through the pack's
//! held-out Harbor harness — producing `Score{SCORE_MAX}` or `Score{0}` — or the
//! miner gets a `NoScore` carrying the reason it could not be graded.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use agent_challenge::{
    grade_to_score_or_absence, score_from_outcome, verify_intake_receipt, AttestationStatus,
    CallOutcome, ExpectedReceiptBind, ExpectedSet, HarborVerifier, HarborVerifierConfig,
    MinerEpochOutcome, ScoreInputs, Verifier,
};
use agent_dispatch::TaskResultV1;
use agent_pack::{load_pack, HarborPack, PACKS_DIR_NAME};
use bundle::{NoScoreReasonCode, ScoreOrAbsence};

use crate::attest::EpochAttestations;
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

/// Everything one epoch's grading pass reads besides the pack source.
pub struct GradeEpoch<'a> {
    /// Subnet netuid (bound into the scorer's I1 gate inputs).
    pub netuid: u16,
    /// Epoch being graded.
    pub epoch: u64,
    /// Challenge id the result envelopes must echo.
    pub challenge_id: &'a str,
    /// Scoring version the result envelopes must echo.
    pub scoring_version: u16,
    /// Expected set `E` — every participant gets an entry or a fallback.
    pub expected: &'a ExpectedSet,
    /// Miner hotkey → dispatch endpoint actually contacted.
    pub endpoints: &'a BTreeMap<Hotkey, String>,
    /// Raw dispatch outcomes.
    pub outcomes: &'a BTreeMap<Hotkey, MinerEpochOutcome>,
    /// Control-plane attestation snapshot for this epoch.
    pub attest: &'a EpochAttestations,
}

/// Grade every outcome that can be graded, keyed by miner hotkey.
///
/// Entries returned here take precedence over the outcome→reason fallback in
/// `score_map_covering_expected`; outcomes left out of the map fall through to
/// that fallback. Miners with no endpoint are recorded as
/// [`NoScoreReasonCode::NotAttempted`] — the challenge genuinely never called
/// them, and that is not an operator fault.
///
/// The attestation gate (I1) runs first for every participant, so an
/// unattested miner is answered before its endpoint, its envelope, or its patch
/// is looked at (`AGENT_CHALLENGE` §7.3 priority 2).
#[must_use]
pub fn grade_outcomes<S: GradeSource + ?Sized>(
    source: &S,
    ep: &GradeEpoch<'_>,
) -> BTreeMap<Hotkey, ScoreOrAbsence> {
    let mut graded = BTreeMap::new();
    for p in &ep.expected.participants {
        let attested = ep.attest.get(&p.hotkey);
        if let Some(gated) = attestation_gate(ep, p.hotkey, attested.status) {
            tracing::info!(
                event = "attestation_gate",
                hotkey = %hex::encode(p.hotkey),
                epoch = ep.epoch,
                status = ?attested.status,
                "no same-epoch Verified attestation; miner not scored"
            );
            graded.insert(p.hotkey, gated);
            continue;
        }
        let Some(receipt_pk) = attested.receipt_pk else {
            continue;
        };
        if !ep.endpoints.contains_key(&p.hotkey) {
            graded.insert(
                p.hotkey,
                ScoreOrAbsence::NoScore {
                    reason: NoScoreReasonCode::NotAttempted,
                },
            );
            continue;
        }
        let Some(MinerEpochOutcome::Completed { pack_id, result }) = ep.outcomes.get(&p.hotkey)
        else {
            continue;
        };
        let bind = ExpectedReceiptBind {
            challenge_id: ep.challenge_id.to_owned(),
            scoring_version: ep.scoring_version,
            epoch: ep.epoch,
            miner_hotkey: p.hotkey,
            pack_id: pack_id.clone(),
            cvm_receipt_pk: receipt_pk,
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

/// I1: `Some(NoScore)` when this miner must not be scored this epoch.
///
/// Routed through the pure scorer so the daemon cannot drift from the spec's
/// attestation precondition; the call outcome is a placeholder because the gate
/// short-circuits before any of it is read.
fn attestation_gate(
    ep: &GradeEpoch<'_>,
    hotkey: Hotkey,
    status: AttestationStatus,
) -> Option<ScoreOrAbsence> {
    let gated = score_from_outcome(&ScoreInputs {
        netuid: ep.netuid,
        epoch: ep.epoch,
        miner_hotkey: hotkey,
        pack_id: Vec::new(),
        expected_model_patch: Vec::new(),
        attestation: status,
        duration_ms: 0,
        outcome: CallOutcome::ChallengeInternal,
    });
    matches!(
        gated,
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::AttestationNotVerified
        }
    )
    .then_some(gated)
}

fn grade_one<S: GradeSource + ?Sized>(
    source: &S,
    bind: &ExpectedReceiptBind,
    result: &TaskResultV1,
) -> Result<ScoreOrAbsence, (NoScoreReasonCode, String)> {
    // A forged or wrong-key receipt is a bad response (§7.3 priority 5), not an
    // operator fault and not silence: the miner still gets a signed leaf.
    let patch = verify_intake_receipt(bind, result)
        .map_err(|e| (NoScoreReasonCode::InvalidResponse, e.to_string()))?
        .model_patch;
    let pack = source
        .pack(&bind.pack_id)
        .map_err(|e| (NoScoreReasonCode::ChallengeInternal, e))?;
    let verifier = source
        .verifier(&bind.pack_id)
        .map_err(|e| (NoScoreReasonCode::ChallengeInternal, e))?;
    Ok(grade_to_score_or_absence(verifier.as_ref(), &pack, &patch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_challenge::SIGNATURE_LEN;
    use agent_challenge::{
        public_key_from_secret, score_map_covering_expected, ExpectedParticipant, Reward,
        VerifyError, KEY_LEN, SCORE_MAX,
    };
    use agent_dispatch::DISPATCH_PROTOCOL;
    use agent_dispatch::{patch_sha256, sign_work_receipt, TaskStatusV1, WorkReceiptBodyV1};
    use agent_pack::HeldOutMaterials;

    use crate::attest::MinerAttestation;

    const MINER: Hotkey = [0xB1; KEY_LEN];
    const ABSENT: Hotkey = [0xC2; KEY_LEN];
    const PACK: &str = "pack-a";
    const CVM_SK: [u8; KEY_LEN] = [0x5A; KEY_LEN];
    const FORGER_SK: [u8; KEY_LEN] = [0x6B; KEY_LEN];
    const NETUID: u16 = 1;
    const EPOCH: u64 = 7;

    fn cvm_pk(sk: &[u8; KEY_LEN]) -> [u8; KEY_LEN] {
        public_key_from_secret(sk).expect("receipt pk")
    }

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

    fn resolves() -> Source {
        Source::Grades(Ok(Reward::try_new(1).expect("reward")))
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

    /// Signature the miner CVM would produce over the receipt body.
    fn receipt_sig(sk: &[u8; KEY_LEN], patch: &[u8]) -> [u8; SIGNATURE_LEN] {
        sign_work_receipt(
            sk,
            WorkReceiptBodyV1 {
                challenge_id: b"agent-v1".to_vec(),
                scoring_version: 2,
                epoch: EPOCH,
                miner_hotkey: MINER,
                pack_id: PACK.as_bytes().to_vec(),
                patch_sha256: patch_sha256(patch),
            },
        )
        .expect("sign receipt")
        .signature
    }

    fn completed(patch: &str, sk: &[u8; KEY_LEN]) -> BTreeMap<Hotkey, MinerEpochOutcome> {
        let bytes = patch.as_bytes().to_vec();
        BTreeMap::from([(
            MINER,
            MinerEpochOutcome::Completed {
                pack_id: PACK.into(),
                result: TaskResultV1 {
                    protocol: DISPATCH_PROTOCOL.into(),
                    challenge_id: "agent-v1".into(),
                    scoring_version: 2,
                    epoch: EPOCH,
                    miner_hotkey_hex: hex::encode(MINER),
                    pack_id: PACK.into(),
                    status: TaskStatusV1::Completed,
                    model_patch: Some(patch.to_owned()),
                    patch_sha256_hex: hex::encode(patch_sha256(&bytes)),
                    receipt_sig_hex: hex::encode(receipt_sig(sk, &bytes)),
                },
            },
        )])
    }

    /// Both participants attested `Verified` against the honest CVM key.
    fn all_verified() -> EpochAttestations {
        let a = MinerAttestation::verified(cvm_pk(&CVM_SK));
        EpochAttestations::new(BTreeMap::from([(MINER, a.clone()), (ABSENT, a)]))
    }

    fn score(
        source: &Source,
        outcomes: &BTreeMap<Hotkey, MinerEpochOutcome>,
        attest: &EpochAttestations,
    ) -> BTreeMap<Hotkey, ScoreOrAbsence> {
        let expected = expected();
        let endpoints = BTreeMap::from([(MINER, "http://miner.invalid".to_owned())]);
        let graded = grade_outcomes(
            source,
            &GradeEpoch {
                netuid: NETUID,
                epoch: EPOCH,
                challenge_id: "agent-v1",
                scoring_version: 2,
                expected: &expected,
                endpoints: &endpoints,
                outcomes,
                attest,
            },
        );
        let scores = score_map_covering_expected(&expected.hotkeys(), &graded, outcomes);
        assert_eq!(scores.len(), expected.participants.len(), "D24 cover of E");
        scores
    }

    fn run(source: &Source, patch: &str) -> BTreeMap<Hotkey, ScoreOrAbsence> {
        score(source, &completed(patch, &CVM_SK), &all_verified())
    }

    /// Regression: the graded map used to be unconditionally empty, so a
    /// resolving patch still scored `NoScore`.
    #[test]
    fn resolving_patch_scores_and_absent_miner_is_not_attempted() {
        let scores = run(&resolves(), "diff");
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
        let mut outcomes = completed("diff", &CVM_SK);
        let Some(MinerEpochOutcome::Completed { result, .. }) = outcomes.get_mut(&MINER) else {
            panic!("completed");
        };
        result.epoch = 8;
        let scores = score(&resolves(), &outcomes, &all_verified());
        assert_eq!(
            scores[&MINER],
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::InvalidResponse
            }
        );
    }

    #[test]
    fn tampered_patch_digest_is_invalid_response() {
        let mut outcomes = completed("diff", &CVM_SK);
        let Some(MinerEpochOutcome::Completed { result, .. }) = outcomes.get_mut(&MINER) else {
            panic!("completed");
        };
        result.model_patch = Some("other".into());
        let scores = score(&resolves(), &outcomes, &all_verified());
        assert_eq!(
            scores[&MINER],
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::InvalidResponse
            }
        );
    }

    /// A well-formed envelope whose receipt is signed by a key the attestation
    /// never pinned used to grade and score `SCORE_MAX`.
    #[test]
    fn receipt_signed_by_the_wrong_key_is_not_scored() {
        let scores = score(&resolves(), &completed("diff", &FORGER_SK), &all_verified());
        assert_eq!(
            scores[&MINER],
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::InvalidResponse
            },
            "unpinned receipt key must not reach the harness"
        );
    }

    /// I1: no same-epoch `Verified` attestation, no score — however good the
    /// patch and however honest the receipt.
    #[test]
    fn unattested_miner_is_not_scored_with_a_perfect_patch() {
        for status in [
            AttestationStatus::Missing,
            AttestationStatus::Parked,
            AttestationStatus::Rejected,
        ] {
            let attest = EpochAttestations::new(BTreeMap::from([(
                MINER,
                MinerAttestation {
                    status,
                    receipt_pk: None,
                },
            )]));
            let scores = score(&resolves(), &completed("diff", &CVM_SK), &attest);
            for hotkey in [MINER, ABSENT] {
                assert_eq!(
                    scores[&hotkey],
                    ScoreOrAbsence::NoScore {
                        reason: NoScoreReasonCode::AttestationNotVerified
                    },
                    "{status:?} must gate before grading"
                );
            }
        }
    }
}
