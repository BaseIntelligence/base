//! Todo 26: result intake, receipt verification, grading.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use agent_challenge::{
    hex32, intake_and_grade, verify_intake_receipt, ExpectedReceiptBind, IntakeOk,
    ReceiptBindError, Reward, ScoreOrAbsence, Verifier, CHALLENGE_ID, SCORE_MAX, SCORING_VERSION,
};
use agent_dispatch::{
    patch_sha256, sign_work_receipt, TaskResultV1, TaskStatusV1, WorkReceiptBodyV1,
    DISPATCH_PROTOCOL,
};
use agent_pack::{HarborPack, HeldOutMaterials};
use crypto::KEY_LEN;
use schnorrkel::MiniSecretKey;

const CVM_SK: [u8; KEY_LEN] = [0x7A; KEY_LEN];
const MINER_A: [u8; KEY_LEN] = [0xA1; KEY_LEN];
const MINER_B: [u8; KEY_LEN] = [0xB2; KEY_LEN];
const PACK: &str = "pack-intake-001";
const PATCH: &str = "diff --git a/x b/x\n+hello-intake\n";

fn cvm_pk() -> [u8; KEY_LEN] {
    MiniSecretKey::from_bytes(&CVM_SK)
        .expect("sk")
        .expand(schnorrkel::ExpansionMode::Ed25519)
        .to_public()
        .to_bytes()
}

fn exp(epoch: u64, miner: [u8; KEY_LEN]) -> ExpectedReceiptBind {
    ExpectedReceiptBind {
        challenge_id: CHALLENGE_ID.into(),
        scoring_version: SCORING_VERSION,
        epoch,
        miner_hotkey: miner,
        pack_id: PACK.into(),
        cvm_receipt_pk: cvm_pk(),
    }
}

fn signed_result(epoch: u64, miner: [u8; KEY_LEN], patch: &str) -> TaskResultV1 {
    let digest = patch_sha256(patch.as_bytes());
    let body = WorkReceiptBodyV1 {
        challenge_id: CHALLENGE_ID.as_bytes().to_vec(),
        scoring_version: SCORING_VERSION,
        epoch,
        miner_hotkey: miner,
        pack_id: PACK.as_bytes().to_vec(),
        patch_sha256: digest,
    };
    let signed = sign_work_receipt(&CVM_SK, body).expect("sign");
    TaskResultV1 {
        protocol: DISPATCH_PROTOCOL.into(),
        challenge_id: CHALLENGE_ID.into(),
        scoring_version: SCORING_VERSION,
        epoch,
        miner_hotkey_hex: hex32(&miner),
        pack_id: PACK.into(),
        status: TaskStatusV1::Completed,
        model_patch: Some(patch.into()),
        patch_sha256_hex: hex::encode(digest),
        receipt_sig_hex: hex::encode(signed.signature),
    }
}

fn dummy_pack() -> HarborPack {
    HarborPack {
        task_id: PACK.into(),
        schema_version: "1".into(),
        repository_url: "https://example.invalid/r".into(),
        base_commit_hash: "abc".into(),
        instruction: "fix".into(),
        dockerfile: b"FROM scratch\n".to_vec(),
        agent_timeout_sec: 60,
        verifier_timeout_sec: Some(30),
        held_out: HeldOutMaterials {
            solution_patch: None,
            test_patch: None,
            grader_py: None,
        },
        files: vec![],
    }
}

struct CountingVerifier {
    calls: Arc<AtomicU32>,
    reward: u8,
}

impl Verifier for CountingVerifier {
    fn grade(
        &self,
        _pack: &HarborPack,
        _model_patch: &[u8],
    ) -> Result<Reward, agent_challenge::VerifyError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Reward::try_new(self.reward)
    }
}

struct FailVerifier {
    calls: Arc<AtomicU32>,
}

impl Verifier for FailVerifier {
    fn grade(
        &self,
        _pack: &HarborPack,
        _model_patch: &[u8],
    ) -> Result<Reward, agent_challenge::VerifyError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(agent_challenge::VerifyError::ApplyFailed {
            message: "boom".into(),
        })
    }
}

/// S1 — valid result → receipt-OK + leaf SCORE_MAX; verifier invoked once.
#[test]
fn s1_valid_intake_grades_leaf() {
    let epoch = 42u64;
    let result = signed_result(epoch, MINER_A, PATCH);
    let e = exp(epoch, MINER_A);
    let calls = Arc::new(AtomicU32::new(0));
    let v = CountingVerifier {
        calls: Arc::clone(&calls),
        reward: 1,
    };
    let pack = dummy_pack();

    let bound = verify_intake_receipt(&e, &result).expect("receipt-OK");
    assert_eq!(bound.patch_sha256, patch_sha256(PATCH.as_bytes()));
    assert_eq!(calls.load(Ordering::SeqCst), 0, "bind must not grade");

    let ok = intake_and_grade(&e, &result, &v, &pack).expect("grade");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(ok.leaf, ScoreOrAbsence::Score { value: SCORE_MAX });
    assert_eq!(ok.model_patch, PATCH.as_bytes());

    let line = format!(
        "receipt-OK patch_sha256={} leaf={:?}\n",
        hex::encode(ok.patch_sha256),
        ok.leaf
    );
    println!("{line}");
    let path = PathBuf::from(
        "/root/.omo/evidence/gbase-agent-challenge-deepagent/task-26-intake-grade-leaf.txt",
    );
    std::fs::write(&path, &line).expect("evidence");
    assert!(path.is_file());
}

/// S2 — patch bytes ≠ receipt hash → PatchHashMismatch, zero verifier calls.
/// S3 — wrong epoch → EpochMismatch, zero verifier calls.
#[test]
fn s2_s3_receipt_mismatch_rejected_pre_grade() {
    let epoch = 99u64;
    let calls = Arc::new(AtomicU32::new(0));
    let v = CountingVerifier {
        calls: Arc::clone(&calls),
        reward: 1,
    };
    let pack = dummy_pack();
    let e = exp(epoch, MINER_A);

    // S2: patch differs from claimed hash (and from signed body digest).
    let mut bad_patch = signed_result(epoch, MINER_A, PATCH);
    bad_patch.model_patch = Some("diff --git a/x b/x\n+TAMPERED\n".into());
    let err = intake_and_grade(&e, &bad_patch, &v, &pack).expect_err("patch mismatch");
    assert_eq!(err, ReceiptBindError::PatchHashMismatch);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "no grade on patch mismatch"
    );

    // S3: receipt/result epoch wrong vs expected.
    let wrong_epoch_result = signed_result(epoch + 1, MINER_A, PATCH);
    let err2 = intake_and_grade(&e, &wrong_epoch_result, &v, &pack).expect_err("epoch");
    assert!(
        matches!(
            err2,
            ReceiptBindError::EpochMismatch {
                expected: 99,
                got: 100
            }
        ),
        "got {err2:?}"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "no grade on epoch mismatch"
    );

    let report = format!(
        "patch_mismatch={err:?}\nepoch_mismatch={err2:?}\nverifier_calls={}\n",
        calls.load(Ordering::SeqCst)
    );
    println!("{report}");
    let path = PathBuf::from(
        "/root/.omo/evidence/gbase-agent-challenge-deepagent/task-26-receipt-mismatch-rejected.txt",
    );
    std::fs::write(&path, &report).expect("evidence");
    assert!(path.is_file());
}

/// S4 — miner A grade failure does not change miner B outcome.
#[test]
fn s4_miner_isolation() {
    let epoch = 7u64;
    let pack = dummy_pack();
    let calls_a = Arc::new(AtomicU32::new(0));
    let calls_b = Arc::new(AtomicU32::new(0));
    let fail_a = FailVerifier {
        calls: Arc::clone(&calls_a),
    };
    let ok_b = CountingVerifier {
        calls: Arc::clone(&calls_b),
        reward: 1,
    };
    let a: IntakeOk = intake_and_grade(
        &exp(epoch, MINER_A),
        &signed_result(epoch, MINER_A, PATCH),
        &fail_a,
        &pack,
    )
    .expect("a");
    let b: IntakeOk = intake_and_grade(
        &exp(epoch, MINER_B),
        &signed_result(epoch, MINER_B, PATCH),
        &ok_b,
        &pack,
    )
    .expect("b");
    assert_eq!(a.leaf, ScoreOrAbsence::Score { value: 0 });
    assert_eq!(b.leaf, ScoreOrAbsence::Score { value: SCORE_MAX });
    assert_eq!(calls_a.load(Ordering::SeqCst), 1);
    assert_eq!(calls_b.load(Ordering::SeqCst), 1);
    assert_ne!(a.leaf, b.leaf);
}
