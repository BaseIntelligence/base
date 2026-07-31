//! Orchestrator ↔ runner task-dispatch wire surface for agent-v1.
//!
//! # Scope (this crate)
//! JSON task descriptor / result envelopes and the SCALE-encoded, domain-separated
//! **work receipt** signed inside the miner CVM. Transport and Harbor execution
//! land in later tasks.
//!
//! # Receipt verification (D19)
//! The **challenge service** verifies [`receipt::SignedWorkReceiptV1`] against the
//! receipt pubkey published in the measured compose. Validators never need the
//! receipt key or domain — they do not re-score agents.
//!
//! # What stays in `agent-challenge`
//! Scoring, `NoScore` / D24 completeness, leaf signing (`gbase-rawweight-v1`), and
//! weight submit remain in `agent-challenge`. The challenge signing key never
//! enters the miner CVM (D18).

#![forbid(unsafe_code)]

mod receipt;
mod wire;

pub use receipt::{
    patch_sha256, receipt_payload, sign_work_receipt, verify_task_result_bind, verify_work_receipt,
    work_receipt_domain, BoundPatch, ExpectedReceiptBind, ReceiptBindError, ReceiptError,
    SignedWorkReceiptV1, WorkReceiptBodyV1,
};
pub use wire::{TaskDescriptorV1, TaskResultV1, TaskStatusV1, DISPATCH_PROTOCOL};

use thiserror::Error;

use std::fmt;

/// Opaque task identifier assigned by the orchestrator.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskId(String);

impl TaskId {
    /// Construct from an already-validated task id string.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the raw id as `str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One unit of work handed from orchestrator to a runner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchRequest {
    /// Task identity.
    pub task_id: TaskId,
    /// Full JSON descriptor for the runner.
    pub descriptor: TaskDescriptorV1,
}

/// Runner outcome reported back to the orchestrator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchResult {
    /// Task identity echoed from the request.
    pub task_id: TaskId,
    /// JSON result envelope (patch + receipt sig).
    pub result: TaskResultV1,
}

/// Failures while enqueueing or collecting dispatch work.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DispatchError {
    /// No runner accepted the task.
    #[error("no runner available for task {0}")]
    NoRunner(String),
    /// Dispatch transport or serialization failure.
    #[error("dispatch failed: {0}")]
    Failed(String),
}

/// Orchestrator-facing handle that sends work to runners.
pub trait TaskDispatcher: Send + Sync {
    /// Enqueue one task for a runner.
    ///
    /// # Errors
    /// Returns [`DispatchError`] when no runner is available or send fails.
    fn dispatch(&self, request: DispatchRequest) -> Result<(), DispatchError>;
}

/// Crate identity for smoke / wiring checks.
#[must_use]
pub fn crate_name() -> &'static str {
    "agent-dispatch"
}

#[cfg(test)]
mod tests {
    use super::{
        crate_name, patch_sha256, receipt_payload, sign_work_receipt, verify_work_receipt,
        work_receipt_domain, DispatchError, DispatchRequest, ReceiptError, TaskDescriptorV1,
        TaskDispatcher, TaskId, TaskResultV1, TaskStatusV1, WorkReceiptBodyV1, DISPATCH_PROTOCOL,
    };
    use crypto::{domain, sign_raw, verify_raw, CryptoError, KEY_LEN, SIGNATURE_LEN};
    use parity_scale_codec::Encode;
    use schnorrkel::MiniSecretKey;

    /// Fixed mini-secret for golden vectors (not a production key).
    const GOLDEN_SK: [u8; KEY_LEN] = [0x42; KEY_LEN];

    fn golden_body() -> WorkReceiptBodyV1 {
        WorkReceiptBodyV1 {
            challenge_id: b"agent-v1".to_vec(),
            scoring_version: 1,
            epoch: 7,
            miner_hotkey: [0x11; KEY_LEN],
            pack_id: b"pack-fixture-001".to_vec(),
            patch_sha256: patch_sha256(b"diff --git a/x b/x\n+hello\n"),
        }
    }

    fn golden_public() -> [u8; KEY_LEN] {
        let mini = MiniSecretKey::from_bytes(&GOLDEN_SK).expect("golden sk");
        mini.expand(schnorrkel::ExpansionMode::Ed25519)
            .to_public()
            .to_bytes()
    }

    /// Pinned SCALE body encoding (fixture file must match).
    const GOLDEN_BODY_HEX: &str = include_str!("../tests/fixtures/work_receipt_body_v1.hex");

    struct RejectAll;

    impl TaskDispatcher for RejectAll {
        fn dispatch(&self, request: DispatchRequest) -> Result<(), DispatchError> {
            Err(DispatchError::NoRunner(request.task_id.to_string()))
        }
    }

    #[test]
    fn crate_name_is_agent_dispatch() {
        assert_eq!(crate_name(), "agent-dispatch");
    }

    #[test]
    fn reject_all_reports_no_runner() {
        let d = RejectAll;
        let req = DispatchRequest {
            task_id: TaskId::new("t-1"),
            descriptor: TaskDescriptorV1::new("agent-v1", 1, 7, "11".repeat(32), "p-1", 0),
        };
        let err = d.dispatch(req).expect_err("reject all");
        assert_eq!(err, DispatchError::NoRunner("t-1".into()));
    }

    /// S1 — receipt body SCALE matches committed golden vector.
    #[test]
    fn s1_receipt_body_matches_golden_vector() {
        let body = golden_body();
        let payload = receipt_payload(&body);
        let got = hex::encode(&payload);
        let want = GOLDEN_BODY_HEX.trim();
        assert_eq!(
            got, want,
            "SCALE body drift — update fixture only with intent"
        );
        // Encode path equals derive Encode.
        assert_eq!(payload, body.encode());
    }

    /// S1b — sign under WORK_RECEIPT and verify.
    #[test]
    fn s1_sign_verify_work_receipt_round_trip() {
        let body = golden_body();
        let signed = sign_work_receipt(&GOLDEN_SK, body.clone()).expect("sign");
        verify_work_receipt(&golden_public(), &signed).expect("verify");
        assert_eq!(signed.body, body);
        assert_eq!(signed.signature.len(), SIGNATURE_LEN);
        assert_eq!(
            work_receipt_domain().as_bytes(),
            b"gbase-agent-work-receipt-v1"
        );
    }

    /// S2 — receipt signed under gbase-attest-v1 fails under WORK_RECEIPT.
    #[test]
    fn s2_cross_domain_attest_signature_rejected() {
        let body = golden_body();
        let payload = receipt_payload(&body);
        let sig = sign_raw(&GOLDEN_SK, domain::ATTEST, &payload).expect("sign attest");
        let err = verify_raw(&golden_public(), work_receipt_domain(), &payload, &sig)
            .expect_err("cross-domain must fail");
        assert_eq!(err, CryptoError::VerificationFailed);

        // Also via receipt helper with a forged SignedWorkReceiptV1.
        let forged = super::SignedWorkReceiptV1 {
            body,
            signature: sig,
        };
        let rerr = verify_work_receipt(&golden_public(), &forged).expect_err("cross-domain");
        assert!(matches!(rerr, ReceiptError::Crypto(_)));
    }

    /// S3 — tampered pack_id after sign fails verify.
    #[test]
    fn s3_tampered_pack_id_rejected() {
        let body = golden_body();
        let mut signed = sign_work_receipt(&GOLDEN_SK, body).expect("sign");
        signed.body.pack_id = b"pack-TAMPERED".to_vec();
        let err = verify_work_receipt(&golden_public(), &signed).expect_err("tamper");
        assert!(matches!(err, ReceiptError::Crypto(_)));
    }

    /// S5 — JSON descriptor / result round-trip.
    #[test]
    fn s5_json_wire_round_trip() {
        let desc = TaskDescriptorV1::new(
            "agent-v1",
            1,
            42,
            "aa".repeat(32),
            "pack-fixture-001",
            1_700_000_000_000,
        );
        assert_eq!(desc.protocol, DISPATCH_PROTOCOL);
        let desc_json = serde_json::to_string(&desc).expect("ser desc");
        let desc2: TaskDescriptorV1 = serde_json::from_str(&desc_json).expect("de desc");
        assert_eq!(desc, desc2);

        let result = TaskResultV1 {
            protocol: DISPATCH_PROTOCOL.into(),
            challenge_id: "agent-v1".into(),
            scoring_version: 1,
            epoch: 42,
            miner_hotkey_hex: "aa".repeat(32),
            pack_id: "pack-fixture-001".into(),
            status: TaskStatusV1::Completed,
            model_patch: Some("diff --git a/x b/x\n+hello\n".into()),
            patch_sha256_hex: hex::encode(patch_sha256(b"diff --git a/x b/x\n+hello\n")),
            receipt_sig_hex: "00".repeat(64),
        };
        let res_json = serde_json::to_string(&result).expect("ser res");
        let res2: TaskResultV1 = serde_json::from_str(&res_json).expect("de res");
        assert_eq!(result, res2);
    }

    /// Domain tag is not ATTEST (D10 separation).
    #[test]
    fn work_receipt_domain_distinct_from_attest() {
        assert_ne!(work_receipt_domain().as_bytes(), domain::ATTEST.as_bytes());
    }
}
