//! Pure task / answer digest generation (`AGENT_CHALLENGE` §5.2–§5.3).
//!
//! # v1 field order (normative, `scoring_version = 1`)
//!
//! ```text
//! task_id = sha256(
//!   b"gbase-agent-task-id-v1" ‖
//!   scale(netuid: u16) ‖ scale(epoch: u64) ‖ miner_hotkey: [u8; 32]
//! )
//! task_blob = sha256(
//!   b"gbase-agent-task-blob-v1" ‖ task_id ‖ scale(scoring_version: u16)
//! )
//! answer_digest = sha256(b"gbase-agent-answer-v1" ‖ task_blob)
//! ```
//!
//! # v2 field order (normative, pack-bound, `scoring_version = 2`)
//!
//! Domain tags are distinct from v1 and from `WORK_RECEIPT` / `ATTEST`.
//!
//! ```text
//! task_id_v2 = sha256(
//!   b"gbase-agent-task-id-v2" ‖
//!   scale(netuid: u16) ‖
//!   scale(epoch: u64) ‖
//!   miner_hotkey: [u8; 32] ‖
//!   scale(pack_id: Vec<u8>) ‖   // UTF-8 pack id bytes
//!   scale(scoring_version: u16) // typically 2
//! )
//! task_blob_v2 = sha256(
//!   b"gbase-agent-task-blob-v2" ‖
//!   task_id_v2 ‖
//!   scale(scoring_version: u16) ‖
//!   scale(pack_id: Vec<u8>)
//! )
//! answer_digest_v2 = sha256(
//!   b"gbase-agent-answer-v2" ‖
//!   model_patch                 // raw returned model.patch bytes
//! )
//! ```
//!
//! All functions are pure: no I/O, no wall clock, no RNG.

use crypto::KEY_LEN;
use parity_scale_codec::Encode;
use sha2::{Digest, Sha256};

/// Normative challenge id.
pub const CHALLENGE_ID: &str = "agent-v1";
/// UTF-8 bytes of [`CHALLENGE_ID`].
pub const CHALLENGE_ID_BYTES: &[u8] = b"agent-v1";
/// Live `challenge_scoring_version` for agent-v1 (pack-bound v2 formulas).
pub const SCORING_VERSION: u16 = 2;
/// Alias kept for call sites that name the v2 constant explicitly.
pub const SCORING_VERSION_V2: u16 = SCORING_VERSION;

/// Placeholder pack id for unit fixtures until the orchestrator selects packs.
pub const FIXTURE_PACK_ID: &[u8] = b"pack-fixture-001";
/// Placeholder model.patch bytes for unit fixtures (matches v2 golden vectors).
pub const FIXTURE_MODEL_PATCH: &[u8] = b"diff --git a/x b/x\n+hello\n";

const TASK_ID_DOMAIN: &[u8] = b"gbase-agent-task-id-v1";
const TASK_BLOB_DOMAIN: &[u8] = b"gbase-agent-task-blob-v1";
const ANSWER_DOMAIN: &[u8] = b"gbase-agent-answer-v1";

/// Domain tag for [`task_id_v2`] (distinct from v1 / WORK_RECEIPT / ATTEST).
pub const TASK_ID_DOMAIN_V2: &[u8] = b"gbase-agent-task-id-v2";
/// Domain tag for [`task_blob_v2`].
pub const TASK_BLOB_DOMAIN_V2: &[u8] = b"gbase-agent-task-blob-v2";
/// Domain tag for [`answer_digest_v2`] over `model.patch`.
pub const ANSWER_DOMAIN_V2: &[u8] = b"gbase-agent-answer-v2";

/// `task_id = sha256(b"gbase-agent-task-id-v1" ‖ scale(netuid) ‖ scale(epoch) ‖ miner_hotkey)`.
#[must_use]
pub fn task_id(netuid: u16, epoch: u64, miner_hotkey: &[u8; KEY_LEN]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(TASK_ID_DOMAIN);
    h.update(netuid.encode());
    h.update(epoch.encode());
    h.update(miner_hotkey);
    finalize32(h)
}

/// `task_blob = sha256(b"gbase-agent-task-blob-v1" ‖ task_id ‖ scale(scoring_version))`.
#[must_use]
pub fn task_blob(task_id: &[u8; 32], scoring_version: u16) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(TASK_BLOB_DOMAIN);
    h.update(task_id);
    h.update(scoring_version.encode());
    finalize32(h)
}

/// `answer_digest = sha256(b"gbase-agent-answer-v1" ‖ task_blob)`.
#[must_use]
pub fn answer_digest(task_blob: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(ANSWER_DOMAIN);
    h.update(task_blob);
    finalize32(h)
}

/// v2 task id: binds netuid, epoch, hotkey, pack_id, and scoring_version.
///
/// `pack_id` is the UTF-8 pack identity bytes (same surface as receipt `pack_id`).
#[must_use]
pub fn task_id_v2(
    netuid: u16,
    epoch: u64,
    miner_hotkey: &[u8; KEY_LEN],
    pack_id: &[u8],
    scoring_version: u16,
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(TASK_ID_DOMAIN_V2);
    h.update(netuid.encode());
    h.update(epoch.encode());
    h.update(miner_hotkey);
    h.update(pack_id.to_vec().encode());
    h.update(scoring_version.encode());
    finalize32(h)
}

/// v2 task blob: binds `task_id_v2`, scoring_version, and pack_id.
#[must_use]
pub fn task_blob_v2(task_id: &[u8; 32], scoring_version: u16, pack_id: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(TASK_BLOB_DOMAIN_V2);
    h.update(task_id);
    h.update(scoring_version.encode());
    h.update(pack_id.to_vec().encode());
    finalize32(h)
}

/// v2 answer digest: domain-tagged SHA-256 over returned `model.patch` bytes.
///
/// Distinct from receipt `patch_sha256` (untagged) and from v1 answer (over task_blob).
#[must_use]
pub fn answer_digest_v2(model_patch: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(ANSWER_DOMAIN_V2);
    h.update(model_patch);
    finalize32(h)
}

fn finalize32(h: Sha256) -> [u8; 32] {
    let d = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&d);
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const FIX_NETUID: u16 = 1;
    const FIX_EPOCH: u64 = 7;
    const FIX_PACK: &[u8] = b"pack-fixture-001";
    const FIX_PATCH: &[u8] = b"diff --git a/x b/x\n+hello\n";

    fn miner11() -> [u8; 32] {
        [0x11u8; 32]
    }

    fn miner22() -> [u8; 32] {
        [0x22u8; 32]
    }

    #[test]
    fn f1_pinned_digests() {
        let miner = miner11();
        let tid = task_id(1, 7, &miner);
        assert_eq!(
            hex::encode(tid),
            "4a590b2abf87da6bccd97d8fbe5d2e774bdbda3ad421119688010537be2b31ec"
        );
        // Historical v1 goldens lock scoring_version = 1 (not live SCORING_VERSION).
        let blob = task_blob(&tid, 1);
        assert_eq!(
            hex::encode(blob),
            "8c5430ceb95b9e422026baf2eaddb4c9c723923c6353164fe9b0905a47f9a29f"
        );
        let ans = answer_digest(&blob);
        assert_eq!(
            hex::encode(ans),
            "83180b08e05630496531a158d174ce69ba857d854d8692087947706c159a487c"
        );
    }

    #[test]
    fn f11_pinned_digests() {
        let miner = miner22();
        let tid = task_id(1, 7, &miner);
        assert_eq!(
            hex::encode(tid),
            "d954306fba3943a86bb69aedfd08f2bca850eb2adabaaf5efe2ad2728dbf3412"
        );
        let blob = task_blob(&tid, 1);
        let ans = answer_digest(&blob);
        assert_eq!(
            hex::encode(ans),
            "05157d001bb1ec9ef5acc7140d0221141d2fbc14a830ce32893793f30470c0aa"
        );
    }

    /// Dual independent computation must agree byte-for-byte (S1).
    #[test]
    fn v2_dual_compute_agrees() {
        let m = miner11();
        let a_tid = task_id_v2(FIX_NETUID, FIX_EPOCH, &m, FIX_PACK, SCORING_VERSION_V2);
        let b_tid = task_id_v2(FIX_NETUID, FIX_EPOCH, &m, FIX_PACK, SCORING_VERSION_V2);
        assert_eq!(a_tid, b_tid);
        let a_blob = task_blob_v2(&a_tid, SCORING_VERSION_V2, FIX_PACK);
        let b_blob = task_blob_v2(&b_tid, SCORING_VERSION_V2, FIX_PACK);
        assert_eq!(a_blob, b_blob);
        let a_ans = answer_digest_v2(FIX_PATCH);
        let b_ans = answer_digest_v2(FIX_PATCH);
        assert_eq!(a_ans, b_ans);
    }

    #[test]
    fn v2_domain_tags_distinct_from_v1_and_crypto_domains() {
        assert_ne!(TASK_ID_DOMAIN_V2, TASK_ID_DOMAIN);
        assert_ne!(TASK_BLOB_DOMAIN_V2, TASK_BLOB_DOMAIN);
        assert_ne!(ANSWER_DOMAIN_V2, ANSWER_DOMAIN);
        assert_ne!(TASK_ID_DOMAIN_V2, b"gbase-agent-work-receipt-v1");
        assert_ne!(TASK_ID_DOMAIN_V2, b"gbase-attest-v1");
        assert_ne!(ANSWER_DOMAIN_V2, b"gbase-agent-work-receipt-v1");
        assert_ne!(ANSWER_DOMAIN_V2, b"gbase-attest-v1");
    }

    #[test]
    fn v2_task_id_differs_from_v1_same_netuid_epoch_hotkey() {
        let m = miner11();
        let v1 = task_id(FIX_NETUID, FIX_EPOCH, &m);
        let v2 = task_id_v2(FIX_NETUID, FIX_EPOCH, &m, FIX_PACK, SCORING_VERSION_V2);
        assert_ne!(v1, v2, "v1 golden must not equal v2 for same netuid/epoch/hotkey");
    }

    #[test]
    fn v2_sensitivity_pack_id_epoch_scoring_version_patch() {
        let m = miner11();
        let base_tid = task_id_v2(FIX_NETUID, FIX_EPOCH, &m, FIX_PACK, SCORING_VERSION_V2);
        let base_blob = task_blob_v2(&base_tid, SCORING_VERSION_V2, FIX_PACK);
        let base_ans = answer_digest_v2(FIX_PATCH);

        let tid_pack = task_id_v2(FIX_NETUID, FIX_EPOCH, &m, b"pack-OTHER", SCORING_VERSION_V2);
        assert_ne!(base_tid, tid_pack);

        let tid_epoch = task_id_v2(FIX_NETUID, FIX_EPOCH + 1, &m, FIX_PACK, SCORING_VERSION_V2);
        assert_ne!(base_tid, tid_epoch);

        let tid_sv = task_id_v2(FIX_NETUID, FIX_EPOCH, &m, FIX_PACK, 1);
        assert_ne!(base_tid, tid_sv);

        let blob_pack = task_blob_v2(&base_tid, SCORING_VERSION_V2, b"pack-OTHER");
        assert_ne!(base_blob, blob_pack);

        let blob_sv = task_blob_v2(&base_tid, 1, FIX_PACK);
        assert_ne!(base_blob, blob_sv);

        let mut patch2 = FIX_PATCH.to_vec();
        patch2.push(b'X');
        assert_ne!(base_ans, answer_digest_v2(&patch2));
    }

    #[test]
    fn v2_answer_not_equal_untagged_patch_sha256() {
        let tagged = answer_digest_v2(FIX_PATCH);
        let untagged = {
            let d = Sha256::digest(FIX_PATCH);
            let mut out = [0u8; 32];
            out.copy_from_slice(&d);
            out
        };
        assert_ne!(tagged, untagged);
    }

    /// Golden vectors for fixture set A (committed under tests/fixtures/).
    #[test]
    fn v2_golden_fixture_a() {
        let m = miner11();
        let tid = task_id_v2(FIX_NETUID, FIX_EPOCH, &m, FIX_PACK, SCORING_VERSION_V2);
        let blob = task_blob_v2(&tid, SCORING_VERSION_V2, FIX_PACK);
        let ans = answer_digest_v2(FIX_PATCH);
        assert_eq!(
            hex::encode(tid),
            include_str!("../tests/fixtures/v2_task_id_a.hex").trim()
        );
        assert_eq!(
            hex::encode(blob),
            include_str!("../tests/fixtures/v2_task_blob_a.hex").trim()
        );
        assert_eq!(
            hex::encode(ans),
            include_str!("../tests/fixtures/v2_answer_digest_a.hex").trim()
        );
    }

    /// Second miner golden (fixture set B).
    #[test]
    fn v2_golden_fixture_b() {
        let m = miner22();
        let tid = task_id_v2(FIX_NETUID, FIX_EPOCH, &m, FIX_PACK, SCORING_VERSION_V2);
        let blob = task_blob_v2(&tid, SCORING_VERSION_V2, FIX_PACK);
        let ans = answer_digest_v2(FIX_PATCH);
        assert_eq!(
            hex::encode(tid),
            include_str!("../tests/fixtures/v2_task_id_b.hex").trim()
        );
        assert_eq!(
            hex::encode(blob),
            include_str!("../tests/fixtures/v2_task_blob_b.hex").trim()
        );
        assert_eq!(
            hex::encode(ans),
            include_str!("../tests/fixtures/v2_answer_digest_b.hex").trim()
        );
        // Same patch → same answer_digest_v2 regardless of miner (patch-only preimage).
        assert_eq!(
            hex::encode(ans),
            include_str!("../tests/fixtures/v2_answer_digest_a.hex").trim()
        );
    }
}
