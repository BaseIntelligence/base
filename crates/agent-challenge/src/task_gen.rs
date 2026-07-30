//! Pure task / answer digest generation (`AGENT_CHALLENGE` §5.2–§5.3).

use crypto::KEY_LEN;
use parity_scale_codec::Encode;
use sha2::{Digest, Sha256};

/// Normative challenge id.
pub const CHALLENGE_ID: &str = "agent-v1";
/// UTF-8 bytes of [`CHALLENGE_ID`].
pub const CHALLENGE_ID_BYTES: &[u8] = b"agent-v1";
/// `challenge_scoring_version` for this crate.
pub const SCORING_VERSION: u16 = 1;

const TASK_ID_DOMAIN: &[u8] = b"gbase-agent-task-id-v1";
const TASK_BLOB_DOMAIN: &[u8] = b"gbase-agent-task-blob-v1";
const ANSWER_DOMAIN: &[u8] = b"gbase-agent-answer-v1";

/// `task_id = sha256(b"gbase-agent-task-id-v1" ‖ scale(netuid) ‖ scale(epoch) ‖ miner_hotkey)`.
#[must_use]
pub fn task_id(netuid: u16, epoch: u64, miner_hotkey: &[u8; KEY_LEN]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(TASK_ID_DOMAIN);
    h.update(netuid.encode());
    h.update(epoch.encode());
    h.update(miner_hotkey);
    let d = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&d);
    out
}

/// `task_blob = sha256(b"gbase-agent-task-blob-v1" ‖ task_id ‖ scale(scoring_version))`.
#[must_use]
pub fn task_blob(task_id: &[u8; 32], scoring_version: u16) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(TASK_BLOB_DOMAIN);
    h.update(task_id);
    h.update(scoring_version.encode());
    let d = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&d);
    out
}

/// `answer_digest = sha256(b"gbase-agent-answer-v1" ‖ task_blob)`.
#[must_use]
pub fn answer_digest(task_blob: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(ANSWER_DOMAIN);
    h.update(task_blob);
    let d = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&d);
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn f1_pinned_digests() {
        let miner = [0x11u8; 32];
        let tid = task_id(1, 7, &miner);
        assert_eq!(
            hex::encode(tid),
            "4a590b2abf87da6bccd97d8fbe5d2e774bdbda3ad421119688010537be2b31ec"
        );
        let blob = task_blob(&tid, SCORING_VERSION);
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
        let miner = [0x22u8; 32];
        let tid = task_id(1, 7, &miner);
        assert_eq!(
            hex::encode(tid),
            "d954306fba3943a86bb69aedfd08f2bca850eb2adabaaf5efe2ad2728dbf3412"
        );
        let blob = task_blob(&tid, SCORING_VERSION);
        let ans = answer_digest(&blob);
        assert_eq!(
            hex::encode(ans),
            "05157d001bb1ec9ef5acc7140d0221141d2fbc14a830ce32893793f30470c0aa"
        );
    }
}
