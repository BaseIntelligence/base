//! Helpers that glue parse / replay / compose-hash into policy inputs.

use gbase_attest_parse::TdReport;
use gbase_attest_replay::{replay_rtmr3, Event, Replay, ReplayError};
use gbase_compose_hash::{mr_config_id, ComposeHash, COMPOSE_HASH_LEN};
use gbase_trustroot::REGISTER_LEN;

/// Replay RTMR3 events and require a 32-byte compose-hash payload.
///
/// # Errors
///
/// Propagates [`ReplayError`]. Returns [`ReplayError::DigestMismatch`] with
/// name `"compose-hash"` when the payload length is not 32 bytes (fail closed).
pub fn replay_compose_hash(events: &[Event]) -> Result<([u8; COMPOSE_HASH_LEN], Replay), ReplayError> {
    let replay = replay_rtmr3(events)?;
    let Some(raw) = replay.compose_hash.as_ref() else {
        return Err(ReplayError::DigestMismatch {
            name: "compose-hash".into(),
        });
    };
    if raw.len() != COMPOSE_HASH_LEN {
        return Err(ReplayError::DigestMismatch {
            name: "compose-hash".into(),
        });
    }
    let mut hash = [0_u8; COMPOSE_HASH_LEN];
    hash.copy_from_slice(raw);
    Ok((hash, replay))
}

/// Whether TD `mr_config_id` matches compose-hash v1 layout (`0x01 ‖ hash ‖ pad`).
#[must_use]
pub fn mr_config_id_matches(td: &TdReport, compose_hash: &ComposeHash) -> bool {
    let expected: [u8; REGISTER_LEN] = mr_config_id(compose_hash);
    td.mr_config_id == expected
}
