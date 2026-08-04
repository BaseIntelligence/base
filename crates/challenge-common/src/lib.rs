//! Shared challenge helpers used by prism (and temporarily agent-v1).
//!
//! - Sealed expected set `E` at `block_B` (I7 / D24)
//! - Exact-E signed leaf emission
//! - Gateway `POST /v1/weights/raw` client

#![forbid(unsafe_code)]

mod expected_set;
mod leaf_emit;
mod submit;

pub use expected_set::{
    expected_set_at, expected_set_at_chain, expected_set_from_optional_pin,
    expected_set_from_pinned_metagraph, hex32, BlockSource, ExpectedParticipant, ExpectedSet,
    ExpectedSetError, PinnedBlockHash,
};
pub use leaf_emit::{
    emit_signed_leaf_set, public_key_from_secret, verify_leaf_sig, Hotkey, LeafEmitError,
};
pub use submit::{
    hotkey_hex, leaf_request_json, submit_signed_leaf_set, GatewayClient, GatewayClientConfig,
    SubmitError, SubmitOutcome, DEFAULT_MAX_RETRIES, DRY_RUN_BASE_URL,
};
