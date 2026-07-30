//! `agent-v1` challenge service (`docs/AGENT_CHALLENGE.md`, `scoring_version` = 1).
//!
//! Pure offline scoring, D24 participant coverage (`NoScore` never silence),
//! sr25519 leaf signing under `gbase-rawweight-v1`, and gateway POST with retry.

#![forbid(unsafe_code)]

mod challenge;
mod keys;
mod score;
mod submit;
mod task_gen;

pub use challenge::{
    correct_http200, silence_is_bug_leaf, AgentV1Challenge, AttestationLookup, Challenge,
    ChallengeError, EpochCtx, Hotkey, MapAttestationLookup, MinerCallOutcome,
};
pub use keys::{load_challenge_secret, public_key_from_secret, ChallengeKeyError};
pub use score::{
    score_from_outcome, score_latency, AttestationStatus, CallOutcome, ScoreInputs, CONNECT_MS,
    HARD_MS, MAX_ATTEMPTS, SCORE_MAX, SOFT_MS,
};
pub use submit::{
    hotkey_hex, leaf_request_json, GatewayClient, GatewayClientConfig, SubmitError, SubmitOutcome,
    DEFAULT_MAX_RETRIES,
};
pub use task_gen::{
    answer_digest, answer_digest_v2, task_blob, task_blob_v2, task_id, task_id_v2,
    ANSWER_DOMAIN_V2, CHALLENGE_ID, CHALLENGE_ID_BYTES, SCORING_VERSION, SCORING_VERSION_V2,
    TASK_BLOB_DOMAIN_V2, TASK_ID_DOMAIN_V2,
};

pub use bundle::{
    make_signed_leaf, raw_weight_payload, LeafV1, NoScoreReasonCode, RawWeightBodyV1,
    ScoreOrAbsence,
};
pub use crypto::{KEY_LEN, SIGNATURE_LEN};
