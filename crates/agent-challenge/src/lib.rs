//! `agent-v1` challenge service (`docs/AGENT_CHALLENGE.md`, `scoring_version` = 2).
//!
//! Pure offline scoring, D24 participant coverage (`NoScore` never silence),
//! sr25519 leaf signing under `gbase-rawweight-v1`, and gateway POST with retry.
//!
//! Live task identity uses pack-bound v2 formulas (`task_id_v2`, `answer_digest_v2`).

#![forbid(unsafe_code)]

mod challenge;
mod keys;
mod score;
mod submit;
mod task_gen;
mod leaf_map;
mod verify;
mod expected_set;

pub use challenge::{
    correct_http200, correct_http200_fixture, leaf_from_verify_result, score_epoch_from_verify,
    silence_is_bug_leaf, AgentV1Challenge, AttestationLookup, Challenge, ChallengeError, EpochCtx,
    Hotkey, MapAttestationLookup, MinerCallOutcome,
};
pub use expected_set::{
    expected_set_at, expected_set_at_chain, expected_set_from_optional_pin,
    expected_set_from_pinned_metagraph, hex32, BlockSource, ExpectedParticipant, ExpectedSet,
    ExpectedSetError, PinnedBlockHash,
};
pub use keys::{load_challenge_secret, public_key_from_secret, ChallengeKeyError};
pub use leaf_map::{
    attempts_within_seal_budget, cover_expected_verify_leaves, grade_to_score_or_absence,
    grade_to_score_or_absence_budgeted, is_operator_fault, is_retryable_operator_fault, map_reward,
    map_verify_error, score_from_verify_result, MAX_VERIFY_ATTEMPTS, MAX_VERIFY_RETRIES,
};
pub use score::{
    score_from_outcome, AttestationStatus, CallOutcome, ScoreInputs, CONNECT_MS, MAX_ATTEMPTS,
    SCORE_MAX,
};
pub use submit::{
    hotkey_hex, leaf_request_json, GatewayClient, GatewayClientConfig, SubmitError, SubmitOutcome,
    DEFAULT_MAX_RETRIES,
};
pub use task_gen::{
    answer_digest, answer_digest_v2, task_blob, task_blob_v2, task_id, task_id_v2,
    ANSWER_DOMAIN_V2, CHALLENGE_ID, CHALLENGE_ID_BYTES, FIXTURE_MODEL_PATCH, FIXTURE_PACK_ID,
    SCORING_VERSION, SCORING_VERSION_V2, TASK_BLOB_DOMAIN_V2, TASK_ID_DOMAIN_V2,
};

pub use bundle::{
    make_signed_leaf, raw_weight_payload, LeafV1, NoScoreReasonCode, RawWeightBodyV1,
    ScoreOrAbsence,
};
pub use crypto::{KEY_LEN, SIGNATURE_LEN};

pub use verify::{
    map_docker_timeout, reward_from_json_bytes, HarborVerifier, HarborVerifierConfig, Reward,
    Verifier, VerifyError, ZeroReason, DEFAULT_VERIFIER_TIMEOUT_SEC,
};
