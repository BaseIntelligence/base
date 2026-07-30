//! Validator glue for peer root cross-check (task 31).
//!
//! Core types live in [`crosscheck`]; this module maps outcomes onto
//! [`crate::recompute::ComparisonOutcome`] / [`NoSubmissionReason`].

use crate::peers::PeerBook;
use crate::recompute::{ComparisonOutcome, NoSubmissionReason};

pub use crosscheck::{
    consensus_root_router, crosscheck_peer_roots, evaluate_sample, fetch_peer_root,
    other_validators_on_metagraph, record_local_root, root_statement_payload, CrossCheckConfig,
    CrossCheckFailKind, CrossCheckOutcome, PeerRef, RootStatementJson, RootStatementStore,
    SharedRootStore, SignedRootStatement, DEFAULT_MIN_SAMPLE,
};

/// Map peer book entries to [`PeerRef`].
#[must_use]
pub fn peer_refs(book: &PeerBook) -> Vec<PeerRef> {
    book.all()
        .iter()
        .map(|p| PeerRef {
            hotkey: p.hotkey.clone(),
            base_url: p.base_url.clone(),
        })
        .collect()
}

/// Map failed cross-check → [`NoSubmissionReason`].
#[must_use]
pub fn no_submission_for_crosscheck(outcome: &CrossCheckOutcome) -> Option<NoSubmissionReason> {
    match outcome {
        CrossCheckOutcome::Agreed { .. } | CrossCheckOutcome::SingleValidatorExempt { .. } => None,
        CrossCheckOutcome::Failed {
            kind:
                CrossCheckFailKind::InsufficientPeers {
                    reachable,
                    required,
                    ..
                },
            ..
        } => Some(NoSubmissionReason::PeerSampleInsufficient {
            reachable: *reachable,
            required: *required,
        }),
        CrossCheckOutcome::Failed {
            kind: CrossCheckFailKind::RootDisagreement { candidate },
            statements,
            epoch,
            ..
        } => Some(NoSubmissionReason::PeerRootDisagreement {
            epoch: *epoch,
            candidate: *candidate,
            sample_len: statements.len(),
        }),
    }
}

/// Gate Match / `VectorMismatch` on a passing cross-check (fail closed).
#[must_use]
pub fn apply_crosscheck_gate(
    comparison: ComparisonOutcome,
    crosscheck: &CrossCheckOutcome,
) -> ComparisonOutcome {
    match &comparison {
        ComparisonOutcome::Match { .. } | ComparisonOutcome::VectorMismatch { .. } => {
            if crosscheck.allows_submission() {
                comparison
            } else if let Some(reason) = no_submission_for_crosscheck(crosscheck) {
                ComparisonOutcome::NoSubmission { reason }
            } else {
                ComparisonOutcome::NoSubmission {
                    reason: NoSubmissionReason::PeerSampleInsufficient {
                        reachable: 0,
                        required: DEFAULT_MIN_SAMPLE,
                    },
                }
            }
        }
        _ => comparison,
    }
}

/// `(epoch, merkle_root)` from a verified comparison.
#[must_use]
pub fn candidate_from_comparison(outcome: &ComparisonOutcome) -> Option<(u64, [u8; 32])> {
    match outcome {
        ComparisonOutcome::Match {
            epoch, merkle_root, ..
        }
        | ComparisonOutcome::VectorMismatch {
            epoch, merkle_root, ..
        } => Some((*epoch, *merkle_root)),
        _ => None,
    }
}

/// Inputs for [`crate::recompute::fetch_compare_and_crosscheck`].
pub struct CrossCheckRun<'a> {
    /// Root statement store (local serve + evidence).
    pub root_store: &'a SharedRootStore,
    /// Sample config.
    pub cfg: &'a CrossCheckConfig,
    /// Optional signing secret for local root.
    pub signing_secret: Option<&'a [u8; 32]>,
    /// Own hotkey public key bytes.
    pub own_hotkey_pk: Option<[u8; 32]>,
}
