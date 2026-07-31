//! Combined anti-noise evaluation (levels + K + dedupe + sanctions).

use crate::dedupe::{DedupeOutcome, FingerprintDedupe};
use crate::error::AntinoisError;
use crate::l1::l1_source_similarity;
use crate::l2::{binary_fingerprint_hex, l2_binary_similarity};
use crate::l3::{l3_telemetry_similarity, TelemetryFingerprint};
use crate::llm::{advisory_only, AdvisoryNote, LlmAdvisory, NoopLlm};
use crate::normalize::normalize_source;
use crate::sanctions::{decide_sanction, Sanction};

/// Candidate submission artifacts for anti-noise.
#[derive(Debug, Clone)]
pub struct CandidateArtifacts<'a> {
    /// Miner identity string.
    pub miner_id: &'a str,
    /// Source text (single-file or concatenated tree).
    pub source: &'a str,
    /// Fixture compiled blob (PTX-like or opaque).
    pub compiled: &'a [u8],
    /// Optional L3 telemetry.
    pub telemetry: Option<&'a TelemetryFingerprint>,
    /// Current segment index for dedupe.
    pub segment_index: u64,
}

/// Champion reference artifacts.
#[derive(Debug, Clone)]
pub struct ChampionArtifacts<'a> {
    /// Champion source.
    pub source: &'a str,
    /// Champion compiled blob.
    pub compiled: &'a [u8],
    /// Optional champion telemetry.
    pub telemetry: Option<&'a TelemetryFingerprint>,
}

/// Full anti-noise report (levels + sanction + optional advisory).
#[derive(Debug, Clone, PartialEq)]
pub struct AntinoisReport {
    /// L1 source similarity.
    pub source_similarity: f64,
    /// L2 binary similarity.
    pub binary_similarity: f64,
    /// L3 telemetry similarity when both sides present.
    pub telemetry_similarity: Option<f64>,
    /// Hex fingerprint of candidate normalized binary.
    pub candidate_fingerprint_hex: String,
    /// Graduated sanction / allow decision.
    pub sanction: Sanction,
    /// Advisory note (never gates).
    pub advisory: AdvisoryNote,
}

impl AntinoisReport {
    /// Whether the pipeline may proceed to measure.
    #[must_use]
    pub fn allows_measure(&self) -> bool {
        self.sanction.allows_measure()
    }
}

/// Evaluate anti-noise for a candidate vs champion, updating dedupe state.
///
/// LLM defaults to [`NoopLlm`] and never blocks.
///
/// # Errors
/// Propagates empty miner / invalid dedupe configuration errors.
pub fn evaluate(
    candidate: &CandidateArtifacts<'_>,
    champion: &ChampionArtifacts<'_>,
    dedupe: &mut FingerprintDedupe,
) -> Result<AntinoisReport, AntinoisError> {
    evaluate_with_llm(candidate, champion, dedupe, &NoopLlm)
}

/// Same as [`evaluate`] with an explicit advisory backend (still non-blocking).
///
/// # Errors
/// Propagates empty miner / invalid dedupe configuration errors.
pub fn evaluate_with_llm<A: LlmAdvisory>(
    candidate: &CandidateArtifacts<'_>,
    champion: &ChampionArtifacts<'_>,
    dedupe: &mut FingerprintDedupe,
    llm: &A,
) -> Result<AntinoisReport, AntinoisError> {
    if candidate.miner_id.is_empty() {
        return Err(AntinoisError::EmptyMinerId);
    }
    if candidate.compiled.is_empty() {
        return Err(AntinoisError::EmptyArtifact("compiled"));
    }
    if champion.compiled.is_empty() {
        return Err(AntinoisError::EmptyArtifact("champion_compiled"));
    }

    let source_similarity = l1_source_similarity(candidate.source, champion.source);
    let binary_similarity = l2_binary_similarity(candidate.compiled, champion.compiled);
    let telemetry_similarity = match (candidate.telemetry, champion.telemetry) {
        (Some(a), Some(b)) => Some(l3_telemetry_similarity(a, b)),
        _ => None,
    };

    let fp_hex = binary_fingerprint_hex(candidate.compiled);

    // Dedupe before measure path — same fingerprint within N segments.
    let dedupe_out =
        dedupe.check_and_record(candidate.miner_id, &fp_hex, candidate.segment_index)?;
    if let DedupeOutcome::Rejected {
        last_segment,
        window_n,
    } = dedupe_out
    {
        let norm_diff = normalize_source(candidate.source);
        let advisory = advisory_only(llm, &norm_diff);
        return Ok(AntinoisReport {
            source_similarity,
            binary_similarity,
            telemetry_similarity,
            candidate_fingerprint_hex: fp_hex,
            sanction: Sanction::DedupeReject {
                last_segment,
                window_n,
            },
            advisory,
        });
    }

    let sanction = decide_sanction(source_similarity, binary_similarity, 5);
    let norm_diff = normalize_source(candidate.source);
    let mut advisory = advisory_only(llm, &norm_diff);
    // If levels disagree strongly, attach explanation (still non-binding).
    if (source_similarity - binary_similarity).abs() > 0.4 {
        let expl = llm.explain_level_disagreement(source_similarity, binary_similarity, &norm_diff);
        if advisory.explanation.is_empty() {
            advisory = expl;
        }
    }

    Ok(AntinoisReport {
        source_similarity,
        binary_similarity,
        telemetry_similarity,
        candidate_fingerprint_hex: fp_hex,
        sanction,
        advisory,
    })
}
