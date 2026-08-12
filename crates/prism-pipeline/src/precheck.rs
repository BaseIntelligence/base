//! Miner-facing similarity precheck (copy gate only — no pod, no LLM).

use challenge_agentic::{
    copy_gate, same_miner_identity, CopyGateHit, CorpusEntry, GateCorpusEntry,
};
use prism_recipe::BASELINE_ARCHITECTURE_PY;
use prism_store::SubmissionState;
use serde::Serialize;
use serde_json::{json, Value};

/// Max precheck attempts per coldkey (or hotkey fallback) per UTC day.
pub const PRECHECK_DAILY_LIMIT: u32 = 3;

/// Quota key axis reported to miners.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecheckIdentityKind {
    /// Metagraph Owner coldkey.
    Coldkey,
    /// Hotkey used when coldkey is unknown (zeros / no snapshot).
    HotkeyFallback,
}

/// Daily precheck budget snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PrecheckQuotaView {
    /// UTC day `YYYY-MM-DD`.
    pub day: String,
    /// Attempts consumed (including the current one when allowed).
    pub used: u32,
    /// Hard cap ([`PRECHECK_DAILY_LIMIT`]).
    pub limit: u32,
    /// Attempts left today.
    pub remaining: u32,
    /// Whether the counter key is coldkey or hotkey fallback.
    pub identity: PrecheckIdentityKind,
}

/// Result of the advisory copy-gate precheck.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PrecheckResult {
    /// True when the architecture would hard-reject at intake copy gate.
    pub similar: bool,
    /// `clean` | `copied` | `skipped`.
    pub verdict: &'static str,
    /// Corpus id of the nearest earlier hit (never full competitor source).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_against: Option<String>,
    /// Similarity in `[0, 1]` (`bps / 10000`), when compared.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Human-readable guidance (no competitor source).
    pub message: String,
    /// Present on copy hits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_identical: Option<bool>,
    /// Daily budget after this call (or at rejection).
    pub quota: PrecheckQuotaView,
}

/// UTC calendar day `YYYY-MM-DD` from unix seconds.
#[must_use]
pub fn utc_day(secs: u64) -> String {
    let days = secs / 86_400;
    let z = i64::try_from(days).unwrap_or(i64::MAX) + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = u64::try_from(z - era * 146_097).unwrap_or(0);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = i64::try_from(yoe).unwrap_or(0) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// Resolve the quota counter key: prefer coldkey, else hotkey (both 64 hex).
#[must_use]
pub fn quota_identity(hotkey: &str, coldkey: Option<&str>) -> (String, PrecheckIdentityKind) {
    match coldkey.map(str::trim).filter(|s| !s.is_empty()) {
        Some(ck) => (ck.to_ascii_lowercase(), PrecheckIdentityKind::Coldkey),
        None => (
            hotkey.trim().to_ascii_lowercase(),
            PrecheckIdentityKind::HotkeyFallback,
        ),
    }
}

/// Build a quota view from used count.
#[must_use]
pub fn quota_view(
    day: impl Into<String>,
    used: u32,
    identity: PrecheckIdentityKind,
) -> PrecheckQuotaView {
    let limit = PRECHECK_DAILY_LIMIT;
    PrecheckQuotaView {
        day: day.into(),
        used,
        limit,
        remaining: limit.saturating_sub(used),
        identity,
    }
}

/// True when `other` is the same economic miner as `candidate`.
#[must_use]
pub fn same_miner(candidate: &SubmissionState, other: &SubmissionState) -> bool {
    same_miner_identity(
        &candidate.miner_hotkey,
        candidate.miner_coldkey.as_deref(),
        &other.miner_hotkey,
        other.miner_coldkey.as_deref(),
    )
}

/// Pre-LLM copy-gate corpus: other miners' **champion** prior art only
/// (caller should pass `PrismStore::list_champions`; hotkey + coldkey excluded).
#[must_use]
pub fn gate_corpus_from_rows(
    candidate: &SubmissionState,
    recent: &[SubmissionState],
) -> Vec<GateCorpusEntry> {
    recent
        .iter()
        .filter(|r| r.id != candidate.id && !same_miner(candidate, r))
        .map(|r| GateCorpusEntry {
            id: format!("subm:{}", r.id),
            source: r.architecture_py.clone(),
            created_at_ms: r.created_at_ms,
        })
        .collect()
}

/// Baseline + champion submissions as agentic corpus entries.
///
/// Architecture.py only; same-hotkey and same-coldkey prior art excluded.
#[must_use]
pub fn corpus_from_rows(
    candidate: &SubmissionState,
    recent: &[SubmissionState],
    exempt_arch: Option<&str>,
) -> Vec<CorpusEntry> {
    let mut v = vec![CorpusEntry {
        id: "baseline".into(),
        source: BASELINE_ARCHITECTURE_PY.into(),
    }];
    for r in recent {
        if r.id == candidate.id || same_miner(candidate, r) {
            continue;
        }
        if Some(r.architecture_py.as_str()) == exempt_arch {
            continue;
        }
        let label = if r.id.len() >= 8 {
            format!("subm:{}", &r.id[..8])
        } else {
            format!("subm:{}", r.id)
        };
        v.push(CorpusEntry {
            id: label,
            source: r.architecture_py.clone(),
        });
    }
    v
}

/// Ephemeral candidate row for precheck corpus filtering (not persisted).
#[must_use]
pub fn ephemeral_candidate(
    hotkey: &str,
    coldkey: Option<String>,
    architecture_py: &str,
    created_at_ms: u64,
) -> SubmissionState {
    SubmissionState {
        id: format!("precheck:{hotkey}:{created_at_ms}"),
        miner_hotkey: hotkey.to_owned(),
        miner_coldkey: coldkey,
        epoch: 0,
        netuid: 0,
        status: prism_store::Stage::Queued,
        architecture_py: architecture_py.to_owned(),
        training_py: String::new(),
        tree_blob: None,
        label: None,
        pod_id: None,
        pod_provider: None,
        receipt: None,
        metrics_json: None,
        bpb: None,
        arch_id: None,
        review: None,
        similarity: None,
        final_score: None,
        retry_count: 0,
        error_detail: None,
        created_at_ms,
        updated_at_ms: created_at_ms,
    }
}

fn from_hit(hit: &CopyGateHit, quota: PrecheckQuotaView) -> PrecheckResult {
    let score = f64::from(hit.similarity_bps) / 10_000.0;
    let message = if hit.byte_identical {
        "architecture matches an earlier submission byte-for-byte; revise before submitting".into()
    } else {
        "architecture is AST-near an earlier submission; revise before submitting".into()
    };
    PrecheckResult {
        similar: true,
        verdict: "copied",
        matched_against: Some(hit.nearest_id.clone()),
        score: Some(score),
        message,
        byte_identical: Some(hit.byte_identical),
        quota,
    }
}

/// Run the intake copy gate against prior art. Training-only callers should
/// short-circuit with [`precheck_skipped`] instead.
#[must_use]
pub fn evaluate_copy_precheck(
    candidate: &SubmissionState,
    recent: &[SubmissionState],
    quota: PrecheckQuotaView,
) -> PrecheckResult {
    let corpus = gate_corpus_from_rows(candidate, recent);
    match copy_gate(
        &candidate.architecture_py,
        candidate.created_at_ms,
        &corpus,
    ) {
        Some(hit) => from_hit(&hit, quota),
        None => PrecheckResult {
            similar: false,
            verdict: "clean",
            matched_against: None,
            score: None,
            message: "no earlier architecture copy detected by the pre-LLM gate; full submit still runs similarity + agentic".into(),
            byte_identical: None,
            quota,
        },
    }
}

/// Training-only / registry-arch rows skip architecture copy comparison.
#[must_use]
pub fn precheck_skipped(quota: PrecheckQuotaView) -> PrecheckResult {
    PrecheckResult {
        similar: false,
        verdict: "skipped",
        matched_against: None,
        score: None,
        message: "training-only entries skip architecture copy precheck (registry arch)".into(),
        byte_identical: None,
        quota,
    }
}

/// JSON body for a successful precheck response.
#[must_use]
pub fn precheck_json(result: &PrecheckResult) -> Value {
    json!(result)
}

/// JSON body when the daily precheck quota is exhausted.
#[must_use]
pub fn precheck_quota_exceeded_json(quota: &PrecheckQuotaView) -> Value {
    json!({
        "error": "precheck daily quota exceeded (3/coldkey/UTC day)",
        "code": "precheck_quota_exceeded",
        "quota": quota,
        "similar": null,
        "verdict": null,
        "message": "retry after the next UTC day; precheck does not create a submission",
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use prism_store::{FinalScore, Stage};

    fn row(
        id: &str,
        hotkey: &str,
        coldkey: Option<&str>,
        arch: &str,
        created_at_ms: u64,
    ) -> SubmissionState {
        SubmissionState {
            id: id.into(),
            miner_hotkey: hotkey.into(),
            miner_coldkey: coldkey.map(str::to_owned),
            epoch: 1,
            netuid: 1,
            status: Stage::Terminated,
            architecture_py: arch.into(),
            training_py: "train".into(),
            tree_blob: None,
            label: None,
            pod_id: None,
            pod_provider: None,
            receipt: None,
            metrics_json: None,
            bpb: Some(1.0),
            arch_id: None,
            review: None,
            similarity: None,
            final_score: Some(FinalScore::Score(1)),
            retry_count: 0,
            error_detail: None,
            created_at_ms,
            updated_at_ms: created_at_ms,
        }
    }

    #[test]
    fn utc_day_epoch() {
        assert_eq!(utc_day(0), "1970-01-01");
        assert_eq!(utc_day(86_400), "1970-01-02");
    }

    #[test]
    fn same_coldkey_excluded_from_gate_corpus() {
        let prior = row("aaaaaaaa", "aa", Some("11"), "arch_a", 1_000);
        let next = row("bbbbbbbb", "bb", Some("11"), "arch_b", 2_000);
        assert!(gate_corpus_from_rows(&next, &[prior, next.clone()]).is_empty());
    }

    #[test]
    fn copy_hit_marks_similar() {
        let victim = row(
            "aaaaaaaa",
            "aa",
            Some("11"),
            "def build_model(ctx):\n    return 1\n",
            1_000,
        );
        let copier = ephemeral_candidate(
            &"bb".repeat(32),
            Some("22".repeat(32)),
            "def build_model(ctx):\n    return 1\n",
            2_000,
        );
        let q = quota_view("2026-08-08", 1, PrecheckIdentityKind::Coldkey);
        let r = evaluate_copy_precheck(&copier, &[victim], q);
        assert!(r.similar);
        assert_eq!(r.verdict, "copied");
        assert_eq!(r.matched_against.as_deref(), Some("subm:aaaaaaaa"));
    }
}
