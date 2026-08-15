//! In-memory [`EvalStore`] for offline tests/sim.

use std::collections::BTreeMap;
use std::sync::Mutex;

use async_trait::async_trait;
use prism_store::eval::{
    AnchorSetRecord, EvalGroupRecord, EvalMetricRecord, EvalRunRecord, EvalStore,
    MetricReportRecord, MirrorPairRecord, PreregRecord,
};
use prism_store::StoreError;
use prism_zoneb::PreparedReport;
use serde_json::Value;

type RunBundle = (
    EvalRunRecord,
    Vec<EvalGroupRecord>,
    Vec<EvalMetricRecord>,
    Vec<MirrorPairRecord>,
);

/// In-memory eval store (test/sim counterpart of [`crate::DbEvalStore`]).
#[derive(Debug, Default)]
pub struct MemoryEvalStore {
    inner: Mutex<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    runs: BTreeMap<String, RunBundle>,
    anchors: BTreeMap<i32, AnchorSetRecord>,
    preregs: Vec<PreregRecord>,
    reports: BTreeMap<String, Vec<MetricReportRecord>>,
    next_run: u64,
}

impl MemoryEvalStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

fn now_text() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or_else(|_| "0".into(), |d| d.as_secs().to_string())
}

fn lock(inner: &Mutex<Inner>) -> Result<std::sync::MutexGuard<'_, Inner>, StoreError> {
    inner.lock().map_err(|e| StoreError::Backend(e.to_string()))
}

fn schema_of(payload: &Value) -> String {
    payload
        .get("schema_version")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned()
}

#[async_trait]
impl EvalStore for MemoryEvalStore {
    async fn save_eval_run(
        &self,
        run: &EvalRunRecord,
        groups: &[EvalGroupRecord],
        metrics: &[EvalMetricRecord],
        mirrors: &[MirrorPairRecord],
    ) -> Result<(), StoreError> {
        let mut g = lock(&self.inner)?;
        g.next_run += 1;
        let mut row = run.clone();
        row.run_id = format!("mem-run-{}", g.next_run);
        row.created_at = now_text();
        g.runs.insert(
            run.submission_id.clone(),
            (row, groups.to_vec(), metrics.to_vec(), mirrors.to_vec()),
        );
        Ok(())
    }

    async fn eval_run(&self, submission_id: &str) -> Result<Option<EvalRunRecord>, StoreError> {
        Ok(lock(&self.inner)?
            .runs
            .get(submission_id)
            .map(|b| b.0.clone()))
    }

    async fn eval_groups(&self, run_id: &str) -> Result<Vec<EvalGroupRecord>, StoreError> {
        Ok(lock(&self.inner)?
            .runs
            .values()
            .find(|b| b.0.run_id == run_id)
            .map_or_else(Vec::new, |b| b.1.clone()))
    }

    async fn eval_metrics(&self, run_id: &str) -> Result<Vec<EvalMetricRecord>, StoreError> {
        Ok(lock(&self.inner)?
            .runs
            .values()
            .find(|b| b.0.run_id == run_id)
            .map_or_else(Vec::new, |b| b.2.clone()))
    }

    async fn eval_mirrors(&self, run_id: &str) -> Result<Vec<MirrorPairRecord>, StoreError> {
        Ok(lock(&self.inner)?
            .runs
            .values()
            .find(|b| b.0.run_id == run_id)
            .map_or_else(Vec::new, |b| b.3.clone()))
    }

    async fn upsert_anchor_set(&self, set: &AnchorSetRecord) -> Result<(), StoreError> {
        let mut g = lock(&self.inner)?;
        if let Some(e) = g.anchors.get_mut(&set.version) {
            let flipped = e.status != "active" && set.status == "active";
            let activated = if flipped {
                Some(now_text())
            } else {
                e.activated_at.clone()
            };
            *e = set.clone();
            e.activated_at = activated;
        } else {
            let mut s = set.clone();
            if s.status == "active" && s.activated_at.is_none() {
                s.activated_at = Some(now_text());
            }
            g.anchors.insert(set.version, s);
        }
        Ok(())
    }

    async fn anchor_sets(&self) -> Result<Vec<AnchorSetRecord>, StoreError> {
        Ok(lock(&self.inner)?.anchors.values().cloned().collect())
    }

    async fn record_prereg(&self, entry: &PreregRecord) -> Result<(), StoreError> {
        let mut g = lock(&self.inner)?;
        if !g
            .preregs
            .iter()
            .any(|p| p.version == entry.version && p.hash == entry.hash)
        {
            let mut row = entry.clone();
            row.committed_at = now_text();
            g.preregs.push(row);
        }
        Ok(())
    }

    async fn preregistrations(&self) -> Result<Vec<PreregRecord>, StoreError> {
        let mut v = lock(&self.inner)?.preregs.clone();
        v.sort_by_key(|p| p.version);
        Ok(v)
    }

    async fn insert_metric_report(
        &self,
        submission_id: &str,
        report: &PreparedReport,
    ) -> Result<(), StoreError> {
        let mut g = lock(&self.inner)?;
        let reports = g.reports.entry(submission_id.to_owned()).or_default();
        let seq = i64::try_from(report.seq).unwrap_or(i64::MAX);
        if reports
            .iter()
            .any(|r| r.seq == seq || r.report_hash == report.report_hash)
        {
            return Err(StoreError::Backend("duplicate zone b report".into()));
        }
        reports.push(MetricReportRecord {
            seq,
            schema_version: schema_of(&report.payload),
            prev_hash: report.prev_hash.clone(),
            report_hash: report.report_hash.clone(),
            payload: report.payload.clone(),
            verdict: report.verdict,
            verdict_reasons: report.reasons.clone(),
            created_at: now_text(),
        });
        Ok(())
    }

    async fn metric_reports(
        &self,
        submission_id: &str,
    ) -> Result<Vec<MetricReportRecord>, StoreError> {
        Ok(lock(&self.inner)?
            .reports
            .get(submission_id)
            .cloned()
            .unwrap_or_default())
    }

    async fn latest_metric_report(
        &self,
        submission_id: &str,
    ) -> Result<Option<MetricReportRecord>, StoreError> {
        Ok(lock(&self.inner)?
            .reports
            .get(submission_id)
            .and_then(|v| v.last())
            .cloned())
    }

    async fn cohort_reports(
        &self,
        exclude_submission: &str,
    ) -> Result<Vec<MetricReportRecord>, StoreError> {
        Ok(lock(&self.inner)?
            .reports
            .iter()
            .filter(|(id, _)| id.as_str() != exclude_submission)
            .flat_map(|(_, v)| v.iter().cloned())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use prism_zoneb::Verdict;
    use serde_json::json;

    fn run(sub: &str) -> EvalRunRecord {
        EvalRunRecord {
            run_id: String::new(),
            submission_id: sub.into(),
            anchor_version: 0,
            prereg_hash: "ff".repeat(32),
            scoring_mode: "benchmarks".into(),
            pod_manifest: None,
            harness_files_sha256: None,
            netns: None,
            eval_tier: None,
            outcome_json: json!({"status": "ineligible"}),
            created_at: String::new(),
        }
    }

    fn group(g: f64) -> EvalGroupRecord {
        EvalGroupRecord {
            grp: "g1".into(),
            g,
            ci_lo: None,
            ci_hi: None,
            mirror_penalty: 0.0,
        }
    }

    fn prepared(seq: u64, hash: &str) -> PreparedReport {
        PreparedReport {
            seq,
            prev_hash: "sub".into(),
            report_hash: hash.into(),
            payload: json!({"schema_version": "1.3.0", "metrics": {}}),
            verdict: Verdict::Ok,
            reasons: vec![],
        }
    }

    #[tokio::test]
    async fn save_replaces_wholesale() {
        let st = MemoryEvalStore::new();
        st.save_eval_run(&run("s"), &[group(0.5)], &[], &[])
            .await
            .unwrap();
        let first = st.eval_run("s").await.unwrap().unwrap();
        st.save_eval_run(&run("s"), &[group(0.9), group(0.8)], &[], &[])
            .await
            .unwrap();
        let second = st.eval_run("s").await.unwrap().unwrap();
        assert_ne!(first.run_id, second.run_id);
        assert!(st.eval_groups(&first.run_id).await.unwrap().is_empty());
        assert_eq!(st.eval_groups(&second.run_id).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn report_chain_and_duplicate_reject() {
        let st = MemoryEvalStore::new();
        st.insert_metric_report("s", &prepared(0, "aa"))
            .await
            .unwrap();
        st.insert_metric_report("s", &prepared(1, "bb"))
            .await
            .unwrap();
        assert!(st
            .insert_metric_report("s", &prepared(1, "cc"))
            .await
            .is_err());
        assert!(st
            .insert_metric_report("s", &prepared(2, "bb"))
            .await
            .is_err());
        assert_eq!(st.latest_metric_report("s").await.unwrap().unwrap().seq, 1);
        assert_eq!(st.metric_reports("s").await.unwrap().len(), 2);
        assert_eq!(st.cohort_reports("s").await.unwrap().len(), 0);
        assert_eq!(st.cohort_reports("other").await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn prereg_idempotent_and_anchor_activation() {
        let st = MemoryEvalStore::new();
        let entry = PreregRecord {
            version: 0,
            hash: "ff".repeat(32),
            committed_at: String::new(),
            notes: None,
        };
        st.record_prereg(&entry).await.unwrap();
        st.record_prereg(&entry).await.unwrap();
        assert_eq!(st.preregistrations().await.unwrap().len(), 1);

        let set = AnchorSetRecord {
            version: 0,
            json: json!({"version": 0}),
            prereg_hash: "ff".repeat(32),
            status: "placeholder".into(),
            activated_at: None,
        };
        st.upsert_anchor_set(&set).await.unwrap();
        assert!(st.anchor_sets().await.unwrap()[0].activated_at.is_none());
        let mut active = set.clone();
        active.status = "active".into();
        st.upsert_anchor_set(&active).await.unwrap();
        assert!(st.anchor_sets().await.unwrap()[0].activated_at.is_some());
    }
}
