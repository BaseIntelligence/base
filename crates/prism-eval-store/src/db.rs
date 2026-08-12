//! Postgres [`EvalStore`] over migration 0017.
//!
//! The SQL lives in this crate — not in `db::prism_store` — because the `db`
//! package is at its non-test LOC cap (same pattern as
//! `prism_store::telemetry`).

use async_trait::async_trait;
use db::PgPool;
use prism_store::eval::{
    AnchorSetRecord, EvalGroupRecord, EvalMetricRecord, EvalRunRecord, EvalStore,
    MetricReportRecord, MirrorPairRecord, PreregRecord,
};
use prism_store::StoreError;
use prism_zoneb::{PreparedReport, Verdict, VerdictReason};
use serde_json::Value;

/// Postgres eval store (production).
#[derive(Debug, Clone)]
pub struct DbEvalStore {
    pool: PgPool,
}

impl DbEvalStore {
    /// Wrap an existing pool.
    #[must_use]
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

fn backend<E: std::fmt::Display>(e: E) -> StoreError {
    StoreError::Backend(e.to_string())
}

fn verdict_str(v: Verdict) -> &'static str {
    match v {
        Verdict::Ok => "ok",
        Verdict::Flagged => "flagged",
        Verdict::Quarantined => "quarantined",
    }
}

fn verdict_from(s: &str) -> Result<Verdict, StoreError> {
    match s {
        "ok" => Ok(Verdict::Ok),
        "flagged" => Ok(Verdict::Flagged),
        "quarantined" => Ok(Verdict::Quarantined),
        other => Err(backend(format!("unknown verdict {other}"))),
    }
}

type RunRow = (
    String,
    String,
    i32,
    String,
    String,
    Option<Value>,
    Option<String>,
    Option<bool>,
    Option<String>,
    Value,
    String,
);

fn run_from(r: RunRow) -> EvalRunRecord {
    let (
        run_id,
        submission_id,
        anchor_version,
        prereg_hash,
        scoring_mode,
        pod_manifest,
        harness_files_sha256,
        netns,
        eval_tier,
        outcome_json,
        created_at,
    ) = r;
    EvalRunRecord {
        run_id,
        submission_id,
        anchor_version,
        prereg_hash,
        scoring_mode,
        pod_manifest,
        harness_files_sha256,
        netns,
        eval_tier,
        outcome_json,
        created_at,
    }
}

type GroupRow = (String, f64, Option<f64>, Option<f64>, f64);
type MetricRow = (String, f64, Option<Value>);
type MirrorRow = (String, String, f64, f64, Option<Value>, Option<Value>);
type AnchorRow = (i32, Value, String, String, Option<String>);
type PreregRow = (i32, String, String, Option<String>);
type ReportRow = (i64, String, String, String, Value, String, Value, String);

fn report_from(r: ReportRow) -> Result<MetricReportRecord, StoreError> {
    let (seq, schema_version, prev_hash, report_hash, payload, verdict, reasons, created_at) = r;
    Ok(MetricReportRecord {
        seq,
        schema_version,
        prev_hash,
        report_hash,
        payload,
        verdict: verdict_from(&verdict)?,
        verdict_reasons: serde_json::from_value::<Vec<VerdictReason>>(reasons).map_err(backend)?,
        created_at,
    })
}

const REPORT_COLS: &str =
    "seq, schema_version, prev_hash, report_hash, payload, verdict, verdict_reasons, \
     created_at::text";

#[async_trait]
impl EvalStore for DbEvalStore {
    async fn save_eval_run(
        &self,
        run: &EvalRunRecord,
        groups: &[EvalGroupRecord],
        metrics: &[EvalMetricRecord],
        mirrors: &[MirrorPairRecord],
    ) -> Result<(), StoreError> {
        let mut tx = self.pool.begin().await.map_err(backend)?;
        // Replace-wholesale on re-score: children cascade from the run row.
        sqlx::query("DELETE FROM prism_eval_run WHERE submission_id = $1")
            .bind(&run.submission_id)
            .execute(&mut *tx)
            .await
            .map_err(backend)?;
        let (run_id,): (String,) = sqlx::query_as(
            "INSERT INTO prism_eval_run (submission_id, anchor_version, prereg_hash, \
             scoring_mode, pod_manifest, harness_files_sha256, netns, eval_tier, outcome_json) \
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING run_id",
        )
        .bind(&run.submission_id)
        .bind(run.anchor_version)
        .bind(&run.prereg_hash)
        .bind(&run.scoring_mode)
        .bind(&run.pod_manifest)
        .bind(&run.harness_files_sha256)
        .bind(run.netns)
        .bind(&run.eval_tier)
        .bind(&run.outcome_json)
        .fetch_one(&mut *tx)
        .await
        .map_err(backend)?;
        if !groups.is_empty() {
            let mut qb = sqlx::QueryBuilder::new(
                "INSERT INTO prism_eval_group (run_id, grp, g, ci_lo, ci_hi, mirror_penalty) ",
            );
            qb.push_values(groups, |mut b, r| {
                b.push_bind(&run_id)
                    .push_bind(&r.grp)
                    .push_bind(r.g)
                    .push_bind(r.ci_lo)
                    .push_bind(r.ci_hi)
                    .push_bind(r.mirror_penalty);
            });
            qb.build().execute(&mut *tx).await.map_err(backend)?;
        }
        if !metrics.is_empty() {
            let mut qb = sqlx::QueryBuilder::new(
                "INSERT INTO prism_eval_metric (run_id, key, value, clusters) ",
            );
            qb.push_values(metrics, |mut b, r| {
                b.push_bind(&run_id)
                    .push_bind(&r.key)
                    .push_bind(r.value)
                    .push_bind(&r.clusters);
            });
            qb.build().execute(&mut *tx).await.map_err(backend)?;
        }
        if !mirrors.is_empty() {
            let mut qb = sqlx::QueryBuilder::new(
                "INSERT INTO prism_mirror_pair (run_id, grp, metric, public_value, \
                 mirror_value, public_clusters, mirror_clusters) ",
            );
            qb.push_values(mirrors, |mut b, r| {
                b.push_bind(&run_id)
                    .push_bind(&r.grp)
                    .push_bind(&r.metric)
                    .push_bind(r.public_value)
                    .push_bind(r.mirror_value)
                    .push_bind(&r.public_clusters)
                    .push_bind(&r.mirror_clusters);
            });
            qb.build().execute(&mut *tx).await.map_err(backend)?;
        }
        tx.commit().await.map_err(backend)
    }

    async fn eval_run(&self, submission_id: &str) -> Result<Option<EvalRunRecord>, StoreError> {
        let row: Option<RunRow> = sqlx::query_as(
            "SELECT run_id, submission_id, anchor_version, prereg_hash, scoring_mode, \
             pod_manifest, harness_files_sha256, netns, eval_tier, outcome_json, created_at::text \
             FROM prism_eval_run WHERE submission_id = $1",
        )
        .bind(submission_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(backend)?;
        Ok(row.map(run_from))
    }

    async fn eval_groups(&self, run_id: &str) -> Result<Vec<EvalGroupRecord>, StoreError> {
        let rows: Vec<GroupRow> = sqlx::query_as(
            "SELECT grp, g, ci_lo, ci_hi, mirror_penalty FROM prism_eval_group \
             WHERE run_id = $1 ORDER BY grp",
        )
        .bind(run_id)
        .fetch_all(&self.pool)
        .await
        .map_err(backend)?;
        Ok(rows
            .into_iter()
            .map(|(grp, g, ci_lo, ci_hi, mirror_penalty)| EvalGroupRecord {
                grp,
                g,
                ci_lo,
                ci_hi,
                mirror_penalty,
            })
            .collect())
    }

    async fn eval_metrics(&self, run_id: &str) -> Result<Vec<EvalMetricRecord>, StoreError> {
        let rows: Vec<MetricRow> = sqlx::query_as(
            "SELECT key, value, clusters FROM prism_eval_metric WHERE run_id = $1 ORDER BY key",
        )
        .bind(run_id)
        .fetch_all(&self.pool)
        .await
        .map_err(backend)?;
        Ok(rows
            .into_iter()
            .map(|(key, value, clusters)| EvalMetricRecord {
                key,
                value,
                clusters,
            })
            .collect())
    }

    async fn eval_mirrors(&self, run_id: &str) -> Result<Vec<MirrorPairRecord>, StoreError> {
        let rows: Vec<MirrorRow> = sqlx::query_as(
            "SELECT grp, metric, public_value, mirror_value, public_clusters, mirror_clusters \
             FROM prism_mirror_pair WHERE run_id = $1 ORDER BY grp, metric",
        )
        .bind(run_id)
        .fetch_all(&self.pool)
        .await
        .map_err(backend)?;
        Ok(rows
            .into_iter()
            .map(
                |(grp, metric, public_value, mirror_value, public_clusters, mirror_clusters)| {
                    MirrorPairRecord {
                        grp,
                        metric,
                        public_value,
                        mirror_value,
                        public_clusters,
                        mirror_clusters,
                    }
                },
            )
            .collect())
    }

    async fn upsert_anchor_set(&self, set: &AnchorSetRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO prism_anchor_set (version, json, prereg_hash, status) \
             VALUES ($1,$2,$3,$4) \
             ON CONFLICT (version) DO UPDATE SET json = EXCLUDED.json, \
             prereg_hash = EXCLUDED.prereg_hash, status = EXCLUDED.status, \
             activated_at = CASE WHEN prism_anchor_set.status <> 'active' \
             AND EXCLUDED.status = 'active' THEN now() \
             ELSE prism_anchor_set.activated_at END",
        )
        .bind(set.version)
        .bind(&set.json)
        .bind(&set.prereg_hash)
        .bind(&set.status)
        .execute(&self.pool)
        .await
        .map_err(backend)?;
        Ok(())
    }

    async fn anchor_sets(&self) -> Result<Vec<AnchorSetRecord>, StoreError> {
        let rows: Vec<AnchorRow> = sqlx::query_as(
            "SELECT version, json, prereg_hash, status, activated_at::text \
             FROM prism_anchor_set ORDER BY version",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(backend)?;
        Ok(rows
            .into_iter()
            .map(
                |(version, json, prereg_hash, status, activated_at)| AnchorSetRecord {
                    version,
                    json,
                    prereg_hash,
                    status,
                    activated_at,
                },
            )
            .collect())
    }

    async fn record_prereg(&self, entry: &PreregRecord) -> Result<(), StoreError> {
        sqlx::query(
            "INSERT INTO prism_prereg (version, hash, notes) VALUES ($1,$2,$3) \
             ON CONFLICT DO NOTHING",
        )
        .bind(entry.version)
        .bind(&entry.hash)
        .bind(&entry.notes)
        .execute(&self.pool)
        .await
        .map_err(backend)?;
        Ok(())
    }

    async fn preregistrations(&self) -> Result<Vec<PreregRecord>, StoreError> {
        let rows: Vec<PreregRow> = sqlx::query_as(
            "SELECT version, hash, committed_at::text, notes FROM prism_prereg \
             ORDER BY version, committed_at",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(backend)?;
        Ok(rows
            .into_iter()
            .map(|(version, hash, committed_at, notes)| PreregRecord {
                version,
                hash,
                committed_at,
                notes,
            })
            .collect())
    }

    async fn insert_metric_report(
        &self,
        submission_id: &str,
        report: &PreparedReport,
    ) -> Result<(), StoreError> {
        let schema = report
            .payload
            .get("schema_version")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let reasons = serde_json::to_value(&report.reasons).map_err(backend)?;
        sqlx::query(
            "INSERT INTO prism_metric_report (submission_id, seq, schema_version, prev_hash, \
             report_hash, payload, verdict, verdict_reasons) \
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
        )
        .bind(submission_id)
        .bind(i64::try_from(report.seq).unwrap_or(i64::MAX))
        .bind(schema)
        .bind(&report.prev_hash)
        .bind(&report.report_hash)
        .bind(&report.payload)
        .bind(verdict_str(report.verdict))
        .bind(reasons)
        .execute(&self.pool)
        .await
        .map_err(backend)?;
        Ok(())
    }

    async fn metric_reports(
        &self,
        submission_id: &str,
    ) -> Result<Vec<MetricReportRecord>, StoreError> {
        let rows: Vec<ReportRow> = sqlx::query_as(&format!(
            "SELECT {REPORT_COLS} FROM prism_metric_report WHERE submission_id = $1 ORDER BY seq"
        ))
        .bind(submission_id)
        .fetch_all(&self.pool)
        .await
        .map_err(backend)?;
        rows.into_iter().map(report_from).collect()
    }

    async fn latest_metric_report(
        &self,
        submission_id: &str,
    ) -> Result<Option<MetricReportRecord>, StoreError> {
        let row: Option<ReportRow> = sqlx::query_as(&format!(
            "SELECT {REPORT_COLS} FROM prism_metric_report WHERE submission_id = $1 \
             ORDER BY seq DESC LIMIT 1"
        ))
        .bind(submission_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(backend)?;
        row.map(report_from).transpose()
    }

    async fn cohort_reports(
        &self,
        exclude_submission: &str,
    ) -> Result<Vec<MetricReportRecord>, StoreError> {
        let rows: Vec<ReportRow> = sqlx::query_as(&format!(
            "SELECT {REPORT_COLS} FROM prism_metric_report WHERE submission_id <> $1 \
             ORDER BY submission_id, seq"
        ))
        .bind(exclude_submission)
        .fetch_all(&self.pool)
        .await
        .map_err(backend)?;
        rows.into_iter().map(report_from).collect()
    }
}
