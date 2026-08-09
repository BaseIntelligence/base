//! `POST /v1/submissions/{id}/zone-b` — the external Zone B self-report
//! intake promised by `docs/external-miner/prism.md`.
//!
//! A miner posts a [`ZoneBEnvelope`] (`schema_version`, optional
//! miner-chained `prev_hash`, `miner.<group>.<name>` metrics) for their own
//! submission. The route runs the exact validation lattice of the internal
//! `train_metrics` lift — `prism_pipeline::zone_b::prepare_report` with the
//! stored chain head, organizer ground truth from the submission's
//! `METRICS_JSON` v2 blob, and the cross-miner cohort — then persists the
//! prepared report to `zone_b_reports` via [`EvalStore`]. Verdicts
//! (`ok`/`flagged`/`quarantined`) are stored, never auto-zero; hard
//! malformed/over-cap envelopes reject with 422 and store nothing.
//!
//! Auth (documented choice, same as `/v1/anchors`, `/v1/preregistration`,
//! and the sibling attribution route): the API ships no bearer middleware
//! at this layer — front the route at the gateway where the deployment
//! needs it. Writes stay safe without one: the report is chained
//! per-submission, validated against organizer ground truth, and a forged
//! or poisoned entry lands as quarantine evidence on the poster's target,
//! never silently scored.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{post, MethodRouter};
use axum::{Json, Router};
use serde_json::json;

use prism_eval_store::{ground_truth, prepare_cohort};
use prism_pipeline::zone_b::{prepare_report, GroundTruth, IngestReject, ZoneBEnvelope};
use prism_store::eval::EvalStore;
use prism_store::PrismStore;

/// Zone B intake state: the submission store (existence + ground truth)
/// and the eval store (report chain).
#[derive(Debug, Clone)]
pub struct ZoneBState {
    /// Submission rows.
    pub submissions: Arc<dyn PrismStore>,
    /// Zone B report chain store.
    pub eval: Arc<dyn EvalStore>,
}

/// Standalone router (offline tests / alternate mounts).
pub fn zone_b_router(st: ZoneBState) -> Router {
    Router::new()
        .route("/v1/submissions/{id}/zone-b", post(post_zone_b))
        .with_state(st)
}

/// The Zone B POST route with its own state applied — mounts into a parent
/// router of any state (the service bin uses this).
pub fn zone_b_route<S>(
    submissions: Arc<dyn PrismStore>,
    eval: Arc<dyn EvalStore>,
) -> MethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    post(post_zone_b).with_state(ZoneBState { submissions, eval })
}

/// `POST /v1/submissions/{id}/zone-b` — validate + chain + persist one
/// miner self-report; the response carries the stored seq/hashes/verdict.
async fn post_zone_b(
    State(st): State<ZoneBState>,
    Path(id): Path<String>,
    body: bytes::Bytes,
) -> Response {
    let row = match st.submissions.get(&id).await {
        Ok(Some(r)) => r,
        Ok(None) => return json_err(StatusCode::NOT_FOUND, "unknown_submission", &id),
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    let env: ZoneBEnvelope = match serde_json::from_slice(&body) {
        Ok(e) => e,
        Err(e) => return json_err(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };
    // Organizer facts come from the run's own METRICS_JSON v2; without it
    // (queued/running rows) only the recipe step cap is known — the
    // ground-truth checks degrade to `flagged`, never crash.
    let gt = row.metrics_json.as_ref().map_or_else(
        || GroundTruth {
            max_steps: u64::from(prism_recipe::MAX_TRAIN_STEPS),
            ..GroundTruth::default()
        },
        ground_truth,
    );
    let head = match st.eval.latest_metric_report(&id).await {
        Ok(r) => r.map(|r| (u64::try_from(r.seq).unwrap_or(0), r.report_hash)),
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    let cohort = match st.eval.cohort_reports(&id).await {
        Ok(reports) => prepare_cohort(&reports),
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    let prepared = match prepare_report(&id, &env, head, &gt, &cohort) {
        Ok(p) => p,
        Err(reject) => {
            let code = match &reject {
                IngestReject::PayloadTooLarge => "payload_too_large",
                IngestReject::TooManyScalars(_) => "too_many_scalars",
                IngestReject::TooManySeries(_) => "too_many_series",
                IngestReject::SeriesTooLong(_) => "series_too_long",
                IngestReject::InvalidName(_) => "invalid_metric_name",
                IngestReject::BadEnvelope(_) => "bad_envelope",
            };
            return json_err(StatusCode::UNPROCESSABLE_ENTITY, code, &reject.to_string());
        }
    };
    if let Err(e) = st.eval.insert_metric_report(&id, &prepared).await {
        return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string());
    }
    (
        StatusCode::OK,
        Json(json!({
            "submission_id": id,
            "seq": prepared.seq,
            "prev_hash": prepared.prev_hash,
            "report_hash": prepared.report_hash,
            "verdict": prepared.verdict,
            "reasons": prepared.reasons,
        })),
    )
        .into_response()
}

fn json_err(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(json!({
            "error": message,
            "code": code,
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use prism_eval_store::MemoryEvalStore;
    use prism_store::{MemoryPrismStore, Stage, SubmissionState};
    use serde_json::Value;
    use tower::ServiceExt;

    async fn app_with_sub(id: &str, metrics_json: Option<Value>) -> Router {
        let submissions = Arc::new(MemoryPrismStore::new());
        let eval = Arc::new(MemoryEvalStore::new());
        submissions
            .insert_queued(&SubmissionState {
                id: id.to_owned(),
                miner_hotkey: "ab".repeat(32),
                miner_coldkey: None,
                epoch: 7,
                netuid: 541,
                status: Stage::Queued,
                architecture_py: String::new(),
                training_py: String::new(),
                tree_blob: None,
                label: None,
                pod_id: None,
                pod_provider: None,
                receipt: None,
                metrics_json,
                bpb: None,
                arch_id: None,
                review: None,
                similarity: None,
                final_score: None,
                retry_count: 0,
                error_detail: None,
                created_at_ms: 1,
                updated_at_ms: 1,
            })
            .await
            .unwrap();
        zone_b_router(ZoneBState { submissions, eval })
    }

    async fn call(app: Router, req: Request<Body>) -> (StatusCode, Value) {
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    fn post(id: &str, body: &Value) -> Request<Body> {
        Request::post(format!("/v1/submissions/{id}/zone-b"))
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(body).unwrap()))
            .unwrap()
    }

    fn scalar_report(v: f64) -> Value {
        json!({
            "schema_version": "1.3.0",
            "metrics": { "miner.train.loss": { "kind": "scalar", "value": v } },
        })
    }

    #[tokio::test]
    async fn zone_b_route_validates_chains_and_persists() {
        let app = app_with_sub("zb-1", None).await;
        // Unknown submission → 404.
        let (s, v) = call(app.clone(), post("nope", &scalar_report(1.0))).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert_eq!(v["code"], "unknown_submission");
        // Malformed JSON → 400.
        let bad = Request::post("/v1/submissions/zb-1/zone-b")
            .header("content-type", "application/json")
            .body(Body::from("{nope"))
            .unwrap();
        let (s, _) = call(app.clone(), bad).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        // Invalid metric name → 422, nothing stored.
        let bad_name = json!({
            "schema_version": "1.3.0",
            "metrics": { "train_steps": { "kind": "scalar", "value": 1.0 } },
        });
        let (s, v) = call(app.clone(), post("zb-1", &bad_name)).await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
        assert_eq!(v["code"], "invalid_metric_name");

        // Clean report → 200 ok, chained to the submission id at seq 0.
        let (s, v) = call(app.clone(), post("zb-1", &scalar_report(2.0))).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["seq"], 0);
        assert_eq!(v["prev_hash"], "zb-1");
        assert_eq!(v["verdict"], "ok");
        // Second report chains to the first report's hash.
        let (s, v2) = call(app.clone(), post("zb-1", &scalar_report(1.5))).await;
        assert_eq!(s, StatusCode::OK, "{v2}");
        assert_eq!(v2["seq"], 1);
        assert_eq!(v2["prev_hash"], v["report_hash"]);

        // Miner-emitted org.* → quarantine verdict, key stripped.
        let forged = json!({
            "schema_version": "1.3.0",
            "metrics": {
                "miner.train.loss": { "kind": "scalar", "value": 1.4 },
                "org.g1.bpb_code": { "kind": "scalar", "value": 0.01 },
            },
        });
        let (s, v) = call(app.clone(), post("zb-1", &forged)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["verdict"], "quarantined");
        assert!(v["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["reason"] == "org_namespace"));
    }

    #[tokio::test]
    async fn zone_b_route_uses_organizer_ground_truth() {
        // With a stored METRICS_JSON v2 the token-accounting check runs for
        // real: claims above the harness counter quarantine.
        let blob = json!({
            "bpb": 2.0,
            "tokens_seen": 1_000_000_u64,
            "tokens_seen_source": "train_stream",
            "wall_clock_seconds": 100.0,
            "n_params": 300_000_000_u64,
            "gpu_type": "NVIDIA H100",
        });
        let app = app_with_sub("zb-2", Some(blob)).await;
        let lying = json!({
            "schema_version": "1.3.0",
            "metrics": { "miner.train.tokens_seen": { "kind": "scalar", "value": 2_000_000.0 } },
        });
        let (s, v) = call(app.clone(), post("zb-2", &lying)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["verdict"], "quarantined");
        assert!(v["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["reason"] == "token_count_exceeds_observed"));
        // Honest claim → ok.
        let honest = json!({
            "schema_version": "1.3.0",
            "metrics": { "miner.train.tokens_seen": { "kind": "scalar", "value": 900_000.0 } },
        });
        let (s, v) = call(app, post("zb-2", &honest)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["verdict"], "ok");
    }
}
