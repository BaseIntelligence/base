//! `prism-attribution` — two `POST /v1/submissions/{id}/*` routes split out
//! of `prism-challenge` for the per-crate non-test LOC cap:
//!
//! - `POST /v1/submissions/{id}/attribution` — planner for the 2×2
//!   attribution matrix off-diagonal runs (submission arch × reference
//!   kernels, reference arch × submission kernels) per
//!   `prism_recipe::attribution`.
//! - `POST /v1/submissions/{id}/zone-b` — external Zone B miner self-report
//!   intake (see [`zone_b`]).
//!
//! Documented choice (E7): the [`prism_recipe::AttributionRun`] plans are
//! RETURNED as JSON, not enqueued — attribution runs are operator-triggered
//! jobs executed through the normal intake, and this route stays a pure
//! planner (no store writes, no pipeline side effects). Auth: the API ships
//! no bearer middleware; like `/anchors` and `/preregistration` this route
//! is operator-facing but unauthenticated at this layer (front it at the
//! gateway where the deployment needs it).
//!
//! Split out of `prism-challenge` for the per-crate non-test LOC cap (the
//! repo's standard crate-split pattern, same as `prism-eval-store`); it
//! mounts into the challenge API router with its own state (the submission
//! store).

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]

mod zone_b;

pub use zone_b::{zone_b_route, zone_b_router, ZoneBState};

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{post, MethodRouter};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use prism_store::PrismStore;

/// Router over the attribution planner surface (own state: the submission
/// store). Standalone mount / offline tests.
pub fn router(store: Arc<dyn PrismStore>) -> Router {
    Router::new()
        .route("/v1/submissions/{id}/attribution", post(post_attribution))
        .with_state(store)
}

/// The attribution planner route with its own state (the submission store)
/// applied — mounts into a parent router of any state (the challenge API
/// uses this for `POST /v1/submissions/{id}/attribution`).
pub fn attribution_route<S>(store: Arc<dyn PrismStore>) -> MethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    post(post_attribution).with_state(store)
}

/// `POST /v1/submissions/{id}/attribution` request body.
#[derive(Deserialize)]
struct AttributionRequest {
    /// Embedded reference baseline (`transformer-pp` | `hybrid-delta`) when
    /// `reference_zip_base64` is absent.
    reference: Option<String>,
    /// Full source-tree ZIP (base64) of the submission. When absent the
    /// store-projected two-script tree is used — attribution then honestly
    /// reports `no_submission_kernels`, because the intake row retains only
    /// the projected seam files (`architecture_py` / `training_py`), never
    /// the full tree.
    submission_zip_base64: Option<String>,
    /// Full source-tree ZIP (base64) overriding the embedded reference. The
    /// embedded v3 baselines ship no `kernels/`; the swap needs a
    /// kernel-carrying reference tree (per `KERNEL_INTERFACE.md`).
    reference_zip_base64: Option<String>,
}

fn decode_zip_b64(b64: &str) -> Result<Vec<u8>, Box<Response>> {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    STANDARD.decode(b64.trim()).map_err(|e| {
        Box::new(json_err(
            StatusCode::BAD_REQUEST,
            "zip_base64",
            &format!("base64: {e}"),
        ))
    })
}

fn tree_from_b64(b64: &str) -> Result<prism_recipe::SourceTree, Box<Response>> {
    let bytes = decode_zip_b64(b64)?;
    prism_recipe::tree_from_zip(&bytes)
        .map_err(|e| Box::new(json_err(StatusCode::BAD_REQUEST, "zip", &e.to_string())))
}

/// Embedded v3 baseline as a source tree (entry `training.py`).
fn baseline_tree(reference: &str) -> Result<prism_recipe::SourceTree, Box<Response>> {
    let files = match reference {
        "transformer-pp" | "transformer_pp" => {
            prism_recipe::baselines::BASELINE_V3_TRANSFORMER_PP_FILES
        }
        "hybrid-delta" | "hybrid_delta" => prism_recipe::baselines::BASELINE_V3_HYBRID_DELTA_FILES,
        other => {
            return Err(Box::new(json_err(
                StatusCode::BAD_REQUEST,
                "unknown_reference",
                &format!(
                    "reference must be transformer-pp | hybrid-delta (or pass reference_zip_base64), got {other:?}"
                ),
            )));
        }
    };
    let map = files
        .iter()
        .map(|(p, c)| ((*p).to_owned(), c.as_bytes().to_vec()))
        .collect();
    Ok(prism_recipe::SourceTree::from_files(
        map,
        "training.py".to_owned(),
    ))
}

/// `POST /v1/submissions/{id}/attribution` — build the 2×2 attribution
/// matrix off-diagonal runs (submission arch × reference kernels, reference
/// arch × submission kernels) per `prism_recipe::attribution`.
async fn post_attribution(
    State(st): State<Arc<dyn PrismStore>>,
    Path(id): Path<String>,
    body: bytes::Bytes,
) -> Response {
    let row = match st.get(&id).await {
        Ok(Some(r)) => r,
        Ok(None) => return json_err(StatusCode::NOT_FOUND, "unknown_submission", &id),
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string()),
    };
    let req: AttributionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return json_err(StatusCode::BAD_REQUEST, "invalid_json", &e.to_string()),
    };

    // Submission tree: a request ZIP validated against the stored seam
    // projections, else the store-projected two-script tree.
    let submission = match req.submission_zip_base64.as_deref() {
        Some(b64) => match tree_from_b64(b64) {
            Ok(tree) => {
                let matches = match (tree.architecture_projection(), tree.training_projection()) {
                    (Ok(a), Ok(t)) => a == row.architecture_py && t == row.training_py,
                    _ => false,
                };
                if !matches {
                    return json_err(
                        StatusCode::CONFLICT,
                        "tree_mismatch",
                        "submission ZIP does not project to the stored architecture.py / training.py",
                    );
                }
                tree
            }
            Err(resp) => return *resp,
        },
        None => prism_recipe::SourceTree::from_files(
            [
                (
                    "architecture.py".to_owned(),
                    row.architecture_py.clone().into_bytes(),
                ),
                (
                    "training.py".to_owned(),
                    row.training_py.clone().into_bytes(),
                ),
            ]
            .into_iter()
            .collect(),
            "training.py".to_owned(),
        ),
    };

    // Reference tree: request ZIP, else an embedded v3 baseline.
    let reference = match req.reference_zip_base64.as_deref() {
        Some(b64) => match tree_from_b64(b64) {
            Ok(tree) => tree,
            Err(resp) => return *resp,
        },
        None => match baseline_tree(req.reference.as_deref().unwrap_or("transformer-pp")) {
            Ok(tree) => tree,
            Err(resp) => return *resp,
        },
    };

    match prism_recipe::build_attribution_runs(&id, &submission, &reference) {
        Ok(runs) => {
            let plans: Vec<Value> = runs
                .iter()
                .map(|r| {
                    json!({
                        "run_id": r.run_id,
                        "parent_submission_id": r.parent_submission_id,
                        "cell": r.cell.label(),
                        "entry": r.tree.entry(),
                        "canonical_sha256": r.tree.canonical_sha256(),
                        "files": r.tree.files().keys().collect::<Vec<_>>(),
                    })
                })
                .collect();
            Json(json!({
                "parent_submission_id": id,
                "enqueued": false,
                "note": "run plans only — execute via the normal intake; swapped cells are gated on the hidden-shape correctness suite before scoring",
                "runs": plans,
            }))
            .into_response()
        }
        Err(e) => {
            let code = match &e {
                prism_recipe::AttributionError::NoSubmissionKernels => "no_submission_kernels",
                prism_recipe::AttributionError::NoReferenceKernels => "no_reference_kernels",
                prism_recipe::AttributionError::InvalidCell(_) => "invalid_cell",
            };
            json_err(StatusCode::UNPROCESSABLE_ENTITY, code, &e.to_string())
        }
    }
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
    use prism_store::{MemoryPrismStore, Stage, SubmissionState};
    use tower::ServiceExt;

    async fn call(app: Router, req: Request<Body>) -> (StatusCode, Value) {
        let res = app.oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let v: Value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or(Value::Null)
        };
        (status, v)
    }

    fn zip_b64(files: &[(&str, &str)]) -> String {
        use base64::Engine;
        use std::io::Write;
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut buf);
            let opts = zip::write::SimpleFileOptions::default();
            for (n, b) in files {
                w.start_file(*n, opts).unwrap();
                w.write_all(b.as_bytes()).unwrap();
            }
            w.finish().unwrap();
        }
        base64::engine::general_purpose::STANDARD.encode(buf.into_inner())
    }

    /// Store seeded with a baseline-contract row (seams = the embedded
    /// baseline sources, so a baseline+`kernels/` ZIP projects cleanly).
    async fn seeded_app() -> (Router, String) {
        let store = Arc::new(MemoryPrismStore::new());
        let id = "attr-sub-1".to_owned();
        store
            .insert_queued(&SubmissionState {
                id: id.clone(),
                miner_hotkey: "ab".repeat(32),
                miner_coldkey: None,
                epoch: 7,
                netuid: 541,
                status: Stage::Queued,
                architecture_py: prism_recipe::BASELINE_ARCHITECTURE_PY.to_owned(),
                training_py: prism_recipe::BASELINE_TRAINING_PY.to_owned(),
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
                created_at_ms: 1,
                updated_at_ms: 1,
            })
            .await
            .unwrap();
        (router(store), id)
    }

    #[tokio::test]
    async fn attribution_route_returns_two_off_diagonal_plans() {
        let (app, id) = seeded_app().await;
        // The submission tree projects to the stored seams (same arch +
        // training bytes) plus a kernels/ dir; the reference tree carries
        // its own kernels implementing the same op set.
        let sub_zip = zip_b64(&[
            ("architecture.py", prism_recipe::BASELINE_ARCHITECTURE_PY),
            ("training.py", prism_recipe::BASELINE_TRAINING_PY),
            ("kernels/rmsnorm.py", "OP = 'rmsnorm'  # submission\n"),
        ]);
        let ref_zip = zip_b64(&[
            ("architecture.py", prism_recipe::BASELINE_ARCHITECTURE_PY),
            ("training.py", prism_recipe::BASELINE_TRAINING_PY),
            ("kernels/rmsnorm.py", "OP = 'rmsnorm'  # reference\n"),
        ]);
        let body = serde_json::json!({
            "submission_zip_base64": sub_zip,
            "reference_zip_base64": ref_zip,
        });
        let (s, v) = call(
            app,
            Request::post(format!("/v1/submissions/{id}/attribution"))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["enqueued"], false);
        let runs = v["runs"].as_array().unwrap();
        assert_eq!(runs.len(), 2, "{v}");
        assert_eq!(runs[0]["cell"], "a_sub_k_ref");
        assert_eq!(runs[1]["cell"], "a_ref_k_sub");
        assert!(
            runs[0]["run_id"].as_str().unwrap().starts_with("attr_"),
            "{v}"
        );
        let files: Vec<&str> = runs[0]["files"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_str().unwrap())
            .collect();
        assert!(files.contains(&"kernels/rmsnorm.py"), "{files:?}");
    }

    #[tokio::test]
    async fn attribution_route_errors_are_honest() {
        // Unknown submission → 404.
        let app = router(Arc::new(MemoryPrismStore::new()));
        let (s, _) = call(
            app,
            Request::post("/v1/submissions/nope/attribution")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);

        let (app, id) = seeded_app().await;
        // Store-projected tree (no kernels/) → 422 no_submission_kernels.
        let (s, v) = call(
            app.clone(),
            Request::post(format!("/v1/submissions/{id}/attribution"))
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
        assert_eq!(v["code"], "no_submission_kernels");

        // A kernel-carrying submission against an embedded baseline →
        // 422 no_reference_kernels (the v3 baselines ship no kernels/).
        let sub_zip = zip_b64(&[
            ("architecture.py", prism_recipe::BASELINE_ARCHITECTURE_PY),
            ("training.py", prism_recipe::BASELINE_TRAINING_PY),
            ("kernels/rmsnorm.py", "OP = 'rmsnorm'\n"),
        ]);
        let body = serde_json::json!({
            "submission_zip_base64": sub_zip,
            "reference": "transformer-pp",
        });
        let (s, v) = call(
            app.clone(),
            Request::post(format!("/v1/submissions/{id}/attribution"))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
        assert_eq!(v["code"], "no_reference_kernels");

        // A ZIP that does not project to the stored seams → 409.
        let other_zip = zip_b64(&[
            (
                "architecture.py",
                "import torch\ndef build_model(ctx):\n    return torch.nn.Linear(4, 4)\n",
            ),
            ("training.py", prism_recipe::BASELINE_TRAINING_PY),
        ]);
        let body = serde_json::json!({
            "submission_zip_base64": other_zip,
            "reference": "hybrid-delta",
        });
        let (s, v) = call(
            app,
            Request::post(format!("/v1/submissions/{id}/attribution"))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::CONFLICT, "{v}");
        assert_eq!(v["code"], "tree_mismatch");
    }
}
