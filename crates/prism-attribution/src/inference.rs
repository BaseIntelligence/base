//! `GET /v1/submissions/{id}/inference` — per-item battery traces + optional
//! playground journal.
//!
//! Traces are additive observation from `METRICS_JSON` v2 `inference_traces`
//! (prompt / choices / gold / selected / choice logprobs / generated text).
//! Scoring never reads this surface. Auth matches `/metrics`: public at this
//! layer (competition transparency of organizer-measured eval); front at the
//! gateway where the deployment needs it.
//!
//! Fail-closed size: default page `limit=50` (max 200). Missing / empty
//! traces → empty page (200), not 404. Unknown submission → 404.

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, MethodRouter};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use prism_store::PrismStore;

use crate::json_err;

const DEFAULT_LIMIT: usize = 50;
const MAX_LIMIT: usize = 200;
const MAX_PLAYGROUND_TAIL: usize = 100;

/// Narrow state for the inference-traces route.
#[derive(Clone)]
pub struct InferenceState {
    /// Submission store (`metrics_json` + existence).
    pub store: Arc<dyn PrismStore>,
}

/// Mountable `GET /v1/submissions/{id}/inference`.
pub fn inference_route<S>(store: Arc<dyn PrismStore>) -> MethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    get(get_inference).with_state(InferenceState { store })
}

#[derive(Debug, Deserialize)]
struct InferenceQuery {
    /// Page offset into battery `items` (default 0).
    offset: Option<usize>,
    /// Page size (default 50, max 200).
    limit: Option<usize>,
    /// Optional group filter (`g2` / `g3` / …).
    group: Option<String>,
    /// `battery` (default) | `playground` | `all`.
    source: Option<String>,
}

async fn get_inference(
    State(st): State<InferenceState>,
    Path(id): Path<String>,
    Query(q): Query<InferenceQuery>,
) -> Response {
    let row = match st.store.get(&id).await {
        Ok(Some(r)) => r,
        Ok(None) => {
            return json_err(StatusCode::NOT_FOUND, "not_found", "submission not found");
        }
        Err(e) => {
            return json_err(StatusCode::INTERNAL_SERVER_ERROR, "store", &e.to_string());
        }
    };
    let offset = q.offset.unwrap_or(0);
    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let source = q.source.as_deref().unwrap_or("battery");
    let traces = row
        .metrics_json
        .as_ref()
        .and_then(|m| m.get("inference_traces"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let mut items: Vec<Value> = traces
        .get("items")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if let Some(g) = q.group.as_deref() {
        items.retain(|it| it.get("group").and_then(|v| v.as_str()) == Some(g));
    }
    let total = items.len();
    let page: Vec<Value> = items.into_iter().skip(offset).take(limit).collect();
    let playground = if matches!(source, "playground" | "all") {
        read_playground_journal(&id)
    } else {
        Vec::new()
    };
    let battery_block = if matches!(source, "battery" | "all") {
        json!({
            "truncated": traces
                .get("truncated")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            "caps": traces.get("caps").cloned().unwrap_or(json!({})),
            "n_items": traces.get("n_items").cloned().unwrap_or(json!(total)),
            "bytes_approx": traces.get("bytes_approx").cloned().unwrap_or(json!(0)),
            "offset": offset,
            "limit": limit,
            "total": total,
            "items": page,
        })
    } else {
        json!(null)
    };
    Json(json!({
        "submission_id": id,
        "provenance": "organizer-measured",
        "source": source,
        "battery": battery_block,
        "playground": playground,
    }))
    .into_response()
}

fn playground_journal_path(submission_id: &str) -> PathBuf {
    // Keep path construction local so this crate does not depend on
    // prism-artifacts (loc + dep surface). Mirrors artifact_dir_for layout.
    let root = std::env::var("PRISM_ARTIFACT_DIR")
        .map_or_else(|_| PathBuf::from("/var/lib/prism/artifacts"), PathBuf::from);
    root.join(submission_id).join("playground_journal.jsonl")
}

fn read_playground_journal(submission_id: &str) -> Vec<Value> {
    let path = playground_journal_path(submission_id);
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut rows: Vec<Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    if rows.len() > MAX_PLAYGROUND_TAIL {
        rows = rows.split_off(rows.len() - MAX_PLAYGROUND_TAIL);
    }
    rows
}
