//! `AutoModel` patch diff view over persisted submission trees.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, MethodRouter};
use axum::Json;
use prism_store::PrismStore;
use serde_json::{json, Value};

/// `GET /v1/submissions/{id}/diff` with its narrow store state applied.
pub fn diff_route<S>(store: Arc<dyn PrismStore>) -> MethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    get(get_submission_diff).with_state(store)
}

async fn get_submission_diff(
    State(store): State<Arc<dyn PrismStore>>,
    Path(id): Path<String>,
) -> Response {
    let row = match store.get(&id).await {
        Ok(Some(row)) => row,
        Ok(None) => return json_err(StatusCode::NOT_FOUND, "not_found", "submission not found"),
        Err(error) => {
            return json_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "store",
                &error.to_string(),
            );
        }
    };
    let Some(blob) = row.tree_blob.as_deref() else {
        return json_err(
            StatusCode::NOT_FOUND,
            "no_diff",
            "submission has no AutoModel diff artifacts",
        );
    };
    let staged = match prism_tree::StagedTree::unpack(blob) {
        Ok(tree) => tree,
        Err(error) => {
            return json_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "tree",
                &error.to_string(),
            );
        }
    };
    let Some((patch, diffstat)) = prism_automodel::diff_from_files(staged.files()) else {
        return json_err(
            StatusCode::NOT_FOUND,
            "no_diff",
            "submission tree lacks .prism/automodel.patch",
        );
    };
    let pin_id = staged
        .get(prism_automodel::META_BASE)
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .map_or("", str::trim);
    let files: Vec<Value> = diffstat
        .files
        .iter()
        .map(|file| {
            json!({
                "path": file.path,
                "added": file.added,
                "deleted": file.deleted,
                "class": file.class,
            })
        })
        .collect();
    Json(json!({
        "submission_id": id,
        "pin_id": pin_id,
        "patch": patch,
        "diffstat": {
            "files": files,
            "total_added": diffstat.total_added,
            "total_deleted": diffstat.total_deleted,
        },
    }))
    .into_response()
}

fn json_err(status: StatusCode, code: &str, message: &str) -> Response {
    (status, Json(json!({"error": message, "code": code}))).into_response()
}
