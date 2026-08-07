//! Idempotent backfill of `index.png` screenshots for runs whose capture is
//! missing (predates working capture, or the browser failed at run time).

use std::path::Path;

use design_store::DesignStore;
use tracing::{info, warn};

use crate::screenshot::{capture_full_page_png, png_artifact_tuple};

/// Outcome counts for one backfill pass.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BackfillSummary {
    /// Runs examined.
    pub scanned: u32,
    /// Runs with `index.html` but no `index.png`.
    pub missing: u32,
    /// Screenshots captured and stored.
    pub captured: u32,
    /// Captures that failed (run left without a screenshot).
    pub failed: u32,
}

/// (Re)capture screenshots for the newest `limit` runs that have sanitized
/// pages but no `index.png` artifact.
///
/// Idempotent: runs that already carry a screenshot are skipped, so operators
/// can re-run until `failed == 0`. Artifact inserts upsert on
/// `(run_id, path)`, so a live capture racing this pass cannot error either
/// side.
///
/// # Errors
/// Store errors (per-run browser failures are counted in `failed`).
pub async fn backfill_screenshots(
    store: &dyn DesignStore,
    staging_root: &Path,
    limit: u32,
) -> Result<BackfillSummary, String> {
    let runs = store
        .list_runs(None, limit)
        .await
        .map_err(|e| e.to_string())?;
    let mut out = BackfillSummary::default();
    for run in &runs {
        out.scanned += 1;
        let pages = store.list_pages(&run.id).await.map_err(|e| e.to_string())?;
        if pages.iter().any(|p| p.path == "index.png")
            || !pages.iter().any(|p| p.path == "index.html")
        {
            continue;
        }
        out.missing += 1;
        let Some(html) = store
            .get_page(&run.id, "index.html")
            .await
            .map_err(|e| e.to_string())?
        else {
            warn!(run_id = %run.id, "index.html vanished during backfill");
            continue;
        };
        let dir = staging_root
            .join("screenshots")
            .join(format!("backfill-{}", run.id));
        let png = capture_full_page_png(&html, &dir);
        let _ = std::fs::remove_dir_all(&dir);
        if let Some(bytes) = png {
            store
                .put_artifacts(&run.id, &[png_artifact_tuple(&bytes)])
                .await
                .map_err(|e| e.to_string())?;
            out.captured += 1;
            info!(run_id = %run.id, bytes = bytes.len(), "backfilled design screenshot");
        } else {
            out.failed += 1;
            warn!(run_id = %run.id, "backfill capture failed");
        }
    }
    Ok(out)
}
