//! Idempotent screenshot / re-sanitize backfills for historical design runs.

use std::collections::HashMap;
use std::path::Path;

use design_harness::REQUIRED_PAGES;
use design_sanitize::sanitize_bundle;
use design_store::{DesignStore, StorePatch};
use tracing::{info, warn};

use crate::screenshot::{capture_full_page_png, png_artifact_tuple};

/// Outcome counts for one screenshot-only backfill pass.
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

/// Outcome counts for a re-sanitize + force-screenshot pass.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ResanitizeSummary {
    /// Runs examined.
    pub scanned: u32,
    /// Runs whose sanitized HTML lost `<style>` while raw still has it.
    pub candidates: u32,
    /// Runs whose artifacts were rewritten with current sanitize.
    pub resanitized: u32,
    /// Screenshots captured after re-sanitize.
    pub screenshots: u32,
    /// Per-run failures (sanitize reject or screenshot miss).
    pub failed: u32,
    /// Candidates skipped because dry-run or sanitize left styles absent.
    pub skipped: u32,
}

/// True when `raw` still has a `<style` block but `sanitized` does not.
#[must_use]
pub fn style_stripped(raw: &str, sanitized: &str) -> bool {
    let raw_l = raw.to_ascii_lowercase();
    let san_l = sanitized.to_ascii_lowercase();
    raw_l.contains("<style") && !san_l.contains("<style")
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

/// Re-sanitize from stored `raw_html` with the current sanitizer, then force
/// re-capture `index.png`.
///
/// Targets historical runs where an older sanitizer wiped `<style>` (e.g.
/// `scroll-behavior` false-positive). Existing `backfill-screenshots` only
/// re-renders already-sanitized HTML and skips runs that already have a PNG,
/// so it cannot repair those rows.
///
/// When `run_ids` is non-empty, only those runs are considered (newest-first
/// `limit` scan is skipped). Otherwise the newest `limit` runs are scanned
/// and only style-stripped candidates are rewritten.
///
/// # Errors
/// Store errors (per-run sanitize/capture failures are counted in `failed`).
pub async fn backfill_resanitize(
    store: &dyn DesignStore,
    staging_root: &Path,
    limit: u32,
    run_ids: &[String],
    sleep_ms: u64,
    dry_run: bool,
) -> Result<ResanitizeSummary, String> {
    let runs = if run_ids.is_empty() {
        store
            .list_runs(None, limit)
            .await
            .map_err(|e| e.to_string())?
    } else {
        let mut out = Vec::new();
        for id in run_ids {
            if let Some(r) = store.get_run(id).await.map_err(|e| e.to_string())? {
                out.push(r);
            } else {
                warn!(run_id = %id, "resanitize: run not found");
            }
        }
        out
    };

    let mut out = ResanitizeSummary::default();
    for run in &runs {
        out.scanned += 1;
        let artifacts = store
            .list_artifacts_with_raw(&run.id)
            .await
            .map_err(|e| e.to_string())?;
        let html_arts: Vec<_> = artifacts
            .iter()
            .filter(|(path, _, _, _, _)| REQUIRED_PAGES.contains(&path.as_str()))
            .collect();
        if html_arts.len() < REQUIRED_PAGES.len() {
            continue;
        }
        let needs = html_arts
            .iter()
            .any(|(_, san, raw, _, _)| style_stripped(raw, san));
        if !needs {
            continue;
        }
        out.candidates += 1;

        let mut pages = HashMap::new();
        for (path, _, raw, _, _) in &html_arts {
            pages.insert((*path).clone(), (*raw).clone());
        }
        let sanitized = match sanitize_bundle(&pages) {
            Ok(s) => s,
            Err(e) => {
                out.failed += 1;
                warn!(run_id = %run.id, error = %e, "resanitize sanitize failed");
                continue;
            }
        };
        let restored = sanitized
            .pages
            .iter()
            .any(|p| p.sanitized_html.to_ascii_lowercase().contains("<style"));
        if !restored {
            out.skipped += 1;
            warn!(
                run_id = %run.id,
                "resanitize: current sanitizer still strips all style blocks; leaving artifacts"
            );
            continue;
        }
        if dry_run {
            out.skipped += 1;
            info!(run_id = %run.id, "resanitize dry-run candidate");
            continue;
        }

        let tuples: Vec<_> = sanitized
            .pages
            .iter()
            .map(|p| {
                (
                    p.path.clone(),
                    p.sanitized_html.clone(),
                    p.raw_html.clone(),
                    p.raw_sha256.clone(),
                    p.bytes,
                )
            })
            .collect();
        store
            .put_artifacts(&run.id, &tuples)
            .await
            .map_err(|e| e.to_string())?;
        let report = serde_json::to_value(&sanitized.report).unwrap_or_default();
        store
            .apply_run(
                &run.id,
                &StorePatch {
                    artifact_digest: Some(sanitized.artifact_digest.clone()),
                    sanitize_report: Some(report),
                    ..StorePatch::default()
                },
                None,
            )
            .await
            .map_err(|e| e.to_string())?;
        out.resanitized += 1;
        info!(
            run_id = %run.id,
            digest = %sanitized.artifact_digest,
            "resanitized design artifacts from raw_html"
        );

        let Some(index) = sanitized.pages.iter().find(|p| p.path == "index.html") else {
            out.failed += 1;
            continue;
        };
        let dir = staging_root
            .join("screenshots")
            .join(format!("resanitize-{}", run.id));
        let html = index.sanitized_html.clone();
        let png = capture_full_page_png(&html, &dir);
        let _ = std::fs::remove_dir_all(&dir);
        if let Some(bytes) = png {
            store
                .put_artifacts(&run.id, &[png_artifact_tuple(&bytes)])
                .await
                .map_err(|e| e.to_string())?;
            out.screenshots += 1;
            info!(run_id = %run.id, bytes = bytes.len(), "resanitize screenshot captured");
        } else {
            out.failed += 1;
            warn!(run_id = %run.id, "resanitize screenshot capture failed");
        }
        if sleep_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(sleep_ms)).await;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::style_stripped;

    #[test]
    fn detects_style_stripped() {
        assert!(style_stripped(
            "<html><style>.x{color:red}</style><body/></html>",
            "<html><body/></html>"
        ));
        assert!(!style_stripped(
            "<html><style>.x{color:red}</style><body/></html>",
            "<html><style>.x{color:red}</style><body/></html>"
        ));
        assert!(!style_stripped(
            "<html><body/></html>",
            "<html><body/></html>"
        ));
    }
}
