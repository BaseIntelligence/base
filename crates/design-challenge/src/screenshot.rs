//! Full-page screenshots of sanitized design pages for the public site.

use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::warn;

/// Capture a full-page PNG of `html` (best-effort).
///
/// Prefers the Playwright CLI (`playwright screenshot --full-page`), then
/// falls back to headless Chrome viewport capture. Returns `None` when no
/// browser tool is available or capture fails — runs must not fail for this.
pub fn capture_full_page_png(html: &str, work_dir: &Path) -> Option<Vec<u8>> {
    let _ = std::fs::create_dir_all(work_dir);
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let html_path = work_dir.join(format!("shot-{stamp}.html"));
    let png_path = work_dir.join(format!("shot-{stamp}.png"));
    if std::fs::write(&html_path, html).is_err() {
        return None;
    }
    let file_url = path_to_file_url(&html_path);
    let ok = try_playwright(&file_url, &png_path) || try_chrome(&file_url, &png_path);
    let _ = std::fs::remove_file(&html_path);
    if !ok {
        let _ = std::fs::remove_file(&png_path);
        return None;
    }
    let bytes = std::fs::read(&png_path).ok();
    let _ = std::fs::remove_file(&png_path);
    match bytes {
        Some(b) if !b.is_empty() && b.starts_with(b"\x89PNG") => Some(b),
        Some(_) => {
            warn!("screenshot tool wrote non-PNG output");
            None
        }
        None => None,
    }
}

fn path_to_file_url(path: &Path) -> String {
    let abs = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    format!("file://{}", abs.display())
}

fn try_playwright(url: &str, out: &Path) -> bool {
    let bin = std::env::var("DESIGN_PLAYWRIGHT_BIN").unwrap_or_else(|_| "playwright".into());
    let status = Command::new(&bin)
        .args(["screenshot", "--full-page", url])
        .arg(out)
        .status();
    matches!(status, Ok(s) if s.success() && out.is_file())
}

fn try_chrome(url: &str, out: &Path) -> bool {
    let candidates = [
        std::env::var("DESIGN_CHROME_BIN").unwrap_or_default(),
        "google-chrome".into(),
        "google-chrome-stable".into(),
        "chromium".into(),
        "chromium-browser".into(),
    ];
    for bin in candidates.into_iter().filter(|b| !b.is_empty()) {
        // Chrome `--screenshot` is viewport-sized; still better than no preview.
        let status = Command::new(&bin)
            .args([
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--window-size=1280,4000",
                &format!("--screenshot={}", out.display()),
                url,
            ])
            .status();
        if matches!(status, Ok(s) if s.success() && out.is_file()) {
            return true;
        }
    }
    false
}
