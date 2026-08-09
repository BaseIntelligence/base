//! Full-page screenshots of sanitized design pages for the public site.
//!
//! The capturer renders the sanitized HTML **artifact** (store → temp file →
//! `file://`) in headless Chromium inside the design-challenge container. It
//! never loads the public `/v1/view` URL: view responses carry a locked-down
//! sandbox CSP meant for browser embedding, and the stored artifact is the
//! capture source of truth. `--no-sandbox` is required because Chromium's
//! renderer sandbox needs user namespaces / `CAP_SYS_ADMIN`, which Docker
//! containers do not grant.
//!
//! Network isolation: Chromium shares the design-challenge netns (high-trust
//! `base` network). All `http(s)` subresource / navigation attempts are forced
//! through `design-egress-proxy` (`--proxy-server` + `--proxy-bypass-list=
//! <-loopback>`) so the same internal-target blocklist that guards miner
//! sandboxes also covers screenshot SSRF (gateway admin, metadata, postgres,
//! socket-proxy). Capture documents also carry a nonce-locked CSP that blocks
//! unintended scripts and navigations (CLI Chromium has no Playwright route
//! hooks).

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine;
use sha2::{Digest, Sha256};
use tracing::warn;

/// Capture viewport width in px (matches the site preview column).
const WIDTH: u32 = 1280;
/// Floor for the measured page height in px.
const MIN_HEIGHT: u32 = 480;
/// Height used when the measure pass yields nothing (px).
const FALLBACK_HEIGHT: u32 = 3_200;
/// Default cap on full-page height in px (`DESIGN_SCREENSHOT_MAX_HEIGHT`).
const DEFAULT_MAX_HEIGHT: u32 = 12_000;
/// Default per-process timeout in secs (`DESIGN_SCREENSHOT_TIMEOUT_SECS`).
const DEFAULT_TIMEOUT_SECS: u64 = 90;
/// Virtual-time budget per render so remote images settle (ms).
const VIRTUAL_TIME_BUDGET_MS: u32 = 10_000;
/// Default forward proxy for screenshot Chromium (`DESIGN_SCREENSHOT_PROXY`).
const DEFAULT_SCREENSHOT_PROXY: &str = "http://design-egress-proxy:8094";
/// Marker the height probe writes into `<title>`.
const HEIGHT_MARKER: &str = "SHOTH=";
/// Height probe appended to the throwaway capture document (never shipped).
/// Nonce `designshot1` must match `CAPTURE_CSP` `script-src`.
const MEASURE_SCRIPT: &str = "<script nonce=\"designshot1\">addEventListener('load',function(){setTimeout(function(){var d=document,e=d.documentElement,b=d.body;d.title='SHOTH='+Math.max(e.scrollHeight,b?b.scrollHeight:0)},50)})</script>";
/// Capture-document CSP: nonce script only; block connect/nav; allow public
/// img/font/style so CDN assets still paint (they still traverse the egress
/// proxy blocklist).
const CAPTURE_CSP: &str = "default-src 'none'; \
img-src data: https: http:; \
style-src 'unsafe-inline' data: https: http:; \
font-src data: https: http:; \
script-src 'nonce-designshot1'; \
connect-src 'none'; \
frame-src 'none'; \
object-src 'none'; \
media-src 'none'; \
worker-src 'none'; \
base-uri 'none'; \
form-action 'none'; \
navigate-to 'none'";

/// Capture knobs resolved from env at call time (tests build them directly).
#[derive(Debug, Clone)]
struct CaptureConfig {
    /// Browser binaries to try, in order (`DESIGN_CHROME_BIN` first).
    chrome_bins: Vec<String>,
    /// Per-process timeout.
    timeout: Duration,
    /// Full-page height cap.
    max_height: u32,
    /// Total attempts (initial + retries).
    attempts: u32,
    /// Forward proxy URL (`None` / empty = direct; prod compose always sets
    /// `design-egress-proxy`).
    proxy: Option<String>,
}

impl CaptureConfig {
    fn from_env() -> Self {
        let mut bins: Vec<String> = std::env::var("DESIGN_CHROME_BIN")
            .ok()
            .filter(|b| !b.is_empty())
            .into_iter()
            .collect();
        bins.extend(
            [
                "chromium",
                "chromium-browser",
                "google-chrome",
                "google-chrome-stable",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        Self {
            chrome_bins: bins,
            timeout: Duration::from_secs(env_u64(
                "DESIGN_SCREENSHOT_TIMEOUT_SECS",
                DEFAULT_TIMEOUT_SECS,
            )),
            max_height: env_u32("DESIGN_SCREENSHOT_MAX_HEIGHT", DEFAULT_MAX_HEIGHT),
            attempts: 2,
            proxy: screenshot_proxy_from_env(),
        }
    }
}

/// Resolve `DESIGN_SCREENSHOT_PROXY`. Unset → egress proxy default; empty
/// string → disable (local stub tests / operators debugging without compose).
fn screenshot_proxy_from_env() -> Option<String> {
    match std::env::var("DESIGN_SCREENSHOT_PROXY") {
        Ok(v) if v.is_empty() => None,
        Ok(v) => Some(v),
        Err(_) => Some(DEFAULT_SCREENSHOT_PROXY.to_owned()),
    }
}

/// Capture a full-page PNG of sanitized `html` (best-effort).
///
/// Two Chromium passes per attempt: measure the rendered height (probe script
/// plus `--dump-dom`), then screenshot `WIDTH × height`. Returns `None` when
/// no browser is available or every attempt fails — a run must never fail
/// because its preview is missing.
#[must_use]
pub fn capture_full_page_png(html: &str, work_dir: &Path) -> Option<Vec<u8>> {
    capture_with(&CaptureConfig::from_env(), html, work_dir)
}

/// Artifact row tuple for a captured PNG (`index.png`): base64 payload in the
/// `sanitized_html` column (the viewer base64-decodes it), sha256 of the PNG.
#[must_use]
pub fn png_artifact_tuple(png: &[u8]) -> (String, String, String, String, u32) {
    let mut h = Sha256::new();
    h.update(png);
    (
        "index.png".into(),
        base64::engine::general_purpose::STANDARD.encode(png),
        String::new(),
        hex::encode(h.finalize()),
        u32::try_from(png.len()).unwrap_or(u32::MAX),
    )
}

fn capture_with(cfg: &CaptureConfig, html: &str, work_dir: &Path) -> Option<Vec<u8>> {
    std::fs::create_dir_all(work_dir).ok()?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let html_path = work_dir.join(format!("shot-{stamp}-{}.html", std::process::id()));
    std::fs::write(&html_path, render_doc(html)).ok()?;
    let url = file_url(&html_path);
    let mut out = None;
    for attempt in 1..=cfg.attempts.max(1) {
        if let Some(png) = capture_once(cfg, &url, work_dir, stamp, attempt) {
            out = Some(png);
            break;
        }
        if attempt < cfg.attempts {
            std::thread::sleep(Duration::from_millis(250));
        }
    }
    let _ = std::fs::remove_file(&html_path);
    out
}

fn capture_once(
    cfg: &CaptureConfig,
    url: &str,
    work_dir: &Path,
    stamp: u128,
    attempt: u32,
) -> Option<Vec<u8>> {
    let height = measure_height(cfg, url, work_dir, stamp, attempt)
        .map_or(FALLBACK_HEIGHT, |h| clamp_height(h, cfg.max_height));
    let png_path = work_dir.join(format!("shot-{stamp}-{attempt}.png"));
    let ok = cfg.chrome_bins.iter().any(|bin| {
        shoot(
            bin,
            url,
            &png_path,
            height,
            work_dir,
            stamp,
            attempt,
            cfg.timeout,
            cfg.proxy.as_deref(),
        )
    });
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

fn measure_height(
    cfg: &CaptureConfig,
    url: &str,
    work_dir: &Path,
    stamp: u128,
    attempt: u32,
) -> Option<u32> {
    let bin = cfg.chrome_bins.first()?;
    let profile = profile_dir(work_dir, stamp, attempt, "dom");
    let out = run_with_timeout(
        Command::new(bin)
            .args(base_args(&profile, cfg.proxy.as_deref()))
            .arg("--dump-dom")
            .arg(url),
        cfg.timeout,
    );
    let _ = std::fs::remove_dir_all(&profile);
    let out = out.filter(|o| o.status.success())?;
    parse_height(&String::from_utf8_lossy(&out.stdout))
}

#[allow(clippy::too_many_arguments)]
fn shoot(
    bin: &str,
    url: &str,
    out: &Path,
    height: u32,
    work_dir: &Path,
    stamp: u128,
    attempt: u32,
    timeout: Duration,
    proxy: Option<&str>,
) -> bool {
    let profile = profile_dir(work_dir, stamp, attempt, "png");
    let res = run_with_timeout(
        Command::new(bin)
            .args(base_args(&profile, proxy))
            .arg(format!("--screenshot={}", out.display()))
            .arg(format!("--window-size={WIDTH},{height}"))
            .arg(url),
        timeout,
    );
    let _ = std::fs::remove_dir_all(&profile);
    matches!(res, Some(o) if o.status.success() && out.is_file())
}

/// Shared headless flags. `--no-sandbox`: the renderer sandbox needs userns /
/// `CAP_SYS_ADMIN`, unavailable in Docker — network isolation is the egress
/// proxy + capture CSP below. `--disable-dev-shm-usage`: Docker caps
/// `/dev/shm` at 64MiB, which crashes tall renders.
fn base_args(profile: &Path, proxy: Option<&str>) -> Vec<String> {
    let mut args: Vec<String> = [
        "--headless=new",
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-crash-reporter",
        "--disable-breakpad",
        "--disable-background-networking",
        "--no-first-run",
        "--hide-scrollbars",
        "--force-color-profile=srgb",
        "--run-all-compositor-stages-before-draw",
        "--block-new-web-contents",
    ]
    .into_iter()
    .map(str::to_owned)
    .chain([
        format!("--virtual-time-budget={VIRTUAL_TIME_BUDGET_MS}"),
        format!("--user-data-dir={}", profile.display()),
    ])
    .collect();
    if let Some(p) = proxy.filter(|s| !s.is_empty()) {
        // Route all http(s) — including loopback / link-local — through the
        // design-egress-proxy blocklist. `<-loopback>` removes Chrome's
        // implicit bypass of localhost (and related) targets.
        args.push(format!("--proxy-server={p}"));
        args.push("--proxy-bypass-list=<-loopback>".into());
    }
    args
}

/// Spawn → poll → kill on timeout. `Command::wait_timeout` is unstable, so
/// poll `try_wait` on a short cadence; a hung browser must not wedge a worker.
fn run_with_timeout(cmd: &mut Command, timeout: Duration) -> Option<Output> {
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return child.wait_with_output().ok(),
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(50));
            }
            Ok(None) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

/// Wrap the sanitized fragment (ammonia unwraps the html/head/body shell)
/// into a capture document. The only script is our own height probe (CSP
/// nonce); capture CSP blocks other scripts and navigations.
fn render_doc(fragment: &str) -> String {
    format!(
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">\
<meta http-equiv=\"Content-Security-Policy\" content=\"{CAPTURE_CSP}\">\
<title>shot</title></head><body>{fragment}{MEASURE_SCRIPT}</body></html>"
    )
}

/// Height marker from a `--dump-dom` payload. The probe writes
/// `<title>SHOTH=<px></title>` in `<head>`, ahead of the script's own source.
fn parse_height(dump: &str) -> Option<u32> {
    let start = dump.find(HEIGHT_MARKER)? + HEIGHT_MARKER.len();
    let digits: String = dump[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

fn clamp_height(measured: u32, max_height: u32) -> u32 {
    measured.clamp(MIN_HEIGHT, max_height.max(MIN_HEIGHT))
}

fn profile_dir(work_dir: &Path, stamp: u128, attempt: u32, kind: &str) -> PathBuf {
    // Per-invocation profile: a killed browser leaves a SingletonLock behind,
    // and concurrent workers must never share one profile dir.
    work_dir.join(format!("prof-{stamp}-{attempt}-{kind}"))
}

fn file_url(path: &Path) -> String {
    let abs = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    format!("file://{}", abs.display())
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use std::io::Write;

    const FAKE_PNG: &[u8] = b"\x89PNG\r\n\x1a\nfake-pixels";

    fn test_cfg(bin: &Path) -> CaptureConfig {
        CaptureConfig {
            chrome_bins: vec![bin.to_string_lossy().into_owned()],
            timeout: Duration::from_secs(5),
            max_height: DEFAULT_MAX_HEIGHT,
            attempts: 1,
            // Stub browsers do not need a real proxy; empty env would still
            // default to design-egress-proxy in from_env().
            proxy: None,
        }
    }

    fn write_stub(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("chrome-stub.sh");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        drop(f);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    /// Stub that answers the measure pass and copies `fake.png` on capture.
    fn ok_stub(dir: &Path) -> PathBuf {
        let fake = dir.join("fake.png");
        std::fs::write(&fake, FAKE_PNG).unwrap();
        write_stub(
            dir,
            &format!(
                "#!/bin/sh\nout=\"\"\nfor a in \"$@\"; do case \"$a\" in --screenshot=*) out=\"${{a#--screenshot=}}\";; esac; done\ncase \"$*\" in *--dump-dom*) echo '<title>SHOTH=2400</title>'; exit 0;; esac\nif [ -n \"$out\" ]; then cp \"{}\" \"$out\"; exit 0; fi\nexit 1\n",
                fake.display()
            ),
        )
    }

    #[test]
    fn capture_success_full_pipeline() {
        let dir = std::env::temp_dir().join(format!("shot-ok-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let stub = ok_stub(&dir);
        let png = capture_with(&test_cfg(&stub), "<p>hi</p>", &dir.join("work"));
        assert_eq!(png.as_deref(), Some(FAKE_PNG));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn capture_falls_back_when_marker_missing() {
        let dir = std::env::temp_dir().join(format!("shot-nomarker-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let fake = dir.join("fake.png");
        std::fs::write(&fake, FAKE_PNG).unwrap();
        let stub = write_stub(
            &dir,
            &format!(
                "#!/bin/sh\nout=\"\"\nfor a in \"$@\"; do case \"$a\" in --screenshot=*) out=\"${{a#--screenshot=}}\";; esac; done\ncase \"$*\" in *--dump-dom*) echo '<html></html>'; exit 0;; esac\nif [ -n \"$out\" ]; then cp \"{}\" \"$out\"; exit 0; fi\nexit 1\n",
                fake.display()
            ),
        );
        let png = capture_with(&test_cfg(&stub), "<p>hi</p>", &dir.join("work"));
        assert_eq!(png.as_deref(), Some(FAKE_PNG));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn capture_failure_returns_none() {
        let dir = std::env::temp_dir().join(format!("shot-fail-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let stub = write_stub(&dir, "#!/bin/sh\nexit 1\n");
        let mut cfg = test_cfg(&stub);
        cfg.attempts = 2;
        let png = capture_with(&cfg, "<p>hi</p>", &dir.join("work"));
        assert!(png.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn capture_timeout_kills_browser() {
        let dir = std::env::temp_dir().join(format!("shot-hang-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let stub = write_stub(&dir, "#!/bin/sh\nsleep 30\n");
        let mut cfg = test_cfg(&stub);
        cfg.timeout = Duration::from_secs(1);
        let start = Instant::now();
        let png = capture_with(&cfg, "<p>hi</p>", &dir.join("work"));
        assert!(png.is_none());
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "hung stub must be killed"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn non_png_output_rejected() {
        let dir = std::env::temp_dir().join(format!("shot-notpng-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let stub = write_stub(
            &dir,
            "#!/bin/sh\nout=\"\"\nfor a in \"$@\"; do case \"$a\" in --screenshot=*) out=\"${a#--screenshot=}\";; esac; done\ncase \"$*\" in *--dump-dom*) echo '<title>SHOTH=900</title>'; exit 0;; esac\nif [ -n \"$out\" ]; then echo notapng > \"$out\"; exit 0; fi\nexit 1\n",
        );
        let png = capture_with(&test_cfg(&stub), "<p>hi</p>", &dir.join("work"));
        assert!(png.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_height_variants() {
        assert_eq!(parse_height("<title>SHOTH=2400</title>"), Some(2400));
        assert_eq!(parse_height("junk <title>shot</title>"), None);
        assert_eq!(parse_height("SHOTH="), None);
        assert_eq!(parse_height("SHOTH=12px"), Some(12));
        assert_eq!(parse_height(""), None);
    }

    #[test]
    fn clamp_height_bounds() {
        assert_eq!(clamp_height(10, 12_000), MIN_HEIGHT);
        assert_eq!(clamp_height(99_999, 12_000), 12_000);
        assert_eq!(clamp_height(2_400, 12_000), 2_400);
    }

    #[test]
    fn render_doc_wraps_fragment_with_probe() {
        let doc = render_doc("<main>hello</main>");
        assert!(doc.starts_with("<!DOCTYPE html>"));
        assert!(doc.contains("<main>hello</main>"));
        assert!(doc.contains("SHOTH="));
        assert!(doc.contains("Content-Security-Policy"));
        assert!(doc.contains("nonce-designshot1"));
        assert!(doc.contains("navigate-to 'none'"));
        assert!(doc.contains("nonce=\"designshot1\""));
        assert!(doc.ends_with("</body></html>"));
    }

    #[test]
    fn base_args_force_egress_proxy_including_loopback() {
        let profile = PathBuf::from("/tmp/shot-profile");
        let args = base_args(&profile, Some("http://design-egress-proxy:8094"));
        assert!(args
            .iter()
            .any(|a| a == "--proxy-server=http://design-egress-proxy:8094"));
        assert!(args.iter().any(|a| a == "--proxy-bypass-list=<-loopback>"));
        assert!(args.iter().any(|a| a == "--block-new-web-contents"));
        let direct = base_args(&profile, None);
        assert!(direct.iter().all(|a| !a.starts_with("--proxy-server")));
    }

    #[test]
    fn capture_passes_proxy_flags_to_browser() {
        let dir = std::env::temp_dir().join(format!("shot-proxy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let fake = dir.join("fake.png");
        std::fs::write(&fake, FAKE_PNG).unwrap();
        let args_log = dir.join("args.log");
        let stub = write_stub(
            &dir,
            &format!(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"{log}\"\nout=\"\"\nfor a in \"$@\"; do case \"$a\" in --screenshot=*) out=\"${{a#--screenshot=}}\";; esac; done\ncase \"$*\" in *--dump-dom*) echo '<title>SHOTH=900</title>'; exit 0;; esac\nif [ -n \"$out\" ]; then cp \"{png}\" \"$out\"; exit 0; fi\nexit 1\n",
                log = args_log.display(),
                png = fake.display()
            ),
        );
        let mut cfg = test_cfg(&stub);
        cfg.proxy = Some("http://design-egress-proxy:8094".into());
        let png = capture_with(&cfg, "<p>hi</p>", &dir.join("work"));
        assert_eq!(png.as_deref(), Some(FAKE_PNG));
        let logged = std::fs::read_to_string(&args_log).unwrap();
        assert!(
            logged.contains("--proxy-server=http://design-egress-proxy:8094"),
            "{logged}"
        );
        assert!(
            logged.contains("--proxy-bypass-list=<-loopback>"),
            "{logged}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn png_tuple_sha_and_b64() {
        let (path, b64, raw, sha, bytes) = png_artifact_tuple(FAKE_PNG);
        assert_eq!(path, "index.png");
        assert!(raw.is_empty());
        assert_eq!(bytes as usize, FAKE_PNG.len());
        assert_eq!(sha.len(), 64);
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .unwrap();
        assert_eq!(decoded, FAKE_PNG);
    }
}
