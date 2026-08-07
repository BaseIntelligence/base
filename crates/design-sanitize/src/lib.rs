//! HTML/CSS sanitization, page-bundle validation, viewer response headers.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::implicit_hasher)]
#![allow(clippy::doc_markdown)]

use std::collections::HashMap;

use design_harness::REQUIRED_PAGES;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Sanitize report surfaced to annotators.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SanitizeReport {
    /// Whether any script-like content was stripped.
    pub js_stripped: bool,
    /// Whether dangerous CSS was stripped.
    pub css_stripped: bool,
    /// Notes.
    pub notes: Vec<String>,
}

/// Bundle validation / sanitize errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SanitizeError {
    /// Missing required page.
    #[error("missing page: {0}")]
    MissingPage(String),
    /// Invalid manifest / layout.
    #[error("invalid bundle: {0}")]
    Invalid(String),
}

/// One sanitized page.
#[derive(Debug, Clone)]
pub struct SanitizedPage {
    /// Path (`index.html`, …).
    pub path: String,
    /// Sanitized HTML.
    pub sanitized_html: String,
    /// Original HTML (audit only).
    pub raw_html: String,
    /// Raw sha256 hex.
    pub raw_sha256: String,
    /// Raw byte length.
    pub bytes: u32,
}

/// Full sanitize result.
#[derive(Debug, Clone)]
pub struct SanitizeResult {
    /// Pages.
    pub pages: Vec<SanitizedPage>,
    /// Report.
    pub report: SanitizeReport,
    /// Artifact digest over sanitized pages.
    pub artifact_digest: String,
}

fn ammonia_builder() -> ammonia::Builder<'static> {
    let mut b = ammonia::Builder::default();
    // Drop scriptable / navigational sinks; ammonia also strips on* handlers.
    let _ = b.rm_tags([
        "script", "iframe", "object", "embed", "applet", "base", "form", "link",
    ]);
    // Landing-page structure + CSS. Ammonia fragment parsing still unwraps
    // html/head/body wrappers (see `rewrite_body_wrapper`); allow them anyway
    // and keep presentation tags/attrs reviewers need.
    let _ = b.add_tags([
        "style",
        "main",
        "section",
        "body",
        "html",
        "head",
        "nav",
        "header",
        "footer",
        "article",
        "aside",
        "figure",
        "figcaption",
    ]);
    // Default clean_content_tags includes `style` (content stripped). Keep the
    // tag and its CSS after our `filter_css` pre-pass.
    let _ = b.rm_clean_content_tags(["style"]);
    let _ = b.add_generic_attributes(["class", "id", "style"]);
    let schemes: std::collections::HashSet<&'static str> =
        ["http", "https", "mailto"].into_iter().collect();
    let _ = b.url_schemes(schemes);
    b
}

/// Ammonia parses as an HTML fragment and unwraps `<body>` / `<html>` / `<head>`,
/// dropping their attributes. Rewrite `<body …>` to a presentation `<div>` so
/// body-level `class` / `id` / `style` survive for reviewers.
fn rewrite_body_wrapper(html: &str) -> String {
    let Ok(re_open) = Regex::new(r"(?i)<body(\s[^>]*)?>") else {
        return html.to_owned();
    };
    let Ok(re_close) = Regex::new(r"(?i)</body\s*>") else {
        return html.to_owned();
    };
    let with_open = re_open.replace_all(html, |caps: &regex::Captures<'_>| {
        let attrs = caps.get(1).map_or("", |m| m.as_str());
        // Marker class (ammonia strips unknown data-* attrs by default).
        if attrs.to_ascii_lowercase().contains("class=") {
            format!("<div{attrs}>")
        } else {
            format!("<div class=\"design-body\"{attrs}>")
        }
    });
    re_close.replace_all(&with_open, "</div>").into_owned()
}

/// True when `css` (already lowercased) contains the IE-only `behavior:`
/// property, not a suffix like `scroll-behavior:`.
fn has_ie_behavior_property(lower: &str) -> bool {
    let mut start = 0;
    while let Some(rel) = lower[start..].find("behavior:") {
        let abs = start + rel;
        if abs == 0 {
            return true;
        }
        let prev = lower.as_bytes()[abs - 1];
        // Property names may include `-` / `_` / alphanumerics (`scroll-behavior`).
        if prev.is_ascii_alphanumeric() || prev == b'-' || prev == b'_' {
            start = abs + "behavior:".len();
            continue;
        }
        return true;
    }
    false
}

/// Filter dangerous CSS constructs from a style attribute / block.
///
/// Soft-strips `@import` rules (keeps the rest of the block). Hard-nukes the
/// whole input for XSS-prone constructs (`expression(`, `url(javascript:`,
/// IE `behavior:`, `-moz-binding`). `scroll-behavior:` is presentation CSS
/// and must not match the IE `behavior:` check.
#[must_use]
pub fn filter_css(css: &str) -> (String, bool) {
    let mut stripped = false;
    let mut out = css.to_owned();
    if out.to_ascii_lowercase().contains("@import") {
        if let Ok(re) = Regex::new(r"(?i)@import[^;]*;?") {
            let next = re.replace_all(&out, "");
            if next.as_ref() != out {
                stripped = true;
                out = next.into_owned();
            }
        }
    }
    let lower = out.to_ascii_lowercase();
    let hard_bad = lower.contains("expression(")
        || lower.contains("url(javascript:")
        || has_ie_behavior_property(&lower)
        || lower.contains("-moz-binding");
    if hard_bad {
        (String::new(), true)
    } else {
        (out, stripped)
    }
}

/// Sanitize one HTML document.
#[must_use]
pub fn sanitize_html(raw: &str) -> (String, SanitizeReport) {
    let mut notes = Vec::new();
    let mut js_stripped = false;
    let mut css_stripped = false;

    let lower = raw.to_ascii_lowercase();
    if lower.contains("<script")
        || lower.contains("javascript:")
        || Regex::new(r"(?i)\son\w+\s*=")
            .ok()
            .is_some_and(|re| re.is_match(raw))
    {
        js_stripped = true;
        notes.push("script_or_handler_present".into());
    }
    if lower.contains("http-equiv") && lower.contains("refresh") {
        js_stripped = true;
        notes.push("meta_refresh".into());
    }

    // Pre-filter style blocks: soft-strip `@import`, hard-nuke XSS CSS.
    let mut pre = rewrite_body_wrapper(raw);
    if let Ok(re) = Regex::new(r"(?is)<style[^>]*>(.*?)</style>") {
        let mut stripped = false;
        let mut css_notes = Vec::new();
        pre = re
            .replace_all(&pre, |caps: &regex::Captures<'_>| {
                let original = &caps[1];
                let (filtered, touched) = filter_css(original);
                if touched {
                    stripped = true;
                    css_notes.push("css_blocked".into());
                }
                // Hard-bad → empty filtered; soft-strip keeps remaining rules.
                if filtered.is_empty() && !original.is_empty() {
                    String::new()
                } else {
                    format!("<style>{filtered}</style>")
                }
            })
            .into_owned();
        css_stripped |= stripped;
        notes.extend(css_notes);
    }

    let cleaned = ammonia_builder().clean(&pre).to_string();
    if cleaned.len() < pre.len() {
        // ammonia removed something — treat as js strip signal when tags differ.
        if pre.to_ascii_lowercase().contains("<script")
            || pre.to_ascii_lowercase().contains("<iframe")
        {
            js_stripped = true;
        }
    }

    // Inline `style="…"` attrs are allowed; re-filter dangerous CSS constructs
    // ammonia does not strip by default.
    let (cleaned, inline_css_stripped) = filter_inline_styles(&cleaned);
    css_stripped |= inline_css_stripped;
    if inline_css_stripped {
        notes.push("css_blocked_inline".into());
    }

    (
        cleaned,
        SanitizeReport {
            js_stripped,
            css_stripped,
            notes,
        },
    )
}

/// Drop or empty inline `style` attributes that fail [`filter_css`].
fn filter_inline_styles(html: &str) -> (String, bool) {
    let Ok(re) = Regex::new(r#"(?i)\sstyle\s*=\s*("([^"]*)"|'([^']*)')"#) else {
        return (html.to_owned(), false);
    };
    let mut stripped = false;
    let out = re
        .replace_all(html, |caps: &regex::Captures<'_>| {
            let val = caps
                .get(2)
                .or_else(|| caps.get(3))
                .map_or("", |m| m.as_str());
            let (filtered, touched) = filter_css(val);
            if filtered.is_empty() && !val.is_empty() {
                stripped = true;
                String::new()
            } else if filtered == val {
                caps[0].to_owned()
            } else {
                stripped |= touched;
                format!(" style=\"{filtered}\"")
            }
        })
        .into_owned();
    (out, stripped)
}

/// Validate required pages exist and sanitize them.
pub fn sanitize_bundle(pages: &HashMap<String, String>) -> Result<SanitizeResult, SanitizeError> {
    for req in REQUIRED_PAGES {
        if !pages.contains_key(*req) {
            return Err(SanitizeError::MissingPage((*req).into()));
        }
    }
    let mut out = Vec::new();
    let mut report = SanitizeReport {
        js_stripped: false,
        css_stripped: false,
        notes: vec![],
    };
    let mut digest = Sha256::new();
    for req in REQUIRED_PAGES {
        let raw = pages.get(*req).cloned().unwrap_or_default();
        let (sanitized, r) = sanitize_html(&raw);
        report.js_stripped |= r.js_stripped;
        report.css_stripped |= r.css_stripped;
        report.notes.extend(r.notes);
        let mut h = Sha256::new();
        h.update(raw.as_bytes());
        let raw_sha256 = hex::encode(h.finalize());
        digest.update(req.as_bytes());
        digest.update(sanitized.as_bytes());
        out.push(SanitizedPage {
            path: (*req).into(),
            sanitized_html: sanitized,
            raw_html: raw.clone(),
            raw_sha256,
            bytes: u32::try_from(raw.len()).unwrap_or(u32::MAX),
        });
    }
    Ok(SanitizeResult {
        pages: out,
        report,
        artifact_digest: hex::encode(digest.finalize()),
    })
}

/// Default `frame-ancestors` allowlist for the viewer CSP: the public site,
/// Vercel preview deploys (staging frontend), and local dev servers. View
/// pages are public capability URLs with no cookies or session state, so
/// framing risk is clickjacking-only; the CSP `sandbox` (opaque origin, no
/// scripts) is the primary XSS control.
#[must_use]
pub fn default_frame_ancestors() -> &'static str {
    "'self' https://joinbase.ai https://*.vercel.app http://localhost:*"
}

/// Viewer response headers for **non-PNG** `/v1/view/*` responses (CSP sandbox
/// is the key guarantee against a stale upstream that still served miner HTML).
///
/// The `sandbox` directive is emitted **without** `allow-scripts` and without
/// `allow-same-origin`: the document runs in an opaque origin with script
/// execution disabled, so miner HTML can never touch the serving origin's
/// cookies, storage, or DOM — even when embedded same-origin through a proxy.
///
/// Public screenshots use [`screenshot_headers`] instead (`CORP: cross-origin`)
/// so joinbase.ai can `<img src="https://chain.joinbase.ai/.../index.png">`
/// without proxying PNG bytes through Vercel.
#[must_use]
pub fn viewer_headers(frame_ancestors: &str) -> Vec<(&'static str, String)> {
    let csp = format!(
        "sandbox; default-src 'none'; img-src data: https:; style-src 'unsafe-inline' https:; \
         font-src data: https:; base-uri 'none'; form-action 'none'; frame-ancestors {frame_ancestors}"
    );
    vec![
        ("Content-Security-Policy", csp),
        ("X-Content-Type-Options", "nosniff".into()),
        ("Referrer-Policy", "no-referrer".into()),
        // Non-PNG view responses stay same-origin only (defense in depth if
        // HTML ever leaks through); PNGs use `screenshot_headers`.
        ("Cross-Origin-Resource-Policy", "same-origin".into()),
        ("Cross-Origin-Opener-Policy", "same-origin".into()),
        (
            "Permissions-Policy",
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), \
             microphone=(), payment=(), usb=()"
                .into(),
        ),
        ("Cache-Control", "private, no-store".into()),
    ]
}

/// Headers for public PNG screenshots (`index.png`).
///
/// `Cross-Origin-Resource-Policy: cross-origin` lets marketing/admin UIs load
/// the image with a direct absolute URL to the gateway (no same-origin proxy
/// required). PNGs are not executable documents; cookies are never set.
#[must_use]
pub fn screenshot_headers() -> Vec<(&'static str, String)> {
    vec![
        ("X-Content-Type-Options", "nosniff".into()),
        ("Referrer-Policy", "no-referrer".into()),
        ("Cross-Origin-Resource-Policy", "cross-origin".into()),
        ("Cache-Control", "private, no-store".into()),
    ]
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn xss_corpus_neutralized() {
        let corpus: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("../fixtures/xss_corpus.json")).unwrap();
        for item in corpus {
            let html = item["html"].as_str().unwrap();
            let (out, report) = sanitize_html(html);
            let low = out.to_ascii_lowercase();
            assert!(
                !low.contains("<script"),
                "{} still has script: {out}",
                item["name"]
            );
            assert!(!low.contains("javascript:"), "{} js href", item["name"]);
            assert!(!low.contains("onerror="), "{} onerror", item["name"]);
            assert!(
                report.js_stripped || report.css_stripped || !low.contains("expression"),
                "{} expected strip signal",
                item["name"]
            );
        }
    }

    #[test]
    fn bundle_requires_pages() {
        let mut m = HashMap::new();
        m.insert("index.html".into(), "<p>a</p>".into());
        assert!(sanitize_bundle(&m).is_err());
        m.insert("pricing.html".into(), "<p>b</p>".into());
        m.insert("components.html".into(), "<p>c</p>".into());
        let r = sanitize_bundle(&m).unwrap();
        assert_eq!(r.pages.len(), 3);
        assert_eq!(r.artifact_digest.len(), 64);
    }

    #[test]
    fn viewer_csp_has_sandbox() {
        let h = viewer_headers("'none'");
        let csp = &h[0].1;
        assert!(csp.starts_with("sandbox;"));
        assert!(csp.contains("default-src 'none'"));
    }

    #[test]
    fn viewer_csp_never_allows_scripts_or_same_origin() {
        // The whole point of the viewer: opaque origin, zero script execution.
        // Regression guard — never relax these without an owner directive.
        let h = viewer_headers(default_frame_ancestors());
        let csp = &h[0].1;
        assert!(!csp.contains("allow-scripts"), "{csp}");
        assert!(!csp.contains("allow-same-origin"), "{csp}");
        assert!(csp.contains("style-src 'unsafe-inline' https:"), "{csp}");
        assert!(csp.contains("font-src data: https:"), "{csp}");
        assert!(csp.contains("base-uri 'none'"), "{csp}");
        assert!(csp.contains("form-action 'none'"), "{csp}");
        assert!(
            csp.contains(&format!("frame-ancestors {}", default_frame_ancestors())),
            "{csp}"
        );
    }

    #[test]
    fn viewer_headers_lockdown_set() {
        let h = viewer_headers("'none'");
        let get = |name: &str| h.iter().find(|(k, _)| *k == name).map(|(_, v)| v.as_str());
        assert_eq!(get("X-Content-Type-Options"), Some("nosniff"));
        assert_eq!(get("Referrer-Policy"), Some("no-referrer"));
        assert_eq!(get("Cross-Origin-Resource-Policy"), Some("same-origin"));
        // No cookie may ever be set on miner-content responses.
        assert!(get("Set-Cookie").is_none());
    }

    #[test]
    fn screenshot_headers_allow_cross_origin_img() {
        let h = screenshot_headers();
        let get = |name: &str| h.iter().find(|(k, _)| *k == name).map(|(_, v)| v.as_str());
        assert_eq!(get("Cross-Origin-Resource-Policy"), Some("cross-origin"));
        assert_eq!(get("X-Content-Type-Options"), Some("nosniff"));
        assert!(get("Content-Security-Policy").is_none());
        assert!(get("Cross-Origin-Opener-Policy").is_none());
    }

    #[test]
    fn default_frame_ancestors_allowlist() {
        let fa = default_frame_ancestors();
        assert!(fa.contains("https://joinbase.ai"), "{fa}");
        assert!(fa.contains("https://*.vercel.app"), "{fa}");
        assert!(!fa.contains("'none'"), "{fa}");
    }

    #[test]
    fn preserves_presentation_css_and_layout_tags() {
        let html = r#"<!DOCTYPE html>
<html>
<head><style>.hero { color: #c00; margin: 0; }</style></head>
<body class="page" id="top" style="margin:0;background:#fff">
<main class="wrap"><section id="hero" class="hero" style="padding:2rem">Hello</section></main>
</body>
</html>"#;
        let (out, report) = sanitize_html(html);
        let low = out.to_ascii_lowercase();
        assert!(
            !report.css_stripped,
            "safe CSS must not set css_stripped: {report:?}"
        );
        assert!(low.contains("<style>"), "style tag kept: {out}");
        assert!(out.contains(".hero"), "CSS rules kept: {out}");
        assert!(low.contains("<main"), "main kept: {out}");
        assert!(low.contains("<section"), "section kept: {out}");
        assert!(
            out.contains("class=\"hero\"") || out.contains("class='hero'"),
            "{out}"
        );
        assert!(
            out.contains("id=\"hero\"") || out.contains("id='hero'"),
            "{out}"
        );
        assert!(
            out.contains("style=\"padding:2rem\"") || out.contains("style=\"padding: 2rem\""),
            "inline style kept: {out}"
        );
        // body attrs survive via a wrapper div (ammonia unwraps <body>).
        assert!(
            out.contains("class=\"page\"") && out.contains("id=\"top\""),
            "body presentation preserved: {out}"
        );
    }

    #[test]
    fn still_strips_script_and_dangerous_css() {
        let (out, report) = sanitize_html(
            r#"<style>@import url('https://evil.test/x.css');</style>
<script>alert(1)</script><p class="ok" onclick="evil()">x</p>"#,
        );
        let low = out.to_ascii_lowercase();
        assert!(!low.contains("<script"), "{out}");
        assert!(!low.contains("onclick"), "{out}");
        assert!(!out.contains("@import"), "{out}");
        assert!(report.js_stripped || report.css_stripped);
        assert!(
            out.contains("class=\"ok\"") || out.contains("class='ok'"),
            "{out}"
        );
    }

    #[test]
    fn scroll_behavior_must_not_nuke_style_block() {
        let html = r#"<style>
html{scroll-behavior:smooth}
body{margin:0;color:#172220;background:#f4f7f6}
.hero{padding:2rem}
</style><main class="hero">Hello</main>"#;
        let (out, report) = sanitize_html(html);
        assert!(
            !report.css_stripped,
            "scroll-behavior is safe presentation CSS: {report:?}"
        );
        assert!(
            out.to_ascii_lowercase().contains("<style>"),
            "style kept: {out}"
        );
        assert!(out.contains("scroll-behavior"), "{out}");
        assert!(out.contains(".hero"), "{out}");
    }

    #[test]
    fn amp_in_css_comment_must_not_drop_style() {
        let html = r#"<style>
/* --- RESET & VARIABLES --- */
:root { --bg: #07090E; }
body { margin: 0; background: var(--bg); }
</style><p class="x">hi</p>"#;
        let (out, report) = sanitize_html(html);
        assert!(!report.css_stripped, "{report:?}");
        assert!(out.to_ascii_lowercase().contains("<style>"), "{out}");
        assert!(out.contains("--bg"), "{out}");
    }

    #[test]
    fn import_soft_stripped_keeps_remaining_rules() {
        let (out, report) = sanitize_html(
            r#"<style>@import url('https://fonts.example/x.css');
.hero{color:red}</style><p class="hero">x</p>"#,
        );
        assert!(report.css_stripped, "{report:?}");
        assert!(!out.contains("@import"), "{out}");
        assert!(out.to_ascii_lowercase().contains("<style>"), "{out}");
        assert!(out.contains(".hero"), "{out}");
    }

    #[test]
    fn ie_behavior_still_nukes_block() {
        let (out, report) =
            sanitize_html(r"<style>body{behavior:url(evil.htc);color:red}</style><p>x</p>");
        assert!(report.css_stripped, "{report:?}");
        assert!(!out.to_ascii_lowercase().contains("<style>"), "{out}");
        assert!(!out.contains("evil.htc"), "{out}");
    }
}
