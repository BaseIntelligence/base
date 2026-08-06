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
    let schemes: std::collections::HashSet<&'static str> =
        ["http", "https", "mailto"].into_iter().collect();
    let _ = b.url_schemes(schemes);
    b
}

/// Filter dangerous CSS constructs from a style attribute / block.
#[must_use]
pub fn filter_css(css: &str) -> (String, bool) {
    let lower = css.to_ascii_lowercase();
    let bad = lower.contains("@import")
        || lower.contains("expression(")
        || lower.contains("url(javascript:")
        || lower.contains("behavior:")
        || lower.contains("-moz-binding");
    if bad {
        (String::new(), true)
    } else {
        (css.to_owned(), false)
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

    // Pre-strip style blocks with dangerous CSS.
    let mut pre = raw.to_owned();
    if let Ok(re) = Regex::new(r"(?is)<style[^>]*>(.*?)</style>") {
        let mut stripped = false;
        let mut css_notes = Vec::new();
        pre = re
            .replace_all(&pre, |caps: &regex::Captures<'_>| {
                let (filtered, bad) = filter_css(&caps[1]);
                if bad {
                    stripped = true;
                    css_notes.push("css_blocked".into());
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

    (
        cleaned,
        SanitizeReport {
            js_stripped,
            css_stripped,
            notes,
        },
    )
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

/// Viewer response headers (CSP sandbox is the key guarantee).
///
/// The `sandbox` directive is emitted **without** `allow-scripts` and without
/// `allow-same-origin`: the document runs in an opaque origin with script
/// execution disabled, so miner HTML can never touch the serving origin's
/// cookies, storage, or DOM — even when embedded same-origin through a proxy.
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
        // Viewer responses are only ever embedded same-origin (site proxies
        // the gateway under its own origin); cross-origin embedders get nothing.
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
    fn default_frame_ancestors_allowlist() {
        let fa = default_frame_ancestors();
        assert!(fa.contains("https://joinbase.ai"), "{fa}");
        assert!(fa.contains("https://*.vercel.app"), "{fa}");
        assert!(!fa.contains("'none'"), "{fa}");
    }
}
