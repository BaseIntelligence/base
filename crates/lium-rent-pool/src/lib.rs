//! Lium rent **helpers** for BYOK / operator clients.
//!
//! Live Prism eval is miner-funded (`X-Lium-Api-Key`). Each miner key has its
//! own Lium rate budget — there is **no** process-wide rent serialize queue.
//! This crate classifies 429 / no-capacity rent failures and recovery.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]

/// Autonomous recovery looks back this far for failed 429 submissions.
pub const RECOVERY_WINDOW_MS: u64 = 6 * 60 * 60 * 1000;

/// Miner-facing text when Lium has no matching 1× B200 offer.
pub const CAPACITY_NOTE: &str =
    "B200s are currently out of capacity on Lium; this job is queued until an offer appears.";

/// Always-on policy (intake / recipe / `/v1/status`).
pub const CAPACITY_POLICY: &str = "When Lium has no matching 1× B200 offer, the job stays queued and retries until an offer appears (sold out is not Score(0)). Bad ZIP, auth, and template-permission errors still fail.";

/// True when a failed row should re-enter the rent queue.
#[must_use]
pub fn should_recover(error_detail: &str, updated_at_ms: u64, now_ms: u64) -> bool {
    let l = error_detail.to_ascii_lowercase();
    // HuggingFace / FineWeb 429s are not Lium rent 429s — re-renting an
    // 8×5090 pod does not fix a dataset CDN throttle.
    if l.contains("huggingface") || l.contains("fineweb") || l.contains("\"stage\": \"dataset\"") {
        return false;
    }
    if is_no_capacity(error_detail) {
        return true;
    }
    is_rate_limited(error_detail) && now_ms.saturating_sub(updated_at_ms) <= RECOVERY_WINDOW_MS
}

/// Parse Lium / Retry-After wait hints from a 429 body or header value.
#[must_use]
pub fn parse_retry_secs(text: &str) -> Option<u64> {
    if let Some(i) = text.find("try again in ") {
        let rest = &text[i + "try again in ".len()..];
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        if let Ok(n) = digits.parse::<u64>() {
            if n > 0 {
                return Some(n.min(7200));
            }
        }
    }
    let t = text.trim();
    if !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()) {
        if let Ok(n) = t.parse::<u64>() {
            if n > 0 {
                return Some(n.min(7200));
            }
        }
    }
    if text.contains("per 1 hour") {
        return Some(120);
    }
    if text.contains("per 5 seconds") {
        return Some(5);
    }
    None
}

/// True when an error string is a Lium rate-limit (HTTP 429).
#[must_use]
pub fn is_rate_limited(msg: &str) -> bool {
    let l = msg.to_ascii_lowercase();
    l.contains("429") || l.contains("too many requests") || l.contains("rate limit")
}

/// Auth / template-permission / missing BYOK — never treat as sold-out.
#[must_use]
pub fn is_auth_or_permission(msg: &str) -> bool {
    let l = msg.to_ascii_lowercase();
    l.contains("permission")
        || l.contains("unauthorized")
        || l.contains("forbidden")
        || l.contains("401")
        || l.contains("invalid api key")
        || l.contains("missing_lium_api_key")
        || l.contains("api key missing")
}

/// No matching Lium offer / B200 sold out (not miner ZIP, not auth).
#[must_use]
pub fn is_no_capacity(msg: &str) -> bool {
    if is_auth_or_permission(msg) {
        return false;
    }
    let l = msg.to_ascii_lowercase();
    l.contains("no_capacity")
        || l.contains("no lium offer")
        || l.contains("no offer matches")
        || l.contains("no matching offer")
        || l.contains("sold out")
        || l.contains("out of capacity")
        || l.contains("lack of b200")
        || (l.contains("b200") && (l.contains("unavailable") || l.contains("no offer")))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn parses_try_again_in_seconds() {
        let s = r#"{"message":"Too many requests. You can make 60 requests per 1 hour. Please try again in 1845 seconds."}"#;
        assert_eq!(parse_retry_secs(s), Some(1845));
        assert_eq!(
            parse_retry_secs("Too many requests. You can make 3 requests per 5 seconds."),
            Some(5)
        );
        assert!(is_rate_limited(
            "lium api: POST /rent -> 429 Too Many Requests"
        ));
        assert!(should_recover("provision: 429 rate limit", 100, 100 + 1));
        assert!(!should_recover("provision: 429", 0, RECOVERY_WINDOW_MS + 1));
        assert!(!should_recover(
            "measure: exec: HTTP Error 429: huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
            100,
            100 + 1
        ));
        assert!(is_no_capacity(
            "measure: provision: no Lium offer matches GPU preference and price caps (no_capacity)"
        ));
        assert!(is_no_capacity("lium rent sold out: no matching offer"));
        assert!(!is_no_capacity(
            "lium api: POST /rent -> 400 permission to rent this template"
        ));
        assert!(should_recover(
            "provision: no_capacity (no matching B200 offer)",
            0,
            RECOVERY_WINDOW_MS + 1
        ));
        assert!(!should_recover(
            "lium api: 400 permission to rent this template",
            100,
            100 + 1
        ));
    }
}
