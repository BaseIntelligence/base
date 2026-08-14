//! Lium rent **helpers** for BYOK / operator clients.
//!
//! Live Prism eval is miner-funded (`X-Lium-Api-Key`). Each miner key has its
//! own Lium rate budget — there is **no** process-wide rent serialize queue.
//! This crate only classifies 429 bodies and decides autonomous recovery.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]

/// Autonomous recovery looks back this far for failed 429 submissions.
pub const RECOVERY_WINDOW_MS: u64 = 6 * 60 * 60 * 1000;

/// True when a failed row should re-enter the rent queue (429 within window).
#[must_use]
pub fn should_recover(error_detail: &str, updated_at_ms: u64, now_ms: u64) -> bool {
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
    }
}
