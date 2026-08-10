//! Autonomous Lium rent budget: **3 / 5s** burst + **60 / hour**, serialized
//! rents, and cooldown from `429` bodies (`Please try again in N seconds`).

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tokio::sync::Semaphore;
use tokio::time::sleep;
use tracing::{info, warn};

/// Lium burst ceiling on `POST /executors/{id}/rent`.
pub const BURST_MAX: usize = 3;
/// Burst window.
pub const BURST_WINDOW: Duration = Duration::from_secs(5);
/// Lium hourly rent ceiling.
pub const HOUR_MAX: usize = 60;
/// Hourly window.
pub const HOUR_WINDOW: Duration = Duration::from_hours(1);
/// Cap a single acquire sleep (hourly cooldowns are long).
pub const MAX_ACQUIRE_WAIT: Duration = Duration::from_hours(1);
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

struct Inner {
    burst: VecDeque<Instant>,
    hour: VecDeque<Instant>,
    cooldown_until: Option<Instant>,
}

impl Inner {
    fn purge(&mut self, now: Instant) {
        while self
            .burst
            .front()
            .is_some_and(|t| now.duration_since(*t) >= BURST_WINDOW)
        {
            self.burst.pop_front();
        }
        while self
            .hour
            .front()
            .is_some_and(|t| now.duration_since(*t) >= HOUR_WINDOW)
        {
            self.hour.pop_front();
        }
        if self.cooldown_until.is_some_and(|u| now >= u) {
            self.cooldown_until = None;
        }
    }

    fn wait_hint(&self, now: Instant) -> Option<Duration> {
        if let Some(until) = self.cooldown_until {
            if now < until {
                return Some(until.saturating_duration_since(now));
            }
        }
        if self.burst.len() >= BURST_MAX {
            if let Some(oldest) = self.burst.front() {
                let elapsed = now.duration_since(*oldest);
                if elapsed < BURST_WINDOW {
                    return BURST_WINDOW.checked_sub(elapsed);
                }
            }
        }
        if self.hour.len() >= HOUR_MAX {
            if let Some(oldest) = self.hour.front() {
                let elapsed = now.duration_since(*oldest);
                if elapsed < HOUR_WINDOW {
                    return HOUR_WINDOW
                        .checked_sub(elapsed)
                        .map(|d| d.min(MAX_ACQUIRE_WAIT));
                }
            }
        }
        None
    }

    fn record(&mut self, now: Instant) {
        self.burst.push_back(now);
        self.hour.push_back(now);
    }
}

/// Process-wide rent gate: one in-flight rent + sliding-window budgets.
pub struct RentPool {
    inner: Mutex<Inner>,
    gate: Semaphore,
}

impl Default for RentPool {
    fn default() -> Self {
        Self::new()
    }
}

impl RentPool {
    /// Empty pool (budgets open).
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                burst: VecDeque::with_capacity(BURST_MAX + 1),
                hour: VecDeque::with_capacity(HOUR_MAX + 1),
                cooldown_until: None,
            }),
            gate: Semaphore::new(1),
        }
    }

    /// Wait until a rent is allowed; hold the serialize lock until permit drop.
    pub async fn take(&self) -> RentPermit<'_> {
        let owned = loop {
            match self.gate.acquire().await {
                Ok(p) => break p,
                Err(_) => sleep(Duration::from_secs(1)).await,
            }
        };
        loop {
            let wait = match self.inner.lock() {
                Ok(mut g) => {
                    let now = Instant::now();
                    g.purge(now);
                    g.wait_hint(now)
                }
                Err(_) => Some(Duration::from_millis(50)),
            };
            if let Some(w) = wait {
                let w = w.max(Duration::from_millis(200)).min(MAX_ACQUIRE_WAIT);
                info!(wait_ms = w.as_millis(), "lium rent pool waiting");
                sleep(w).await;
                continue;
            }
            if let Ok(mut g) = self.inner.lock() {
                g.record(Instant::now());
            }
            return RentPermit {
                pool: self,
                _permit: owned,
            };
        }
    }

    /// Apply a 429 cooldown (from body / Retry-After).
    pub fn note_429(&self, retry_secs: u64) {
        let secs = retry_secs.clamp(1, 7200);
        if let Ok(mut g) = self.inner.lock() {
            let until = Instant::now() + Duration::from_secs(secs);
            g.cooldown_until = Some(match g.cooldown_until {
                Some(prev) if prev > until => prev,
                _ => until,
            });
        }
        warn!(retry_secs = secs, "lium rent pool cooldown from 429");
    }

    /// Snapshot `(burst_used, hour_used, cooldown_secs)`.
    #[must_use]
    pub fn stats(&self) -> (usize, usize, Option<u64>) {
        let Ok(mut g) = self.inner.lock() else {
            return (0, 0, None);
        };
        let now = Instant::now();
        g.purge(now);
        let cool = g
            .cooldown_until
            .map(|u| u.saturating_duration_since(now).as_secs());
        (g.burst.len(), g.hour.len(), cool)
    }
}

/// RAII permit: drops the serialize lock when the rent HTTP call finishes.
pub struct RentPermit<'a> {
    pool: &'a RentPool,
    _permit: tokio::sync::SemaphorePermit<'a>,
}

impl RentPermit<'_> {
    /// Forward a 429 into the pool cooldown.
    pub fn rate_limited(&self, msg: &str) {
        self.pool.note_429(parse_retry_secs(msg).unwrap_or(5));
    }
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

    #[tokio::test]
    async fn burst_forces_wait() {
        let pool = RentPool::new();
        for _ in 0..BURST_MAX {
            drop(pool.take().await);
        }
        let start = Instant::now();
        drop(pool.take().await);
        assert!(start.elapsed() >= Duration::from_millis(200));
    }
}
