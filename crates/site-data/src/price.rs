//! TAO/USD quote with a short server-side cache.
//!
//! Source: CoinGecko `simple/price` (override with `SITE_TAO_PRICE_URL` for
//! mirrors). The quote is a display convenience for the currency toggle,
//! never a scoring input: failures keep the last good value and a cold miss
//! yields 0, which the frontend renders as "—".

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Default quote endpoint.
const DEFAULT_URL: &str =
    "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd";
/// Cache lifetime.
const TTL: Duration = Duration::from_mins(10);
/// Hard ceiling on one quote fetch — a hung upstream must not stall stats.
const FETCH_TIMEOUT: Duration = Duration::from_secs(4);
/// Outbound identity: CoinGecko/Cloudflare 403 requests with no User-Agent.
const USER_AGENT: &str = "base-site/3.3 (+https://joinbase.ai)";

/// Cached TAO/USD quote.
#[derive(Debug, Clone, Copy)]
pub struct TaoPriceEntry {
    /// USD per TAO.
    pub usd: f64,
    /// Fetch instant.
    pub at: Instant,
}

/// Shared quote cache handle (one per process).
pub type TaoPriceCache = Mutex<Option<TaoPriceEntry>>;

fn price_url() -> String {
    std::env::var("SITE_TAO_PRICE_URL")
        .ok()
        .filter(|u| !u.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_URL.to_owned())
}

fn lock(cache: &TaoPriceCache) -> MutexGuard<'_, Option<TaoPriceEntry>> {
    cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// USD per TAO; 0 when no quote has ever succeeded.
pub async fn tao_price_usd(client: &reqwest::Client, cache: &TaoPriceCache) -> f64 {
    tao_price_from(client, cache, &price_url()).await
}

/// [`tao_price_usd`] with an explicit endpoint (tests).
async fn tao_price_from(client: &reqwest::Client, cache: &TaoPriceCache, url: &str) -> f64 {
    if let Some(entry) = *lock(cache) {
        if entry.at.elapsed() < TTL {
            return entry.usd;
        }
    }
    let fetched = fetch_once(client, url).await;
    let mut guard = lock(cache);
    match fetched {
        Some(usd) => {
            *guard = Some(TaoPriceEntry {
                usd,
                at: Instant::now(),
            });
            usd
        }
        None => guard.map_or(0.0, |e| e.usd),
    }
}

async fn fetch_once(client: &reqwest::Client, url: &str) -> Option<f64> {
    let res = tokio::time::timeout(
        FETCH_TIMEOUT,
        client
            .get(url)
            .header(reqwest::header::USER_AGENT, USER_AGENT)
            .send(),
    )
    .await
    .ok()?
    .ok()?;
    if !res.status().is_success() {
        return None;
    }
    let body: serde_json::Value = res.json().await.ok()?;
    body.pointer("/bittensor/usd")
        .and_then(serde_json::Value::as_f64)
        .filter(|v| v.is_finite() && *v > 0.0)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn sends_user_agent() {
        let server = MockServer::start().await;
        // CoinGecko/Cloudflare 403 header-less requests; the quote fetch must
        // always identify itself.
        Mock::given(method("GET"))
            .and(path("/simple/price"))
            .and(header("user-agent", USER_AGENT))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "bittensor": { "usd": 191.62 }
            })))
            .expect(1)
            .mount(&server)
            .await;
        let client = reqwest::Client::new();
        let cache = TaoPriceCache::new(None);
        let url = format!("{}/simple/price", server.uri());

        let v = tao_price_from(&client, &cache, &url).await;
        assert!((v - 191.62).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn caches_and_fails_closed() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/simple/price"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "bittensor": { "usd": 412.5 }
            })))
            .expect(1)
            .mount(&server)
            .await;
        let client = reqwest::Client::new();
        let cache = TaoPriceCache::new(None);
        let url = format!("{}/simple/price", server.uri());

        let a = tao_price_from(&client, &cache, &url).await;
        assert!((a - 412.5).abs() < f64::EPSILON);
        // Second call within TTL must not hit the server again (expect(1)).
        let b = tao_price_from(&client, &cache, &url).await;
        assert!((b - 412.5).abs() < f64::EPSILON);

        // Stale value survives a failed refresh beyond TTL.
        let entry = lock(&cache).unwrap();
        *lock(&cache) = Some(TaoPriceEntry {
            usd: entry.usd,
            at: Instant::now()
                .checked_sub(TTL + Duration::from_secs(1))
                .unwrap(),
        });
        let c = tao_price_from(&client, &cache, &format!("{}/gone", server.uri())).await;
        assert!((c - 412.5).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn cold_miss_is_zero() {
        let client = reqwest::Client::new();
        let cache = TaoPriceCache::new(None);
        // Nothing listening on 127.0.0.1:9 (discard) → fast failure → 0.
        let v = tao_price_from(&client, &cache, "http://127.0.0.1:9/price").await;
        assert!(v.abs() < f64::EPSILON);
    }
}
