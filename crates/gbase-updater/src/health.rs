//! HTTP health gate against a service `/readyz` (or full URL).

use thiserror::Error;

/// Health-check failures.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum HealthError {
    /// Transport error.
    #[error("health request failed: {0}")]
    Transport(String),
    /// Non-success status.
    #[error("health not ready: HTTP {status}")]
    NotReady {
        /// Status code.
        status: u16,
    },
}

/// GET `url` and require 2xx.
///
/// # Errors
/// [`HealthError`] on transport failure or non-success status.
pub fn check_readyz(url: &str) -> Result<(), HealthError> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(url)
        .send()
        .map_err(|e| HealthError::Transport(e.to_string()))?;
    let status = resp.status();
    if status.is_success() {
        Ok(())
    } else {
        Err(HealthError::NotReady {
            status: status.as_u16(),
        })
    }
}

/// Poll until success or `timeout` elapses.
///
/// # Errors
/// Last [`HealthError`] when the timeout expires without success.
pub fn wait_readyz(
    url: &str,
    timeout: std::time::Duration,
    interval: std::time::Duration,
) -> Result<(), HealthError> {
    let start = std::time::Instant::now();
    let mut last = HealthError::NotReady { status: 0 };
    while start.elapsed() < timeout {
        match check_readyz(url) {
            Ok(()) => return Ok(()),
            Err(e) => last = e,
        }
        std::thread::sleep(interval);
    }
    Err(last)
}

/// Test double: scripted readiness results.
#[derive(Debug, Clone, Default)]
pub struct ScriptedHealth {
    results: std::sync::Arc<std::sync::Mutex<Vec<Result<(), HealthError>>>>,
}

impl ScriptedHealth {
    /// Queue of outcomes consumed FIFO by [`Self::check`].
    #[must_use]
    pub fn new(results: Vec<Result<(), HealthError>>) -> Self {
        Self {
            results: std::sync::Arc::new(std::sync::Mutex::new(results)),
        }
    }

    /// Pop next scripted result (defaults to Ok when empty).
    ///
    /// # Errors
    /// Scripted error variants.
    pub fn check(&self) -> Result<(), HealthError> {
        let mut g = self
            .results
            .lock()
            .map_err(|e| HealthError::Transport(e.to_string()))?;
        if g.is_empty() {
            Ok(())
        } else {
            g.remove(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn wiremock_readyz_200_and_503() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/readyz"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string("ok"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        let url = format!("{}/readyz", server.uri());
        let u = url.clone();
        tokio::task::spawn_blocking(move || check_readyz(&u))
            .await
            .expect("join")
            .expect("ready");

        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/readyz"))
            .respond_with(wiremock::ResponseTemplate::new(503))
            .mount(&server)
            .await;
        let u2 = url;
        let err = tokio::task::spawn_blocking(move || check_readyz(&u2))
            .await
            .expect("join")
            .expect_err("should fail");
        assert!(matches!(err, HealthError::NotReady { status: 503 }));

        let server2 = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .respond_with(wiremock::ResponseTemplate::new(503))
            .mount(&server2)
            .await;
        let bad = format!("{}/readyz", server2.uri());
        let err = tokio::task::spawn_blocking(move || {
            wait_readyz(&bad, Duration::from_millis(30), Duration::from_millis(5))
        })
        .await
        .expect("join")
        .expect_err("timeout");
        assert!(matches!(err, HealthError::NotReady { .. }));
    }
}
