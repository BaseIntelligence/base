//! In-memory challenge backend registry (routing only — D18).
//!
//! Stores operational base URLs and health/ejection state. Never accepts or
//! exposes signing keys; those live in owner-signed `config/challenges.toml`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

/// Default consecutive failures before passive ejection.
pub const DEFAULT_FAILURE_THRESHOLD: u32 = 3;

/// Default cooldown before a ejected backend is re-admitted.
pub const DEFAULT_COOLDOWN: Duration = Duration::from_secs(30);

/// Tunables for passive ejection / re-admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryConfig {
    /// Consecutive proxy failures before marking a backend ejected.
    pub failure_threshold: u32,
    /// How long an ejected backend stays out of the healthy pool.
    pub cooldown: Duration,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            failure_threshold: DEFAULT_FAILURE_THRESHOLD,
            cooldown: DEFAULT_COOLDOWN,
        }
    }
}

/// Operational backend row (no key material — D18).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Backend {
    /// Stable id (UUID).
    pub id: Uuid,
    /// Challenge identifier used in `/challenge/{id}/…`.
    pub challenge_id: String,
    /// Upstream base URL (scheme + host[:port], no trailing path required).
    pub base_url: String,
    /// Relative weight (reserved; round-robin treats all equal today).
    pub weight: u32,
    /// Operator / passive-health flag.
    pub healthy: bool,
    /// Consecutive failures since last success.
    pub fail_count: u32,
    /// When set and in the future, backend is excluded from picks.
    #[serde(skip_serializing)]
    pub ejected_until: Option<Instant>,
    /// Wall-clock ejection deadline for API responses (RFC3339-ish debug).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ejected: Option<bool>,
}

impl Backend {
    /// Whether this backend may receive traffic at `now`.
    #[must_use]
    pub fn is_eligible(&self, now: Instant) -> bool {
        if let Some(until) = self.ejected_until {
            if now < until {
                return false;
            }
        }
        self.healthy || self.ejected_until.is_some()
    }
}

/// Body for creating a backend. Unknown fields (e.g. `signing_key`) are rejected.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateBackend {
    /// Challenge id.
    pub challenge_id: String,
    /// Upstream base URL.
    pub base_url: String,
    /// Optional weight (default 1).
    #[serde(default = "default_weight")]
    pub weight: u32,
}

fn default_weight() -> u32 {
    1
}

/// Public view returned by the registry API (never includes key fields).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackendView {
    /// Id.
    pub id: Uuid,
    /// Challenge id.
    pub challenge_id: String,
    /// Base URL.
    pub base_url: String,
    /// Weight.
    pub weight: u32,
    /// Healthy flag after re-admission bookkeeping.
    pub healthy: bool,
    /// Fail count.
    pub fail_count: u32,
    /// True when currently ejected (cooldown not elapsed).
    pub ejected: bool,
}

impl BackendView {
    fn from_backend(b: &Backend, now: Instant) -> Self {
        let ejected = b.ejected_until.is_some_and(|u| now < u);
        Self {
            id: b.id,
            challenge_id: b.challenge_id.clone(),
            base_url: b.base_url.clone(),
            weight: b.weight,
            healthy: b.healthy && !ejected,
            fail_count: b.fail_count,
            ejected,
        }
    }
}

/// Registry failures.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RegistryError {
    /// Duplicate `(challenge_id, base_url)`.
    #[error("backend already registered for challenge_id={challenge_id} base_url={base_url}")]
    Duplicate {
        /// Challenge id.
        challenge_id: String,
        /// Base URL.
        base_url: String,
    },
    /// Unknown backend id.
    #[error("backend not found: {0}")]
    NotFound(Uuid),
    /// Empty or invalid fields.
    #[error("invalid backend: {0}")]
    Invalid(String),
    /// No eligible upstream for the challenge.
    #[error("no healthy backends for challenge_id={0}")]
    NoBackends(String),
}

/// Thread-safe in-memory backend registry with round-robin selection.
#[derive(Debug)]
pub struct Registry {
    cfg: RegistryConfig,
    backends: RwLock<Vec<Backend>>,
    /// Per-challenge round-robin cursor.
    rr: RwLock<HashMap<String, AtomicUsize>>,
}

impl Registry {
    /// Empty registry with the given knobs.
    #[must_use]
    pub fn new(cfg: RegistryConfig) -> Self {
        Self {
            cfg,
            backends: RwLock::new(Vec::new()),
            rr: RwLock::new(HashMap::new()),
        }
    }

    /// Default thresholds (3 failures, 30s cooldown).
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(RegistryConfig::default())
    }

    /// Shared handle.
    #[must_use]
    pub fn shared(cfg: RegistryConfig) -> Arc<Self> {
        Arc::new(Self::new(cfg))
    }

    /// Config snapshot.
    #[must_use]
    pub fn config(&self) -> &RegistryConfig {
        &self.cfg
    }

    /// Register a backend. Rejects empty ids/urls and zero weight.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Invalid`] or [`RegistryError::Duplicate`].
    pub fn create(&self, req: &CreateBackend) -> Result<BackendView, RegistryError> {
        let challenge_id = req.challenge_id.trim().to_owned();
        let base_url = normalize_base_url(req.base_url.as_str())?;
        if challenge_id.is_empty() {
            return Err(RegistryError::Invalid(
                "challenge_id must be non-empty".into(),
            ));
        }
        if req.weight == 0 {
            return Err(RegistryError::Invalid("weight must be > 0".into()));
        }

        let mut guard = self.backends.write();
        if guard
            .iter()
            .any(|b| b.challenge_id == challenge_id && b.base_url == base_url)
        {
            return Err(RegistryError::Duplicate {
                challenge_id,
                base_url,
            });
        }
        let backend = Backend {
            id: Uuid::new_v4(),
            challenge_id,
            base_url,
            weight: req.weight,
            healthy: true,
            fail_count: 0,
            ejected_until: None,
            ejected: None,
        };
        let view = BackendView::from_backend(&backend, Instant::now());
        guard.push(backend);
        Ok(view)
    }

    /// List backends, optionally filtered by challenge id.
    #[must_use]
    pub fn list(&self, challenge_id: Option<&str>) -> Vec<BackendView> {
        let now = Instant::now();
        let guard = self.backends.read();
        guard
            .iter()
            .filter(|b| challenge_id.is_none_or(|c| b.challenge_id == c))
            .map(|b| BackendView::from_backend(b, now))
            .collect()
    }

    /// Fetch one backend by id.
    ///
    /// # Errors
    ///
    /// [`RegistryError::NotFound`].
    pub fn get(&self, id: Uuid) -> Result<BackendView, RegistryError> {
        let now = Instant::now();
        self.backends
            .read()
            .iter()
            .find(|b| b.id == id)
            .map(|b| BackendView::from_backend(b, now))
            .ok_or(RegistryError::NotFound(id))
    }

    /// Remove a backend.
    ///
    /// # Errors
    ///
    /// [`RegistryError::NotFound`].
    pub fn delete(&self, id: Uuid) -> Result<(), RegistryError> {
        let mut guard = self.backends.write();
        let before = guard.len();
        guard.retain(|b| b.id != id);
        if guard.len() == before {
            return Err(RegistryError::NotFound(id));
        }
        Ok(())
    }

    /// Round-robin pick among eligible backends for `challenge_id`.
    ///
    /// Re-admits backends whose cooldown has elapsed (clears ejection, resets
    /// `fail_count`, sets healthy).
    ///
    /// # Errors
    ///
    /// [`RegistryError::NoBackends`] when none are eligible.
    pub fn pick(&self, challenge_id: &str) -> Result<Backend, RegistryError> {
        let now = Instant::now();
        let mut guard = self.backends.write();

        // Re-admit cooled-down backends.
        for b in guard.iter_mut() {
            if b.challenge_id != challenge_id {
                continue;
            }
            if let Some(until) = b.ejected_until {
                if now >= until {
                    b.ejected_until = None;
                    b.fail_count = 0;
                    b.healthy = true;
                }
            }
        }

        let eligible: Vec<usize> = guard
            .iter()
            .enumerate()
            .filter(|(_, b)| b.challenge_id == challenge_id && b.is_eligible(now))
            .map(|(i, _)| i)
            .collect();

        if eligible.is_empty() {
            return Err(RegistryError::NoBackends(challenge_id.to_owned()));
        }

        let cursor = {
            let mut rr = self.rr.write();
            let entry = rr
                .entry(challenge_id.to_owned())
                .or_insert_with(|| AtomicUsize::new(0));
            entry.fetch_add(1, Ordering::Relaxed)
        };
        let idx = eligible[cursor % eligible.len()];
        Ok(guard[idx].clone())
    }

    /// Record a successful proxy hop (reset `fail_count`, mark healthy).
    pub fn record_success(&self, id: Uuid) {
        let mut guard = self.backends.write();
        if let Some(b) = guard.iter_mut().find(|b| b.id == id) {
            b.fail_count = 0;
            b.healthy = true;
            b.ejected_until = None;
        }
    }

    /// Record a failed proxy hop; eject after `failure_threshold`.
    pub fn record_failure(&self, id: Uuid) {
        let now = Instant::now();
        let mut guard = self.backends.write();
        if let Some(b) = guard.iter_mut().find(|b| b.id == id) {
            b.fail_count = b.fail_count.saturating_add(1);
            b.last_touch_failure(now, self.cfg.failure_threshold, self.cfg.cooldown);
        }
    }

    /// Test helper: force-eject immediately (sets `fail_count` to threshold).
    #[cfg(test)]
    pub fn force_eject(&self, id: Uuid) {
        let now = Instant::now();
        let mut guard = self.backends.write();
        if let Some(b) = guard.iter_mut().find(|b| b.id == id) {
            b.fail_count = self.cfg.failure_threshold;
            b.healthy = false;
            b.ejected_until = Some(now + self.cfg.cooldown);
        }
    }
}

impl Backend {
    fn last_touch_failure(&mut self, now: Instant, threshold: u32, cooldown: Duration) {
        if self.fail_count >= threshold {
            self.healthy = false;
            self.ejected_until = Some(now + cooldown);
        }
    }
}

fn normalize_base_url(raw: &str) -> Result<String, RegistryError> {
    let s = raw.trim().trim_end_matches('/').to_owned();
    if s.is_empty() {
        return Err(RegistryError::Invalid("base_url must be non-empty".into()));
    }
    if !(s.starts_with("http://") || s.starts_with("https://")) {
        return Err(RegistryError::Invalid(
            "base_url must start with http:// or https://".into(),
        ));
    }
    Ok(s)
}

/// Parse a boot-seed list: `challenge_id=url` entries separated by commas
/// and/or newlines. Empty / whitespace-only input yields an empty vec.
///
/// Optional `;weight` suffix (default 1): `design=http://design:8093;2`.
///
/// # Errors
///
/// [`RegistryError::Invalid`] on malformed entries or bad URLs.
pub fn parse_backend_seed_list(raw: &str) -> Result<Vec<CreateBackend>, RegistryError> {
    let mut out = Vec::new();
    for part in raw.split([',', '\n', '\r']) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (challenge_id, rest) = part.split_once('=').ok_or_else(|| {
            RegistryError::Invalid(format!(
                "backend seed entry must be challenge_id=url[,…]; got `{part}`"
            ))
        })?;
        let challenge_id = challenge_id.trim();
        if challenge_id.is_empty() {
            return Err(RegistryError::Invalid(
                "backend seed challenge_id must be non-empty".into(),
            ));
        }
        let (url_raw, weight) = match rest.rsplit_once(';') {
            Some((url, w)) if w.trim().is_empty() => (url, 1u32),
            Some((url, w)) if !w.contains("://") => {
                let weight: u32 = w.trim().parse().map_err(|_| {
                    RegistryError::Invalid(format!(
                        "backend seed weight must be u32; got `{}`",
                        w.trim()
                    ))
                })?;
                (url, weight)
            }
            _ => (rest, 1u32),
        };
        let base_url = normalize_base_url(url_raw)?;
        out.push(CreateBackend {
            challenge_id: challenge_id.to_owned(),
            base_url,
            weight,
        });
    }
    Ok(out)
}

impl Registry {
    /// Insert seed backends; skip rows already present (`Duplicate`).
    ///
    /// Returns how many rows were newly created.
    ///
    /// # Errors
    ///
    /// Propagates non-duplicate [`RegistryError`] from [`Self::create`].
    pub fn seed(&self, backends: &[CreateBackend]) -> Result<usize, RegistryError> {
        let mut created = 0usize;
        for req in backends {
            match self.create(req) {
                Ok(_) => created += 1,
                Err(RegistryError::Duplicate { .. }) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(created)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg_fast() -> Registry {
        Registry::new(RegistryConfig {
            failure_threshold: 2,
            cooldown: Duration::from_millis(80),
        })
    }

    fn add(reg: &Registry, challenge: &str, url: &str) -> BackendView {
        reg.create(&CreateBackend {
            challenge_id: challenge.into(),
            base_url: url.into(),
            weight: 1,
        })
        .expect("create")
    }

    #[test]
    fn round_robin_alternates_two_backends() {
        let reg = reg_fast();
        let a = add(&reg, "c1", "http://127.0.0.1:1");
        let b = add(&reg, "c1", "http://127.0.0.1:2");
        let mut seen = Vec::new();
        for _ in 0..6 {
            seen.push(reg.pick("c1").unwrap().id);
        }
        assert_eq!(seen[0], a.id);
        assert_eq!(seen[1], b.id);
        assert_eq!(seen[2], a.id);
        assert_eq!(seen[3], b.id);
    }

    #[test]
    fn ejection_after_threshold_leaves_survivor() {
        let reg = reg_fast();
        let a = add(&reg, "c1", "http://127.0.0.1:1");
        let b = add(&reg, "c1", "http://127.0.0.1:2");
        reg.record_failure(a.id);
        reg.record_failure(a.id); // threshold 2 → eject a
        for _ in 0..8 {
            let p = reg.pick("c1").unwrap();
            assert_eq!(p.id, b.id, "only survivor should be picked");
        }
    }

    #[test]
    fn re_admit_after_cooldown() {
        let reg = reg_fast();
        let a = add(&reg, "c1", "http://127.0.0.1:1");
        let _b = add(&reg, "c1", "http://127.0.0.1:2");
        reg.force_eject(a.id);
        // Immediately: a not eligible
        for _ in 0..4 {
            assert_ne!(reg.pick("c1").unwrap().id, a.id);
        }
        std::thread::sleep(Duration::from_millis(100));
        // After cooldown, a is eligible again — pick should eventually return a
        let mut saw_a = false;
        for _ in 0..10 {
            if reg.pick("c1").unwrap().id == a.id {
                saw_a = true;
                break;
            }
        }
        assert!(saw_a, "ejected backend must re-admit after cooldown");
    }

    #[test]
    fn create_rejects_empty_and_duplicate() {
        let reg = reg_fast();
        let err = reg
            .create(&CreateBackend {
                challenge_id: " ".into(),
                base_url: "http://x".into(),
                weight: 1,
            })
            .unwrap_err();
        assert!(matches!(err, RegistryError::Invalid(_)));
        add(&reg, "c1", "http://x/");
        let err = reg
            .create(&CreateBackend {
                challenge_id: "c1".into(),
                base_url: "http://x".into(), // trailing slash normalized
                weight: 1,
            })
            .unwrap_err();
        assert!(matches!(err, RegistryError::Duplicate { .. }));
    }

    #[test]
    fn deny_unknown_fields_signing_key() {
        let err = serde_json::from_str::<CreateBackend>(
            r#"{"challenge_id":"c","base_url":"http://x","signing_key":"deadbeef"}"#,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("signing_key") || err.to_string().contains("unknown field"),
            "got {err}"
        );
    }

    #[test]
    fn backend_view_has_no_key_shaped_fields() {
        let reg = reg_fast();
        let v = add(&reg, "c1", "http://127.0.0.1:9");
        let json = serde_json::to_value(&v).unwrap();
        let obj = json.as_object().unwrap();
        for k in obj.keys() {
            assert!(
                !k.to_lowercase().contains("key") && !k.to_lowercase().contains("secret"),
                "D18: unexpected field {k}"
            );
        }
        assert!(obj.contains_key("base_url"));
        assert!(obj.contains_key("challenge_id"));
    }

    /// Multi-challenge registration: prism + design coexist (must not hardcode a single id).
    #[test]
    fn create_multi_challenge_backends() {
        let reg = Registry::with_defaults();
        let prism = reg
            .create(&CreateBackend {
                challenge_id: "prism".into(),
                base_url: "http://prism-challenge:8092/".into(),
                weight: 1,
            })
            .expect("create prism backend");
        assert_eq!(prism.challenge_id, "prism");
        assert_eq!(prism.base_url, "http://prism-challenge:8092");
        assert!(prism.healthy);
        assert!(!prism.ejected);

        let design = reg
            .create(&CreateBackend {
                challenge_id: "design".into(),
                base_url: "http://design-challenge:8093".into(),
                weight: 1,
            })
            .expect("create design backend");
        assert_eq!(design.challenge_id, "design");

        let listed = reg.list(Some("prism"));
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, prism.id);

        let picked = reg.pick("prism").expect("pick prism");
        assert_eq!(picked.id, prism.id);
        assert_eq!(picked.base_url, "http://prism-challenge:8092");

        let design_pick = reg.pick("design").expect("pick design");
        assert_eq!(design_pick.id, design.id);
    }

    /// PRISM compose registration: `challenge_id` + service `base_url` port 8092 (no Phala CVM).
    #[test]
    fn create_prism_backend_compose_url() {
        let reg = Registry::with_defaults();
        let view = reg
            .create(&CreateBackend {
                challenge_id: "prism".into(),
                base_url: "http://prism-challenge:8092/".into(),
                weight: 1,
            })
            .expect("create prism backend");
        assert_eq!(view.challenge_id, "prism");
        assert_eq!(view.base_url, "http://prism-challenge:8092");
        let picked = reg.pick("prism").expect("pick prism");
        assert_eq!(picked.base_url, "http://prism-challenge:8092");
    }

    #[test]
    fn parse_and_seed_compose_backends() {
        let list = parse_backend_seed_list(
            "prism=http://prism-challenge:8092, design=http://design-challenge:8093\n",
        )
        .expect("parse");
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].challenge_id, "prism");
        assert_eq!(list[0].base_url, "http://prism-challenge:8092");
        assert_eq!(list[1].challenge_id, "design");
        assert_eq!(list[1].weight, 1);

        let reg = Registry::with_defaults();
        assert_eq!(reg.seed(&list).expect("seed"), 2);
        assert_eq!(reg.seed(&list).expect("idempotent"), 0);
        assert_eq!(reg.list(None).len(), 2);
        assert_eq!(
            reg.pick("design").unwrap().base_url,
            "http://design-challenge:8093"
        );
    }

    #[test]
    fn parse_backend_seed_rejects_bad_entry() {
        let err = parse_backend_seed_list("not-a-pair").unwrap_err();
        assert!(matches!(err, RegistryError::Invalid(_)));
    }
}
