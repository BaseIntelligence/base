//! Rollout state machine: idle → resolving → pulling → recreating → verifying → committing.

use std::collections::HashMap;
use std::fmt;

use crate::config::UpdaterConfig;
use crate::digest::{extract_digest, parse_pinned_image, PinnedImage};
use crate::docker::{ContainerSummary, DockerApi};
use crate::error::UpdaterError;
use crate::health::{wait_readyz, HealthError, ScriptedHealth};
use crate::pin_store::{commit_pins, load_pins, PinRecord, PinStore};

/// Durable / in-memory rollout phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// No active rollout.
    Idle,
    /// Parsing / validating desired digest.
    Resolving,
    /// Pulling image by digest.
    Pulling,
    /// Stopping old + creating/starting new container.
    Recreating,
    /// Waiting on `/readyz`.
    Verifying,
    /// Writing pin files after success.
    Committing,
    /// Restoring previous digest after failure.
    RollingBack,
    /// Waiting before retry.
    Backoff,
    /// Retries exhausted for this digest.
    Exhausted,
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Idle => "idle",
            Self::Resolving => "resolving",
            Self::Pulling => "pulling",
            Self::Recreating => "recreating",
            Self::Verifying => "verifying",
            Self::Committing => "committing",
            Self::RollingBack => "rolling_back",
            Self::Backoff => "backoff",
            Self::Exhausted => "exhausted",
        };
        f.write_str(s)
    }
}

/// Result of one [`tick`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TickOutcome {
    /// Already running desired digest.
    AlreadyCurrent,
    /// Successfully rolled to desired digest.
    Rolled {
        /// New digest.
        digest: String,
    },
    /// Health failed; rolled back to previous.
    RolledBack {
        /// Reason for failure.
        reason: String,
    },
    /// Refused because target is self (D14).
    RefusedSelf {
        /// Container name.
        name: String,
    },
    /// Pull failed; running container left untouched.
    PullFailed {
        /// Error text.
        reason: String,
    },
    /// Desired image rejected (not digest-pinned).
    RejectedImage {
        /// Image string.
        image: String,
    },
}

/// Health probe used by the machine (live HTTP or scripted).
pub trait HealthProbe: Send + Sync {
    /// Check readiness once (or poll internally).
    ///
    /// # Errors
    /// [`HealthError`] when not ready.
    fn wait_ready(
        &self,
        timeout: std::time::Duration,
        interval: std::time::Duration,
    ) -> Result<(), HealthError>;
}

/// Live HTTP probe against config health URL.
pub struct HttpHealthProbe {
    url: String,
}

impl HttpHealthProbe {
    /// Build probe for `url`.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }
}

impl HealthProbe for HttpHealthProbe {
    fn wait_ready(
        &self,
        timeout: std::time::Duration,
        interval: std::time::Duration,
    ) -> Result<(), HealthError> {
        wait_readyz(&self.url, timeout, interval)
    }
}

impl HealthProbe for ScriptedHealth {
    fn wait_ready(
        &self,
        _timeout: std::time::Duration,
        _interval: std::time::Duration,
    ) -> Result<(), HealthError> {
        self.check()
    }
}

/// Updater holding config + last phase (for observability).
#[derive(Debug)]
pub struct Updater {
    /// Configuration.
    pub config: UpdaterConfig,
    /// Last phase reached.
    pub phase: Phase,
}

impl Updater {
    /// Construct from config.
    #[must_use]
    pub fn new(config: UpdaterConfig) -> Self {
        Self {
            config,
            phase: Phase::Idle,
        }
    }
}

/// Run one rollout tick against `docker` and `health`.
///
/// # Errors
/// [`UpdaterError`] for hard failures (missing target, pin I/O, etc.).
#[allow(clippy::too_many_lines)]
pub fn tick<D: DockerApi, H: HealthProbe>(
    updater: &mut Updater,
    docker: &D,
    health: &H,
) -> Result<TickOutcome, UpdaterError> {
    updater.phase = Phase::Resolving;

    let Ok(desired) = parse_pinned_image(&updater.config.desired_image) else {
        updater.phase = Phase::Idle;
        return Ok(TickOutcome::RejectedImage {
            image: updater.config.desired_image.clone(),
        });
    };

    let target = find_target(
        docker,
        &updater.config.compose_project,
        &updater.config.service_name,
    )?;

    // D14: never recreate our own container automatically.
    if names_match(&target.name, &updater.config.self_container_name) {
        updater.phase = Phase::Idle;
        return Ok(TickOutcome::RefusedSelf { name: target.name });
    }

    let running_digest = extract_digest(&target.image);
    if running_digest.as_deref() == Some(desired.digest.as_str()) {
        // Ensure pins reflect reality.
        let store = PinStore::new(&updater.config.state_dir);
        let (current, _) = load_pins(&store)?;
        if current.as_ref().map(|c| c.digest.as_str()) != Some(desired.digest.as_str()) {
            let rec = pin_record(&updater.config.service_name, &desired);
            commit_pins(&store, current.as_ref(), &rec)?;
        }
        updater.phase = Phase::Idle;
        return Ok(TickOutcome::AlreadyCurrent);
    }

    let store = PinStore::new(&updater.config.state_dir);
    let (current_pin, _) = load_pins(&store)?;
    let rollback_image = current_pin
        .as_ref()
        .map(|p| p.image.clone())
        .filter(|img| parse_pinned_image(img).is_ok())
        .or_else(|| extract_digest(&target.image).map(|_| target.image.clone()));

    // Pull
    updater.phase = Phase::Pulling;
    if let Err(e) = docker.pull_image(&desired.as_ref_string()) {
        updater.phase = Phase::Backoff;
        return Ok(TickOutcome::PullFailed {
            reason: e.to_string(),
        });
    }

    // Recreate
    updater.phase = Phase::Recreating;
    let labels = compose_labels(
        &updater.config.compose_project,
        &updater.config.service_name,
    );
    let old_name = target.name.clone();
    let backup_name = format!("{old_name}.pre-update");

    if let Err(e) = recreate_container(docker, &target, &desired, &labels, &backup_name) {
        updater.phase = Phase::RollingBack;
        if let Some(ref rb) = rollback_image {
            let _ = restore_from_backup_or_image(docker, &old_name, &backup_name, rb, &labels);
        }
        updater.phase = Phase::Backoff;
        return Ok(TickOutcome::RolledBack {
            reason: e.to_string(),
        });
    }

    // Verify health
    updater.phase = Phase::Verifying;
    if let Err(e) = health.wait_ready(
        updater.config.health_timeout,
        updater.config.health_poll_interval,
    ) {
        updater.phase = Phase::RollingBack;
        let reason = e.to_string();
        if let Some(ref rb) = rollback_image {
            let _ = rollback_container(docker, &old_name, rb, &labels, &backup_name);
        } else {
            let _ = rollback_container(docker, &old_name, &target.image, &labels, &backup_name);
        }
        updater.phase = Phase::Backoff;
        return Ok(TickOutcome::RolledBack { reason });
    }

    // Commit pins
    updater.phase = Phase::Committing;
    let new_pin = pin_record(&updater.config.service_name, &desired);
    let old_for_prev = current_pin.or_else(|| {
        extract_digest(&target.image).map(|d| PinRecord {
            service: updater.config.service_name.clone(),
            image: target.image.clone(),
            digest: d,
            updated_at: now_rfc3339(),
        })
    });
    commit_pins(&store, old_for_prev.as_ref(), &new_pin)?;
    updater.phase = Phase::Idle;
    Ok(TickOutcome::Rolled {
        digest: desired.digest,
    })
}

fn names_match(a: &str, b: &str) -> bool {
    let na = a.trim_start_matches('/');
    let nb = b.trim_start_matches('/');
    na == nb
}

fn find_target<D: DockerApi>(
    docker: &D,
    project: &str,
    service: &str,
) -> Result<ContainerSummary, UpdaterError> {
    let list = docker.list_containers()?;
    list.into_iter()
        .find(|c| {
            c.compose_project.as_deref() == Some(project)
                && c.compose_service.as_deref() == Some(service)
        })
        .ok_or_else(|| UpdaterError::TargetNotFound {
            project: project.to_owned(),
            service: service.to_owned(),
        })
}

fn compose_labels(project: &str, service: &str) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("com.docker.compose.project".into(), project.to_owned());
    m.insert("com.docker.compose.service".into(), service.to_owned());
    m
}

fn pin_record(service: &str, pinned: &PinnedImage) -> PinRecord {
    PinRecord {
        service: service.to_owned(),
        image: pinned.as_ref_string(),
        digest: pinned.digest.clone(),
        updated_at: now_rfc3339(),
    }
}

fn now_rfc3339() -> String {
    // Avoid extra chrono dep: use unix secs placeholder format acceptable for pins.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format!("{secs}")
}

fn recreate_container<D: DockerApi>(
    docker: &D,
    target: &ContainerSummary,
    desired: &PinnedImage,
    labels: &HashMap<String, String>,
    backup_name: &str,
) -> Result<(), UpdaterError> {
    docker.stop_container(&target.id)?;
    docker.rename_container(&target.id, backup_name)?;
    let new_id = docker.create_container(&target.name, &desired.as_ref_string(), labels)?;
    docker.start_container(&new_id)?;
    Ok(())
}

fn rollback_container<D: DockerApi>(
    docker: &D,
    original_name: &str,
    rollback_image: &str,
    labels: &HashMap<String, String>,
    backup_name: &str,
) -> Result<(), UpdaterError> {
    // Stop failed new container if present under original_name.
    let _ = docker.stop_container(original_name);
    // Try rename new out of the way.
    let failed = format!("{original_name}.failed");
    let _ = docker.rename_container(original_name, &failed);
    // Prefer restoring backup container name.
    if docker.rename_container(backup_name, original_name).is_ok() {
        let _ = docker.start_container(original_name);
        return Ok(());
    }
    // Else recreate from rollback image.
    restore_from_backup_or_image(docker, original_name, backup_name, rollback_image, labels)
}

fn restore_from_backup_or_image<D: DockerApi>(
    docker: &D,
    original_name: &str,
    backup_name: &str,
    image: &str,
    labels: &HashMap<String, String>,
) -> Result<(), UpdaterError> {
    if docker.rename_container(backup_name, original_name).is_ok() {
        docker.start_container(original_name)?;
        return Ok(());
    }
    let id = docker.create_container(original_name, image, labels)?;
    docker.start_container(&id)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docker::MockDocker;
    use crate::health::HealthError;
    use tempfile::tempdir;

    fn d(hex_byte: char) -> String {
        format!("sha256:{}", hex_byte.to_string().repeat(64))
    }

    fn cfg(dir: &std::path::Path, desired: &str, self_name: &str) -> UpdaterConfig {
        let mut c = UpdaterConfig::new(
            "http://proxy:2375",
            "gbase",
            "validator",
            "http://validator:8080/readyz",
            dir.to_path_buf(),
            desired,
            self_name,
        );
        c.health_timeout = std::time::Duration::from_millis(50);
        c.health_poll_interval = std::time::Duration::from_millis(5);
        c
    }

    #[test]
    fn s1_good_digest_adopted() {
        let dir = tempdir().expect("tmp");
        let old_d = d('a');
        let new_d = d('b');
        let old_img = format!("ghcr.io/org/val:1.0.0@{old_d}");
        let new_img = format!("ghcr.io/org/val:1.0.1@{new_d}");

        let docker = MockDocker::new();
        docker.seed(ContainerSummary {
            id: "c1".into(),
            name: "validator-1".into(),
            image: old_img.clone(),
            compose_project: Some("gbase".into()),
            compose_service: Some("validator".into()),
        });

        // Seed current pin as old.
        let store = PinStore::new(dir.path());
        commit_pins(
            &store,
            None,
            &PinRecord {
                service: "validator".into(),
                image: old_img,
                digest: old_d,
                updated_at: "1".into(),
            },
        )
        .expect("seed pin");

        let mut updater = Updater::new(cfg(dir.path(), &new_img, "updater"));
        let health = ScriptedHealth::new(vec![Ok(())]);
        let out = tick(&mut updater, &docker, &health).expect("tick");
        assert_eq!(
            out,
            TickOutcome::Rolled {
                digest: new_d.clone()
            }
        );
        let (cur, prev) = load_pins(&store).expect("pins");
        assert_eq!(cur.unwrap().digest, new_d);
        assert_eq!(prev.unwrap().digest, d('a'));
        assert_eq!(updater.phase, Phase::Idle);
        // Recreate path used start.
        assert!(docker
            .calls()
            .iter()
            .any(|(m, p)| m == "POST" && p.contains("/start")));
    }

    #[test]
    fn s2_unhealthy_rolled_back() {
        let dir = tempdir().expect("tmp");
        let old_d = d('a');
        let new_d = d('b');
        let old_img = format!("ghcr.io/org/val:1.0.0@{old_d}");
        let new_img = format!("ghcr.io/org/val:1.0.1@{new_d}");

        let docker = MockDocker::new();
        docker.seed(ContainerSummary {
            id: "c1".into(),
            name: "validator-1".into(),
            image: old_img.clone(),
            compose_project: Some("gbase".into()),
            compose_service: Some("validator".into()),
        });
        let store = PinStore::new(dir.path());
        commit_pins(
            &store,
            None,
            &PinRecord {
                service: "validator".into(),
                image: old_img.clone(),
                digest: old_d.clone(),
                updated_at: "1".into(),
            },
        )
        .expect("seed");

        let mut updater = Updater::new(cfg(dir.path(), &new_img, "updater"));
        let health = ScriptedHealth::new(vec![Err(HealthError::NotReady { status: 503 })]);
        let out = tick(&mut updater, &docker, &health).expect("tick");
        assert!(matches!(out, TickOutcome::RolledBack { .. }));
        let (cur, _) = load_pins(&store).expect("pins");
        // current.json must remain the old digest (commit never happened).
        assert_eq!(cur.unwrap().digest, old_d);
    }

    #[test]
    fn s3_non_allowlisted_methods_rejected() {
        let docker = MockDocker::new();
        let err = docker.raw_call("DELETE", "/volumes/data").unwrap_err();
        assert!(matches!(
            err,
            crate::docker::DockerError::NotAllowlisted { .. }
        ));
        let err2 = docker.raw_call("POST", "/networks/create").unwrap_err();
        assert!(matches!(
            err2,
            crate::docker::DockerError::NotAllowlisted { .. }
        ));
        // No calls recorded when rejected before push... raw_call only pushes on success.
        assert!(docker.calls().is_empty());
    }

    #[test]
    fn s4_never_recreates_self() {
        let dir = tempdir().expect("tmp");
        let new_d = d('b');
        let new_img = format!("ghcr.io/org/updater:1@{new_d}");
        let docker = MockDocker::new();
        docker.seed(ContainerSummary {
            id: "up1".into(),
            name: "updater".into(),
            image: format!("ghcr.io/org/updater:0@{}", d('a')),
            compose_project: Some("gbase".into()),
            compose_service: Some("validator".into()), // misconfig still protected by name
        });
        let mut updater = Updater::new(cfg(dir.path(), &new_img, "updater"));
        let health = ScriptedHealth::new(vec![Ok(())]);
        let out = tick(&mut updater, &docker, &health).expect("tick");
        assert_eq!(
            out,
            TickOutcome::RefusedSelf {
                name: "updater".into()
            }
        );
        assert!(
            !docker
                .calls()
                .iter()
                .any(|(m, p)| m == "POST" && p.contains("/create")),
            "must not create/recreate self: {:?}",
            docker.calls()
        );
    }

    #[test]
    fn s5_rejects_unpinned_desired() {
        let dir = tempdir().expect("tmp");
        let docker = MockDocker::new();
        docker.seed(ContainerSummary {
            id: "c1".into(),
            name: "v".into(),
            image: format!("x@{}", d('a')),
            compose_project: Some("gbase".into()),
            compose_service: Some("validator".into()),
        });
        let mut updater = Updater::new(cfg(dir.path(), "ghcr.io/org/val:latest", "updater"));
        let health = ScriptedHealth::new(vec![]);
        let out = tick(&mut updater, &docker, &health).expect("tick");
        assert!(matches!(out, TickOutcome::RejectedImage { .. }));
    }

    #[test]
    fn already_current_short_circuit() {
        let dir = tempdir().expect("tmp");
        let dig = d('c');
        let img = format!("ghcr.io/org/val@{dig}");
        let docker = MockDocker::new();
        docker.seed(ContainerSummary {
            id: "c1".into(),
            name: "v".into(),
            image: img.clone(),
            compose_project: Some("gbase".into()),
            compose_service: Some("validator".into()),
        });
        let mut updater = Updater::new(cfg(dir.path(), &img, "updater"));
        let health = ScriptedHealth::new(vec![]);
        let out = tick(&mut updater, &docker, &health).expect("tick");
        assert_eq!(out, TickOutcome::AlreadyCurrent);
        assert!(!docker
            .calls()
            .iter()
            .any(|(m, p)| m == "POST" && p.contains("images/create")));
    }
}
