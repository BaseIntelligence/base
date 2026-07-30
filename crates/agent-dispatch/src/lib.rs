//! Orchestrator ↔ runner task-dispatch wire surface for agent-v1.
//!
//! # Scope (this crate)
//! Task identifiers, dispatch envelopes, and the dispatcher trait used between
//! the challenge orchestrator and Harbor runners. Concrete protocol framing
//! and transport land in a later task.
//!
//! # What stays in `agent-challenge`
//! Scoring, NoScore / D24 completeness, signing, and weight submit remain in
//! `agent-challenge`. Dispatch only moves work units; it does not score them.
//!
//! Skeletons only — no network or queue implementation yet.

#![forbid(unsafe_code)]

use thiserror::Error;

use std::fmt;

/// Opaque task identifier assigned by the orchestrator.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskId(String);

impl TaskId {
    /// Construct from an already-validated task id string.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the raw id as `str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One unit of work handed from orchestrator to a runner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchRequest {
    /// Task identity.
    pub task_id: TaskId,
    /// Pack id string the runner must load (typed pack crate owns validation).
    pub pack_id: String,
}

/// Runner outcome reported back to the orchestrator (payload shape later).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchResult {
    /// Task identity echoed from the request.
    pub task_id: TaskId,
    /// Whether the runner completed without transport/runtime failure.
    pub ok: bool,
}

/// Failures while enqueueing or collecting dispatch work.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DispatchError {
    /// No runner accepted the task.
    #[error("no runner available for task {0}")]
    NoRunner(String),
    /// Dispatch transport or serialization failure (implementation later).
    #[error("dispatch failed: {0}")]
    Failed(String),
}

/// Orchestrator-facing handle that sends work to runners.
pub trait TaskDispatcher: Send + Sync {
    /// Enqueue one task for a runner.
    ///
    /// # Errors
    /// Returns [`DispatchError`] when no runner is available or send fails.
    fn dispatch(&self, request: DispatchRequest) -> Result<(), DispatchError>;
}

/// Crate identity for smoke / wiring checks.
#[must_use]
pub fn crate_name() -> &'static str {
    "agent-dispatch"
}

#[cfg(test)]
mod tests {
    use super::{
        crate_name, DispatchError, DispatchRequest, TaskDispatcher, TaskId,
    };

    struct RejectAll;

    impl TaskDispatcher for RejectAll {
        fn dispatch(&self, request: DispatchRequest) -> Result<(), DispatchError> {
            Err(DispatchError::NoRunner(request.task_id.to_string()))
        }
    }

    #[test]
    fn crate_name_is_agent_dispatch() {
        assert_eq!(crate_name(), "agent-dispatch");
    }

    #[test]
    fn reject_all_reports_no_runner() {
        let d = RejectAll;
        let req = DispatchRequest {
            task_id: TaskId::new("t-1"),
            pack_id: "p-1".into(),
        };
        let err = d.dispatch(req).expect_err("reject all");
        assert_eq!(err, DispatchError::NoRunner("t-1".into()));
    }
}
