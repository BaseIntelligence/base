//! Contamination-check policy: is a scored run's mirror defence live?
//!
//! **The defect this addresses.** The mirror-gap penalty is Prism's designed
//! contamination detector, and in the `public_dev` tier it is **inert by
//! construction**: `harness/eval/rollup.py::build_mirrors` sets
//! `mirror = dict(public)` when no private asset differs, so the gap is
//! identically 0 and the penalty deducts nothing however contaminated the
//! submission is. That was honestly labelled in a docstring, but nothing in
//! the run output said so — a scored `public_dev` run *looked*
//! contamination-checked when no check had run.
//!
//! **Two halves, and only one of them lives here.** The harness now emits
//! `battery.mirror_defence` (`contamination_checked`, `inert_pairs`,
//! `live_pairs`, a reason string) and logs a warning when the defence is
//! inert — that is the loud half. This module is the *policy* half: reading
//! that flag and deciding whether a run may be scored, or may hold the
//! protected champion share.
//!
//! **Why the policy is enforced at emission, not only at finalize.** The
//! strictest option would refuse to persist a composite at all for an
//! unchecked run. That call site is
//! `prism_eval_store::finalize::finalize_composite`, and the one-line change
//! is stated verbatim in [`FINALIZE_GATE_PATCH`] for whoever owns that file.
//! But the gate must not depend on that: significance-gated emission grants
//! a **protected 60 % champion share on measured evidence**, so it is the
//! emission path that must not hand statistical authority to a number whose
//! contamination detector was switched off. [`crate::sig`] therefore
//! fail-closes on its own — an unchecked round pays the champion nothing and
//! burns instead (see [`crate::sig::SigContext::contamination_checked`]).
//!
//! **Fail-closed on silence.** An absent flag reads as *unchecked*. An older
//! harness that cannot report the flag has not proven a check ran, and the
//! whole point of the defect is that absence of evidence was being read as
//! evidence of absence.
//!
//! **Why the `require` knob is not on by default.** `public_dev` is the tier
//! CI, Sim and local-e2e all run in, and every one of those runs is
//! *supposed* to have an inert mirror — they have no private pack staged. A
//! default-on gate would fail-closed on the entire existing test matrix
//! rather than on a real contamination risk. So the default is off, the
//! significance mode implies it, and the strictest available posture that
//! does not break CI is: **loud always, refused when it matters.**

/// The finalize-time gate, for the owner of `prism-eval-store`.
///
/// Refusing to *persist* an unchecked composite is strictly stronger than
/// refusing to pay it, because it also keeps the unchecked number out of
/// the carry set and off the public leaderboard. It is a one-line insert at
/// the top of `finalize_composite`, after `submission_metrics` resolves:
///
/// ```text
/// if prism_competition::contamination::require_check()
///     && !prism_competition::contamination::checked(metrics_v2)
/// {
///     return Err(FinalizeError::ContaminationUnchecked);
/// }
/// ```
///
/// plus a `ContaminationUnchecked` variant on `FinalizeError`. Recorded here
/// rather than applied because that crate is owned elsewhere; the emission
/// path in [`crate::sig`] does not wait for it.
pub const FINALIZE_GATE_PATCH: &str = "finalize_composite: refuse when require_check() && !checked";

/// Env knob: refuse to score/pay a run with no contamination evidence.
///
/// `PRISM_EVAL_REQUIRE_PRIVATE=1` turns the check on explicitly;
/// `PRISM_EMISSION_MODE=sig` **implies** it, because the significance rule
/// is the thing that makes an unchecked number dangerous. Both off by
/// default, so CI / Sim / local-e2e / the live `public_dev` path are
/// unaffected until an operator opts in.
#[must_use]
pub fn require_check() -> bool {
    static REQUIRE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *REQUIRE.get_or_init(|| {
        let explicit = std::env::var("PRISM_EVAL_REQUIRE_PRIVATE").is_ok_and(|v| v.trim() == "1");
        let implied = std::env::var("PRISM_EMISSION_MODE").is_ok_and(|v| v.trim() == "sig");
        explicit || implied
    })
}

/// Whether a `METRICS_JSON` v2 blob reports a **live** contamination check.
///
/// Reads `battery.mirror_defence.contamination_checked`. Absent, non-bool,
/// or `false` ⇒ `false`. Kept as a `&str`-keyed walk over
/// [`serde_json::Value`] so this crate does not need the eval-store types.
#[must_use]
pub fn checked(metrics_v2: &serde_json::Value) -> bool {
    metrics_v2
        .get("battery")
        .and_then(|b| b.get("mirror_defence"))
        .and_then(|m| m.get("contamination_checked"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

/// Human-readable reason the harness gave for the defence's state, if any.
/// Surfaced in operator views so "penalty = 0" is never read as "clean".
#[must_use]
pub fn reason(metrics_v2: &serde_json::Value) -> Option<&str> {
    metrics_v2
        .get("battery")?
        .get("mirror_defence")?
        .get("reason")?
        .as_str()
}

/// Whether this run may be scored at all under the current policy.
///
/// `true` when the check is not required, or when it is required and the
/// run proved it ran.
#[must_use]
pub fn scoreable(metrics_v2: &serde_json::Value) -> bool {
    !require_check() || checked(metrics_v2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn blob(flag: Option<bool>) -> serde_json::Value {
        match flag {
            Some(f) => json!({"battery": {"mirror_defence": {
                "contamination_checked": f,
                "inert": !f,
                "reason": if f { "all 4 mirror pairs are live" } else { "every mirror pair is degenerate" },
            }}}),
            None => json!({"battery": {"tier": "public_dev"}}),
        }
    }

    #[test]
    fn flag_reader_is_strict_about_shape() {
        assert!(checked(&blob(Some(true))));
        assert!(!checked(&blob(Some(false))));
        assert!(!checked(&blob(None)), "absent flag must fail closed");
        assert!(!checked(&json!({})));
        // A truthy string must not read as a live check.
        assert!(!checked(&json!({
            "battery": {"mirror_defence": {"contamination_checked": "yes"}}
        })));
        // Nor a truthy number.
        assert!(!checked(&json!({
            "battery": {"mirror_defence": {"contamination_checked": 1}}
        })));
    }

    #[test]
    fn reason_is_surfaced_when_present() {
        assert_eq!(
            reason(&blob(Some(false))),
            Some("every mirror pair is degenerate")
        );
        assert_eq!(reason(&json!({})), None);
    }

    #[test]
    fn default_policy_scores_everything() {
        // The env is unset in unit tests, so the gate is off and an inert
        // run still scores — this is the property that keeps CI / Sim /
        // local-e2e green. `require_check()` memoizes, so the on-path is
        // asserted in `tests/contamination_gate.rs`, which owns its env.
        assert!(!require_check());
        assert!(scoreable(&blob(Some(false))));
        assert!(scoreable(&blob(None)));
        assert!(scoreable(&blob(Some(true))));
    }
}
