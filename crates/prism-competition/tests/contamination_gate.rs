//! Contamination gate with the env knob **on**.
//!
//! Own integration binary because
//! [`prism_competition::contamination::require_check`] memoizes in a
//! `OnceLock` — the default-off behavior is covered by the crate's unit
//! tests, where the env is unset, and cannot share a process with this.

#![allow(clippy::expect_used)]

use std::collections::BTreeMap;

use prism_competition::contamination::{checked, require_check, scoreable};
use prism_competition::sig::{plan_emission, SigContext};
use prism_store::FinalScore;
use serde_json::json;

fn blob(flag: Option<bool>) -> serde_json::Value {
    match flag {
        Some(f) => json!({"battery": {"tier": "public_dev", "mirror_defence": {
            "contamination_checked": f, "inert": !f, "pairs": 4,
            "inert_pairs": if f {0} else {4}, "live_pairs": if f {4} else {0},
        }}}),
        None => json!({"battery": {"tier": "public_dev"}}),
    }
}

#[test]
fn require_knob_gates_scoring_and_fails_closed_on_silence() {
    std::env::set_var("PRISM_EVAL_REQUIRE_PRIVATE", "1");
    assert!(require_check(), "the explicit knob must turn the gate on");

    // A live check scores.
    assert!(scoreable(&blob(Some(true))));
    // An inert check does not — this is the whole point of the fix.
    assert!(
        !scoreable(&blob(Some(false))),
        "an inert mirror defence must not be scoreable when required"
    );
    // Neither does an older harness that cannot report the flag.
    assert!(
        !scoreable(&blob(None)),
        "absent evidence must fail closed, not pass by silence"
    );
    assert!(!checked(&blob(None)));

    // And the emission path refuses independently of the finalize gate:
    // an unchecked round allocates nothing and burns the whole share.
    let credits: BTreeMap<String, FinalScore> = [
        ("champ".to_owned(), FinalScore::Score(900_000)),
        ("chal".to_owned(), FinalScore::Score(800_000)),
    ]
    .into_iter()
    .collect();
    let unchecked = SigContext {
        contamination_checked: checked(&blob(Some(false))),
        ..SigContext::default()
    };
    let plan = plan_emission(&credits, &unchecked);
    assert!(plan.shares.is_empty(), "unchecked round paid: {plan:?}");
    assert_eq!(plan.burn_bps, 10_000);
    assert!(plan.conserves());

    // The same round with live evidence pays, so the refusal is
    // attributable to the missing contamination check and nothing else.
    let live = SigContext {
        contamination_checked: checked(&blob(Some(true))),
        ..SigContext::default()
    };
    let ok = plan_emission(&credits, &live);
    assert!(!ok.shares.is_empty(), "a checked round must pay");
    assert!(ok.conserves());
}
