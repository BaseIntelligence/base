//! Cheap source-only cheat screens (no GPU, no private eval assets).
//!
//! Run these **before** renting a Lium pod so a bad submission fails fast
//! instead of burning hours of GPU. Metrics/receipt consistency checks stay
//! post-eval (they need harness output).

use crate::types::CheatCode;

/// One static source finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticCheatHit {
    /// Cheat taxonomy code.
    pub code: CheatCode,
    /// Human-readable reason (safe to surface in error_detail).
    pub rationale: String,
}

/// Scan miner sources for cheap, deterministic cheat patterns.
///
/// Order: hardcoded `METRICS_JSON=` short-circuit first, then missing Prism
/// telemetry hooks in `training.py`. Returns the first hit.
#[must_use]
pub fn static_source_cheat(architecture_py: &str, training_py: &str) -> Option<StaticCheatHit> {
    for (path, src) in [
        ("architecture.py", architecture_py),
        ("training.py", training_py),
    ] {
        if src.contains("METRICS_JSON=") {
            return Some(StaticCheatHit {
                code: CheatCode::EvalShortCircuit,
                rationale: format!("static: hardcoded METRICS_JSON in {path}"),
            });
        }
    }
    if !training_has_telemetry_hooks(training_py) {
        return Some(StaticCheatHit {
            code: CheatCode::MissingTelemetryHooks,
            rationale: "static: training.py missing prism_telemetry report/finish_evaluation hooks"
                .into(),
        });
    }
    None
}

/// Prism telemetry-hook contract (recipe ≥ 1.1.0).
#[must_use]
pub fn training_has_telemetry_hooks(training_py: &str) -> bool {
    let imports_shim = training_py.contains("prism_telemetry")
        || training_py.contains("ctx[\"telemetry\"]")
        || training_py.contains("ctx['telemetry']");
    let calls_report = training_py.contains(".report(");
    let calls_finish = training_py.contains("finish_evaluation(");
    imports_shim && calls_report && calls_finish
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_json_short_circuit() {
        let hit = static_source_cheat(
            "def build_model(ctx):\n    pass\n",
            "def train(m, ctx):\n    print('METRICS_JSON={}')\n",
        )
        .expect("hit");
        assert_eq!(hit.code, CheatCode::EvalShortCircuit);
    }

    #[test]
    fn missing_hooks() {
        let hit = static_source_cheat(
            "def build_model(ctx):\n    pass\n",
            "def train(m, ctx):\n    return {}\n",
        )
        .expect("hit");
        assert_eq!(hit.code, CheatCode::MissingTelemetryHooks);
    }

    #[test]
    fn clean_hooks() {
        let train = concat!(
            "import prism_telemetry\n",
            "def train(m, ctx):\n",
            "    prism_telemetry.report(loss=1.0, step=1)\n",
            "    prism_telemetry.finish_evaluation()\n",
            "    return {}\n",
        );
        assert!(static_source_cheat("def build_model(ctx):\n    pass\n", train).is_none());
    }
}
