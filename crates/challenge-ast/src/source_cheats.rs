//! Cheap source-only cheat screens (no GPU, no private eval assets).

/// Kind of static source hit (maps to agentic `CheatCode` at the call site).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceCheatKind {
    /// Hardcoded `METRICS_JSON=` short-circuit.
    EvalShortCircuit,
    /// Missing Prism telemetry hooks in `training.py`.
    MissingTelemetryHooks,
}

/// One static source finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceCheatHit {
    /// Cheat kind.
    pub kind: SourceCheatKind,
    /// Human-readable reason (safe to surface in `error_detail`).
    pub rationale: String,
}

/// Scan miner sources for cheap, deterministic cheat patterns.
#[must_use]
pub fn static_source_cheat(architecture_py: &str, training_py: &str) -> Option<SourceCheatHit> {
    for (path, src) in [
        ("architecture.py", architecture_py),
        ("training.py", training_py),
    ] {
        if src.contains("METRICS_JSON=") {
            return Some(SourceCheatHit {
                kind: SourceCheatKind::EvalShortCircuit,
                rationale: format!("static: hardcoded METRICS_JSON in {path}"),
            });
        }
    }
    if !training_has_telemetry_hooks(training_py) {
        return Some(SourceCheatHit {
            kind: SourceCheatKind::MissingTelemetryHooks,
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
    training_py.contains(".report(") && training_py.contains("finish_evaluation(") && imports_shim
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
        assert_eq!(hit.kind, SourceCheatKind::EvalShortCircuit);
    }

    #[test]
    fn missing_hooks() {
        let hit = static_source_cheat(
            "def build_model(ctx):\n    pass\n",
            "def train(m, ctx):\n    return {}\n",
        )
        .expect("hit");
        assert_eq!(hit.kind, SourceCheatKind::MissingTelemetryHooks);
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
