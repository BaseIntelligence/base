//! Cheap source-only cheat screens (no GPU, no private eval assets).

/// Kind of static source hit (maps to agentic `CheatCode` at the call site).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceCheatKind {
    /// Hardcoded `METRICS_JSON=` short-circuit.
    EvalShortCircuit,
    /// Missing Prism telemetry hooks in `training.py`.
    MissingTelemetryHooks,
    /// Dense sequence/time mix without a causal mask (MLP-Mixer / `TokenMix`
    /// label leak into next-token CE).
    NonCausalLabelLeak,
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
    if let Some(why) = noncausal_seq_mix_reason(architecture_py) {
        return Some(SourceCheatHit {
            kind: SourceCheatKind::NonCausalLabelLeak,
            rationale: format!("static: non-causal sequence mix label leak — {why}"),
        });
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

/// Detect MLP-Mixer / `TokenMix`-style dense mixes across the time axis with no
/// causal mask. Under next-token CE, position `t` can see tokens `t+1…`
/// (including the label), so val BPB collapses without real language modeling.
///
/// Heuristic (pre-pod, fail-closed on this class):
/// - time↔feature `transpose(1, 2)` (or equivalent), and
/// - a sequence-mixer `Linear` / `TokenMix` naming pattern, and
/// - no obvious causal mask (`triu` / `tril` / `is_causal` / `causal_mask`).
#[must_use]
pub fn arch_has_noncausal_seq_mix(architecture_py: &str) -> bool {
    noncausal_seq_mix_reason(architecture_py).is_some()
}

fn noncausal_seq_mix_reason(architecture_py: &str) -> Option<&'static str> {
    let src = architecture_py;
    if src.trim().is_empty() {
        return None;
    }
    // Allow-list: any explicit causal / attention mask construction.
    let has_causal = src.contains("torch.triu")
        || src.contains("torch.tril")
        || src.contains("F.triu")
        || src.contains("F.tril")
        || src.contains("is_causal")
        || src.contains("causal_mask")
        || src.contains("create_causal")
        || src.contains("generate_square_subsequent_mask")
        || src.contains("attn_mask")
        || src.contains("attention_mask");
    if has_causal {
        return None;
    }

    let has_time_transpose = src.contains("transpose(1, 2)")
        || src.contains("transpose(1,2)")
        || src.contains("transpose(-1, -2)")
        || src.contains("transpose(-2, -1)")
        || src.contains(".mT")
        || (src.contains("einops.rearrange")
            && src.contains("b t d")
            && src.contains("b d t"));

    if !has_time_transpose {
        return None;
    }

    let named_mixer = src.contains("TokenMix")
        || src.contains("token_mix")
        || src.contains("TokenMixing")
        || src.contains("t_mix")
        || src.contains("seq_mix")
        || src.contains("time_mix")
        || src.contains("MixerBlock")
        || src.contains("MLPMixer")
        || src.contains("mlp_mixer");

    let seq_linear = src.contains("nn.Linear(seq")
        || src.contains("nn.Linear(block")
        || src.contains("Linear(seq")
        || src.contains("Linear(block")
        || src.contains("Linear(self.block")
        || src.contains("Linear(self.seq")
        || src.contains("Linear(self.block_size")
        || src.contains("Linear(self.max_seq");

    if named_mixer || seq_linear {
        return Some(
            "dense Linear/MLP mixes the full sequence axis after transpose without a causal mask",
        );
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

    /// Sanitized `TokenMix` label-leak fixture (prod `b99a7047` class).
    const TOKENMIX_LEAK: &str = r"
import torch
import torch.nn as nn

class TokenMix(nn.Module):
    def __init__(self, seq: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq, hidden),
            nn.GELU(),
            nn.Linear(hidden, seq),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)

def build_model(ctx):
    return TokenMix(512, 1024)
";

    const CLEAN_TRAIN: &str = concat!(
        "import prism_telemetry\n",
        "def train(m, ctx):\n",
        "    prism_telemetry.report(loss=1.0, step=1)\n",
        "    prism_telemetry.finish_evaluation()\n",
        "    return {}\n",
    );

    #[test]
    fn tokenmix_label_leak_is_static_cheat() {
        let hit = static_source_cheat(TOKENMIX_LEAK, CLEAN_TRAIN).expect("hit");
        assert_eq!(hit.kind, SourceCheatKind::NonCausalLabelLeak);
        assert!(arch_has_noncausal_seq_mix(TOKENMIX_LEAK));
    }

    #[test]
    fn mixer_t_mix_pattern_is_static_cheat() {
        let arch = r"
class MixerBlock(nn.Module):
    def __init__(self, d, block):
        super().__init__()
        self.t_mix = nn.Sequential(
            nn.Linear(block, block * 2), nn.GELU(), nn.Linear(block * 2, block)
        )
    def forward(self, x):
        h = x.transpose(1, 2)
        h = self.t_mix(h)
        return x + h.transpose(1, 2)
def build_model(ctx):
    return MixerBlock(192, 512)
";
        let hit = static_source_cheat(arch, CLEAN_TRAIN).expect("hit");
        assert_eq!(hit.kind, SourceCheatKind::NonCausalLabelLeak);
    }

    #[test]
    fn causal_transformer_baseline_style_is_clean() {
        let arch = r"
import torch
import torch.nn as nn
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.tr = nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
    def forward(self, x):
        causal = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool()
        return self.tr(x, src_mask=causal)
def build_model(ctx):
    return Tiny()
";
        assert!(static_source_cheat(arch, CLEAN_TRAIN).is_none());
        assert!(!arch_has_noncausal_seq_mix(arch));
    }
}
