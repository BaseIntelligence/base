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
    /// Recipe 2.0 delta disables / mocks telemetry hooks.
    TelemetryDisable,
    /// Recipe 2.0 delta introduces network / exfil primitives.
    NetworkExfil,
    /// Recipe 2.0 delta references held-out / private eval assets.
    EvalSetLeak,
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
    static_source_cheat_inner(architecture_py, training_py, false)
}

/// Recipe 2.0 `AutoModel` path: scan the **delta** (patch + touched Python).
///
/// Skips the 1.x `training.py` telemetry-hook requirement (harness wraps the
/// `AutoModel` entry). Still hard-rejects telemetry disable / network exfil /
/// eval-set leakage patterns in added hunk lines, plus `METRICS_JSON` and
/// non-causal mixes in touched architecture Python.
#[must_use]
pub fn static_automodel_delta_cheat(
    patch_text: &str,
    touched_architecture_py: &str,
) -> Option<SourceCheatHit> {
    if let Some(hit) = static_patch_cheat(patch_text) {
        return Some(hit);
    }
    static_source_cheat_inner(touched_architecture_py, "", true)
}

/// Unified-diff delta scan (added lines only).
#[must_use]
pub fn static_patch_cheat(patch_text: &str) -> Option<SourceCheatHit> {
    let added = added_hunk_text(patch_text);
    if added.is_empty() {
        return None;
    }
    let lower = added.to_ascii_lowercase();

    for needle in [
        "disable_telemetry",
        "telemetry_disabled",
        "prism_telemetry = none",
        "prism_telemetry=none",
        "sys.modules['prism_telemetry']",
        "sys.modules[\"prism_telemetry\"]",
        "report = lambda",
        "finish_evaluation = lambda",
    ] {
        if lower.contains(needle) {
            return Some(SourceCheatHit {
                kind: SourceCheatKind::TelemetryDisable,
                rationale: format!("static delta: telemetry disable/mock ({needle})"),
            });
        }
    }

    for needle in [
        "import socket",
        "from socket ",
        "import urllib",
        "from urllib",
        "import requests",
        "from requests",
        "http.client",
        "aiohttp",
        "subprocess.",
        "subprocess.call",
        "subprocess.popen",
        "os.system(",
        "curl ",
        "wget ",
    ] {
        if lower.contains(needle) || added.contains(needle) {
            return Some(SourceCheatHit {
                kind: SourceCheatKind::NetworkExfil,
                rationale: format!("static delta: network/exfil primitive ({needle})"),
            });
        }
    }

    for needle in [
        "eval_pack",
        "held_out",
        "held-out",
        "private_eval",
        "private/eval",
        "/eval/private",
        "g8_private",
        "eval_tier=private",
        "eval_tier = private",
    ] {
        if lower.contains(needle) {
            return Some(SourceCheatHit {
                kind: SourceCheatKind::EvalSetLeak,
                rationale: format!("static delta: eval-set leakage pattern ({needle})"),
            });
        }
    }

    if added.contains("METRICS_JSON=") {
        return Some(SourceCheatHit {
            kind: SourceCheatKind::EvalShortCircuit,
            rationale: "static delta: hardcoded METRICS_JSON in patch".into(),
        });
    }
    None
}

fn static_source_cheat_inner(
    architecture_py: &str,
    training_py: &str,
    automodel_delta: bool,
) -> Option<SourceCheatHit> {
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
    if automodel_delta {
        // Harness wraps AutoModel train entry; miner training.py hooks N/A.
        return None;
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

/// Concatenate added hunk lines from a unified diff (skip `+++` headers).
fn added_hunk_text(patch_text: &str) -> String {
    let mut out = String::new();
    for line in patch_text.lines() {
        if line.starts_with('+') && !line.starts_with("+++") {
            out.push_str(&line[1..]);
            out.push('\n');
        }
    }
    out
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
        || (src.contains("einops.rearrange") && src.contains("b t d") && src.contains("b d t"));

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

    #[test]
    fn patch_telemetry_disable_is_cheat() {
        let patch = "\
diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -0,0 +1,2 @@
+import sys
+sys.modules['prism_telemetry'] = None
";
        let hit = static_patch_cheat(patch).expect("hit");
        assert_eq!(hit.kind, SourceCheatKind::TelemetryDisable);
    }

    #[test]
    fn patch_network_exfil_is_cheat() {
        let patch = "\
diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -0,0 +1,1 @@
+import socket
";
        let hit = static_patch_cheat(patch).expect("hit");
        assert_eq!(hit.kind, SourceCheatKind::NetworkExfil);
    }

    #[test]
    fn patch_eval_leak_is_cheat() {
        let patch = "\
diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -0,0 +1,1 @@
+path = 'private_eval/held_out.json'
";
        let hit = static_patch_cheat(patch).expect("hit");
        assert_eq!(hit.kind, SourceCheatKind::EvalSetLeak);
    }

    #[test]
    fn automodel_delta_skips_training_hooks() {
        assert!(static_automodel_delta_cheat(
            "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -0,0 +1,1 @@\n+x = 1\n",
            "def scale(x):\n    return x\n",
        )
        .is_none());
    }
}
