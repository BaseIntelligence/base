//! Shared types for offers, specs, instances, exec results.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Rentable GPU offer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Offer {
    /// Provider offer / executor id.
    pub id: String,
    /// GPU type string from provider (e.g. `NVIDIA A100-SXM4-80GB`).
    pub gpu_type: String,
    /// GPUs on the offer.
    pub gpu_count: u32,
    /// Price per GPU-hour (USD).
    pub price_per_hour: f64,
    /// Provider label.
    pub provider: String,
}

/// Plausible discrete GPU counts on marketplace offers (not SKU numbers).
fn plausible_gpu_count(n: u32) -> bool {
    (2..=16).contains(&n)
}

/// GPUs to rent per Prism eval pod, from `PRISM_POD_GPU_COUNT`.
///
/// Bounded to `1..=8`; anything absent, unparseable or out of range falls back
/// to the default **4** (recipe-v10 rents 4×RTX 5090).
///
/// Multi-GPU contract: miners may train across all 4 GPUs (the harness exposes
/// `gpu_count` in the miner `ctx`), but the eval battery stays pinned to GPU 0
/// so G7 timings stay comparable across submissions.
#[must_use]
pub fn pod_gpu_count_from_env() -> u32 {
    parse_pod_gpu_count(std::env::var("PRISM_POD_GPU_COUNT").ok().as_deref())
}

/// Default GPUs per Prism eval pod when unset (recipe-v10 rents 4×RTX 5090).
pub const DEFAULT_POD_GPU_COUNT: u32 = 4;

/// Pure core of [`pod_gpu_count_from_env`] (kept separate so bounds and
/// garbage-fallback are testable without mutating process env).
#[must_use]
pub fn parse_pod_gpu_count(raw: Option<&str>) -> u32 {
    raw.and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|n| (1..=8).contains(n))
        .unwrap_or(DEFAULT_POD_GPU_COUNT)
}

/// Parse a multi-GPU multiplier from a provider label (`8x RTX 5090`,
/// `RTX 5090 x8`, `8×GeForce`, `8 x RTX`, …). Returns `None` when the label
/// does not clearly encode a count. SKU digits like `5090` / `H100` are
/// ignored via [`plausible_gpu_count`].
#[must_use]
pub fn gpu_count_from_label(label: &str) -> Option<u32> {
    let s = label.to_ascii_lowercase().replace('×', "x");
    let bytes = s.as_bytes();
    let mut best: Option<u32> = None;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'x' {
            // Leading: `8x` / `8 x`
            let mut l = i;
            while l > 0 && bytes[l - 1].is_ascii_whitespace() {
                l -= 1;
            }
            let end_num = l;
            while l > 0 && bytes[l - 1].is_ascii_digit() {
                l -= 1;
            }
            if l < end_num {
                if let Ok(n) = s[l..end_num].parse::<u32>() {
                    if plausible_gpu_count(n) {
                        best = Some(best.map_or(n, |b| b.max(n)));
                    }
                }
            }
            // Trailing: `x8` / `x 8`
            let mut r = i + 1;
            while r < bytes.len() && bytes[r].is_ascii_whitespace() {
                r += 1;
            }
            let start_num = r;
            while r < bytes.len() && bytes[r].is_ascii_digit() {
                r += 1;
            }
            if start_num < r {
                if let Ok(n) = s[start_num..r].parse::<u32>() {
                    if plausible_gpu_count(n) {
                        best = Some(best.map_or(n, |b| b.max(n)));
                    }
                }
            }
        }
        i += 1;
    }
    best
}

/// Effective GPU count for selection: max of the numeric field and any
/// multiplier encoded in the type / machine label. Defaults to at least 1.
#[must_use]
pub fn effective_gpu_count(gpu_count: u32, gpu_type: &str) -> u32 {
    let from_label = gpu_count_from_label(gpu_type).unwrap_or(1);
    gpu_count.max(1).max(from_label)
}

impl Offer {
    /// True when this offer is multi-GPU (field or label).
    #[must_use]
    pub fn is_multi_gpu(&self) -> bool {
        effective_gpu_count(self.gpu_count, &self.gpu_type) > 1
    }

    /// Whether this offer may be rented for `requested` GPUs.
    ///
    /// A 1-GPU request still hard-rejects multi-GPU hosts (live SKU pin).
    /// A multi-GPU request accepts a larger host (`effective >= requested`)
    /// when no exact 4× offer is listed. Lium rejects GPU splitting, so the
    /// client rents `gpu_count = effective` (the whole host); the miner caps
    /// DDP at the requested width.
    #[must_use]
    pub fn matches_gpu_count(&self, requested: u32) -> bool {
        let effective = effective_gpu_count(self.gpu_count, &self.gpu_type);
        if requested <= 1 {
            return effective == 1;
        }
        effective >= requested
    }
}

/// Provision request with mandatory cost guardrails.
#[derive(Debug, Clone)]
pub struct InstanceSpec {
    /// Pod name (unique per rent).
    pub name: String,
    /// Max lifetime hours (≥ 1).
    pub max_lifetime_hours: f64,
    /// Max price per GPU-hour.
    pub max_price_per_hour: f64,
    /// GPU count requested.
    pub gpu_count: u32,
    /// Optional image / template digest pin (integrity).
    pub image_digest: Option<String>,
    /// SSH public keys (required for real Lium rent).
    pub ssh_public_keys: Vec<String>,
    /// Optional Lium SSH key name for `ensure_ssh_key`.
    pub ssh_key_name: Option<String>,
    /// Optional preferred offer id (skip cheapest-of-filter).
    pub preferred_offer_id: Option<String>,
    /// Optional Lium template id (required by rent API if no dockerfile).
    pub template_id: Option<String>,
    /// Optional template name to ensure/resolve (e.g. prism-mission-e2e).
    pub template_name: Option<String>,
}

impl Default for InstanceSpec {
    fn default() -> Self {
        Self {
            name: "prism-eval".into(),
            max_lifetime_hours: 1.0,
            max_price_per_hour: 2.5,
            gpu_count: DEFAULT_POD_GPU_COUNT,
            image_digest: None,
            ssh_public_keys: vec![],
            ssh_key_name: Some("prism-mission-worker".into()),
            preferred_offer_id: None,
            template_id: None,
            template_name: Some("prism-mission-e2e".into()),
        }
    }
}

/// Provisioned instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instance {
    /// Pod id.
    pub id: String,
    /// Status string (`RUNNING`, `CREATION_FAILED`, …).
    pub status: String,
    /// Provider label.
    pub provider: String,
    /// GPU type if known from pod payload.
    #[serde(default)]
    pub gpu_type: Option<String>,
    /// Raw `ssh_connect_cmd` when present (no secrets).
    #[serde(default)]
    pub ssh_connect_cmd: Option<String>,
}

/// Ordered GPU preference list — Prism live rents are a **hard SKU pin**.
#[derive(Debug, Clone, Default)]
pub struct GpuPreference {
    /// Substrings matched against `Offer.gpu_type` (case-insensitive), first wins.
    pub prefer: Vec<String>,
}

impl GpuPreference {
    /// Default PRISM SKU pin: **RTX 5090 only** (fail-closed).
    ///
    /// Ranking fairness requires a single SKU: wall-capped trains on a slower
    /// card (e.g. 4090) see fewer tokens → worse bpb. GPU width is enforced
    /// separately by [`Offer::matches_gpu_count`].
    #[must_use]
    pub fn default_prism() -> Self {
        Self {
            prefer: vec!["RTX 5090".into()],
        }
    }

    /// True when `gpu_type` matches any pin needle (case-insensitive substring).
    #[must_use]
    pub fn matches_pin(&self, gpu_type: &str) -> bool {
        let upper = gpu_type.to_ascii_uppercase();
        self.prefer
            .iter()
            .any(|needle| upper.contains(&needle.to_ascii_uppercase()))
    }

    /// Rank offers: lower is better. Unmatched get large rank.
    #[must_use]
    pub fn rank(&self, gpu_type: &str) -> usize {
        let upper = gpu_type.to_ascii_uppercase();
        for (i, needle) in self.prefer.iter().enumerate() {
            if upper.contains(&needle.to_ascii_uppercase()) {
                return i;
            }
        }
        usize::MAX / 2
    }

    /// Keep only pin + gpu-count matches, then sort (pin rank, count, price).
    pub fn filter_sort_offers(&self, offers: &mut Vec<Offer>, requested_gpus: u32) {
        offers.retain(|o| o.matches_gpu_count(requested_gpus) && self.matches_pin(&o.gpu_type));
        offers.sort_by(|a, b| {
            self.rank(&a.gpu_type)
                .cmp(&self.rank(&b.gpu_type))
                .then_with(|| {
                    let ac = effective_gpu_count(a.gpu_count, &a.gpu_type);
                    let bc = effective_gpu_count(b.gpu_count, &b.gpu_type);
                    ac.cmp(&bc).then_with(|| {
                        a.price_per_hour
                            .partial_cmp(&b.price_per_hour)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                })
        });
    }
}

/// One telemetry point reported by the miner's `training.py` through the
/// harness-provided `prism_telemetry` shim (`report(loss=, step=, …)`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryPoint {
    /// Optimizer step index reported by the miner.
    pub step: u64,
    /// Training loss at `step`.
    pub loss: f64,
    /// Global gradient norm when the miner reported it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grad_norm: Option<f64>,
    /// Seconds since train start on the pod clock.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at_secs: Option<f64>,
    /// Per-layer gradient/activation stats (miner-declared, bounded in-pod).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_stats: Option<serde_json::Value>,
}

/// Telemetry bundle captured by the harness for one eval.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EvalTelemetry {
    /// Loss series (decimated in-pod to a bounded length).
    #[serde(default)]
    pub loss_series: Vec<TelemetryPoint>,
    /// Why training ended: `finish_evaluation` (miner signal) or
    /// `train_returned`. The wall-clock cap is a failure path, not an end
    /// reason, so it never appears here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Total `report()` calls, including decimated-away points.
    #[serde(default)]
    pub report_count: u64,
}

/// One G6 intermediate-probe point captured by the harness during training
/// (recipe ≥1.3.0): the harness evaluates its fixed probe texts on the
/// in-memory model every `PRISM_PROBE_EVERY`-th telemetry report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbePoint {
    /// Optimizer step at probe time (as reported by the miner).
    pub step: u64,
    /// Harness-counted train tokens consumed at probe time.
    pub tokens_seen: u64,
    /// Wall seconds since train start (pod clock).
    pub wall_s: f64,
    /// Teacher-forced CE on the harness-held probe texts.
    pub probe_loss: f64,
}

/// Result of a remote (or sim) PRISM eval execution.
///
/// Deserialization accepts both METRICS_JSON v1 (recipe ≤1.2.x) and v2
/// (recipe ≥1.3.0): every v2 field is optional and defaults to `None`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RemoteExecResult {
    /// Finite quality metric used for scoring (higher is better after inversion if needed).
    /// PRISM prequential BPB: lower is better; we store raw bpb and map in score layer.
    pub bpb: f64,
    /// Optional tokens seen.
    pub tokens_seen: u64,
    /// Optional wall-clock seconds.
    pub wall_clock_seconds: f64,
    /// GPU type used (if known).
    pub gpu_type: Option<String>,
    /// Free-form notes (no secrets).
    pub notes: String,
    /// Model parameter count measured in-pod after `build_model`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_params: Option<u64>,
    /// Frozen val rows scored by the harness.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub val_rows: Option<u64>,
    /// Miner-reported training telemetry (recipe ≥1.1.0 harnesses).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<EvalTelemetry>,
    /// METRICS_JSON schema version (`2` for recipe ≥1.3.0 harnesses).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics_version: Option<u32>,
    /// Where `tokens_seen` came from: `"train_stream"` (authoritative
    /// harness counter) or `"legacy"` (miner bypassed the stream; the
    /// pre-1.3.0 row-count fallback).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_seen_source: Option<String>,
    /// G6 intermediate-probe curve (loss vs tokens during training).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe_curve: Option<Vec<ProbePoint>>,
    /// Pod manifest: nvidia-smi -q snapshot, versions, netns/unshare facts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pod_manifest: Option<serde_json::Value>,
    /// Whether the miner subprocess ran inside an empty network namespace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub netns: Option<bool>,
    /// SHA-256 hex of the harness file set uploaded to the pod.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harness_files_sha256: Option<String>,
    /// Eval tier realized by the v3 two-phase harness flow: `"private"`
    /// when operator eval assets were staged post-train, `"public_dev"`
    /// otherwise. Absent on v1 payloads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_tier: Option<String>,
    /// Forward-compatible passthrough of METRICS_JSON v2+ fields this crate
    /// keeps opaque (`battery`, `train_metrics`, `cap_exceeded`, …): keys
    /// not modeled above are retained verbatim so master-side consumers
    /// (`prism-eval-store::finalize_composite`, the orchestrator cap guard)
    /// read them from the re-serialized blob. Absent on v1 payloads.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, serde_json::Value>,
}

/// Optional Real-client SSH configuration (paths only; never logs key material).
#[derive(Debug, Clone, Default)]
pub struct LiumSshConfig {
    /// Path to ed25519 private key used for pod SSH.
    pub private_key_path: Option<PathBuf>,
    /// Max seconds waiting for RUNNING after rent.
    pub running_timeout_secs: u64,
    /// SSH attempts while pod networking settles.
    pub ssh_attempts: u32,
    /// Seconds between SSH retries.
    pub ssh_retry_secs: u64,
    /// Kitchen-timer training cap (hours) the harness enforces in-pod.
    pub train_hours_cap: f64,
}

impl LiumSshConfig {
    /// Defaults matching historical `live_lium_e2e` smoke.
    #[must_use]
    pub fn default_live() -> Self {
        Self {
            private_key_path: None, // resolve via LIUM_SSH_PRIVATE_KEY / default path
            // The digest-pinned CUDA/TE image is large and a cold provider
            // cache can spend several minutes pulling layers before RUNNING.
            running_timeout_secs: 900,
            ssh_attempts: 8,
            ssh_retry_secs: 5,
            train_hours_cap: prism_recipe::train_hours_cap(),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn default_prism_hard_pins_5090_only() {
        let p = GpuPreference::default_prism();
        assert_eq!(p.prefer.as_slice(), ["RTX 5090"]);
        assert!(p.matches_pin("NVIDIA GeForce RTX 5090"));
        assert!(!p.matches_pin("NVIDIA GeForce RTX 4090"));
        assert!(!p.matches_pin("NVIDIA H100"));
        assert!(!p.matches_pin("NVIDIA A100-SXM4-80GB"));
        assert!(p.rank("NVIDIA GeForce RTX 5090") < p.rank("NVIDIA GeForce RTX 4090"));
    }

    #[test]
    fn gpu_count_from_label_parses_multipliers() {
        assert_eq!(gpu_count_from_label("8x RTX 5090"), Some(8));
        assert_eq!(gpu_count_from_label("8× NVIDIA GeForce RTX 5090"), Some(8));
        assert_eq!(gpu_count_from_label("8 x RTX 5090"), Some(8));
        assert_eq!(gpu_count_from_label("RTX 5090 x8"), Some(8));
        assert_eq!(gpu_count_from_label("RTX 5090 x 8"), Some(8));
        assert_eq!(gpu_count_from_label("NVIDIA GeForce RTX 5090"), None);
        assert_eq!(gpu_count_from_label("H100x8"), Some(8));
    }

    #[test]
    fn pod_gpu_count_defaults_to_four_and_bounds() {
        // Default when unset / empty / garbage (recipe-v10 rents 4×RTX 5090).
        assert_eq!(parse_pod_gpu_count(None), 4);
        assert_eq!(parse_pod_gpu_count(Some("")), 4);
        assert_eq!(parse_pod_gpu_count(Some("   ")), 4);
        assert_eq!(parse_pod_gpu_count(Some("four")), 4);
        assert_eq!(parse_pod_gpu_count(Some("2.5")), 4);
        assert_eq!(parse_pod_gpu_count(Some("-1")), 4);
        // Out of the 1..=8 band falls back to the default, never clamps.
        assert_eq!(parse_pod_gpu_count(Some("0")), 4);
        assert_eq!(parse_pod_gpu_count(Some("9")), 4);
        assert_eq!(parse_pod_gpu_count(Some("64")), 4);
        // In-band values are honored (with surrounding whitespace).
        assert_eq!(parse_pod_gpu_count(Some("1")), 1);
        assert_eq!(parse_pod_gpu_count(Some("4")), 4);
        assert_eq!(parse_pod_gpu_count(Some("8")), 8);
        assert_eq!(parse_pod_gpu_count(Some(" 2 ")), 2);
        // The env wrapper agrees with the pure core for the unset case.
        assert_eq!(DEFAULT_POD_GPU_COUNT, 4);
    }

    #[test]
    fn four_gpu_request_matches_only_four_gpu_offers() {
        let mk = |id: &str, gpu_type: &str, gpu_count: u32, price: f64| Offer {
            id: id.into(),
            gpu_type: gpu_type.into(),
            gpu_count,
            price_per_hour: price,
            provider: "lium".into(),
        };
        let one = mk("1x", "NVIDIA GeForce RTX 5090", 1, 2.0);
        let four = mk("4x", "NVIDIA GeForce RTX 5090", 4, 1.0);
        let four_label = mk("4x-label", "4x RTX 5090", 1, 0.9);
        let eight = mk("8x", "NVIDIA GeForce RTX 5090", 8, 0.48);

        // A request for 4 must not silently land on a 1×GPU offer.
        assert!(!one.matches_gpu_count(4));
        assert!(four.matches_gpu_count(4));
        assert!(four_label.matches_gpu_count(4), "label multiplier wins");
        assert!(
            eight.matches_gpu_count(4),
            "8x host can rent 4 cards when no exact 4× offer exists"
        );

        // Prefer exact 4× (smaller count first, then cheaper). 8× is fallback.
        let pref = GpuPreference::default_prism();
        let mut offers = vec![one.clone(), four.clone(), four_label.clone(), eight.clone()];
        pref.filter_sort_offers(&mut offers, 4);
        let ids: Vec<&str> = offers.iter().map(|o| o.id.as_str()).collect();
        assert_eq!(
            ids,
            ["4x-label", "4x", "8x"],
            "exact 4× first, then larger 5090 hosts"
        );

        // Sanity: the historical single-GPU path is unchanged.
        let mut single_req = vec![one, four, four_label, eight];
        pref.filter_sort_offers(&mut single_req, 1);
        let ids: Vec<&str> = single_req.iter().map(|o| o.id.as_str()).collect();
        assert_eq!(ids, ["1x"]);
    }

    #[test]
    fn offer_hard_rejects_multi_gpu_for_single_request() {
        let single = Offer {
            id: "1x".into(),
            gpu_type: "NVIDIA GeForce RTX 5090".into(),
            gpu_count: 1,
            price_per_hour: 2.0,
            provider: "lium".into(),
        };
        let eight = Offer {
            id: "8x".into(),
            gpu_type: "NVIDIA GeForce RTX 5090".into(),
            gpu_count: 8,
            price_per_hour: 0.48,
            provider: "lium".into(),
        };
        let eight_label = Offer {
            id: "8x-label".into(),
            gpu_type: "8x RTX 5090".into(),
            gpu_count: 1, // lying field; label wins
            price_per_hour: 0.48,
            provider: "lium".into(),
        };
        assert!(single.matches_gpu_count(1));
        assert!(!eight.matches_gpu_count(1));
        assert!(!eight_label.matches_gpu_count(1));
        assert!(eight.is_multi_gpu());
    }

    #[test]
    fn remote_exec_result_parses_pre_telemetry_harness_json() {
        // Recipe ≤1.0.2 harnesses print neither telemetry nor n_params.
        let v: RemoteExecResult = serde_json::from_str(
            r#"{"bpb":1.5,"tokens_seen":2048,"wall_clock_seconds":12.0,
                "gpu_type":"NVIDIA A100","notes":"recipe-v1","recipe":"1.0.2"}"#,
        )
        .unwrap();
        assert!(v.telemetry.is_none());
        assert!(v.n_params.is_none());
        assert!(v.val_rows.is_none());
    }

    #[test]
    fn remote_exec_result_parses_telemetry_payload() {
        let v: RemoteExecResult = serde_json::from_str(
            r#"{"bpb":1.5,"tokens_seen":2048,"wall_clock_seconds":12.0,
                "gpu_type":"NVIDIA RTX 5090","notes":"recipe-v1","n_params":12400000,
                "val_rows":256,
                "telemetry":{"finish_reason":"finish_evaluation","report_count":2,
                  "loss_series":[{"step":1,"loss":4.0,"grad_norm":0.5,"at_secs":0.2,
                    "layer_stats":{"head":0.1}},{"step":2,"loss":3.0}]}}"#,
        )
        .unwrap();
        let t = v.telemetry.expect("telemetry");
        assert_eq!(t.finish_reason.as_deref(), Some("finish_evaluation"));
        assert_eq!(t.loss_series.len(), 2);
        assert_eq!(t.loss_series[0].step, 1);
        assert!(t.loss_series[1].grad_norm.is_none());
        assert_eq!(v.n_params, Some(12_400_000));
    }

    #[test]
    fn remote_exec_result_parses_metrics_v2_payload() {
        let v: RemoteExecResult = serde_json::from_str(
            r#"{"bpb":2.5,"tokens_seen":1048576,"wall_clock_seconds":780.0,
                "gpu_type":"NVIDIA GeForce RTX 5090","notes":"recipe-v2 val_ce->bpb",
                "val_rows":256,"n_params":12400000,"recipe":"1.3.0",
                "metrics_version":2,"tokens_seen_source":"train_stream","netns":true,
                "harness_files_sha256":"91a8737e",
                "probe_curve":[{"step":25,"tokens_seen":102400,"wall_s":12.5,"probe_loss":4.25}],
                "train_metrics":{"train_loss":3.1,"train_steps":500},
                "pod_manifest":{"python":"3.12.3","netns":true,
                  "unshare":{"available":true,"detail":"probe ok"}},
                "telemetry":{"finish_reason":"train_returned","report_count":20,
                  "loss_series":[]}}"#,
        )
        .unwrap();
        assert_eq!(v.metrics_version, Some(2));
        assert_eq!(v.tokens_seen_source.as_deref(), Some("train_stream"));
        assert_eq!(v.netns, Some(true));
        assert_eq!(v.harness_files_sha256.as_deref(), Some("91a8737e"));
        let curve = v.probe_curve.expect("probe_curve");
        assert_eq!(curve.len(), 1);
        assert_eq!(curve[0].step, 25);
        assert_eq!(curve[0].tokens_seen, 102_400);
        assert!(curve[0].probe_loss > 0.0);
        let pm = v.pod_manifest.expect("pod_manifest");
        assert_eq!(pm["python"], "3.12.3");
        // v1 keys still intact on the v2 payload.
        assert!(v.bpb.is_finite() && v.bpb > 0.0);
        assert_eq!(v.tokens_seen, 1_048_576);
        assert!(v.telemetry.is_some());
    }

    #[test]
    fn remote_exec_result_v2_fields_default_absent() {
        // Mixed-version payloads (e.g. metrics_version set, other v2 keys
        // absent) must still parse — every v2 field is optional.
        let v: RemoteExecResult = serde_json::from_str(
            r#"{"bpb":1.5,"tokens_seen":2048,"wall_clock_seconds":12.0,
                "gpu_type":"SIM","notes":"sim-eval","metrics_version":2}"#,
        )
        .unwrap();
        assert_eq!(v.metrics_version, Some(2));
        assert!(v.tokens_seen_source.is_none());
        assert!(v.probe_curve.is_none());
        assert!(v.pod_manifest.is_none());
        assert!(v.netns.is_none());
        assert!(v.harness_files_sha256.is_none());
    }
}
