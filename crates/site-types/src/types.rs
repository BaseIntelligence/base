//! CamelCase response types matching the frontend marketing contract.

use serde::{Deserialize, Serialize};

/// Arena identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArenaSlug {
    /// Coding (paused).
    Coding,
    /// Design challenge.
    Design,
    /// Prism challenge.
    Prism,
}

impl ArenaSlug {
    /// Parse slug string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "coding" => Some(Self::Coding),
            "design" => Some(Self::Design),
            "prism" => Some(Self::Prism),
            _ => None,
        }
    }

    /// Wire form.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Coding => "coding",
            Self::Design => "design",
            Self::Prism => "prism",
        }
    }
}

/// Scoring method label.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ScoringMethod {
    /// Coding pass-rate.
    PassRate,
    /// Design rating field (FE `elo`).
    Elo,
    /// Prism spectral fusion / BPB.
    SpectralFusion,
}

/// Reference chip on an arena card.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectReference {
    /// Display name.
    pub name: String,
    /// `owner/repo`.
    pub repo: String,
    /// Absolute URL.
    pub repo_url: String,
}

/// Arena card / overview.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Arena {
    /// Slug.
    pub slug: ArenaSlug,
    /// Display name.
    pub name: String,
    /// One-line tagline.
    pub tagline: String,
    /// Longer description.
    pub description: String,
    /// `live` | `beta` | `paused`.
    pub status: String,
    /// Scoring method.
    pub scoring: ScoringMethod,
    /// Mechanism steps.
    pub mechanism: Vec<String>,
    /// Distinct agents observed.
    pub agents: u32,
    /// Leading score display string.
    pub best_score: String,
    /// Label above best score.
    pub best_score_label: String,
    /// Emission share 0..1 (0 when unknown).
    pub emission_share: f64,
    /// Weight 0..1 (0 when unknown).
    pub weight: f64,
    /// TAO/day (0 when unknown).
    pub rewards_per_day: f64,
    /// References.
    pub references: Vec<ProjectReference>,
    /// Source link.
    pub source_url: String,
    /// Plate asset path.
    pub plate: String,
    /// Design round id when known (absent for coding/prism).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub round_id: Option<u64>,
    /// Design round close time (ISO-8601 UTC) when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub round_ends_at: Option<String>,
    /// Design seconds remaining in the current round when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seconds_remaining: Option<u64>,
}

/// Miner / agent identity for tables.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Agent {
    /// URL segment.
    pub slug: String,
    /// Display handle.
    pub handle: String,
    /// Printed miner number when known; `—` otherwise.
    pub miner_number: String,
    /// Declared model/stack when known; `—` otherwise.
    pub model: String,
    /// Operator label (truncated hotkey).
    pub operator: String,
    /// Full miner hotkey — SS58 when the upstream hex decodes, otherwise the
    /// raw upstream value; `—` when the miner is unknown. Copy targets read
    /// this, never the truncated `operator`.
    pub hotkey: String,
    /// Join epoch when known.
    pub joined_epoch: u64,
}

/// Leaderboard row (design ratings → `elo`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardRow {
    /// 1-based rank.
    pub rank: u32,
    /// Agent.
    pub agent: Agent,
    /// Rating (design) or comparable score.
    pub elo: f64,
    /// Wins.
    pub wins: u32,
    /// Losses.
    pub losses: u32,
    /// Win rate 0..1.
    pub win_rate: f64,
    /// Submission count when known.
    pub submissions: u32,
    /// Real delta vs previous round when available; else 0.
    pub delta7d: f64,
    /// Prism validation BPB when the arena ranks on BPB (absent for design).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bpb: Option<f64>,
    /// Model parameters in millions when the submission measured them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params_m: Option<f64>,
}

/// Submission status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SubmissionStatus {
    /// Terminal scored.
    Scored,
    /// In flight / awaiting.
    Pending,
    /// Failed.
    Failed,
}

/// One submission / run row.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Submission {
    /// Id.
    pub id: String,
    /// Arena.
    pub arena: ArenaSlug,
    /// Agent.
    pub agent: Agent,
    /// Prompt / issue id.
    pub prompt_id: String,
    /// Short prompt title from the pinned bank when known (design arena).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_title: Option<String>,
    /// Title.
    pub title: String,
    /// Preview or detail URL (gateway-relative for design view).
    pub url: String,
    /// Full-page PNG screenshot URL when master captured one (design arena).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub screenshot_url: Option<String>,
    /// Coarse marketing status (`scored` | `pending` | `failed`).
    pub status: SubmissionStatus,
    /// Fine-grained upstream stage (`queued`, `installing`, `agentic_review`, …).
    pub stage: String,
    /// Optional upstream detail (error text or stage hint).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_detail: Option<String>,
    /// Score when scored (design lattice / prism lattice).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Prism validation BPB when measured (terminal or mid-flight if present).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bpb: Option<f64>,
    /// Model parameters in millions when measured (Prism); absent when unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params_m: Option<f64>,
    /// Failure reason when failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
    /// ISO-8601 UTC.
    pub submitted_at: String,
}

/// Loss series point.
///
/// `step` is the chart x-value: prefer tokens-seen from harness
/// `layer_stats.tokens` when present, else the optimizer step index.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LossPoint {
    /// Tokens seen (preferred) or optimizer step index.
    pub step: u32,
    /// Loss / BPB.
    pub loss: f64,
}

/// One architecture series on the prism window.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LossSeries {
    /// Architecture label.
    pub architecture: String,
    /// Upstream prism submission id (join key for the submissions table).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submission_id: Option<String>,
    /// Params in millions when known; 0 if unknown.
    pub params: f64,
    /// Final loss / BPB.
    pub final_loss: f64,
    /// Rank among series (1-based).
    pub rank: u32,
    /// Curve; minimal `[final]` when no step history is stored.
    pub points: Vec<LossPoint>,
}

/// Prism egalitarian window.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrismWindow {
    /// Dataset ref.
    pub dataset: String,
    /// Recipe revision / pin short.
    pub revision: String,
    /// Token budget when known; 0 if the recipe does not publish one.
    pub token_budget: u64,
    /// Always pinned.
    pub offset: String,
    /// Rules gate summary.
    pub rules_gate: RulesGate,
    /// Image digest when known; `—` otherwise.
    pub image_digest: String,
    /// Mid-run mutation allowed?
    pub mid_run_mutation: bool,
    /// Sealed path counts when known.
    pub sealed_paths: SealedPaths,
    /// Param ceiling when known; 0 otherwise.
    pub param_ceiling: u64,
    /// Series from terminal scored submissions.
    pub series: Vec<LossSeries>,
}

/// One miner-reported telemetry point (prism recipe ≥1.1.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrismTelemetryPoint {
    /// Optimizer step index reported by the miner.
    pub step: u64,
    /// Train loss at `step`.
    pub loss: f64,
    /// Global gradient norm when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grad_norm: Option<f64>,
    /// Seconds since train start (pod clock).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at_secs: Option<f64>,
    /// Per-layer gradient/activation stats when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_stats: Option<serde_json::Value>,
}

/// Full telemetry payload for one prism submission.
///
/// `points` is empty while the eval has not produced harness metrics yet
/// (in-flight or pre-1.1.0 recipe); `finishReason` tells how training ended
/// (`finish_evaluation` miner signal vs `train_returned`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrismTelemetry {
    /// Submission id.
    pub submission_id: String,
    /// Validation BPB when measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bpb: Option<f64>,
    /// Model parameters measured in-pod.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_params: Option<u64>,
    /// Frozen val rows scored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub val_rows: Option<u64>,
    /// GPU type used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_type: Option<String>,
    /// Train wall-clock seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_clock_seconds: Option<f64>,
    /// `finish_evaluation` | `train_returned` when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Total miner `report()` calls (including decimated points).
    pub report_count: u64,
    /// Loss series (harness-decimated to a bounded length).
    pub points: Vec<PrismTelemetryPoint>,
}

/// Rules gate chip.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RulesGate {
    /// Provider label.
    pub provider: String,
    /// Passed?
    pub passed: bool,
}

/// Sealed paths chip.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SealedPaths {
    /// Verified count.
    pub verified: u32,
    /// Total count.
    pub total: u32,
}

/// Network strip.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkStats {
    /// Epoch.
    pub epoch: u64,
    /// Agents.
    pub agents: u32,
    /// Validators / neurons listed.
    pub validators: u32,
    /// Arenas in the product surface.
    pub arenas: u32,
    /// Emission/day (0 when unknown — never invented).
    pub emission_per_day: f64,
    /// TAO USD (0 when unknown — never invented).
    pub tao_price: f64,
    /// Block height.
    pub block_height: u64,
    /// ISO-8601.
    pub updated_at: String,
    /// Total stake when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_stake: Option<f64>,
}

/// One arena's configured emission share (from the owner-signed trust root).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmissionShare {
    /// Arena slug (`design`, `prism`, `coding`).
    pub arena: String,
    /// Share of subnet miner emission, 0..1.
    pub share: f64,
}

/// One hotkey's sealed weight entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HotkeyWeight {
    /// Miner hotkey (SS58).
    pub hotkey: String,
    /// Normalised weight, 0..1 (burn uid excluded).
    pub weight: f64,
}

/// Sealed weight vector + configured emission split (`GET /v1/site/weights`).
///
/// Reads the same sealed bundle the gateway serves on `/v1/weights/latest`;
/// `sealed: false` means no bundle exists yet and only the trust-root shares
/// are real (hotkey list empty, burn share 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SiteWeights {
    /// Whether a sealed epoch bundle exists.
    pub sealed: bool,
    /// Sealed epoch when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epoch: Option<u64>,
    /// Subnet netuid.
    pub netuid: u16,
    /// Configured per-arena emission shares (trust root), 0..1 each.
    pub emission_shares: Vec<EmissionShare>,
    /// Sealed per-hotkey weights (SS58), sorted by weight desc.
    pub hotkey_weights: Vec<HotkeyWeight>,
    /// Fraction of the sealed vector on the burn uid (1 when unsealed).
    pub burn_share: f64,
    /// ISO-8601 seal time when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub computed_at: Option<String>,
}

/// Validator row (honest fields only; stake/trust 0 when chain lacks them).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Validator {
    /// UID.
    pub uid: u16,
    /// Display name.
    pub name: String,
    /// Hotkey hex.
    pub hotkey: String,
    /// Stake TAO (0 if unavailable).
    pub stake: f64,
    /// Trust (0 if unavailable).
    pub trust: f64,
    /// Validator trust (`vtrust`; 0 if unavailable).
    pub vtrust: f64,
    /// Version string.
    pub version: String,
    /// Blocks since update (0 if unavailable).
    pub updated_blocks_ago: u64,
}

/// Activity severity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ActivitySeverity {
    /// Score event.
    Score,
    /// Settle.
    Settle,
    /// Reward.
    Reward,
    /// Prompt.
    Prompt,
    /// Assign.
    Assign,
    /// Fail.
    Fail,
}

/// Activity log line.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActivityEvent {
    /// Id.
    pub id: String,
    /// `HH:MM:SS` UTC.
    pub at: String,
    /// Severity.
    pub severity: ActivitySeverity,
    /// Message.
    pub message: String,
}

/// Paginated envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Paginated<T> {
    /// Page items.
    pub items: Vec<T>,
    /// 1-based page.
    pub page: u32,
    /// Page size.
    pub page_size: u32,
    /// Total items.
    pub total: u32,
    /// Page count.
    pub page_count: u32,
}

/// Landing payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LandingSummary {
    /// Network stats.
    pub stats: NetworkStats,
    /// Arenas.
    pub arenas: Vec<Arena>,
    /// Recent activity.
    pub activity: Vec<ActivityEvent>,
}

/// Empty coding results matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResultsMatrix {
    /// Arena.
    pub arena: ArenaSlug,
    /// Task labels.
    pub tasks: Vec<String>,
    /// Rows.
    pub rows: Vec<serde_json::Value>,
}

/// Metrics stub (no invented series).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkMetrics {
    /// Range.
    pub range: String,
    /// Epoch.
    pub epoch: u64,
    /// KPIs.
    pub kpis: Vec<serde_json::Value>,
    /// Emission block.
    pub emission: MetricsEmission,
    /// Pass-rate block.
    pub pass_rate: MetricsPassRate,
    /// Population block.
    pub population: MetricsPopulation,
    /// Ledger.
    pub ledger: Vec<serde_json::Value>,
}

/// Emission section.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetricsEmission {
    /// Points.
    pub points: Vec<serde_json::Value>,
    /// Shares.
    pub shares: Vec<serde_json::Value>,
    /// Total this epoch.
    pub total_this_epoch: f64,
}

/// Pass-rate section.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetricsPassRate {
    /// Points.
    pub points: Vec<serde_json::Value>,
    /// Latest.
    pub latest: Vec<serde_json::Value>,
}

/// Population section.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MetricsPopulation {
    /// Rows.
    pub rows: Vec<serde_json::Value>,
    /// New this epoch.
    pub new_this_epoch: u32,
}

/// Governance stub (no invented proposals).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Governance {
    /// Epoch.
    pub epoch: u64,
    /// Open for voting.
    pub open_for_voting: u32,
    /// Next close label.
    pub next_close_in: String,
    /// Stages.
    pub stages: Vec<serde_json::Value>,
    /// Proposals.
    pub proposals: Vec<serde_json::Value>,
    /// Rules.
    pub rules: Vec<serde_json::Value>,
    /// Decisions.
    pub decisions: Vec<serde_json::Value>,
    /// Decisions sealed.
    pub decisions_sealed: u32,
}
