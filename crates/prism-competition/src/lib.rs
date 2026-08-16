//! Prism emission competition math (epoch-local, lattice-preserving).
//!
//! Exact rule (mirrors `docs/PRISM.md` § competition / WTA):
//!
//! - Input: every scored submission row in the competition set (fresh outbox
//!   + active positive carry). Multiple rows per hotkey are possible.
//! - **Submitter credit** (normative): a hotkey's own rows — `max(Score)`
//!   over submissions posted by that `miner_hotkey`. Best BPB ⇒ highest
//!   lattice score ⇒ that **submitter** wins.
//! - **Architecture-owner credit**: gated off by
//!   [`OWNER_ARCH_CREDIT_ENABLED`] (`false`). Arch owners must not receive
//!   emission credit for challenger trains; the dead path remains only so a
//!   future explicit product flip can restore it.
//! - **Per-hotkey credit**: own score only while the flag is false.
//! - **WTA leaf emission**: [`apply_wta`] keeps a single positive `Score`
//!   (argmax; lexicographically smallest hotkey on ties). Prism's emission
//!   share goes to that submitter.
//! - **Recipe 2.0 / AutoModel only**: rows with `weight_eligible == false`
//!   (legacy 1.x) contribute `Score(0)` only — never win WTA. If every
//!   positive score is ineligible, emission fail-closes to an all-zero /
//!   burn projection (no 1.x winner).

use std::collections::BTreeMap;

use prism_store::{EpochScoreRow, FinalScore};

/// Architecture-owner emission credit — **must stay `false`**.
///
/// Product rule: Prism WTA goes to the **submitter** of the best-BPB run
/// (`miner_hotkey` on the scored row), never the architecture registry owner.
/// With this flag `false`, leaf emission uses own BPB-derived lattice scores
/// only, so a non-training / off-metagraph arch owner cannot steal or burn
/// Prism's share via lex-tie.
///
/// Do not flip to `true` without an explicit product decision; `docs/PRISM.md`
/// documents owner credit as disabled for emission.
pub const OWNER_ARCH_CREDIT_ENABLED: bool = false;

/// Compute per-hotkey emission for one epoch.
///
/// `arch_owners` is ignored while [`OWNER_ARCH_CREDIT_ENABLED`] is `false`.
/// Legacy (non-[`EpochScoreRow::weight_eligible`]) positive scores are
/// treated as `Score(0)` so they cannot win WTA or carry emission.
#[must_use]
pub fn competition_scores(
    rows: &[EpochScoreRow],
    arch_owners: &BTreeMap<String, String>,
) -> BTreeMap<String, FinalScore> {
    // Arch epoch best (max lattice score among linked rows, any trainer).
    let mut arch_best: BTreeMap<&str, u64> = BTreeMap::new();
    // Own credits per hotkey.
    let mut own: BTreeMap<String, u64> = BTreeMap::new();
    // Absence fallback for hotkeys with no score rows at all.
    let mut absence: BTreeMap<String, u8> = BTreeMap::new();

    for r in rows {
        match &r.final_score {
            FinalScore::Score(v) => {
                // Fail-closed: legacy 1.x never receives emission credit.
                let v = if r.weight_eligible { *v } else { 0 };
                let e = own.entry(r.miner_hotkey.clone()).or_insert(0);
                *e = (*e).max(v);
                if OWNER_ARCH_CREDIT_ENABLED && r.weight_eligible {
                    if let Some(a) = r.arch_id.as_deref() {
                        let e = arch_best.entry(a).or_insert(0);
                        *e = (*e).max(v);
                    }
                }
            }
            FinalScore::NoScore(reason) => {
                absence.entry(r.miner_hotkey.clone()).or_insert(*reason);
            }
        }
    }

    // Owner credits: arch epoch best → arch owner (gated).
    let mut owner_credit: BTreeMap<String, u64> = BTreeMap::new();
    if OWNER_ARCH_CREDIT_ENABLED {
        for (arch_id, best) in &arch_best {
            if let Some(owner) = arch_owners.get(*arch_id) {
                let e = owner_credit.entry(owner.clone()).or_insert(0);
                *e = (*e).max(*best);
            }
        }
    } else {
        // Silence unused-param lint while the kill-switch is off; callers
        // still pass the registry map so re-enable is a one-line flip.
        let _ = arch_owners;
    }

    let mut out: BTreeMap<String, FinalScore> = BTreeMap::new();
    for hk in own.keys().chain(owner_credit.keys()).chain(absence.keys()) {
        if out.contains_key(hk) {
            continue;
        }
        let score = own
            .get(hk)
            .copied()
            .unwrap_or(0)
            .max(owner_credit.get(hk).copied().unwrap_or(0));
        if score > 0 || own.contains_key(hk) {
            out.insert(hk.clone(), FinalScore::Score(score));
        } else if let Some(reason) = absence.get(hk) {
            out.insert(hk.clone(), FinalScore::NoScore(*reason));
        }
    }
    out
}

/// Winner-take-all collapse: keep only the single highest positive score.
///
/// Ties break by lexicographically smallest hotkey (stable, hex-encoded).
/// Non-positive `Score(0)` and `NoScore` rows are preserved unchanged;
/// every other positive `Score` is zeroed so the aggregator cannot soft-
/// allocate Prism's share across multiple hotkeys.
#[must_use]
pub fn apply_wta(scores: BTreeMap<String, FinalScore>) -> BTreeMap<String, FinalScore> {
    let winner = top_positive(&scores).map(|(hk, _)| hk.to_owned());
    let Some(winner) = winner else {
        return scores;
    };
    scores
        .into_iter()
        .map(|(hk, s)| match &s {
            FinalScore::Score(v) if *v > 0 && hk != winner => (hk, FinalScore::Score(0)),
            _ => (hk, s),
        })
        .collect()
}

/// Highest positive credit under the emission tie convention (higher score
/// wins; on equal score the lexicographically smaller hotkey).
fn top_positive(scores: &BTreeMap<String, FinalScore>) -> Option<(&str, u64)> {
    scores
        .iter()
        .filter_map(|(hk, s)| match s {
            FinalScore::Score(v) if *v > 0 => Some((hk.as_str(), *v)),
            _ => None,
        })
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(a.0)))
}

/// Prism **v2.1** emission mode (`PRISM_EMISSION_MODE`).
///
/// `wta` (default, bit-identical to the historical behavior) keeps a single
/// positive leaf; `top3` keeps the top three positive credits at a decaying
/// 100 % / 50 % / 25 % scale so exploration behind the champion still earns
/// — a product lever against WTA's exploit-only miner meta. Anything else
/// (unknown values included) is `wta`, fail-safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmissionMode {
    /// Winner-take-all ([`apply_wta`]) — the live default.
    #[default]
    Wta,
    /// Top-3 decaying split ([`apply_top3_decay`]) — opt-in via env.
    Top3Decay,
}

/// Decay (bps of the rank's own score) for ranks 1..=3 under
/// [`EmissionMode::Top3Decay`].
pub const TOP3_DECAY_BPS: [u64; 3] = [10_000, 5_000, 2_500];

impl EmissionMode {
    /// Parse the raw env value: exactly `top3` selects the decaying split.
    #[must_use]
    pub fn parse(raw: Option<&str>) -> Self {
        match raw {
            Some("top3") => Self::Top3Decay,
            _ => Self::Wta,
        }
    }

    /// Mode for this process, read once from `PRISM_EMISSION_MODE`.
    #[must_use]
    pub fn from_env() -> Self {
        static MODE: std::sync::OnceLock<EmissionMode> = std::sync::OnceLock::new();
        *MODE.get_or_init(|| Self::parse(std::env::var("PRISM_EMISSION_MODE").ok().as_deref()))
    }
}

/// Top-3 decaying collapse: rank positive credits (same tie convention as
/// [`apply_wta`]), keep rank r scaled by [`TOP3_DECAY_BPS`]`[r]`, zero the
/// rest. A positive score never rounds below 1 so a ranked hotkey cannot be
/// silently dropped; `Score(0)` / `NoScore` rows pass through unchanged.
#[must_use]
pub fn apply_top3_decay(scores: BTreeMap<String, FinalScore>) -> BTreeMap<String, FinalScore> {
    let mut ranked: Vec<(&str, u64)> = scores
        .iter()
        .filter_map(|(hk, s)| match s {
            FinalScore::Score(v) if *v > 0 => Some((hk.as_str(), *v)),
            _ => None,
        })
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    let scaled: BTreeMap<String, u64> = ranked
        .iter()
        .take(TOP3_DECAY_BPS.len())
        .enumerate()
        .map(|(rank, (hk, v))| {
            (
                (*hk).to_owned(),
                ((v * TOP3_DECAY_BPS[rank]) / 10_000).max(1),
            )
        })
        .collect();
    scores
        .into_iter()
        .map(|(hk, s)| match &s {
            FinalScore::Score(v) if *v > 0 => {
                let kept = scaled.get(&hk).copied().unwrap_or(0);
                (hk, FinalScore::Score(kept))
            }
            _ => (hk, s),
        })
        .collect()
}

/// Dispatch the configured emission collapse.
#[must_use]
pub fn apply_emission(
    mode: EmissionMode,
    scores: BTreeMap<String, FinalScore>,
) -> BTreeMap<String, FinalScore> {
    match mode {
        EmissionMode::Wta => apply_wta(scores),
        EmissionMode::Top3Decay => apply_top3_decay(scores),
    }
}

/// Prism **v2.1** architecture-owner split (`PRISM_OWNER_ARCH_CREDIT_BPS`).
///
/// 0 (default/absent/unparseable) keeps the historical behavior — no owner
/// credit. Positive values are clamped to 5 000 bps so the owner can never
/// out-earn the winning submitter from the split alone.
#[must_use]
pub fn owner_split_bps_from_env() -> u64 {
    static BPS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BPS.get_or_init(|| {
        std::env::var("PRISM_OWNER_ARCH_CREDIT_BPS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
            .min(5_000)
    })
}

/// Carve an owner credit out of the emission winner's score (v2.1, opt-in).
///
/// Unlike the dead [`OWNER_ARCH_CREDIT_ENABLED`] pre-WTA path, this split
/// runs **after** the emission collapse and only redistributes the winner's
/// own leaf: winner keeps `v − cut`, the registry owner of the winning
/// architecture receives `cut = v × bps / 10_000`. No-ops (fail-safe) when
/// `bps == 0`, the winner has no linked published arch, the owner **is**
/// the winner, or the cut rounds to 0. An off-metagraph owner's leaf is
/// silently dropped downstream by the D24 expected-set filter, burning the
/// cut rather than re-routing it — the lex-tie theft vector of the legacy
/// path stays closed.
pub fn apply_owner_split(
    scores: &mut BTreeMap<String, FinalScore>,
    winner_arch_owner: &BTreeMap<String, String>,
    batch: &[EpochScoreRow],
    bps: u64,
) {
    if bps == 0 {
        return;
    }
    let Some((winner, v)) = top_positive(scores).map(|(hk, v)| (hk.to_owned(), v)) else {
        return;
    };
    // The winning hotkey's best weight-eligible positive row carries the
    // architecture the split credits.
    let arch = batch
        .iter()
        .filter(|r| r.miner_hotkey == winner && r.weight_eligible)
        .filter_map(|r| match &r.final_score {
            FinalScore::Score(v) if *v > 0 => Some((*v, r.arch_id.as_deref()?)),
            _ => None,
        })
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, arch)| arch);
    let Some(owner) = arch.and_then(|a| winner_arch_owner.get(a)) else {
        return;
    };
    if *owner == winner {
        return;
    }
    let cut = (v * bps) / 10_000;
    if cut == 0 {
        return;
    }
    scores.insert(winner, FinalScore::Score(v - cut));
    scores.insert(owner.clone(), FinalScore::Score(cut));
}

/// Full v2.1 emission projection: competition credits → configured collapse
/// → optional owner split. With `EmissionMode::Wta` and `bps == 0` (the
/// defaults) this is bit-identical to
/// `apply_wta(competition_scores(batch, arch_owners))`.
#[must_use]
pub fn emission_leaves(
    batch: &[EpochScoreRow],
    arch_owners: &BTreeMap<String, String>,
    mode: EmissionMode,
    owner_split_bps: u64,
) -> BTreeMap<String, FinalScore> {
    let mut scores = apply_emission(mode, competition_scores(batch, arch_owners));
    apply_owner_split(&mut scores, arch_owners, batch, owner_split_bps);
    scores
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn row(hk: &str, arch: Option<&str>, score: u64) -> EpochScoreRow {
        EpochScoreRow {
            miner_hotkey: hk.into(),
            arch_id: arch.map(str::to_owned),
            final_score: FinalScore::Score(score),
            weight_eligible: true,
        }
    }

    fn legacy_row(hk: &str, score: u64) -> EpochScoreRow {
        EpochScoreRow {
            miner_hotkey: hk.into(),
            arch_id: None,
            final_score: FinalScore::Score(score),
            weight_eligible: false,
        }
    }

    fn owners(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(a, o)| ((*a).to_owned(), (*o).to_owned()))
            .collect()
    }

    #[test]
    fn owner_arch_credit_temporarily_disabled() {
        const {
            assert!(
                !OWNER_ARCH_CREDIT_ENABLED,
                "flip this test when restoring owner-arch credit"
            );
        }
        // Challenger BB trains owner AA's arch to 900k. With owner credit
        // off, AA keeps only their own 400k — BB (best BPB/score) wins WTA.
        let rows = vec![
            row("aa", Some("arch_x"), 400_000),
            row("bb", Some("arch_x"), 900_000),
        ];
        let out = competition_scores(&rows, &owners(&[("arch_x", "aa")]));
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(400_000)));
        assert_eq!(out.get("bb"), Some(&FinalScore::Score(900_000)));
        let wta = apply_wta(out);
        assert_eq!(wta.get("bb"), Some(&FinalScore::Score(900_000)));
        assert_eq!(wta.get("aa"), Some(&FinalScore::Score(0)));
    }

    #[test]
    fn owner_not_credited_across_archs_while_disabled() {
        let rows = vec![
            row("bb", Some("arch_x"), 300_000),
            row("cc", Some("arch_y"), 500_000),
        ];
        let out = competition_scores(&rows, &owners(&[("arch_x", "aa"), ("arch_y", "aa")]));
        assert!(!out.contains_key("aa"));
        assert_eq!(out.get("bb"), Some(&FinalScore::Score(300_000)));
        assert_eq!(out.get("cc"), Some(&FinalScore::Score(500_000)));
    }

    #[test]
    fn zero_scores_do_not_invent_owner_rows() {
        let rows = vec![row("bb", Some("arch_x"), 0), row("aa", Some("arch_x"), 0)];
        let out = competition_scores(&rows, &owners(&[("arch_x", "aa")]));
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(0)));
        assert_eq!(out.get("bb"), Some(&FinalScore::Score(0)));
    }

    #[test]
    fn unlinked_rows_only_own_credit() {
        let rows = vec![row("aa", None, 700_000), row("bb", None, 200_000)];
        let out = competition_scores(&rows, &BTreeMap::new());
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(700_000)));
        assert_eq!(out.get("bb"), Some(&FinalScore::Score(200_000)));
    }

    #[test]
    fn absence_preserved_for_scoreless_hotkeys() {
        let rows = vec![
            EpochScoreRow {
                miner_hotkey: "cc".into(),
                arch_id: None,
                final_score: FinalScore::NoScore(6),
                weight_eligible: true,
            },
            row("aa", None, 100_000),
        ];
        let out = competition_scores(&rows, &BTreeMap::new());
        assert_eq!(out.get("cc"), Some(&FinalScore::NoScore(6)));
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(100_000)));
    }

    #[test]
    fn challenger_keeps_own_score_without_owner_boost() {
        // A owns arch X but owner credit is off — only own rows count.
        let rows = vec![
            row("bb", Some("arch_x"), 300_000),
            row("aa", Some("arch_y"), 800_000),
        ];
        let out = competition_scores(&rows, &owners(&[("arch_x", "aa"), ("arch_y", "cc")]));
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(800_000)));
        assert_eq!(out.get("bb"), Some(&FinalScore::Score(300_000)));
        assert!(!out.contains_key("cc"));
    }

    #[test]
    fn wta_keeps_only_the_argmax_score() {
        let rows = vec![row("aa", None, 177_155), row("bb", Some("arch_x"), 111_595)];
        let credits = competition_scores(&rows, &owners(&[("arch_x", "cc")]));
        // Own-only: aa=177155, bb=111595 — owner cc gets nothing.
        let wta = apply_wta(credits);
        assert_eq!(wta.get("aa"), Some(&FinalScore::Score(177_155)));
        assert_eq!(wta.get("bb"), Some(&FinalScore::Score(0)));
        assert!(!wta.contains_key("cc"));
    }

    #[test]
    fn wta_tie_breaks_by_lexicographically_smallest_hotkey() {
        let mut credits = BTreeMap::new();
        credits.insert("bb".into(), FinalScore::Score(900_000));
        credits.insert("aa".into(), FinalScore::Score(900_000));
        let wta = apply_wta(credits);
        assert_eq!(wta.get("aa"), Some(&FinalScore::Score(900_000)));
        assert_eq!(wta.get("bb"), Some(&FinalScore::Score(0)));
    }

    #[test]
    fn legacy_positive_scores_cannot_win_wta() {
        // Legacy 1.x tops the lattice but is weight-ineligible; AutoModel
        // runner with a lower score still wins. Fail-closed vs 1.x emission.
        let rows = vec![legacy_row("legacy", 900_000), row("auto", None, 100_000)];
        let out = competition_scores(&rows, &BTreeMap::new());
        assert_eq!(out.get("legacy"), Some(&FinalScore::Score(0)));
        assert_eq!(out.get("auto"), Some(&FinalScore::Score(100_000)));
        let wta = apply_wta(out);
        assert_eq!(wta.get("auto"), Some(&FinalScore::Score(100_000)));
        assert_eq!(wta.get("legacy"), Some(&FinalScore::Score(0)));
    }

    #[test]
    fn only_legacy_tops_fail_closed_to_burn() {
        // No AutoModel-eligible positive → no WTA winner (burn / hold).
        let rows = vec![legacy_row("aa", 900_000), legacy_row("bb", 800_000)];
        let out = competition_scores(&rows, &BTreeMap::new());
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(0)));
        assert_eq!(out.get("bb"), Some(&FinalScore::Score(0)));
        let wta = apply_wta(out);
        assert!(wta.values().all(|s| matches!(s, FinalScore::Score(0))));
    }

    // ---- v2.1: emission modes ----

    #[test]
    fn emission_mode_parse_defaults_to_wta() {
        assert_eq!(EmissionMode::parse(None), EmissionMode::Wta);
        assert_eq!(EmissionMode::parse(Some("wta")), EmissionMode::Wta);
        assert_eq!(EmissionMode::parse(Some("top3")), EmissionMode::Top3Decay);
        assert_eq!(EmissionMode::parse(Some("TOP3")), EmissionMode::Wta);
        assert_eq!(EmissionMode::parse(Some("garbage")), EmissionMode::Wta);
    }

    #[test]
    fn top3_decay_keeps_three_ranks_and_zeroes_the_rest() {
        let mut credits = BTreeMap::new();
        credits.insert("aa".into(), FinalScore::Score(800_000));
        credits.insert("bb".into(), FinalScore::Score(600_000));
        credits.insert("cc".into(), FinalScore::Score(400_000));
        credits.insert("dd".into(), FinalScore::Score(200_000));
        credits.insert("ee".into(), FinalScore::NoScore(0));
        let out = apply_top3_decay(credits);
        assert_eq!(
            out.get("aa"),
            Some(&FinalScore::Score(800_000)),
            "rank 1: 100%"
        );
        assert_eq!(
            out.get("bb"),
            Some(&FinalScore::Score(300_000)),
            "rank 2: 50%"
        );
        assert_eq!(
            out.get("cc"),
            Some(&FinalScore::Score(100_000)),
            "rank 3: 25%"
        );
        assert_eq!(out.get("dd"), Some(&FinalScore::Score(0)), "rank 4 zeroed");
        assert_eq!(out.get("ee"), Some(&FinalScore::NoScore(0)), "absence kept");
    }

    #[test]
    fn top3_decay_tie_break_and_tiny_scores_stay_positive() {
        let mut credits = BTreeMap::new();
        credits.insert("bb".into(), FinalScore::Score(2));
        credits.insert("aa".into(), FinalScore::Score(2));
        credits.insert("cc".into(), FinalScore::Score(1));
        let out = apply_top3_decay(credits);
        // Tie at the top: lex-smallest first (same convention as WTA).
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(2)));
        assert_eq!(out.get("bb"), Some(&FinalScore::Score(1)), "2×50% = 1");
        assert_eq!(out.get("cc"), Some(&FinalScore::Score(1)), "floor at 1");
    }

    #[test]
    fn apply_emission_wta_is_bit_identical() {
        let rows = vec![row("aa", None, 500_000), row("bb", None, 900_000)];
        let credits = competition_scores(&rows, &BTreeMap::new());
        assert_eq!(
            apply_emission(EmissionMode::Wta, credits.clone()),
            apply_wta(credits)
        );
    }

    // ---- v2.1: owner split ----

    #[test]
    fn owner_split_carves_bps_out_of_the_winner() {
        let rows = vec![
            row("bb", Some("arch_x"), 900_000),
            row("aa", Some("arch_y"), 400_000),
        ];
        let owners = owners(&[("arch_x", "cc")]);
        let out = emission_leaves(&rows, &owners, EmissionMode::Wta, 1_000);
        assert_eq!(out.get("bb"), Some(&FinalScore::Score(810_000)), "90%");
        assert_eq!(out.get("cc"), Some(&FinalScore::Score(90_000)), "10% cut");
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(0)), "WTA holds");
    }

    #[test]
    fn owner_split_noops_when_disabled_or_self_owned() {
        let rows = vec![row("bb", Some("arch_x"), 900_000)];
        let self_owned = owners(&[("arch_x", "bb")]);
        let out = emission_leaves(&rows, &self_owned, EmissionMode::Wta, 1_000);
        assert_eq!(out.get("bb"), Some(&FinalScore::Score(900_000)));
        let third = owners(&[("arch_x", "cc")]);
        let off = emission_leaves(&rows, &third, EmissionMode::Wta, 0);
        assert_eq!(off.get("cc"), None, "bps 0 → no owner leaf");
        assert_eq!(off.get("bb"), Some(&FinalScore::Score(900_000)));
    }

    #[test]
    fn owner_split_never_credits_unlinked_or_legacy_winners() {
        let unlinked = vec![row("bb", None, 900_000)];
        let owners_map = owners(&[("arch_x", "cc")]);
        let out = emission_leaves(&unlinked, &owners_map, EmissionMode::Wta, 1_000);
        assert_eq!(out.get("cc"), None, "no arch on the winning row");
        let legacy = vec![legacy_row("bb", 900_000), row("aa", Some("arch_x"), 100)];
        let out = emission_leaves(&legacy, &owners_map, EmissionMode::Wta, 1_000);
        // aa wins (legacy ineligible); its arch owner cc gets the cut.
        assert_eq!(out.get("aa"), Some(&FinalScore::Score(90)));
        assert_eq!(out.get("cc"), Some(&FinalScore::Score(10)));
    }

    #[test]
    fn emission_leaves_default_knobs_match_legacy_wta() {
        let rows = vec![
            row("aa", Some("arch_x"), 400_000),
            row("bb", Some("arch_x"), 900_000),
        ];
        let owners = owners(&[("arch_x", "aa")]);
        let legacy = apply_wta(competition_scores(&rows, &owners));
        let v21 = emission_leaves(&rows, &owners, EmissionMode::Wta, 0);
        assert_eq!(legacy, v21, "defaults are bit-identical");
    }
}
