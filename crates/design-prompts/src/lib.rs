//! Pinned design prompt bank + deterministic weighted round selection.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::cast_possible_truncation)]

use design_challenge_task::{prompts_per_round, ROUND_ID_DOMAIN};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Embedded prompt bank (v1) — repo-pinned; no human approval API.
const PROMPT_BANK_JSON: &str = include_str!("../prompts/bank_v1.json");

/// One prompt in the pinned bank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prompt {
    /// Stable id (`spr01` …).
    pub id: String,
    /// Category key (`saas_pr_review`, `marketplace`, …).
    pub category: String,
    /// Short title.
    pub title: String,
    /// Draw weight (higher → more likely). Minimum effective weight is 1.
    pub weight: u32,
    /// Diversity temperature: boosts unseen categories during multi-draw.
    pub temperature: u32,
    /// Full brief for the miner harness (index / pricing / components).
    pub prompt: String,
}

/// Prompt-bank errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PromptError {
    /// Embedded JSON failed to parse.
    #[error("prompt bank parse: {0}")]
    Parse(String),
    /// Bank too small for selection.
    #[error("prompt bank too small: need {need}, have {have}")]
    TooSmall { need: usize, have: usize },
    /// Weights collapsed to zero (should not happen with validated bank).
    #[error("prompt bank has no positive weight mass")]
    ZeroWeight,
}

/// Load the pinned prompt bank.
///
/// # Errors
/// Parse failures.
pub fn load_bank() -> Result<Vec<Prompt>, PromptError> {
    serde_json::from_str(PROMPT_BANK_JSON).map_err(|e| PromptError::Parse(e.to_string()))
}

/// Alias for [`load_bank`] (HTTP / legacy callers).
///
/// # Errors
/// Parse failures.
pub fn load_prompt_set() -> Result<Vec<Prompt>, PromptError> {
    load_bank()
}

/// SHA-256 digest (hex) of the canonical bank JSON bytes.
#[must_use]
pub fn bank_digest() -> String {
    let mut h = Sha256::new();
    h.update(PROMPT_BANK_JSON.as_bytes());
    hex::encode(h.finalize())
}

/// Alias for [`bank_digest`] (DB column / HTTP field `prompt_set_digest`).
#[must_use]
pub fn prompt_set_digest() -> String {
    bank_digest()
}

/// Deterministic weighted selection of [`prompts_per_round`] prompts for `round_id`.
///
/// Seed: `SHA256(ROUND_ID_DOMAIN || round_id_be || bank_digest_bytes)`.
/// Draws without replacement; unseen categories are boosted by `temperature`.
///
/// # Errors
/// Empty/too-small bank.
pub fn select_prompts_for_round(round_id: u64) -> Result<Vec<Prompt>, PromptError> {
    let bank = load_bank()?;
    select_from(&bank, round_id, &bank_digest())
}

/// Core selection (testable with injected bank/digest).
///
/// # Errors
/// Too-small bank or zero weight mass.
pub fn select_from(
    bank: &[Prompt],
    round_id: u64,
    digest_hex: &str,
) -> Result<Vec<Prompt>, PromptError> {
    let per_round = prompts_per_round();
    if bank.len() < per_round {
        return Err(PromptError::TooSmall {
            need: per_round,
            have: bank.len(),
        });
    }
    let digest_bytes = hex::decode(digest_hex).unwrap_or_default();
    let mut seed = Sha256::new();
    seed.update(ROUND_ID_DOMAIN);
    seed.update(round_id.to_be_bytes());
    seed.update(&digest_bytes);
    let mut state = seed.finalize().to_vec();

    let mut available: Vec<usize> = (0..bank.len()).collect();
    let mut selected_categories: Vec<String> = Vec::new();
    let mut out = Vec::with_capacity(per_round);

    for draw in 0..per_round {
        let weights: Vec<u64> = available
            .iter()
            .map(|&i| effective_weight(&bank[i], &selected_categories))
            .collect();
        let total: u64 = weights.iter().sum();
        if total == 0 {
            return Err(PromptError::ZeroWeight);
        }
        let mut h = Sha256::new();
        h.update(&state);
        h.update((draw as u64).to_be_bytes());
        state = h.finalize().to_vec();
        let n = u64::from_be_bytes(state[0..8].try_into().unwrap_or([0; 8]));
        let mut r = n % total;
        let mut pick = 0usize;
        for (j, &w) in weights.iter().enumerate() {
            if r < w {
                pick = j;
                break;
            }
            r -= w;
        }
        let idx = available.remove(pick);
        selected_categories.push(bank[idx].category.clone());
        out.push(bank[idx].clone());
    }
    Ok(out)
}

fn effective_weight(p: &Prompt, selected_categories: &[String]) -> u64 {
    let w = u64::from(p.weight.max(1));
    let t = u64::from(p.temperature.max(1));
    if selected_categories.iter().any(|c| c == &p.category) {
        w
    } else {
        w.saturating_mul(t)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use design_challenge_task::PROMPTS_PER_ROUND;
    use std::collections::BTreeSet;

    #[test]
    fn bank_loads_and_digest_stable() {
        let bank = load_bank().unwrap();
        assert!(bank.len() >= 40, "expected dozens of bank entries");
        assert!(bank.len() >= PROMPTS_PER_ROUND);
        let cats: BTreeSet<_> = bank.iter().map(|p| p.category.as_str()).collect();
        assert!(cats.len() >= 8, "expected multi-category bank");
        for p in &bank {
            assert!(!p.id.is_empty());
            assert!(!p.prompt.is_empty());
            assert!(p.prompt.contains("index.html") || p.prompt.contains("three-page"));
            assert!(p.weight >= 1);
            assert!(p.temperature >= 1);
        }
        let d = bank_digest();
        assert_eq!(d.len(), 64);
        assert_eq!(d, prompt_set_digest());
        assert_eq!(d, bank_digest());
    }

    #[test]
    fn selection_deterministic_and_sized() {
        let a = select_prompts_for_round(42).unwrap();
        let b = select_prompts_for_round(42).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), PROMPTS_PER_ROUND);
        let ids: BTreeSet<_> = a.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids.len(), PROMPTS_PER_ROUND);
        let c = select_prompts_for_round(43).unwrap();
        assert_ne!(
            a.iter().map(|p| &p.id).collect::<Vec<_>>(),
            c.iter().map(|p| &p.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn weighted_draw_respects_injected_weights() {
        let bank: Vec<Prompt> = (0..6)
            .map(|i| Prompt {
                id: format!("t{i}"),
                category: format!("c{i}"),
                title: format!("T{i}"),
                weight: if i == 0 { 1_000 } else { 1 },
                temperature: 1,
                prompt: format!("prompt {i} three-page index.html"),
            })
            .collect();
        let digest = "00".repeat(32);
        let mut heavy = 0u32;
        for rid in 0..200u64 {
            let sel = select_from(&bank, rid, &digest).unwrap();
            if sel.iter().any(|p| p.id == "t0") {
                heavy += 1;
            }
        }
        // Heavy entry should appear in a clear majority of rounds.
        assert!(heavy > 100, "heavy weight under-selected: {heavy}/200");
    }
}
