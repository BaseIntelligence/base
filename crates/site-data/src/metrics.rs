//! Metrics-page frames built from live reads (no epoch-close history store
//! exists yet — trend points stay empty rather than invented).

use serde_json::{json, Value};
use site_types::{Arena, SiteWeights};

/// KPI strip: registered agents, validators, TAO quote, chain tip, seal state.
#[must_use]
pub fn kpis(
    agents: u32,
    validators: usize,
    tao_price: f64,
    block: u64,
    weights: &SiteWeights,
) -> Vec<Value> {
    vec![
        json!({
            "label": "REGISTERED AGENTS",
            "value": agents.to_string(),
            "note": "Distinct miner hotkeys with a harness or submission",
            "direction": "flat",
        }),
        json!({
            "label": "VALIDATORS",
            "value": validators.to_string(),
            "note": "Neurons in the metagraph cache",
            "direction": "flat",
        }),
        json!({
            "label": "TAO / USD",
            "value": if tao_price > 0.0 { format!("${tao_price:.2}") } else { "—".into() },
            "note": "CoinGecko simple/price · 10 min cache",
            "direction": "flat",
        }),
        json!({
            "label": "BLOCK HEIGHT",
            "value": block.to_string(),
            "note": "Chain tip read by the gateway",
            "direction": "flat",
        }),
        json!({
            "label": "SEALED EPOCH",
            "value": weights.epoch.map_or_else(|| "UNSEALED".into(), |e| e.to_string()),
            "note": if weights.sealed { "Latest sealed weight vector" } else { "No sealed bundle — burn fallback" },
            "direction": "flat",
        }),
    ]
}

/// Current emission split by arena (trust-root configured shares).
#[must_use]
pub fn emission_shares(arenas: &[Arena]) -> Vec<Value> {
    arenas
        .iter()
        .map(|a| {
            json!({
                "arena": a.slug.as_str(),
                "name": a.name,
                "tao": 0.0,
                "share": a.emission_share,
            })
        })
        .collect()
}

/// Registered agents per arena; `ratio` scales against the largest arena.
#[must_use]
pub fn population_rows(arenas: &[Arena]) -> Vec<Value> {
    let max_agents = arenas.iter().map(|a| a.agents).max().unwrap_or(0);
    arenas
        .iter()
        .map(|a| {
            json!({
                "arena": a.slug.as_str(),
                "name": a.name,
                "agents": a.agents,
                "ratio": if max_agents > 0 {
                    f64::from(a.agents) / f64::from(max_agents)
                } else {
                    0.0
                },
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use site_types::{design_frame, prism_frame};

    #[test]
    fn population_ratio_normalises_to_max() {
        let mut design = design_frame();
        design.agents = 4;
        let mut prism = prism_frame();
        prism.agents = 8;
        prism.emission_share = 0.8;
        let rows = population_rows(&[design, prism]);
        assert_eq!(rows[1]["agents"], 8);
        assert!((rows[0]["ratio"].as_f64().unwrap() - 0.5).abs() < f64::EPSILON);
        assert!((rows[1]["ratio"].as_f64().unwrap() - 1.0).abs() < f64::EPSILON);
        let shares = emission_shares(&[prism_frame()]);
        assert_eq!(shares[0]["arena"], "prism");
    }
}
