//! Search for a compiled blob with strictly lower sim wallclock than champion.

use hypertraining_cluster::{ClusterBackend, SegmentConfig, SegmentSeeds, SimBackend, Topology};
use sha2::{Digest, Sha256};

/// Fingerprint of compiled blob (domain-tagged SHA-256).
#[must_use]
pub fn code_fingerprint(compiled: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"base-hypertraining-code-fp-v1");
    h.update(compiled);
    h.finalize().into()
}

/// Search for a compiled blob whose sim wallclock is strictly faster than `champ_compiled`.
///
/// # Errors
/// When no faster fingerprint is found within the search budget, or cluster fails.
pub fn find_faster_compiled(
    champ_compiled: &[u8],
    budget_tokens: u64,
    seeds: &SegmentSeeds,
    topology: Topology,
    search_limit: u32,
) -> Result<Vec<u8>, String> {
    let mut cluster = SimBackend::new();
    let champ_ms = measure(
        &mut cluster,
        code_fingerprint(champ_compiled),
        budget_tokens,
        seeds,
        topology,
        1,
    )?;

    for i in 0..search_limit {
        let mut blob = b"novel-cand-v1\n".to_vec();
        blob.extend_from_slice(&i.to_le_bytes());
        blob.extend_from_slice(b"\n.entry faster { add.u32 %r1, %r2, %r3; }\n");
        let ms = measure(
            &mut cluster,
            code_fingerprint(&blob),
            budget_tokens,
            seeds,
            topology,
            2u16.wrapping_add((i % 1000) as u16),
        )?;
        if ms < champ_ms {
            return Ok(blob);
        }
    }
    Err(format!(
        "no faster cand in {search_limit} tries (champ_ms={champ_ms})"
    ))
}

fn measure(
    cluster: &mut SimBackend,
    fp: [u8; 32],
    budget_tokens: u64,
    seeds: &SegmentSeeds,
    topology: Topology,
    pkey_id: u16,
) -> Result<u64, String> {
    cluster
        .run_segment(&SegmentConfig {
            code_fingerprint: fp,
            budget_tokens,
            seeds: seeds.clone(),
            master_topology: topology,
            slot_topology: topology,
            pkey_id,
            noise_ms: 0,
        })
        .map(|r| r.wallclock_ms)
        .map_err(|e| e.to_string())
}
