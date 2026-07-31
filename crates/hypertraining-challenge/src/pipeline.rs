//! Sim tournament pipeline: build → kernel → antinois → cluster → eval → pay.

use crate::pipeline_types::{PipelineError, SimPipelineInput, SimPipelineResult};
use crate::score::PipelineOutcome;
use crate::sim_search::code_fingerprint;
use hypertraining_antinois::{
    evaluate, AntinoisReport, CandidateArtifacts, ChampionArtifacts, FingerprintDedupe,
    DEFAULT_DEDUPE_SEGMENTS,
};
use hypertraining_build::{
    AdmittedSource, BuildRequest, FixtureBuilder, HermeticBuilder, LockMaterial, ValidatorLock,
    Wheelhouse,
};
use hypertraining_cluster::{
    ClusterBackend, SegmentConfig, SegmentResult, SegmentSeeds, SimBackend,
};
use hypertraining_eval::{
    evaluate_candidate, AnalyticModel, EpsilonParams, EvalVerdict, PhysicsTelemetry,
};
use hypertraining_kernel_gate::{
    fixtures::{fixture_attestation_ok, fixture_baseline, fixture_good_kernel, fixture_reference},
    gate_kernel, validate_attestation, AttestationPolicy, KAPPA,
};
use hypertraining_pay::{score_from_pay_inputs, PayInputs};

/// Run build→kernel→antinois→sim→eval→pay for one candidate (sim backend).
///
/// # Errors
/// Stage failures that prevent a terminal outcome.
pub fn run_sim_pipeline(
    input: &SimPipelineInput<'_>,
    dedupe: &mut FingerprintDedupe,
    cluster: &mut SimBackend,
) -> Result<SimPipelineResult, PipelineError> {
    if input.budget_tokens == 0 {
        return Err(PipelineError::Invalid("budget_tokens must be > 0".into()));
    }

    let image_digest = hermetic_build(input)?;
    kernel_gate_pass()?;
    let antinois = run_antinois(input, dedupe)?;

    if !antinois.allows_measure() {
        return Ok(rejected_antinois(antinois, image_digest));
    }

    let (cand_segment, t_champ_ms) = run_segments(input, cluster)?;
    let physics =
        PhysicsTelemetry::from_segment(&cand_segment.telemetry, cand_segment.wallclock_ms);
    let model = AnalyticModel::from_telemetry_baseline(&physics, 3_000);
    let eval = evaluate_candidate(
        &input.champ_loss,
        &input.cand_loss,
        &physics,
        &model,
        &EpsilonParams::must_calibrate_defaults(),
    )
    .map_err(|e| PipelineError::Eval(e.to_string()))?;

    let guards_passed = eval.promote_allowed();
    let score_u64 = score_from_pay_inputs(&PayInputs {
        t_champ_ms,
        t_cand_ms: cand_segment.wallclock_ms,
        guards_passed,
    });
    let outcome = PipelineOutcome::Measured {
        t_champ_ms,
        t_cand_ms: cand_segment.wallclock_ms,
        guards_passed,
    };

    Ok(SimPipelineResult {
        antinois,
        cand_segment,
        t_champ_ms,
        eval,
        kernel_ok: true,
        image_digest,
        score_u64,
        outcome,
    })
}

fn hermetic_build(input: &SimPipelineInput<'_>) -> Result<String, PipelineError> {
    let source = AdmittedSource::new(input.admitted_files.clone())
        .map_err(|e| PipelineError::Build(e.to_string()))?;
    let lock = ValidatorLock::new(input.validator_lock.to_vec())
        .map_err(|e| PipelineError::Build(e.to_string()))?;
    let artifact = FixtureBuilder::new()
        .build(&BuildRequest {
            source,
            lock: LockMaterial::Validator(lock),
            wheelhouse: Wheelhouse::empty(),
        })
        .map_err(|e| PipelineError::Build(e.to_string()))?;
    Ok(artifact.image_digest)
}

fn kernel_gate_pass() -> Result<(), PipelineError> {
    let att = fixture_attestation_ok();
    validate_attestation(&att, &AttestationPolicy::default())
        .map_err(|e| PipelineError::Kernel(e.to_string()))?;
    gate_kernel(
        &fixture_good_kernel(),
        &fixture_baseline(),
        &fixture_reference(),
        KAPPA,
    )
    .map_err(|e| PipelineError::Kernel(e.to_string()))?;
    Ok(())
}

fn run_antinois(
    input: &SimPipelineInput<'_>,
    dedupe: &mut FingerprintDedupe,
) -> Result<AntinoisReport, PipelineError> {
    let cand = CandidateArtifacts {
        miner_id: input.miner_id,
        source: input.cand_source,
        compiled: input.cand_compiled,
        telemetry: None,
        segment_index: input.segment_index,
    };
    let champ = ChampionArtifacts {
        source: input.champ_source,
        compiled: input.champ_compiled,
        telemetry: None,
    };
    evaluate(&cand, &champ, dedupe).map_err(|e| PipelineError::Antinois(e.to_string()))
}

fn run_segments(
    input: &SimPipelineInput<'_>,
    cluster: &mut SimBackend,
) -> Result<(SegmentResult, u64), PipelineError> {
    let topo = input.topology;
    let cand_cfg = SegmentConfig {
        code_fingerprint: code_fingerprint(input.cand_compiled),
        budget_tokens: input.budget_tokens,
        seeds: input.seeds.clone(),
        master_topology: topo,
        slot_topology: topo,
        pkey_id: input.pkey_id,
        noise_ms: input.noise_ms,
    };
    let cand_segment = cluster
        .run_segment(&cand_cfg)
        .map_err(|e| PipelineError::Cluster(e.to_string()))?;

    let t_champ_ms = if let Some(ms) = input.t_champ_ms_override {
        ms
    } else {
        let champ_cfg = SegmentConfig {
            code_fingerprint: code_fingerprint(input.champ_compiled),
            budget_tokens: input.budget_tokens,
            seeds: input.seeds.clone(),
            master_topology: topo,
            slot_topology: topo,
            pkey_id: input.pkey_id.wrapping_add(1),
            noise_ms: input.noise_ms,
        };
        cluster
            .run_segment(&champ_cfg)
            .map_err(|e| PipelineError::Cluster(e.to_string()))?
            .wallclock_ms
    };
    Ok((cand_segment, t_champ_ms))
}

fn rejected_antinois(antinois: AntinoisReport, image_digest: String) -> SimPipelineResult {
    use hypertraining_cluster::{CheckpointHash, MmaFamily, SegmentTelemetry};
    SimPipelineResult {
        antinois,
        cand_segment: SegmentResult {
            wallclock_ms: 0,
            checkpoint_hash: CheckpointHash::default(),
            telemetry: SegmentTelemetry {
                tokens_processed: 0,
                steps: 0,
                backend: "sim-skipped",
                pkey_id: 0,
                slot_handle: 0,
                dram_bytes: 0,
                tensor_ops: 0,
                mma_family: MmaFamily::None,
                peak_dram_bandwidth_bytes_per_s: 0,
            },
            seeds: SegmentSeeds {
                run_seed: 0,
                aux_seed: 0,
            },
        },
        t_champ_ms: 0,
        eval: EvalVerdict {
            quality_ok: false,
            physics_ok: false,
            reasons: vec![],
        },
        kernel_ok: true,
        image_digest,
        score_u64: 0,
        outcome: PipelineOutcome::MinerZero,
    }
}

/// Default dedupe ledger for sim tests.
///
/// # Errors
/// Propagates antinois constructor errors.
pub fn default_dedupe() -> Result<FingerprintDedupe, PipelineError> {
    FingerprintDedupe::new(DEFAULT_DEDUPE_SEGMENTS)
        .map_err(|e| PipelineError::Antinois(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim_search::find_faster_compiled;
    use hypertraining_cluster::Topology;
    use hypertraining_eval::fixtures::fixture_equal_quality_pairs;

    const CHAMP_SRC: &str = "def fused_gemm(a, b):\n    return a @ b\n";
    const CHAMP_BIN: &[u8] =
        b".version 7.0\n.entry gemm {\nadd.u32 %r1, %r2, %r3;\nmul.f32 %f1, %f2, %f3;\n}\n";

    #[test]
    fn faster_cand_sim_pipeline_positive_score() {
        let seeds = SegmentSeeds {
            run_seed: 7,
            aux_seed: 3,
        };
        let topo = Topology::new(2, 1, 1, 1);
        let budget = 5_000_000u64;
        let faster =
            find_faster_compiled(CHAMP_BIN, budget, &seeds, topo, 50_000).expect("find faster");
        let mut dedupe = default_dedupe().expect("dedupe");
        let mut cluster = SimBackend::new();
        let (champ_loss, cand_loss) = fixture_equal_quality_pairs();
        let input = SimPipelineInput {
            cand_source: "def pipeline_overlap(x, y):\n    return compute(x)+y\n",
            cand_compiled: &faster,
            champ_source: CHAMP_SRC,
            champ_compiled: CHAMP_BIN,
            miner_id: "m-fast",
            segment_index: 1,
            budget_tokens: budget,
            seeds,
            topology: topo,
            pkey_id: 10,
            noise_ms: 0,
            validator_lock: b"validator-lock-v1\n",
            admitted_files: vec![(
                "megatron/core/fusions/softmax.py".into(),
                b"x = 1\n".to_vec(),
            )],
            t_champ_ms_override: None,
            champ_loss,
            cand_loss,
        };
        let result = run_sim_pipeline(&input, &mut dedupe, &mut cluster).expect("pipeline");
        assert!(result.kernel_ok);
        assert!(result.antinois.allows_measure());
        assert!(
            result.cand_segment.wallclock_ms < result.t_champ_ms,
            "cand {} champ {}",
            result.cand_segment.wallclock_ms,
            result.t_champ_ms
        );
        assert!(
            result.eval.promote_allowed(),
            "guards should pass: {:?}",
            result.eval
        );
        assert!(result.score_u64 > 0, "score {}", result.score_u64);
    }
}
