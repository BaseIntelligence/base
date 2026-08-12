//! 2×2 attribution matrix (research/08 §7): decompose a submission's gain
//! over the reference into an **architecture** delta and a **kernel** delta.
//!
//! The matrix:
//! - `(A_ref, K_ref)` — reference arch + reference kernels: the published
//!   baseline run (already exists).
//! - `(A_sub, K_sub)` — the submission itself (already evaluated by the
//!   normal pipeline).
//! - `(A_sub, K_ref)` — submission arch on reference kernels: **new run**,
//!   isolates the architecture contribution.
//! - `(A_ref, K_sub)` — reference arch + submission kernels: **new run**,
//!   isolates the kernel contribution.
//!
//! [`build_attribution_runs`] is an operator-triggered job builder: given the
//! parent submission's validated [`SourceTree`] and the reference tree, it
//! assembles the two new cells as seam-validated trees ready for the normal
//! eval pipeline (`POST /v1/submissions/{id}/attribution` returns these
//! plans as JSON; executing them stays an operator action). The kernel swap
//! is a wholesale `kernels/`
//! directory replacement, sound because both sides implement the stable
//! [`KERNEL_INTERFACE_MD`] contract. Swapped cells are gated on the
//! [`HIDDEN_SHAPE_SUITE_SPEC`] correctness battery before scoring.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::zip_submit::{SourceTree, ZipSubmitError};

/// `KERNEL_INTERFACE.md` convention — the stable contract a submission's
/// `kernels/` directory implements so kernels are mechanically swappable
/// between trees (also published to miners; keep in sync with the docs repo).
pub const KERNEL_INTERFACE_MD: &str = "\
KERNEL_INTERFACE.md (Prism v3 custom-kernel contract)

Layout: `kernels/<op_name>.py`, one file per op; `<op_name>` is
`[a-z0-9_]+`. The directory is the swap unit: attribution cells replace a
tree's whole `kernels/` dir with another tree's, so file names and exported
symbols MUST stay identical across implementations of the same op set.

Each `kernels/<op_name>.py` exports:
- `OP: str` — the op name (must equal `<op_name>`).
- `class Kernel` with:
  - `bind(model, ctx) -> model` — wire the optimized op into a model
    (replace/wrap submodules in place and return the model).
  - `reference(x, *args, **kwargs) -> torch.Tensor` — pure-PyTorch
    reference implementation of the op (present in EVERY tree: it is what
    the correctness suite compares against and what `K_ref` cells run).
  - `optimized(x, *args, **kwargs) -> torch.Tensor` — the custom op.
  - `min_shapes() -> list[tuple]` — declared shape coverage; the hidden
    suite draws shapes outside this list to catch shape-specialization.

Rules: pure Python + torch; native code only via in-tree source-built
`torch.utils.cpp_extension.load_inline` (intake bans prebuilt binaries and
`ctypes`). No I/O, no threads/streams/timers, no RNG outside the passed
ctx. Kernels run under the harness netns sandbox like all miner code.";

/// Hidden-shape correctness suite — gate applied to every kernel-swapped
/// attribution cell (and available to the main eval for `kernels/` trees)
/// BEFORE the scored run. Spec for the harness-side implementation:
///
/// 1. For each `kernels/<op>.py`, instantiate `Kernel` and draw shapes from
///    a seed-lattice grid (batch × seq × dim ∈ hidden ranges derived from
///    the eval seed), explicitly excluding `min_shapes()` entries.
/// 2. fp64 reference cast: `want = reference(x.double())`,
///    `got = optimized(x)`; require finite and
///    `allclose(got.double(), want, rtol=1e-4, atol=1e-5)`.
/// 3. Determinism: run `optimized` twice on one shape; require bitwise
///    equality (mirrors cheatguard's determinism probe).
/// 4. Any mismatch → cell tagged `correctness_failed`, score withheld, and
///    the finding is recorded under `attribution.correctness` in the run
///    report (never silently scored).
pub const HIDDEN_SHAPE_SUITE_SPEC: &str = "\
HIDDEN-SHAPE CORRECTNESS SUITE (gate for kernel-swapped cells):
per kernels/<op>.py draw hidden (batch, seq, dim) shapes from the eval-seed
lattice outside Kernel.min_shapes(); compare optimized(x) vs reference(x)
in fp64 with rtol=1e-4/atol=1e-5; require finite outputs and a bitwise-equal
second run. Any failure tags the cell correctness_failed and withholds its
score (recorded under attribution.correctness in METRICS_JSON).";

/// Which of the two off-diagonal matrix cells a run realizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributionCell {
    /// Submission architecture + reference kernels (isolates the arch delta).
    SubmissionArchReferenceKernels,
    /// Reference architecture + submission kernels (isolates the kernel delta).
    ReferenceArchSubmissionKernels,
}

impl AttributionCell {
    /// Stable label (used in run ids + DB rows).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::SubmissionArchReferenceKernels => "a_sub_k_ref",
            Self::ReferenceArchSubmissionKernels => "a_ref_k_sub",
        }
    }
}

/// One off-diagonal attribution cell as a runnable plan.
#[derive(Debug, Clone)]
pub struct AttributionRun {
    /// Deterministic plan id: `attr_` + first 16 hex of
    /// `sha256("prism-attribution-v1\0" ‖ parent_id ‖ cell ‖ tree hash)`.
    pub run_id: String,
    /// Submission this attribution was triggered for.
    pub parent_submission_id: String,
    /// Matrix cell realized by this run.
    pub cell: AttributionCell,
    /// Assembled + seam-validated source tree for the run.
    pub tree: SourceTree,
}

/// Attribution builder errors.
#[derive(Debug, Error)]
pub enum AttributionError {
    /// Submission tree carries no `kernels/` — attribution is N/A.
    #[error("submission has no kernels/ directory")]
    NoSubmissionKernels,
    /// Reference tree carries no `kernels/` to swap in.
    #[error("reference tree has no kernels/ directory")]
    NoReferenceKernels,
    /// An assembled cell fails the source-tree contract.
    #[error("assembled cell invalid: {0}")]
    InvalidCell(#[from] ZipSubmitError),
}

/// Build the two off-diagonal attribution runs for a submission.
///
/// `submission` / `reference` are validated [`SourceTree`]s (intake /
/// recipe baseline). The swap replaces `kernels/` wholesale; seam files and
/// entries are inherited from the architecture side of each cell.
///
/// # Errors
/// [`AttributionError`] when either side lacks `kernels/` or an assembled
/// cell no longer satisfies the tree contract.
pub fn build_attribution_runs(
    parent_submission_id: &str,
    submission: &SourceTree,
    reference: &SourceTree,
) -> Result<Vec<AttributionRun>, AttributionError> {
    let sub_kernels = submission.kernels();
    if sub_kernels.is_empty() {
        return Err(AttributionError::NoSubmissionKernels);
    }
    let ref_kernels = reference.kernels();
    if ref_kernels.is_empty() {
        return Err(AttributionError::NoReferenceKernels);
    }
    let cells = [
        (
            AttributionCell::SubmissionArchReferenceKernels,
            swap_kernels(submission, &ref_kernels),
            submission.entry(),
        ),
        (
            AttributionCell::ReferenceArchSubmissionKernels,
            swap_kernels(reference, &sub_kernels),
            reference.entry(),
        ),
    ];
    let mut runs = Vec::with_capacity(2);
    for (cell, files, entry) in cells {
        let tree = SourceTree::from_files(files, entry.to_owned());
        tree.validate().map_err(AttributionError::InvalidCell)?;
        let run_id = attribution_run_id(parent_submission_id, cell, &tree);
        runs.push(AttributionRun {
            run_id,
            parent_submission_id: parent_submission_id.to_owned(),
            cell,
            tree,
        });
    }
    Ok(runs)
}

/// Replace the tree's whole `kernels/` dir with `kernels` (swap unit).
fn swap_kernels(
    base: &SourceTree,
    kernels: &BTreeMap<String, Vec<u8>>,
) -> BTreeMap<String, Vec<u8>> {
    let mut files: BTreeMap<String, Vec<u8>> = base
        .files()
        .iter()
        .filter(|(p, _)| !p.starts_with("kernels/"))
        .map(|(p, b)| (p.clone(), b.clone()))
        .collect();
    files.extend(kernels.iter().map(|(p, b)| (p.clone(), b.clone())));
    files
}

fn attribution_run_id(parent: &str, cell: AttributionCell, tree: &SourceTree) -> String {
    let mut h = Sha256::new();
    h.update(b"prism-attribution-v1\0");
    h.update(parent.as_bytes());
    h.update([0x00]);
    h.update(cell.label().as_bytes());
    h.update([0x00]);
    h.update(tree.canonical_sha256().as_bytes());
    let digest = hex::encode(h.finalize());
    format!("attr_{}", &digest[..16])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    const ARCH: &str = "import torch\ndef build_model(ctx):\n    return torch.nn.Linear(8, 8)\n";
    const TRAIN: &str = "def train(model, ctx):\n    return {}\n";

    fn tree(kernel_body: &str) -> SourceTree {
        SourceTree::from_files(
            [
                ("architecture.py".to_owned(), ARCH.as_bytes().to_vec()),
                ("train.py".to_owned(), TRAIN.as_bytes().to_vec()),
                (
                    "kernels/rmsnorm.py".to_owned(),
                    kernel_body.as_bytes().to_vec(),
                ),
            ]
            .into_iter()
            .collect(),
            "train.py".to_owned(),
        )
    }

    #[test]
    fn builds_two_off_diagonal_cells() {
        let sub = tree("OP = 'rmsnorm'  # submission kernels\n");
        let ref_tree = tree("OP = 'rmsnorm'  # reference kernels\n");
        let runs = build_attribution_runs("sub_abc", &sub, &ref_tree).unwrap();
        assert_eq!(runs.len(), 2);
        let a_sub_k_ref = &runs[0];
        assert_eq!(
            a_sub_k_ref.cell,
            AttributionCell::SubmissionArchReferenceKernels
        );
        assert_eq!(
            a_sub_k_ref.tree.files()["kernels/rmsnorm.py"],
            b"OP = 'rmsnorm'  # reference kernels\n"
        );
        assert_eq!(a_sub_k_ref.tree.entry(), "train.py");
        let a_ref_k_sub = &runs[1];
        assert_eq!(
            a_ref_k_sub.cell,
            AttributionCell::ReferenceArchSubmissionKernels
        );
        assert_eq!(
            a_ref_k_sub.tree.files()["kernels/rmsnorm.py"],
            b"OP = 'rmsnorm'  # submission kernels\n"
        );
        for run in &runs {
            assert_eq!(run.parent_submission_id, "sub_abc");
            assert!(run.run_id.starts_with("attr_"));
            assert_eq!(run.run_id.len(), 5 + 16);
        }
    }

    #[test]
    fn run_ids_are_deterministic() {
        let sub = tree("OP = 'rmsnorm'  # s\n");
        let ref_tree = tree("OP = 'rmsnorm'  # r\n");
        let a = build_attribution_runs("sub_x", &sub, &ref_tree).unwrap();
        let b = build_attribution_runs("sub_x", &sub, &ref_tree).unwrap();
        assert_eq!(a[0].run_id, b[0].run_id);
        assert_eq!(a[1].run_id, b[1].run_id);
        assert_ne!(a[0].run_id, a[1].run_id);
    }

    #[test]
    fn errors_without_kernels() {
        let bare = SourceTree::from_files(
            [
                ("architecture.py".to_owned(), ARCH.as_bytes().to_vec()),
                ("train.py".to_owned(), TRAIN.as_bytes().to_vec()),
            ]
            .into_iter()
            .collect(),
            "train.py".to_owned(),
        );
        let kern = tree("OP = 'rmsnorm'\n");
        assert!(matches!(
            build_attribution_runs("s", &bare, &kern),
            Err(AttributionError::NoSubmissionKernels)
        ));
        assert!(matches!(
            build_attribution_runs("s", &kern, &bare),
            Err(AttributionError::NoReferenceKernels)
        ));
    }
}
