//! Result intake: receipt bind (I5) then Harbor grade → leaf.

use agent_dispatch::{verify_task_result_bind, BoundPatch, TaskResultV1};
use agent_pack::HarborPack;
use bundle::ScoreOrAbsence;

use crate::leaf_map::grade_to_score_or_absence;
use crate::verify::Verifier;

pub use agent_dispatch::{BoundPatch as IntakePatch, ExpectedReceiptBind, ReceiptBindError};

/// Successful intake: bound patch + mapped leaf value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntakeOk {
    pub model_patch: Vec<u8>,
    pub patch_sha256: [u8; 32],
    pub leaf: ScoreOrAbsence,
}

/// Verify receipt binds only — never grades.
///
/// # Errors
/// [`ReceiptBindError`] on bind failure.
pub fn verify_intake_receipt(
    exp: &ExpectedReceiptBind,
    result: &TaskResultV1,
) -> Result<BoundPatch, ReceiptBindError> {
    verify_task_result_bind(exp, result)
}

/// Verify receipt then grade. Bad receipts never call `verifier.grade`.
///
/// # Errors
/// [`ReceiptBindError`] when bind checks fail.
pub fn intake_and_grade(
    exp: &ExpectedReceiptBind,
    result: &TaskResultV1,
    verifier: &dyn Verifier,
    pack: &HarborPack,
) -> Result<IntakeOk, ReceiptBindError> {
    let b = verify_task_result_bind(exp, result)?;
    Ok(IntakeOk {
        leaf: grade_to_score_or_absence(verifier, pack, &b.model_patch),
        model_patch: b.model_patch,
        patch_sha256: b.patch_sha256,
    })
}
