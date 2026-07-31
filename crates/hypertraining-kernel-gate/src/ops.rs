//! Pure-Rust synthetic GEMV ops used as CPU stand-ins for kernel checks.

use crate::gate::KernelTensors;

/// Round `v` to BF16-representable f32 (truncate mantissa to 7 bits).
#[must_use]
pub fn round_bf16(v: f32) -> f32 {
    let bits = v.to_bits() & 0xFFFF_0000;
    f32::from_bits(bits)
}

/// Round `v` toward a TF32-like 10-bit mantissa (for diagnostics only).
#[must_use]
pub fn round_tf32(v: f32) -> f32 {
    // IEEE f32: 1 sign + 8 exp + 23 mantissa. TF32 keeps 10 mantissa bits.
    let bits = v.to_bits();
    let truncated = bits & 0xFFFF_E000;
    f32::from_bits(truncated)
}

fn check_dims(weight: &[f32], input: &[f32], rows: usize, cols: usize) {
    debug_assert_eq!(weight.len(), rows * cols, "weight length");
    debug_assert_eq!(input.len(), cols, "input length");
}

/// Reference FP32 GEMV with f64 accumulation: `y = W x`.
#[must_use]
pub fn gemv_ref_fp32(weight: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    check_dims(weight, input, rows, cols);
    let mut out = vec![0.0_f32; rows];
    for r in 0..rows {
        let mut acc = 0.0_f64;
        let row = &weight[r * cols..(r + 1) * cols];
        for (w, x) in row.iter().zip(input.iter()) {
            acc += f64::from(*w) * f64::from(*x);
        }
        #[allow(clippy::cast_possible_truncation)] // intentional f64 acc → f32 store
        {
            out[r] = acc as f32;
        }
    }
    out
}

/// Baseline same-dtype path: BF16-rounded operands, f32 accumulate (full reduction).
#[must_use]
pub fn gemv_baseline_bf16(weight: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    check_dims(weight, input, rows, cols);
    let mut out = vec![0.0_f32; rows];
    for r in 0..rows {
        let mut acc = 0.0_f32;
        let row = &weight[r * cols..(r + 1) * cols];
        for (w, x) in row.iter().zip(input.iter()) {
            acc += round_bf16(*w) * round_bf16(*x);
        }
        out[r] = acc;
    }
    out
}

/// Good candidate kernel: matches the BF16 baseline reduction (legitimate fused path).
#[must_use]
pub fn gemv_good_kernel(weight: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    gemv_baseline_bf16(weight, input, rows, cols)
}

/// Degraded candidate: truncated reduction — only the first `cols/2` products per row.
///
/// This systematically under-counts and exceeds the κ=2 budget vs the BF16 baseline.
#[must_use]
pub fn gemv_degraded_truncated(
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    check_dims(weight, input, rows, cols);
    let keep = cols / 2;
    let mut out = vec![0.0_f32; rows];
    for r in 0..rows {
        let mut acc = 0.0_f32;
        let row = &weight[r * cols..(r + 1) * cols];
        for j in 0..keep {
            acc += round_bf16(row[j]) * round_bf16(input[j]);
        }
        out[r] = acc;
    }
    out
}

/// Backward for `L = sum(y)` after `y = W x` (dL/dy = 1).
///
/// `grad_input[j] = sum_r W[r,j]`, `grad_weight[r,j] = x[j]` (with dtype rounding applied
/// the same way as the matching forward path).
#[must_use]
pub fn gemv_bwd_ref_fp32(
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>) {
    check_dims(weight, input, rows, cols);
    let mut grad_input = vec![0.0_f32; cols];
    let mut grad_weight = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for j in 0..cols {
            let w = weight[r * cols + j];
            grad_input[j] += w;
            grad_weight[r * cols + j] = input[j];
        }
    }
    (grad_input, grad_weight)
}

/// BF16-rounded backward matching [`gemv_baseline_bf16`].
#[must_use]
pub fn gemv_bwd_baseline_bf16(
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>) {
    check_dims(weight, input, rows, cols);
    let mut grad_input = vec![0.0_f32; cols];
    let mut grad_weight = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for j in 0..cols {
            let w = round_bf16(weight[r * cols + j]);
            let x = round_bf16(input[j]);
            grad_input[j] += w;
            grad_weight[r * cols + j] = x;
        }
    }
    (grad_input, grad_weight)
}

/// Good bwd = baseline bwd.
#[must_use]
pub fn gemv_bwd_good_kernel(
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>) {
    gemv_bwd_baseline_bf16(weight, input, rows, cols)
}

/// Truncated bwd: only first half of columns contribute to grads (matches degraded fwd).
#[must_use]
pub fn gemv_bwd_degraded_truncated(
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>) {
    check_dims(weight, input, rows, cols);
    let keep = cols / 2;
    let mut grad_input = vec![0.0_f32; cols];
    let mut grad_weight = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for j in 0..keep {
            let w = round_bf16(weight[r * cols + j]);
            let x = round_bf16(input[j]);
            grad_input[j] += w;
            grad_weight[r * cols + j] = x;
        }
    }
    (grad_input, grad_weight)
}

/// Bundle forward + backward tensors for a named kernel style.
#[must_use]
pub fn kernel_tensors_ref(
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> KernelTensors {
    let output = gemv_ref_fp32(weight, input, rows, cols);
    let (grad_input, grad_weight) = gemv_bwd_ref_fp32(weight, input, rows, cols);
    KernelTensors {
        output,
        grad_input,
        grad_weight,
    }
}

/// Baseline same-dtype bundle.
#[must_use]
pub fn kernel_tensors_baseline(
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> KernelTensors {
    let output = gemv_baseline_bf16(weight, input, rows, cols);
    let (grad_input, grad_weight) = gemv_bwd_baseline_bf16(weight, input, rows, cols);
    KernelTensors {
        output,
        grad_input,
        grad_weight,
    }
}

/// Good candidate bundle.
#[must_use]
pub fn kernel_tensors_good(
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> KernelTensors {
    let output = gemv_good_kernel(weight, input, rows, cols);
    let (grad_input, grad_weight) = gemv_bwd_good_kernel(weight, input, rows, cols);
    KernelTensors {
        output,
        grad_input,
        grad_weight,
    }
}

/// Degraded (truncated reduction) candidate bundle.
#[must_use]
pub fn kernel_tensors_degraded(
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) -> KernelTensors {
    let output = gemv_degraded_truncated(weight, input, rows, cols);
    let (grad_input, grad_weight) = gemv_bwd_degraded_truncated(weight, input, rows, cols);
    KernelTensors {
        output,
        grad_input,
        grad_weight,
    }
}
