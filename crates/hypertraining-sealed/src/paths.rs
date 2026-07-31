//! Allowlist / denylist path rules from challenge brief §6.2–6.3.

/// Default allowlist globs / exact paths (brief §6.2).
///
/// `**` matches zero or more path segments. Exact files have no `*`.
pub const DEFAULT_ALLOWLIST_GLOBS: &[&str] = &[
    "megatron/core/fusions/**",
    "megatron/core/extensions/**",
    "megatron/core/transformer/**",
    "megatron/core/tensor_parallel/**",
    "megatron/core/pipeline_parallel/**",
    "megatron/core/distributed/**",
    "megatron/core/optimizer/**",
    "megatron/core/parallel_state.py",
    "megatron/core/model_parallel_config.py",
    "miner_ext/**",
];

/// Paths under `megatron/core/transformer/**` that are NOT allowlisted.
const TRANSFORMER_ALLOWLIST_EXCEPTIONS: &[&str] = &["megatron/core/transformer/moe_logging.py"];

/// Default denylist directory globs (brief §6.3).
pub const DEFAULT_DENYLIST_GLOBS: &[&str] = &[
    "megatron/core/datasets/**",
    "megatron/core/dist_checkpointing/**",
    "megatron/bridge/data/**",
    "3rdparty/**",
];

/// Default denylist exact files (brief §6.3).
pub const DEFAULT_DENYLIST_PATHS: &[&str] = &[
    "megatron/core/num_microbatches_calculator.py",
    "megatron/training/checkpointing.py",
    "megatron/bridge/training/eval.py",
    "pyproject.toml",
    "uv.lock",
];

/// Normalize a repo-relative path: strip `./`, backslashes → `/`, drop trailing `/`.
#[must_use]
pub fn normalize_path(path: &str) -> String {
    let mut p = path.trim().replace('\\', "/");
    while p.starts_with("./") {
        p = p[2..].to_owned();
    }
    while p.starts_with('/') {
        p = p[1..].to_owned();
    }
    while p.ends_with('/') && p.len() > 1 {
        p.pop();
    }
    p
}

/// True when `path` matches the default denylist.
#[must_use]
pub fn is_denylisted(path: &str) -> bool {
    let p = normalize_path(path);
    if DEFAULT_DENYLIST_PATHS.iter().any(|d| *d == p) {
        return true;
    }
    DEFAULT_DENYLIST_GLOBS.iter().any(|g| glob_match(g, &p))
}

/// True when `path` is on the default allowlist (and not an allowlist exception).
///
/// Denylist takes precedence at admission time; callers should check denylist first.
#[must_use]
pub fn is_allowlisted(path: &str) -> bool {
    let p = normalize_path(path);
    if TRANSFORMER_ALLOWLIST_EXCEPTIONS.iter().any(|e| *e == p) {
        return false;
    }
    DEFAULT_ALLOWLIST_GLOBS.iter().any(|g| glob_match(g, &p))
}

/// Minimal glob: `**` = any suffix (including empty), `*` = one segment chars without `/`.
fn glob_match(pattern: &str, path: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix("/**") {
        if path == prefix {
            return true;
        }
        let with_slash = format!("{prefix}/");
        return path.starts_with(&with_slash);
    }
    if pattern.contains('*') {
        return simple_star_match(pattern, path);
    }
    pattern == path
}

fn simple_star_match(pattern: &str, path: &str) -> bool {
    // Only used if we add single-star patterns later; exact equality fallback.
    if !pattern.contains('*') {
        return pattern == path;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == path;
    }
    let mut rest = path;
    if !parts[0].is_empty() {
        if !rest.starts_with(parts[0]) {
            return false;
        }
        rest = &rest[parts[0].len()..];
    }
    for (i, part) in parts.iter().enumerate().skip(1) {
        if part.is_empty() {
            if i == parts.len() - 1 {
                return true;
            }
            continue;
        }
        if i == parts.len() - 1 {
            return rest.ends_with(part);
        }
        match rest.find(part) {
            Some(idx) => rest = &rest[idx + part.len()..],
            None => return false,
        }
    }
    true
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_fusions_and_deny_datasets() {
        assert!(is_allowlisted("megatron/core/fusions/softmax.py"));
        assert!(is_allowlisted("megatron/core/parallel_state.py"));
        assert!(is_allowlisted("miner_ext/kernels/foo.py"));
        assert!(!is_allowlisted("megatron/core/transformer/moe_logging.py"));
        assert!(is_allowlisted("megatron/core/transformer/moe/router.py"));
        assert!(is_denylisted("megatron/core/datasets/blended.py"));
        assert!(is_denylisted("pyproject.toml"));
        assert!(is_denylisted("megatron/training/checkpointing.py"));
        assert!(!is_denylisted("megatron/core/fusions/x.py"));
    }
}
