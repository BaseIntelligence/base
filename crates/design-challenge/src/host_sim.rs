//! Host `SimSandbox` gate — fail-closed outside explicit non-prod CI opt-in.
//! Prod/staging must use Docker; host sim needs `BASE_ALLOW_HOST_SIM` + non-prod.

/// Mainnet netuid / explicit deploy env → prod (host Sim forbidden).
#[must_use]
pub fn is_prod_env(netuid: u16, deploy_env: Option<&str>) -> bool {
    netuid == 100 || matches!(deploy_env, Some("prod" | "production"))
}

/// Whether host `SimSandbox` may be selected.
#[must_use]
pub fn host_sim_allowed(netuid: u16, allow_host_sim: bool, deploy_env: Option<&str>) -> bool {
    allow_host_sim && !is_prod_env(netuid, deploy_env)
}

/// Error when force-sim is requested without host-sim opt-in.
#[must_use]
pub fn force_sim_refusal_reason() -> &'static str {
    "DESIGN_FORCE_SIM requires BASE_ALLOW_HOST_SIM=1 and non-prod \
     (refusing host SimSandbox)"
}

/// Validate an explicit force-sim request.
///
/// # Errors
/// When force-sim is requested but host sim is not allowed.
pub fn require_host_sim_for_force(
    force_sim: bool,
    netuid: u16,
    allow_host_sim: bool,
    deploy_env: Option<&str>,
) -> Result<(), String> {
    if force_sim && !host_sim_allowed(netuid, allow_host_sim, deploy_env) {
        return Err(force_sim_refusal_reason().into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_sim_forbidden_without_allow() {
        assert!(!host_sim_allowed(541, false, Some("staging")));
        assert!(require_host_sim_for_force(true, 541, false, Some("staging")).is_err());
    }

    #[test]
    fn host_sim_forbidden_on_prod_even_with_allow() {
        assert!(!host_sim_allowed(100, true, None));
        assert!(!host_sim_allowed(541, true, Some("prod")));
        assert!(require_host_sim_for_force(true, 100, true, None).is_err());
        assert!(require_host_sim_for_force(true, 541, true, Some("production")).is_err());
    }

    #[test]
    fn host_sim_allowed_on_non_prod_ci() {
        assert!(host_sim_allowed(541, true, Some("staging")));
        assert!(host_sim_allowed(1, true, None));
        assert!(require_host_sim_for_force(true, 541, true, Some("local")).is_ok());
        assert!(require_host_sim_for_force(false, 541, false, None).is_ok());
    }
}
