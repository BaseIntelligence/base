"""Compose + deploy-plan wiring for the live-registry orchestrator/task refs.

* The pullable canonical digest threads through
  ``prepare_deployment``/``build_deploy_plan``/``generate_app_compose`` and the
  compose pins it (assert_digest_pinned passes; no bare tag).
* Optionally the compose points the in-CVM orchestrator at the live-registry side
  manifest via the ``CHALLENGE_OWN_RUNNER_LIVE_REGISTRY`` static env; with no live
  manifest configured the compose is byte-identical (compose-hash stable).
"""

from __future__ import annotations

import yaml

from agent_challenge.canonical import compose as c
from agent_challenge.canonical import live_registry as lr
from agent_challenge.selfdeploy import plan as p

PULLABLE = "docker.io/mathiiss/agent-challenge-canonical@sha256:" + ("a" * 64)
KEY_URL = "https://validator.example/key-release"


def _orchestrator_env(compose) -> list[str]:
    parsed = yaml.safe_load(compose["docker_compose_file"])
    return parsed["services"][c.ORCHESTRATOR_SERVICE]["environment"]


# --------------------------------------------------------------------------- #
# compose: default byte-identical; live env only when configured
# --------------------------------------------------------------------------- #
def test_compose_default_has_no_live_registry_env():
    compose = c.generate_app_compose(orchestrator_image=PULLABLE)
    assert not any(e.startswith(f"{lr.LIVE_REGISTRY_ENV}=") for e in _orchestrator_env(compose))


def test_compose_default_is_byte_identical_to_before():
    # Adding the optional param must not perturb the default compose bytes/hash.
    base = c.generate_app_compose(orchestrator_image=PULLABLE)
    also = c.generate_app_compose(orchestrator_image=PULLABLE, live_registry_manifest_path=None)
    assert c.render_app_compose(base) == c.render_app_compose(also)
    assert c.app_compose_hash(base) == c.app_compose_hash(also)


def test_compose_live_registry_env_present_when_configured():
    manifest_path = "/opt/agent-challenge/golden/live-registry-refs.json"
    compose = c.generate_app_compose(
        orchestrator_image=PULLABLE, live_registry_manifest_path=manifest_path
    )
    env = _orchestrator_env(compose)
    assert f"{lr.LIVE_REGISTRY_ENV}={manifest_path}" in env
    # Different from the default compose-hash (a distinct, still-deterministic run).
    assert c.app_compose_hash(compose) != c.app_compose_hash(
        c.generate_app_compose(orchestrator_image=PULLABLE)
    )


def test_compose_with_live_registry_is_deterministic():
    manifest_path = "/opt/agent-challenge/golden/live-registry-refs.json"
    a = c.generate_app_compose(
        orchestrator_image=PULLABLE, live_registry_manifest_path=manifest_path
    )
    b = c.generate_app_compose(
        orchestrator_image=PULLABLE, live_registry_manifest_path=manifest_path
    )
    assert c.render_app_compose(a) == c.render_app_compose(b)
    # render == normalize verbatim invariant preserved.
    from agent_challenge.canonical import measurement as m

    assert c.render_app_compose(a) == m.normalize_app_compose(a)


# --------------------------------------------------------------------------- #
# plan: thread the pullable orchestrator digest + resolve recorded image
# --------------------------------------------------------------------------- #
def test_prepare_deployment_pins_pushed_digest():
    prepared = p.prepare_deployment(image=PULLABLE, key_release_url=KEY_URL)
    assert prepared.image == PULLABLE
    assert c.assert_digest_pinned(prepared.image) == PULLABLE
    assert PULLABLE in prepared.compose_text


def test_resolve_orchestrator_image_prefers_explicit():
    assert p.resolve_orchestrator_image(image=PULLABLE) == PULLABLE


def test_resolve_orchestrator_image_from_recorded_manifest():
    reg_path = lr.DEFAULT_LIVE_REGISTRY_PATH
    resolved = p.resolve_orchestrator_image(image=None, live_registry_path=reg_path)
    reg = lr.load_live_registry(reg_path)
    assert resolved == reg.orchestrator_image
    assert c.assert_digest_pinned(resolved) == resolved


def test_resolve_orchestrator_image_requires_a_source():
    import pytest

    with pytest.raises(p.PrepareError):
        p.resolve_orchestrator_image(image=None, live_registry_path=None)


def test_build_deploy_plan_pins_pushed_digest_and_live_env():
    plan = p.build_deploy_plan(
        image=PULLABLE,
        key_release_url=KEY_URL,
        live_registry_manifest_path="/opt/agent-challenge/golden/live-registry-refs.json",
    )
    assert plan.image == PULLABLE
    env = _orchestrator_env(plan.prepared.compose)
    assert any(e.startswith(f"{lr.LIVE_REGISTRY_ENV}=") for e in env)
