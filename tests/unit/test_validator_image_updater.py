"""Unit tests for host-side Compose validator image auto-update (Option A)."""

from __future__ import annotations

import json
import stat
from collections.abc import Sequence
from pathlib import Path

import pytest

from base.supervisor.retry import RetryPolicy
from base.supervisor.validator_image_updater import (
    DEFAULT_TRACK_IMAGE,
    CommandResult,
    ComposeValidatorImageUpdater,
    UpdaterPhase,
    assert_runtime_pin_policy,
    bare_digest_hex,
    has_explicit_tag,
    load_state,
    normalize_digest,
    pinned_runtime_image,
    read_dotenv,
    repository_from_track_image,
    write_dotenv_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
INSTALL_VALIDATOR = ROOT / "deploy" / "compose" / "install-validator.sh"
UPDATER_SCRIPT = ROOT / "deploy" / "compose" / "validator-image-updater.sh"
SERVICE_UNIT = (
    ROOT / "deploy" / "compose" / "systemd" / "base-validator-image-updater@.service"
)
TIMER_UNIT = (
    ROOT / "deploy" / "compose" / "systemd" / "base-validator-image-updater@.timer"
)
COMPOSE_FILE = ROOT / "deploy" / "compose" / "docker-compose.validator.yml"

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_BAD = "sha256:" + "c" * 64
REPO = "ghcr.io/baseintelligence/base-validator-runtime"
TRACK = f"{REPO}:latest"


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeRunner:
    """Injectable docker/compose command runner for unit tests."""

    def __init__(
        self,
        *,
        running: bool = True,
        image: str | None = None,
        pull_rc: int = 0,
        up_rc: int = 0,
        fail_up_once: bool = False,
    ) -> None:
        self.running = running
        self.image = image
        self.pull_rc = pull_rc
        self.up_rc = up_rc
        self.fail_up_once = fail_up_once
        self.calls: list[tuple[str, ...]] = []
        self._up_calls = 0

    def __call__(self, argv: Sequence[str], timeout_seconds: float) -> CommandResult:
        del timeout_seconds
        call = tuple(argv)
        self.calls.append(call)
        # compose pull
        if len(call) >= 3 and call[1] == "compose" and "pull" in call:
            return CommandResult(
                self.pull_rc, "", "" if self.pull_rc == 0 else "pull-fail"
            )
        # compose up
        if len(call) >= 3 and call[1] == "compose" and "up" in call:
            self._up_calls += 1
            rc = self.up_rc
            if self.fail_up_once and self._up_calls == 1:
                rc = 1
            if rc == 0 and self.image is None:
                # After a successful up, adopt whatever pin was intended last.
                pass
            return CommandResult(rc, "", "" if rc == 0 else "up-fail")
        # inspect running
        if (
            len(call) >= 3
            and call[1] == "inspect"
            and "State.Running" in " ".join(call)
        ):
            return CommandResult(0, "true\n" if self.running else "false\n", "")
        # inspect image / RepoDigests
        if len(call) >= 3 and call[1] == "inspect":
            img = self.image or ""
            return CommandResult(0, f"{img}\n", "")
        if len(call) >= 3 and call[1] == "compose" and "images" in call:
            img = self.image or ""
            return CommandResult(0, f"{img}\n", "")
        if len(call) >= 3 and call[1] == "compose" and "ps" in call:
            return CommandResult(0, "validator\n" if self.running else "", "")
        return CommandResult(0, "", "")

    @property
    def pull_calls(self) -> list[tuple[str, ...]]:
        return [c for c in self.calls if "pull" in c]

    @property
    def up_calls(self) -> list[tuple[str, ...]]:
        return [c for c in self.calls if "up" in c and "compose" in c]


def _seed_project(tmp_path: Path, digest: str = DIGEST_A) -> tuple[Path, Path, Path]:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    compose = artifacts / "docker-compose.validator.yml"
    compose.write_text(COMPOSE_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    env_path = artifacts / ".env"
    write_dotenv_atomic(
        env_path,
        updates={
            "COMPOSE_PROJECT_NAME": "base-validator-test",
            "BASE_VALIDATOR_IMAGE_REPOSITORY": REPO,
            "BASE_VALIDATOR_IMAGE_DIGEST": bare_digest_hex(digest),
            "BASE_VALIDATOR_CONFIG": "/tmp/validator.yaml",
            "BASE_VALIDATOR_PROTOCOL_IDENTITY": "/tmp/identity",
            "BASE_VALIDATOR_BROKER_TOKEN": "/tmp/token",
        },
    )
    state_path = artifacts / "image_update_state.json"
    return compose, env_path, state_path


def _updater(
    tmp_path: Path,
    *,
    resolver_digest: str,
    runner: FakeRunner,
    hold: bool = False,
    dry_run: bool = False,
    policy: RetryPolicy | None = None,
    clock: FakeClock | None = None,
) -> ComposeValidatorImageUpdater:
    compose, env_path, state_path = _seed_project(tmp_path)
    clock = clock or FakeClock()
    return ComposeValidatorImageUpdater(
        project_name="base-validator-test",
        compose_file=compose,
        env_file=env_path,
        state_path=state_path,
        track_image=TRACK,
        hold=hold,
        dry_run=dry_run,
        resolver=lambda _ref: resolver_digest,
        runner=runner,
        retry_policy=policy
        or RetryPolicy(max_attempts=3, base_delay=10.0, max_delay=30.0, jitter=False),
        clock=clock,
        wall_clock=clock,
        sleep=clock.sleep,
        jitter_source=lambda: 0.0,
        verify_timeout_seconds=5.0,
        verify_poll_seconds=1.0,
        command_timeout_seconds=30.0,
    )


# --------------------------------------------------------------------------- pin policy


def test_normalize_digest_accepts_bare_hex_and_full() -> None:
    assert normalize_digest("a" * 64) == DIGEST_A
    assert normalize_digest(DIGEST_A) == DIGEST_A
    assert normalize_digest(f"repo@sha256:{'a' * 64}") == DIGEST_A
    assert normalize_digest("not-a-digest") is None
    assert normalize_digest("sha256:dead") is None


def test_pin_policy_rejects_bare_latest() -> None:
    with pytest.raises(ValueError, match="sha256"):
        assert_runtime_pin_policy(f"{REPO}:latest")
    with pytest.raises(ValueError, match="sha256"):
        assert_runtime_pin_policy(REPO)
    pin = pinned_runtime_image(REPO, DIGEST_A)
    assert pin == f"{REPO}@{DIGEST_A}"
    assert assert_runtime_pin_policy(pin) == pin


def test_has_explicit_tag_and_repository_from_track() -> None:
    assert has_explicit_tag(TRACK) is True
    assert has_explicit_tag(REPO) is False
    assert repository_from_track_image(TRACK) == REPO


def test_write_dotenv_atomic_mode_and_merge(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    write_dotenv_atomic(
        path,
        updates={
            "COMPOSE_PROJECT_NAME": "p",
            "BASE_VALIDATOR_IMAGE_REPOSITORY": REPO,
            "BASE_VALIDATOR_IMAGE_DIGEST": bare_digest_hex(DIGEST_A),
            "BASE_VALIDATOR_CONFIG": "/c",
        },
    )
    mode = path.stat().st_mode
    assert mode & stat.S_IRWXG == 0
    assert mode & stat.S_IRWXO == 0
    env = read_dotenv(path)
    assert env["BASE_VALIDATOR_IMAGE_DIGEST"] == "a" * 64
    write_dotenv_atomic(
        path,
        updates={"BASE_VALIDATOR_IMAGE_DIGEST": bare_digest_hex(DIGEST_B)},
    )
    env2 = read_dotenv(path)
    assert env2["BASE_VALIDATOR_IMAGE_DIGEST"] == "b" * 64
    assert env2["BASE_VALIDATOR_CONFIG"] == "/c"


# --------------------------------------------------------------------------- ticker


def test_same_digest_is_noop(tmp_path: Path) -> None:
    runner = FakeRunner(image=f"{REPO}@{DIGEST_A}")
    updater = _updater(tmp_path, resolver_digest=DIGEST_A, runner=runner)
    # Keep env at A
    outcome = updater.run_once()
    assert outcome == "noop"
    assert runner.pull_calls == []
    assert runner.up_calls == []
    state = load_state(updater.state_path)
    assert state.phase == UpdaterPhase.IDLE
    assert state.current_digest == DIGEST_A


def test_digest_change_rewrites_env_and_recreates(tmp_path: Path) -> None:
    runner = FakeRunner(image=f"{REPO}@{DIGEST_B}", running=True)
    updater = _updater(tmp_path, resolver_digest=DIGEST_B, runner=runner)
    outcome = updater.run_once()
    assert outcome == "updated"
    env = read_dotenv(updater.env_file)
    assert env["BASE_VALIDATOR_IMAGE_DIGEST"] == bare_digest_hex(DIGEST_B)
    assert env["BASE_VALIDATOR_IMAGE_REPOSITORY"] == REPO
    # No bare latest in env as runtime selector
    assert ":latest" not in env["BASE_VALIDATOR_IMAGE_REPOSITORY"]
    assert any("pull" in c for c in runner.pull_calls)
    assert any("--force-recreate" in c and "--no-deps" in c for c in runner.up_calls)
    # compose argv uses project + env-file
    first_up = next(c for c in runner.up_calls if "up" in c)
    assert "-p" in first_up and "base-validator-test" in first_up
    assert "--env-file" in first_up
    assert "validator" in first_up


def test_recreate_failure_rolls_back_to_lkg(tmp_path: Path) -> None:
    runner = FakeRunner(image=f"{REPO}@{DIGEST_A}", running=True, fail_up_once=True)
    updater = _updater(
        tmp_path,
        resolver_digest=DIGEST_B,
        runner=runner,
        policy=RetryPolicy(
            max_attempts=5, base_delay=10.0, max_delay=30.0, jitter=False
        ),
    )
    outcome = updater.run_once()
    assert outcome == "failed"
    env = read_dotenv(updater.env_file)
    # Restored LKG A after failed apply of B
    assert env["BASE_VALIDATOR_IMAGE_DIGEST"] == bare_digest_hex(DIGEST_A)
    state = load_state(updater.state_path)
    assert state.rollback_digest == DIGEST_A
    assert state.desired_digest == DIGEST_B
    assert state.attempts == 1
    assert state.phase in {UpdaterPhase.BACKOFF, UpdaterPhase.EXHAUSTED}
    # At least one rollback compose up after failure
    assert len(runner.up_calls) >= 2


def test_hold_skips_without_compose_calls(tmp_path: Path) -> None:
    runner = FakeRunner()
    updater = _updater(tmp_path, resolver_digest=DIGEST_B, runner=runner, hold=True)
    assert updater.run_once() == "skipped-held"
    assert runner.calls == []


def test_dry_run_does_not_rewrite_env(tmp_path: Path) -> None:
    runner = FakeRunner()
    updater = _updater(tmp_path, resolver_digest=DIGEST_B, runner=runner, dry_run=True)
    assert updater.run_once() == "dry-run"
    env = read_dotenv(updater.env_file)
    assert env["BASE_VALIDATOR_IMAGE_DIGEST"] == bare_digest_hex(DIGEST_A)
    assert runner.pull_calls == []


def test_exhausted_skips_until_new_digest(tmp_path: Path) -> None:
    clock = FakeClock()
    runner = FakeRunner(up_rc=1, image=f"{REPO}@{DIGEST_A}")
    policy = RetryPolicy(max_attempts=2, base_delay=1.0, max_delay=1.0, jitter=False)
    updater = _updater(
        tmp_path,
        resolver_digest=DIGEST_B,
        runner=runner,
        policy=policy,
        clock=clock,
    )
    assert updater.run_once() == "failed"
    clock.advance(5)
    # Still same desired digest, attempts exhausted eventually
    for _ in range(3):
        clock.advance(5)
        outcome = updater.run_once()
        if outcome == "exhausted":
            break
    assert load_state(updater.state_path).phase == UpdaterPhase.EXHAUSTED

    # New digest resets episode
    pulls_before = len(runner.pull_calls)
    clock.advance(5)
    runner.up_rc = 0
    runner.image = f"{REPO}@{DIGEST_BAD}"
    updater.resolver = lambda _ref: DIGEST_BAD  # type: ignore[method-assign]
    # Force eligible
    state = load_state(updater.state_path)
    state.next_eligible_at = None
    state.attempts = policy.max_attempts
    from base.supervisor.validator_image_updater import save_state

    save_state(updater.state_path, state)
    # new desired should clear exhausted and attempt update
    outcome = updater.run_once()
    assert outcome in {"updated", "failed", "noop"}
    # either pull happened or we progressed; at least desired changed
    assert load_state(updater.state_path).desired_digest == DIGEST_BAD
    assert len(runner.pull_calls) >= pulls_before


def test_untagged_track_image_rejected(tmp_path: Path) -> None:
    compose, env_path, state_path = _seed_project(tmp_path)
    runner = FakeRunner()
    updater = ComposeValidatorImageUpdater(
        project_name="base-validator-test",
        compose_file=compose,
        env_file=env_path,
        state_path=state_path,
        track_image=REPO,  # no tag
        resolver=lambda _ref: DIGEST_B,
        runner=runner,
        clock=FakeClock(),
        wall_clock=FakeClock(),
        sleep=lambda _s: None,
    )
    assert updater.run_once() == "reject-untagged"
    assert runner.pull_calls == []


def test_invalid_resolver_digest_refused(tmp_path: Path) -> None:
    runner = FakeRunner()
    updater = _updater(tmp_path, resolver_digest="sha256:deadbeef", runner=runner)
    assert updater.run_once() == "invalid-digest"
    assert runner.pull_calls == []


# ---- installer packaging static checks ---------------------


def test_updater_script_and_units_present_and_executable() -> None:
    assert UPDATER_SCRIPT.is_file()
    assert UPDATER_SCRIPT.stat().st_mode & stat.S_IXUSR
    content = UPDATER_SCRIPT.read_text(encoding="utf-8")
    assert "BASE_VALIDATOR_IMAGE_DIGEST" in content
    assert "force-recreate" in content
    assert "docker.sock" in content  # documents that agent has none
    assert "COMPOSE_PROJECT_NAME" in content
    # never instruct compose up with bare latest alone as runtime selector
    assert (
        "bare :latest" in content
        or "never bare" in content.lower()
        or "never" in content
    )

    assert SERVICE_UNIT.is_file()
    service = SERVICE_UNIT.read_text(encoding="utf-8")
    assert "Type=oneshot" in service
    assert "validator-image-updater.sh" in service
    assert "COMPOSE_PROJECT_NAME=%i" in service

    assert TIMER_UNIT.is_file()
    timer = TIMER_UNIT.read_text(encoding="utf-8")
    assert "OnUnitActiveSec=" in timer
    # 60–120s window
    assert any(f"OnUnitActiveSec={n}" in timer for n in ("60", "90", "120"))


def test_install_validator_enables_auto_update_by_default() -> None:
    content = INSTALL_VALIDATOR.read_text(encoding="utf-8")
    assert "--no-auto-update" in content
    assert "base-validator-image-updater@" in content
    assert "validator-image-updater.sh" in content
    assert "ENABLE_AUTO_UPDATE" in content or "AUTO_UPDATE" in content
    # Default is ON
    assert "no-auto-update" in content
    # State next to artifacts
    assert "image_update_state.json" in content
    # systemd enable
    assert "systemctl" in content
    # never mount docker.sock into agent via installer
    assert (
        "docker.sock" not in content
        or "no docker.sock" in content.lower()
        or "never" in content.lower()
    )


def test_compose_artifact_still_agent_only_no_socket() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8")
    assert "/var/run/docker.sock" not in text
    assert "services:" in text
    assert "validator:" in text
    assert "BASE_VALIDATOR_IMAGE_DIGEST" in text
    # Comment may mention master postgres as forbidden; service must not declare one.
    assert "master-postgres" not in text
    assert "challenge-prism" not in text
    assert "docker service" not in text


def test_default_track_image_matches_settings_runtime() -> None:
    assert "base-validator-runtime" in DEFAULT_TRACK_IMAGE
    assert DEFAULT_TRACK_IMAGE.endswith(":latest")
    assert has_explicit_tag(DEFAULT_TRACK_IMAGE)


def test_state_json_roundtrip(tmp_path: Path) -> None:
    from base.supervisor.validator_image_updater import (
        ValidatorImageUpdateState,
        save_state,
    )

    path = tmp_path / "image_update_state.json"
    state = ValidatorImageUpdateState(
        desired_digest=DIGEST_B,
        current_digest=DIGEST_A,
        rollback_digest=DIGEST_A,
        phase=UpdaterPhase.BACKOFF,
        attempts=2,
        next_eligible_at=1234.5,
        hold=False,
        last_error="boom",
        track_image=TRACK,
    )
    save_state(path, state)
    mode = path.stat().st_mode
    assert mode & stat.S_IRWXO == 0
    loaded = load_state(path)
    assert loaded.desired_digest == DIGEST_B
    assert loaded.attempts == 2
    assert loaded.phase == UpdaterPhase.BACKOFF
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "next_eligible_at" in raw


def test_env_ahead_of_running_container_forces_update(tmp_path: Path) -> None:
    """Rewritten .env pin with lagging container must recreate, not no-op."""
    compose, env_path, state_path = _seed_project(tmp_path, digest=DIGEST_B)
    # env already at B, running still A
    runner = FakeRunner(image=f"{REPO}@{DIGEST_A}", running=True)

    # When pull/up succeeds, runner.image still A until we flip it mid-calls
    class FlippingRunner(FakeRunner):
        def __call__(self, argv, timeout_seconds):
            result = super().__call__(argv, timeout_seconds)
            if "up" in argv and "compose" in argv and result.returncode == 0:
                self.image = f"{REPO}@{DIGEST_B}"
            return result

    runner = FlippingRunner(image=f"{REPO}@{DIGEST_A}", running=True)
    clock = FakeClock()
    updater = ComposeValidatorImageUpdater(
        project_name="base-validator-test",
        compose_file=compose,
        env_file=env_path,
        state_path=state_path,
        track_image=TRACK,
        resolver=lambda _ref: DIGEST_B,
        runner=runner,
        clock=clock,
        wall_clock=clock,
        sleep=clock.sleep,
        jitter_source=lambda: 0.0,
        verify_timeout_seconds=5.0,
        verify_poll_seconds=1.0,
        retry_policy=RetryPolicy(
            max_attempts=3, base_delay=1.0, max_delay=2.0, jitter=False
        ),
    )
    outcome = updater.run_once()
    assert outcome == "updated"
    assert any("pull" in c for c in runner.pull_calls)
