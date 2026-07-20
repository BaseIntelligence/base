"""In-container terminal (tmux) session manager for the own-runner backend (Task 12).

The own-runner drives a Terminal-Bench task by giving the agent an interactive
shell to act in: a single ``tmux`` session+pane living inside the task
container. This module reproduces harbor 0.13.1's terminal-session model --
``harbor.agents.terminus_2.tmux_session.TmuxSession`` -- so that an agent driven
through our runner behaves identically to one driven through stock harbor.

Faithful semantics reproduced from harbor's ``TmuxSession``:

* **Pane lifecycle (create):** ``export TERM=xterm-256color && export
  SHELL=/bin/bash && tmux new-session [-e KEY=value ...] -x <w> -y <h> -d -s
  <name> 'bash --login'`` followed by ``tmux set-option -g history-limit
  10000000`` to retain scrollback. Pane geometry defaults to harbor's 160x40.
* **Command injection (send-keys):** ``tmux send-keys -t <name> -- <keys...>``
  with each key ``shlex.quote``-d. tmux silently drops a single ``send-keys``
  command exceeding ~16 KB (its internal buffer), so oversized payloads are
  split across multiple commands -- byte-for-byte the splitting harbor performs.
  Blocking sends append harbor's completion sentinel ``; tmux wait -S done`` +
  ``Enter`` and then block on ``timeout <sec>s tmux wait done`` (a non-zero
  return -> ``TimeoutError`` with harbor's message).
* **Capture:** ``tmux capture-pane -p [-S -] -t <name>`` (``-S -`` captures the
  entire scrollback); empty output decodes to ``""``.
* **Liveness:** ``tmux has-session -t <name>`` (return code 0 == alive).
* **Teardown:** ``tmux kill-session -t <name>`` -- idempotent, leaving NO
  residual sessions or panes.

Everything that gets a command INTO the container goes through the Task 10
exec-bridge contract (:class:`DockerExecEnvironment` / any object satisfying
:class:`TerminalEnvironment`). This module does NOT reimplement docker transport
or change the exec-bridge signature; it builds tmux command strings and runs
them via ``environment.exec(...)``.

NOTE: cross-module package wiring (``__init__`` / ``pyproject``) is deferred to
the package-wiring task; this module only owns the session model.
"""

from __future__ import annotations

import asyncio
import re
import shlex
import time
from types import TracebackType
from typing import Protocol

from agent_challenge.evaluation.own_runner.exec_bridge import ExecResult

#: harbor's default pane geometry (``pane_width`` x ``pane_height``).
DEFAULT_PANE_WIDTH = 160
DEFAULT_PANE_HEIGHT = 40

#: harbor bumps scrollback to this many lines after creating the session so the
#: full-buffer capture (``capture-pane -S -``) does not lose early output.
HISTORY_LIMIT = 10_000_000


class TerminalEnvironment(Protocol):
    """Structural type for the Task 10 exec-bridge this session runs on.

    Matches :meth:`DockerExecEnvironment.exec` exactly (kwarg names are
    load-bearing for harbor parity), so a real ``DockerExecEnvironment`` -- or
    any faithful stand-in -- can back a session.
    """

    async def exec(
        self,
        command: str,
        cwd: str | None = ...,
        env: dict[str, str] | None = ...,
        timeout_sec: int | None = ...,
        user: str | int | None = ...,
    ) -> ExecResult: ...


class TmuxSession:
    """A single in-container tmux session+pane the agent acts in.

    Reproduces harbor ``TmuxSession`` pane-lifecycle / capture / teardown
    semantics on top of the own-runner exec-bridge. Construct with a session
    name and an :class:`TerminalEnvironment`, call :meth:`start` to create the
    pane, drive it with :meth:`send_keys` / :meth:`capture_pane`, and tear it
    down with :meth:`stop` (or use the async-context-manager form).
    """

    #: Keys tmux treats as pressing Enter (harbor ``_ENTER_KEYS``).
    _ENTER_KEYS = {"Enter", "C-m", "KPEnter", "C-j", "^M", "^J"}
    _ENDS_WITH_NEWLINE_PATTERN = r"[\r\n]$"
    _NEWLINE_CHARS = "\r\n"
    #: Sentinel appended to blocking commands; ``tmux wait done`` blocks on it.
    _TMUX_COMPLETION_COMMAND = "; tmux wait -S done"
    #: tmux silently drops a single send-keys command above ~16 KB
    #: (https://github.com/tmux/tmux/issues/254); stay under that ceiling.
    _TMUX_SEND_KEYS_MAX_COMMAND_LENGTH = 16_000
    #: Bounded ceiling (seconds) for the ``tmux -V`` availability probe. The
    #: exec-bridge converts a breach into ``RuntimeError("Command timed out...")``
    #: so a missing/unresponsive tmux fails fast instead of hanging.
    _TMUX_PROBE_TIMEOUT_SEC = 10

    def __init__(
        self,
        session_name: str,
        environment: TerminalEnvironment,
        *,
        pane_width: int = DEFAULT_PANE_WIDTH,
        pane_height: int = DEFAULT_PANE_HEIGHT,
        extra_env: dict[str, str] | None = None,
        user: str | int | None = None,
    ) -> None:
        try:
            self._pane_width = int(pane_width)
            self._pane_height = int(pane_height)
        except (ValueError, TypeError):
            raise ValueError("pane_width and pane_height must be valid integers.") from None
        if self._pane_width <= 0 or self._pane_height <= 0:
            raise ValueError("pane_width and pane_height must be positive integers.")
        self._session_name = session_name
        self.environment = environment
        self._extra_env: dict[str, str] = extra_env or {}
        self._user = user

    # -- read-only accessors ----------------------------------------------

    @property
    def session_name(self) -> str:
        return self._session_name

    @property
    def pane_width(self) -> int:
        return self._pane_width

    @property
    def pane_height(self) -> int:
        return self._pane_height

    # -- command builders (pure; byte-faithful to harbor) ------------------

    @property
    def _tmux_start_session(self) -> str:
        # harbor: env vars passed via repeated `-e KEY=value`, shell-quoted.
        env_options = "".join(
            f"-e {shlex.quote(f'{key}={value}')} " for key, value in self._extra_env.items()
        )
        return (
            "export TERM=xterm-256color && "
            "export SHELL=/bin/bash && "
            f"tmux new-session {env_options}"
            f"-x {self._pane_width} -y {self._pane_height} "
            f"-d -s {self._session_name} 'bash --login'"
        )

    @staticmethod
    def _utf8_len(s: str) -> int:
        """UTF-8 byte length; tmux measures command size in bytes, not codepoints."""
        return len(s.encode("utf-8"))

    def _tmux_send_keys(self, keys: list[str]) -> list[str]:
        """Build one or more ``tmux send-keys`` commands for *keys*.

        If the shell-escaped command would exceed the tmux command-length limit,
        the keys are spread across multiple commands so each stays within the
        limit; oversized single keys are split into fitting sub-strings.
        """
        prefix = "tmux send-keys -t " + shlex.quote(self._session_name)
        # `--` explicitly ends options so everything after is treated as keys.
        prefix += " --"
        max_len = self._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH
        _blen = self._utf8_len

        escaped_keys = [shlex.quote(key) for key in keys]
        single = prefix + " " + " ".join(escaped_keys)
        if _blen(single) <= max_len:
            return [single]

        commands: list[str] = []
        current_escaped: list[str] = []
        current_len = _blen(prefix)

        def _flush() -> None:
            nonlocal current_len
            if current_escaped:
                commands.append(prefix + " " + " ".join(current_escaped))
                current_escaped.clear()
                current_len = _blen(prefix)

        for key in keys:
            escaped = shlex.quote(key)
            addition = 1 + _blen(escaped)  # space + quoted key

            if current_len + addition <= max_len:
                current_escaped.append(escaped)
                current_len += addition
            elif _blen(prefix) + addition <= max_len:
                _flush()
                current_escaped.append(escaped)
                current_len = _blen(prefix) + addition
            else:
                _flush()
                max_escaped = max_len - _blen(prefix) - 1
                for chunk_escaped in self._split_key_for_tmux(key, max_escaped):
                    if current_len + 1 + _blen(chunk_escaped) <= max_len:
                        current_escaped.append(chunk_escaped)
                        current_len += 1 + _blen(chunk_escaped)
                    else:
                        _flush()
                        current_escaped.append(chunk_escaped)
                        current_len = _blen(prefix) + 1 + _blen(chunk_escaped)

        _flush()
        return commands

    @staticmethod
    def _split_key_for_tmux(key: str, max_escaped_len: int) -> list[str]:
        """Split *key* into ``shlex.quote``-d chunks each <= *max_escaped_len* bytes."""
        _blen = TmuxSession._utf8_len
        chunks: list[str] = []
        remaining = key
        while remaining:
            lo, hi, best = 1, len(remaining), 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if _blen(shlex.quote(remaining[:mid])) <= max_escaped_len:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            chunks.append(shlex.quote(remaining[:best]))
            remaining = remaining[best:]
        return chunks

    def _tmux_capture_pane(self, capture_entire: bool = False) -> str:
        extra_args = ["-S", "-"] if capture_entire else []
        return " ".join(["tmux", "capture-pane", "-p", *extra_args, "-t", self._session_name])

    # -- key-preparation helpers (harbor parity) ---------------------------

    def _is_enter_key(self, key: str) -> bool:
        return key in self._ENTER_KEYS

    def _ends_with_newline(self, key: str) -> bool:
        return re.search(self._ENDS_WITH_NEWLINE_PATTERN, key) is not None

    def _is_executing_command(self, key: str) -> bool:
        return self._is_enter_key(key) or self._ends_with_newline(key)

    def _prevent_execution(self, keys: list[str]) -> list[str]:
        keys = keys.copy()
        while keys and self._is_executing_command(keys[-1]):
            if self._is_enter_key(keys[-1]):
                keys.pop()
            else:
                stripped_key = keys[-1].rstrip(self._NEWLINE_CHARS)
                if stripped_key:
                    keys[-1] = stripped_key
                else:
                    keys.pop()
        return keys

    def _prepare_keys(self, keys: str | list[str], block: bool) -> tuple[list[str], bool]:
        """Return ``(keys_to_send, is_blocking)`` -- byte-faithful to harbor.

        Blocking is engaged ONLY when ``block`` is set AND the final key
        executes the command (Enter / trailing newline) -- exactly harbor's
        guard. In that case harbor strips the trailing executor, appends the
        completion sentinel + a single Enter, and the caller blocks on
        ``tmux wait done``. A non-terminated key list is a keystroke, never a
        command, so it is sent non-blocking even under ``block=True``.
        """
        if isinstance(keys, str):
            keys = [keys]
        if not block or not keys or not self._is_executing_command(keys[-1]):
            return keys, False
        keys = self._prevent_execution(keys)
        keys.extend([self._TMUX_COMPLETION_COMMAND, "Enter"])
        return keys, True

    # -- lifecycle ---------------------------------------------------------

    async def _ensure_tmux(self) -> None:
        """Verify ``tmux`` is present in the container, failing fast if it is not.

        tmux is baked into the task image at BUILD time (the container builder's
        derived ``*-tmux`` image), so this is a single bounded ``tmux -V`` probe.
        The eval runtime is network-isolated (``--network none``), so there is
        deliberately NO runtime package install here — an offline ``apt-get``/
        ``apk``/… could never succeed and only hung (the universal own-runner
        hang this replaces). If tmux is missing, raise immediately; the driver
        maps that to a fast, typed ``harbor_trial_failed`` instead of an
        unbounded stall.
        """
        probe = await self.environment.exec(
            command="tmux -V", user="root", timeout_sec=self._TMUX_PROBE_TIMEOUT_SEC
        )
        if probe.return_code == 0:
            return
        raise RuntimeError(
            "tmux is not available in the task container and runtime install is "
            "disabled (the eval runtime is network-isolated). tmux must be baked "
            "into the task image at build time "
            f"(probe return_code={probe.return_code}, output={probe.stdout!r})."
        )

    async def start(self) -> None:
        """Create the tmux session+pane (harbor ``new-session`` lifecycle)."""
        await self._ensure_tmux()
        start_result = await self.environment.exec(
            command=self._tmux_start_session, user=self._user
        )
        if start_result.return_code != 0:
            raise RuntimeError(
                f"Failed to start tmux session {self._session_name!r}. "
                f"Output: {start_result.stdout!r}"
            )
        # Retain scrollback so full-buffer capture does not lose early output.
        await self.environment.exec(
            command=f"tmux set-option -g history-limit {HISTORY_LIMIT}",
            user=self._user,
        )

    async def stop(self) -> None:
        """Tear down the session (``kill-session``); idempotent, leaves no residue."""
        await self.environment.exec(
            command=f"tmux kill-session -t {self._session_name}",
            user=self._user,
        )

    async def __aenter__(self) -> TmuxSession:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()

    # -- liveness ----------------------------------------------------------

    async def is_session_alive(self) -> bool:
        """True iff ``tmux has-session`` reports the session is alive."""
        result = await self.environment.exec(
            command=f"tmux has-session -t {self._session_name}",
            user=self._user,
        )
        return result.return_code == 0

    # -- command injection -------------------------------------------------

    async def _send_blocking_keys(self, keys: list[str], max_timeout_sec: float) -> None:
        start_time_sec = time.time()
        for command in self._tmux_send_keys(keys):
            result = await self.environment.exec(command=command, user=self._user)
            if result.return_code != 0:
                raise RuntimeError(
                    f"{self._session_name}: failed to send blocking keys: "
                    f"command={command!r:.100}, return_code={result.return_code}, "
                    f"stdout={result.stdout!r}"
                )
        result = await self.environment.exec(
            command=f"timeout {max_timeout_sec}s tmux wait done", user=self._user
        )
        if result.return_code != 0:
            raise TimeoutError(f"Command timed out after {max_timeout_sec} seconds")
        _ = time.time() - start_time_sec

    async def _send_non_blocking_keys(self, keys: list[str], min_timeout_sec: float) -> None:
        start_time_sec = time.time()
        for command in self._tmux_send_keys(keys):
            result = await self.environment.exec(command=command, user=self._user)
            if result.return_code != 0:
                raise RuntimeError(
                    f"{self._session_name}: failed to send non-blocking keys: "
                    f"command={command!r:.100}, return_code={result.return_code}, "
                    f"stdout={result.stdout!r}"
                )
        elapsed_time_sec = time.time() - start_time_sec
        if elapsed_time_sec < min_timeout_sec:
            await asyncio.sleep(min_timeout_sec - elapsed_time_sec)

    async def send_keys(
        self,
        keys: str | list[str],
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ) -> None:
        """Send *keys* into the pane (harbor ``send_keys`` semantics).

        Args:
            keys: A single string (sent as one key) or a list of tmux keys.
            block: Wait for the command to finish (uses the completion sentinel
                + ``tmux wait done``). The final key must execute the command
                (Enter / trailing newline) -- if it does not, an Enter is added.
            min_timeout_sec: For non-blocking sends, minimum settle time.
            max_timeout_sec: For blocking sends, the wait ceiling; exceeding it
                raises :class:`TimeoutError`.
        """
        prepared_keys, is_blocking = self._prepare_keys(keys=keys, block=block)
        if is_blocking:
            await self._send_blocking_keys(prepared_keys, max_timeout_sec)
        else:
            await self._send_non_blocking_keys(prepared_keys, min_timeout_sec)

    # -- capture -----------------------------------------------------------

    async def capture_pane(self, capture_entire: bool = False) -> str:
        """Return pane contents (``capture_entire`` -> full scrollback)."""
        result = await self.environment.exec(
            command=self._tmux_capture_pane(capture_entire=capture_entire),
            user=self._user,
        )
        return result.stdout or ""
