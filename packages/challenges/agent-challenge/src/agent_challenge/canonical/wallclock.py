"""Non-blocking wallclock for guest dstack RPCs that may hang forever.

``concurrent.futures.ThreadPoolExecutor`` used as a context manager is unsafe
for unbounded RPCs such as live dstack ``GetTlsKey`` / ``get_quote``:

* ``future.result(timeout=T)`` surfaces ``TimeoutError`` on time, **but**
* ``ThreadPoolExecutor.__exit__`` always calls ``shutdown(wait=True)`` and
  re-joins the already-running worker thread.

If the RPC never returns, the guest never emits ``ra_tls_bootstrap stage=fail``
and stays billed. ``future.cancel()`` is a no-op once the worker has started.

This helper runs the callable on a **daemon** thread and joins it **once** with
the deadline. On timeout it returns immediately without joining again; the
abandoned daemon cannot keep the process alive after the fail-closed path
exits.
"""

from __future__ import annotations

import threading
from collections.abc import Callable


class WallclockTimeout(TimeoutError):
    """Raised when a wallclocked call exceeds its hard deadline."""


def call_with_wallclock[T](
    fn: Callable[[], T],
    *,
    timeout_seconds: float,
    label: str = "call",
) -> T:
    """Invoke ``fn`` under a hard wallclock that never re-joins a hung worker.

    Parameters
    ----------
    fn:
        Zero-arg callable. Exceptions raised by ``fn`` are re-raised after the
        worker thread has finished (within the budget).
    timeout_seconds:
        Hard deadline in seconds. Must be positive.
    label:
        Short name included in timeout messages (``GetTlsKey``, ``get_quote``).

    Returns
    -------
    The value returned by ``fn``.

    Raises
    ------
    WallclockTimeout
        If the worker is still alive after ``timeout_seconds``. The hung thread
        is **not** joined again.
    Exception
        Whatever ``fn`` raised, re-raised on the calling thread after a clean
        join within the budget.
    """

    budget = float(timeout_seconds)
    if budget <= 0.0:
        raise WallclockTimeout(f"{label} wallclock must be positive, got {timeout_seconds!r}")

    box: list[T] = []
    errors: list[BaseException] = []

    def _runner() -> None:
        try:
            box.append(fn())
        except BaseException as exc:  # noqa: BLE001 - delivery to caller after join
            errors.append(exc)

    worker = threading.Thread(
        target=_runner,
        name=f"wallclock-{label}",
        daemon=True,
    )
    worker.start()
    # Single join with the deadline. Do NOT call join() again on the timeout
    # path — that reintroduces the ThreadPoolExecutor re-join hang.
    worker.join(timeout=budget)
    if worker.is_alive():
        raise WallclockTimeout(f"{label} exceeded {budget:.0f}s wallclock")
    if errors:
        raise errors[0]
    if not box:
        # Worker finished without result or error (should not happen).
        raise WallclockTimeout(f"{label} finished without result")
    return box[0]
