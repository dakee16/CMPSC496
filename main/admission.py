"""admission.py - refuse work the deployment cannot finish, quickly.

Without a bound, a surge does not get slower evenly: every request starts a
subprocess or a model call, memory and rate limits run out together, and
requests that were going to succeed die with the rest. A slot makes the excess
wait briefly and then fail fast.

Busy is a RuntimeError on purpose, like account_cap.CapReached: model callers
already treat a RuntimeError as an outage (grading says it could not check and
no try is used), so overload needs no new path through them.

Every limit is opt-in. An unset or zero limit admits everything.
"""
import os
import threading
import time
from contextlib import contextmanager

from . import shared

_lock = threading.Lock()
_slots: dict[tuple[str, int], threading.BoundedSemaphore] = {}


class Busy(RuntimeError):
    """Over capacity right now; trying again shortly is the right response."""


def _limit(env: str) -> int:
    try:
        return max(0, int(os.environ.get(env, "") or 0))
    except ValueError:
        return 0


@contextmanager
def slot(name: str, env: str, wait: float):
    """Hold one of the `env` concurrent slots, waiting at most `wait` seconds.
    The limit is per process, so size it as capacity / workers."""
    limit = _limit(env)
    if not limit:
        yield
        return
    with _lock:
        sem = _slots.setdefault((env, limit), threading.BoundedSemaphore(limit))
    if not sem.acquire(timeout=wait):
        raise Busy(f"{name} is at capacity ({limit} in flight)")
    try:
        yield
    finally:
        sem.release()


def require_rate(name: str, env: str) -> None:
    """Enforce a fleet-wide per-minute limit, raising Busy once it is spent.

    Fixed one-minute window. Failing here beats discovering the limit as a
    provider 429, whose retries sleep while holding a worker thread. Needs
    Redis; without it, or if Redis errors, the call proceeds and the
    per-process slot above is the remaining bound."""
    limit = _limit(env)
    r = shared.client()
    if not limit or r is None:
        return
    window = shared.key("rate", name, int(time.time() // 60))
    try:
        pipe = r.pipeline()
        pipe.incr(window)
        pipe.expire(window, 120)
        used = pipe.execute()[0]
    except Exception as e:
        print(f"  ⚠️  rate limiter unavailable: {e!r}"[:160])
        return
    if used > limit:
        raise Busy(f"{name} is at its limit of {limit} per minute")
