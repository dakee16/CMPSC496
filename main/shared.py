"""shared.py - the one Redis client, for state that must span processes.

Unset REDIS_URL and client() is None: every caller then uses its original
in-process store, which is correct for one worker and is what a single-box
deployment still runs. start.sh refuses more than one worker without it.

Callers catch their own errors, because the right answer to Redis being down
differs: the login limiter falls back to counting locally, a spend cap refuses
the call, and a memo just misses.
"""
import os
import threading

_lock = threading.Lock()
_client = None
_injected = None

KEY_PREFIX = "mt:"


def client():
    """The shared client, or None when this deployment has no Redis."""
    global _client
    if _injected is not None:
        return _injected
    url = os.environ.get("REDIS_URL", "").strip()
    if not url:
        return None
    if _client is None:
        with _lock:
            if _client is None:
                import redis
                # Short timeouts on purpose: this sits in front of request
                # handling, and a hung Redis must cost milliseconds, not a
                # worker thread. prepare_bus polls rather than blocking long
                # for the same reason.
                _client = redis.Redis.from_url(
                    url, decode_responses=True, socket_timeout=2.5,
                    socket_connect_timeout=2, health_check_interval=30)
    return _client


def set_client(fake) -> None:
    """Test seam: inject a fake, or None to go back to REDIS_URL."""
    global _injected
    _injected = fake


def key(*parts) -> str:
    return KEY_PREFIX + ":".join(str(p) for p in parts)
