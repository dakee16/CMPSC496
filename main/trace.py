"""Private structured traces for model invocations.

PRIVATE BY CONSTRUCTION: nothing here is ever returned from an API route. The
grading engine must not depend on it - every entry point swallows its own
errors, because a telemetry failure must never change a student's verdict.

Cost is reported ONLY when pricing is explicitly configured. An unconfigured
model yields cost=None rather than a fabricated number.
"""
import json
import logging
import os
import threading
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager

_LOCK = threading.Lock()
# Bounded: a server that stays up for a term makes millions of model calls,
# and an unbounded list here grew until the process was killed. The tail is
# what a human ever looks at; MICROTUTOR_TRACE_FILE keeps the history on
# disk, itself bounded by the rotation below.
# ponytail: fixed ring, swap for a real sink if traces are ever queried.
MAX_EVENTS = 10_000
_SINK: deque = deque(maxlen=MAX_EVENTS)   # in-memory unless MICROTUTOR_TRACE_FILE is set

# THE DURABLE HALF, when MICROTUTOR_TRACE_FILE is set. The ring above is
# bounded because a term of model calls does not fit in memory, and it dies
# with the process - so which tier actually decided a verdict was never a
# question anything could answer after a restart. "Measure how often the judges
# acquit, then delete them" needs that history to exist.
#
# The file is bounded for the same reason the ring is, and a sharper one: it
# shares a volume with the session store and the oracle cache, so a log that
# fills the disk takes grading down with it. RotatingFileHandler is stdlib and
# already correct about the rename, so nothing here hand-rolls rotation. It
# caps the total at (TRACE_BACKUPS + 1) files. Safe with one appender, which
# is what there is: start.sh pins uvicorn to one worker, for this ring among
# other reasons.
TRACE_MAX_BYTES = 64 * 1024 * 1024
TRACE_BACKUPS = 3
_LOG = logging.getLogger("microtutor.trace")
_LOG.propagate = False          # never reaches uvicorn's handlers or a student
_LOG.setLevel(logging.INFO)
_ATTACHED = False

# USD per 1M tokens. Empty by default: absent pricing => cost is None, never 0.
PRICING: dict[str, tuple[float, float]] = {}


def configure_pricing(model: str, input_per_m: float, output_per_m: float) -> None:
    PRICING[model] = (input_per_m, output_per_m)


def _estimate(model, tin, tout):
    if model not in PRICING or tin is None or tout is None:
        return None
    i, o = PRICING[model]
    return round(tin / 1e6 * i + tout / 1e6 * o, 6)


def _file_sink() -> bool:
    """True when the rotating file writer is live. Attached once, lazily.

    Lazily because the path is read from the environment, which a test or a
    warm-up script may set after import. The caller holds _LOCK and swallows
    every error this can raise - a missing directory or an unwritable volume
    must cost the trace, never the grade."""
    global _ATTACHED
    path = os.environ.get("MICROTUTOR_TRACE_FILE")
    if not path:
        return False
    if not _ATTACHED:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=TRACE_MAX_BYTES,
                                      backupCount=TRACE_BACKUPS)
        handler.setFormatter(logging.Formatter("%(message)s"))
        _LOG.addHandler(handler)
        _ATTACHED = True
    return True


def record(**event) -> None:
    """Append one private event. Never raises into the caller."""
    try:
        event.setdefault("ts", time.time())
        with _LOCK:
            # THE RING FIRST, so a file sink that cannot be opened costs only
            # the durable copy. Reversing these two lines would drop the event
            # entirely on a full or unwritable volume.
            _SINK.append(event)
            if _file_sink():
                _LOG.info(json.dumps(event, default=str))
    except Exception:
        pass                    # telemetry must never affect grading


@contextmanager
def model_call(case_id, model, kind, **fields):
    """Time one model invocation and record it whatever happens."""
    t0 = time.perf_counter()
    box = {}
    try:
        yield box
    finally:
        try:
            tin, tout = box.get("input_tokens"), box.get("output_tokens")
            # Distinct kind for the TIMING event so it is never confused with
            # the outcome event that follows it.
            record(case_id=case_id, model=model, kind=f"{kind}_call",
                   latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                   input_tokens=tin, output_tokens=tout,
                   estimated_cost_usd=_estimate(model, tin, tout),
                   **{**fields, **{k: v for k, v in box.items()
                                   if k not in ("input_tokens", "output_tokens")}})
        except Exception:
            pass


def record_adapter(case_id, model, attempt, outcome, **extra):
    """outcome: malformed | unsafe | calibration_failed | bypass_rejected | accepted"""
    record(case_id=case_id, model=model, kind="adapter", attempt=attempt,
           outcome=outcome, **extra)


def record_judge(case_id, model, role, verdict, confidence, **extra):
    """role: primary | verifier"""
    record(case_id=case_id, model=model, kind="judge", role=role,
           verdict=verdict, confidence=confidence, **extra)


def record_route(case_id, final_route, verdict, **extra):
    record(case_id=case_id, kind="route", final_route=final_route,
           verdict=verdict, **extra)


def events():
    with _LOCK:
        return list(_SINK)


def reset():
    with _LOCK:
        _SINK.clear()


if __name__ == "__main__":
    # Self-check for the durable sink.  python -m main.trace
    # No model, no oracle - this writes real files under a temp directory.
    import pathlib
    import tempfile

    _dir = pathlib.Path(tempfile.mkdtemp())
    _path = _dir / "traces" / "model_trace.jsonl"      # directory absent yet
    os.environ["MICROTUTOR_TRACE_FILE"] = str(_path)
    TRACE_MAX_BYTES, TRACE_BACKUPS = 400, 2            # rotate inside the check

    # WHAT THE GAP WAS: a route decision reached the ring and nothing else, so
    # after a restart nobody could say which tier had decided anything.
    record_route("grade-000000000000", "llm-judge", "correct")
    record_judge("grade-000000000000", "model-x", "primary", True, 0.9)
    _lines = [json.loads(ln) for ln in _path.read_text().splitlines()]
    assert [e["kind"] for e in _lines] == ["route", "judge"], _lines
    assert _lines[0]["final_route"] == "llm-judge", _lines[0]
    assert _lines[0]["verdict"] == "correct" and _lines[0]["ts"] > 0, _lines[0]
    # ...and the in-memory ring is unchanged by any of this.
    assert [e["kind"] for e in events()] == ["route", "judge"], events()

    # BOUNDED. /data also holds the session store and the oracle cache, so an
    # unbounded log takes grading down with the disk.
    for _i in range(200):
        record_route(f"grade-{_i:012d}", "execution-bridged", "correct")
    _files = sorted(p.name for p in _path.parent.iterdir())
    assert _files == ["model_trace.jsonl", "model_trace.jsonl.1",
                      "model_trace.jsonl.2"], _files
    assert all(p.stat().st_size < 2 * TRACE_MAX_BYTES
               for p in _path.parent.iterdir()), _files

    # A SINK THAT CANNOT BE OPENED IS NOT A GRADING FAILURE. The event still
    # reaches the ring, and record() returns normally - grading.py wraps these
    # calls too, but the guarantee belongs here.
    _LOG.handlers.clear()
    _ATTACHED = False
    reset()
    os.environ["MICROTUTOR_TRACE_FILE"] = "/dev/null/not-a-directory/t.jsonl"
    record_route("grade-ffffffffffff", "execution-reference", "correct")
    assert len(events()) == 1 and events()[0]["final_route"] == \
        "execution-reference", events()

    # Unset means memory only, exactly as before.
    _LOG.handlers.clear()
    _ATTACHED = False
    del os.environ["MICROTUTOR_TRACE_FILE"]
    assert _file_sink() is False
    record_route("grade-eeeeeeeeeeee", "execution-final", "correct")
    assert len(events()) == 2, events()

    print("trace.py self-check OK")
