"""Private structured traces for model invocations.

PRIVATE BY CONSTRUCTION: nothing here is ever returned from an API route. The
grading engine must not depend on it - every entry point swallows its own
errors, because a telemetry failure must never change a student's verdict.

Cost is reported ONLY when pricing is explicitly configured. An unconfigured
model yields cost=None rather than a fabricated number.
"""
import json
import os
import threading
import time
from contextlib import contextmanager

_LOCK = threading.Lock()
_SINK = []                      # in-memory unless MICROTUTOR_TRACE_FILE is set

# USD per 1M tokens. Empty by default: absent pricing => cost is None, never 0.
PRICING: dict[str, tuple[float, float]] = {}


def configure_pricing(model: str, input_per_m: float, output_per_m: float) -> None:
    PRICING[model] = (input_per_m, output_per_m)


def _estimate(model, tin, tout):
    if model not in PRICING or tin is None or tout is None:
        return None
    i, o = PRICING[model]
    return round(tin / 1e6 * i + tout / 1e6 * o, 6)


def record(**event) -> None:
    """Append one private event. Never raises into the caller."""
    try:
        event.setdefault("ts", time.time())
        with _LOCK:
            _SINK.append(event)
            path = os.environ.get("MICROTUTOR_TRACE_FILE")
            if path:
                with open(path, "a") as f:
                    f.write(json.dumps(event, default=str) + "\n")
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
