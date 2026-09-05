"""
live_playground.py - a REAL pipeline run, narrated live for the browser.

This is the opposite of the read-only showcase endpoints in api_server.py:
those replay cache and never call a model; this one runs the actual pipeline
(oracle generation -> mutation testing -> repair -> verdict -> decomposition ->
Gate 1) and yields a structured event for every step as it happens, so a UI
can render the run in real time.

Design rules, in the same spirit as the rest of the codebase:
  1. NOTHING here re-implements pipeline logic. Every decision is made by the
     same functions production uses (make_oracle_tests, validate_oracle,
     decompose_into_chunks, check_necessity). This module only orchestrates
     and narrates. If the pipeline and the playground ever disagree, the
     playground is wrong by definition - and this structure makes that
     impossible.
  2. The verdict is persisted to tests/tests_cache.json in EXACTLY the format
     get_oracle_tests writes, so a live run leaves the system in the same
     state a warmup run would.
  3. `emit` callbacks never change behaviour - they only report it.

Costs: a live run makes real LLM calls (oracle input generation, one
counterexample search per surviving mutant per round, and the decomposition
itself). Expect the same OpenAI billing as a warmup pass on one problem.
"""
import io
import json
import queue
import sys
import threading
from datetime import datetime, timezone

from tests.sandbox import _load_cache, _save_cache, make_oracle_tests

from . import mutation
from .gates import check_necessity
from .identity import content_hash, get_resolved_entry
from .mutation import validate_oracle


# ── the tunables ──────────────────────────────────────────────────────────
# Every knob the pipeline exposes, in ONE table the UI renders its sliders
# from. The server is the source of truth for names, bounds and defaults; the
# page never hardcodes a constant, so adding a row here is all it takes to
# grow a new slider.
#
# `module`/`attr` name where the value lives; None means the value is passed
# as a call argument by _pipeline instead of patched. All the mutation-module
# constants are read late (module-global lookup at call time - the def-time
# defaults were rebound for exactly this), so patching the attribute for the
# duration of a run is sufficient and is undone in the same `finally` that
# restores stdout. The run holds _RUN_LOCK throughout, so two runs can never
# interleave their patches.
PLAYGROUND_PARAMS = [
    {"key": "oracle_n_tests", "group": "Oracle generation",
     "label": "Oracle inputs requested",
     "help": "How many test inputs the model is asked to propose. The ground "
             "truth computes every expected output; ambiguous inputs are "
             "filtered afterwards, so the kept suite is usually smaller.",
     "min": 4, "max": 30, "step": 1, "default": 12,
     "module": None, "attr": None},

    {"key": "kill_rate_cutoff", "group": "Mutation testing",
     "label": "STRONG cutoff (kill rate)",
     "help": "Fraction of mutants the suite must kill, as handed in, to be "
             "trusted for grading. Lower accepts weaker oracles.",
     "min": 0.5, "max": 1.0, "step": 0.05, "default": mutation.CUTOFF_1_KILL_RATE,
     "module": "mutation", "attr": "CUTOFF_1_KILL_RATE"},

    {"key": "max_expand_rounds", "group": "Mutation testing",
     "label": "Max repair rounds",
     "help": "How many times a weak suite is topped up with fresh verified "
             "tests and re-scored before giving up.",
     "min": 1, "max": 6, "step": 1, "default": mutation.CUTOFF_2_MAX_EXPAND_ROUNDS,
     "module": None, "attr": None},   # passed as validate_oracle(max_rounds=)

    {"key": "counterexample_candidates", "group": "Mutation testing",
     "label": "LLM inputs per survivor",
     "help": "Input guesses the model may propose to kill one surviving "
             "mutant. The model only ever supplies inputs; execution decides.",
     "min": 1, "max": 12, "step": 1,
     "default": mutation.CUTOFF_4_MAX_COUNTEREXAMPLE_CANDIDATES,
     "module": "mutation", "attr": "CUTOFF_4_MAX_COUNTEREXAMPLE_CANDIDATES"},

    {"key": "max_probe_inputs", "group": "Mutation testing",
     "label": "Free boundary probes",
     "help": "Deterministic boundary variants tried on each survivor before "
             "any model is asked - these cost nothing.",
     "min": 0, "max": 40, "step": 2, "default": mutation._MAX_PROBE_INPUTS,
     "module": "mutation", "attr": "_MAX_PROBE_INPUTS"},

    {"key": "mutant_timeout", "group": "Mutation testing",
     "label": "Mutant timeout (seconds)",
     "help": "Wall-clock leash per mutant run. Mutants can loop forever where "
             "the original did not, so this is tighter than a real solution's.",
     "min": 1, "max": 15, "step": 1, "default": mutation._MUTANT_TIMEOUT,
     "module": "mutation", "attr": "_MUTANT_TIMEOUT"},

    {"key": "equivalence_sweep_size", "group": "Mutation testing",
     "label": "Equivalence sweep size",
     "help": "Inputs in the deterministic sweep that must agree EXACTLY before "
             "a survivor is excused as genuinely equivalent. Bigger = more "
             "confident exclusions, slower runs.",
     "min": 25, "max": 500, "step": 25, "default": mutation._EQUIVALENCE_SWEEP_SIZE,
     "module": "mutation", "attr": "_EQUIVALENCE_SWEEP_SIZE"},

    {"key": "min_mutants", "group": "Mutation testing",
     "label": "Minimum mutants to judge",
     "help": "Below this many mutants the solution is too trivial for a kill "
             "rate to mean anything - flagged insufficient, never STRONG.",
     "min": 1, "max": 10, "step": 1, "default": mutation._MIN_MUTANTS,
     "module": "mutation", "attr": "_MIN_MUTANTS"},

    {"key": "decompose_max_tries", "group": "Decomposition",
     "label": "Decomposition attempts",
     "help": "How many times the model may re-split the problem after a "
             "failed assembly or necessity gate before the run is blocked. "
             "Temperature climbs with each retry.",
     "min": 1, "max": 8, "step": 1, "default": 5,
     "module": None, "attr": None},   # passed as decompose_into_chunks(max_tries=)
]

_PARAM_BY_KEY = {p["key"]: p for p in PLAYGROUND_PARAMS}


def clean_overrides(raw: dict | None) -> dict:
    """Clamp caller-supplied overrides to the registry's bounds.

    Unknown keys are dropped rather than erroring: the page and the server can
    be one deploy apart, and a stale slider must not kill the whole run."""
    out = {}
    for key, val in (raw or {}).items():
        spec = _PARAM_BY_KEY.get(key)
        if spec is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        v = max(spec["min"], min(spec["max"], v))
        # An integer knob stays an integer - range(1, 3.0) is a TypeError.
        if float(spec["step"]).is_integer() and float(spec["min"]).is_integer():
            v = int(round(v))
        out[key] = v
    return out


def current_params() -> list[dict]:
    """The registry with LIVE defaults, for the UI to build sliders from."""
    live = []
    for p in PLAYGROUND_PARAMS:
        cur = p["default"]
        if p["module"] == "mutation":
            cur = getattr(mutation, p["attr"], cur)
        live.append({k: p[k] for k in ("key", "group", "label", "help",
                                       "min", "max", "step")} | {"default": cur})
    return live


class _applied_overrides:
    """Patch the mutation-module knobs for the duration of one run, and
    guarantee the originals come back whatever the pipeline does. Only ever
    entered while holding _RUN_LOCK."""

    def __init__(self, overrides: dict):
        self._overrides = overrides
        self._saved = {}

    def __enter__(self):
        for key, val in self._overrides.items():
            spec = _PARAM_BY_KEY[key]
            if spec["module"] == "mutation":
                self._saved[spec["attr"]] = getattr(mutation, spec["attr"])
                setattr(mutation, spec["attr"], val)
        return self

    def __exit__(self, *exc):
        for attr, val in self._saved.items():
            setattr(mutation, attr, val)
        return False


# One live run at a time. Two reasons: (1) sys.stdout is process-global, so
# two concurrent runs would fight over it; (2) each run makes real, billed
# LLM calls - accidental parallel runs are pure waste. FastAPI's default
# threadpool would otherwise happily run several at once.
_RUN_LOCK = threading.Lock()


class _ThreadLineStream(io.TextIOBase):
    """A stdout stand-in that streams the WORKER thread's printed lines as
    log events, and passes every other thread's writes through untouched.

    contextlib.redirect_stdout is process-wide, which is wrong here: while a
    run streamed, prints from other request handlers (or the caller itself)
    would get swallowed into this run's feed - verified live, including a
    feedback loop when the consumer of the stream itself printed. Routing by
    thread id confines capture to the pipeline that owns it.

    The pipeline already narrates itself with print() (attempt counts, gate
    verdicts, oracle stats); capturing those lines shows the decomposition
    retry loop in real time without touching run_phase1."""

    def __init__(self, worker_ident: int, emit, passthrough):
        self._ident = worker_ident
        self._emit = emit
        self._pass = passthrough
        self._buf = ""

    def write(self, s: str) -> int:
        if threading.get_ident() != self._ident:
            return self._pass.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._emit({"type": "log", "text": line.rstrip()})
        return len(s)

    def flush(self):
        if threading.get_ident() != self._ident:
            return self._pass.flush()
        if self._buf.strip():
            self._emit({"type": "log", "text": self._buf.rstrip()})
        self._buf = ""


def _persist_verdict(problem: dict, report: dict) -> None:
    """Write the validation result to the oracle cache in the exact shape
    tests/sandbox.get_oracle_tests writes, so downstream readers (playground
    replay, is_oracle_strong, get_oracle_tests) see a normal validated entry."""
    validated = {
        "final_tests": report["final_tests"],
        "strong": report["strong"],
        "kill_rate": report["kill_rate"],
        "kill_rate_direct": report["kill_rate_direct"],
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "breakdown": {
            "total_mutants": report.get("total_mutants", 0),
            "killed": report.get("killed", 0),
            "killed_on_retry": report.get("killed_on_retry", 0),
            "proven_equivalent": report.get("proven_equivalent", 0),
            "unresolved": report.get("unresolved", 0),
            "mutants": [{"label": m["label"], "status": m["status"]}
                        for m in report.get("mutants", [])],
        },
    }
    cache = _load_cache()
    cache[content_hash(problem)] = {**validated, "slug": problem.get("slug", "")}
    _save_cache(cache)


def _pipeline(problem: dict, emit, overrides: dict | None = None) -> None:
    """The run itself. Raises nothing to the caller - every failure becomes an
    event, because a demo that dies silently teaches nothing."""
    overrides = overrides or {}
    slug = problem.get("slug") or problem.get("title") or "<unnamed>"

    emit({"type": "stage", "name": "start",
          "label": f"Live run: {problem.get('title') or slug}"})
    # The knobs this run is actually using - defaults with the caller's clamped
    # overrides on top. Emitted first so the UI can pin them to the transcript:
    # a run is only comparable to another run if both wear their settings.
    effective = {p["key"]: overrides.get(
                     p["key"],
                     getattr(mutation, p["attr"]) if p["module"] == "mutation"
                     else p["default"])
                 for p in PLAYGROUND_PARAMS}
    emit({"type": "params", "values": effective,
          "overridden": sorted(overrides.keys())})
    emit({"type": "ground_truth", "code": problem.get("solution", "")})

    # ── entry resolution ──────────────────────────────────────────────────
    emit({"type": "stage", "name": "entry",
          "label": "Resolving the entry point (which function to call)"})
    resolved = get_resolved_entry(problem)
    emit({"type": "entry", "entry_name": resolved.get("entry_name"),
          "params": resolved.get("params", [])})

    # ── oracle generation (always fresh - the whole point is to WATCH it) ──
    emit({"type": "stage", "name": "oracle_gen",
          "label": "Generating oracle tests - model proposes INPUTS only, "
                   "the ground truth computes every expected output"})
    tests = make_oracle_tests(problem, n=int(effective["oracle_n_tests"]))
    if not tests:
        emit({"type": "blocked", "at": "oracle_generation",
              "error_type": "NoOracleTests",
              "message": f"No oracle tests could be generated for '{slug}' "
                         f"(inputs may not be JSON-serializable, or the ground "
                         f"truth failed to run). Nothing can be validated or "
                         f"decomposed without an oracle."})
        return
    emit({"type": "oracle_tests", "tests": tests, "origin": "fresh"})

    # ── mutation testing + repair, fully narrated ─────────────────────────
    emit({"type": "stage", "name": "mutation",
          "label": f"Mutation testing - deterministically breaking the ground "
                   f"truth one edit at a time and checking the oracle notices "
                   f"(STRONG needs kill_rate_direct ≥ "
                   f"{mutation.CUTOFF_1_KILL_RATE})"})
    report = validate_oracle(problem, tests, emit=emit,
                             max_rounds=int(effective["max_expand_rounds"]))

    _persist_verdict(problem, report)
    emit({"type": "verdict",
          "strong": report["strong"],
          "kill_rate": report["kill_rate"],
          "kill_rate_direct": report["kill_rate_direct"],
          "cutoff": mutation.CUTOFF_1_KILL_RATE,
          "insufficient_mutants": report.get("insufficient_mutants", False),
          "rounds": report.get("rounds", 1),
          "n_tests": len(report["final_tests"]),
          "detail": "verdict persisted to tests/tests_cache.json - the live "
                    "run leaves the same state a warmup pass would"})

    # ── decomposition + Gate 1 ────────────────────────────────────────────
    # Local import: run_phase1 is heavy and pulls the whole model stack; also
    # keeps this module importable in isolation for tests.
    from .run_phase1 import (DecompositionUnavailableError, NoOracleTestsError,
                             OracleNotStrongError, decompose_into_chunks)

    emit({"type": "stage", "name": "decomposition",
          "label": "Decomposing into 2-3 chunks - exactly the path a student "
                   "request takes, including every retry and gate"})
    try:
        result = decompose_into_chunks(
            problem, max_tries=int(effective["decompose_max_tries"]))
    except (OracleNotStrongError, NoOracleTestsError,
            DecompositionUnavailableError) as e:
        emit({"type": "blocked", "at": "decomposition",
              "error_type": type(e).__name__, "message": str(e)})
        return
    except RuntimeError as e:
        emit({"type": "blocked", "at": "decomposition",
              "error_type": "RuntimeError", "message": str(e)})
        return

    chunks = result["chunks"]
    emit({"type": "chunks", "header": result["header"],
          "chunks": [{"step_id": c.step_id, "prompt": c.prompt,
                      "reference": c.reference or ""} for c in chunks]})

    # decompose_into_chunks only returns decompositions that already PASSED
    # Gate 1; re-running it here is deterministic, model-free and sub-second,
    # and gives the UI the per-chunk knockout detail the return value omits.
    emit({"type": "stage", "name": "necessity",
          "label": "Gate 1 (necessity) - knocking each chunk out in turn; a "
                   "load-bearing chunk's removal must break the assembly"})
    nec = check_necessity(result["header"], chunks, problem)
    emit({"type": "necessity", "status": nec["status"],
          "per_chunk": nec["per_chunk"], "summary": nec["summary"]})

    emit({"type": "stage", "name": "finished", "label": "Run complete"})


def live_run(problem: dict, overrides: dict | None = None):
    """Generator yielding one JSON-serialisable event dict at a time.

    The pipeline runs in a worker thread pushing events into a queue; this
    generator drains it. That is what makes the stream LIVE: FastAPI's
    StreamingResponse pulls from here while the pipeline is still working,
    instead of waiting for one big result at the end. All pipeline print()
    output is also captured and streamed as {"type": "log"} events.

    `overrides` are playground knob values (see PLAYGROUND_PARAMS), clamped to
    their registered bounds and applied only for this run - the module-global
    patches are made and undone while _RUN_LOCK is held, so a concurrent
    student-facing request can never observe them mid-flip on the paths that
    matter (validation runs only from upload, warmup and here)."""
    q: queue.Queue = queue.Queue()
    _DONE = object()
    overrides = clean_overrides(overrides)

    def emit(event: dict) -> None:
        q.put(event)

    def work() -> None:
        acquired = _RUN_LOCK.acquire(timeout=1.0)
        if not acquired:
            emit({"type": "error",
                  "message": "Another live run is already in progress - one at "
                             "a time (each run bills real LLM calls)."})
            q.put(_DONE)
            return
        real_stdout = sys.stdout
        try:
            sys.stdout = _ThreadLineStream(threading.get_ident(), emit, real_stdout)
            with _applied_overrides(overrides):
                _pipeline(problem, emit, overrides)
        except Exception as e:                      # belt and braces
            emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            sys.stdout = real_stdout
            _RUN_LOCK.release()
            q.put(_DONE)

    threading.Thread(target=work, daemon=True).start()

    while True:
        ev = q.get()
        if ev is _DONE:
            break
        yield ev
    yield {"type": "done"}


def ndjson_stream(problem: dict, overrides: dict | None = None):
    """live_run, framed as newline-delimited JSON for a StreamingResponse."""
    for ev in live_run(problem, overrides):
        yield json.dumps(ev, default=str) + "\n"