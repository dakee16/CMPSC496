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

from .gates import check_necessity
from .identity import content_hash, get_resolved_entry
from .mutation import CUTOFF_1_KILL_RATE, validate_oracle


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


def _pipeline(problem: dict, emit) -> None:
    """The run itself. Raises nothing to the caller - every failure becomes an
    event, because a demo that dies silently teaches nothing."""
    slug = problem.get("slug") or problem.get("title") or "<unnamed>"

    emit({"type": "stage", "name": "start",
          "label": f"Live run: {problem.get('title') or slug}"})
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
    tests = make_oracle_tests(problem)
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
                   f"(STRONG needs kill_rate_direct ≥ {CUTOFF_1_KILL_RATE})"})
    report = validate_oracle(problem, tests, emit=emit)

    _persist_verdict(problem, report)
    emit({"type": "verdict",
          "strong": report["strong"],
          "kill_rate": report["kill_rate"],
          "kill_rate_direct": report["kill_rate_direct"],
          "cutoff": CUTOFF_1_KILL_RATE,
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
        result = decompose_into_chunks(problem)
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


def live_run(problem: dict):
    """Generator yielding one JSON-serialisable event dict at a time.

    The pipeline runs in a worker thread pushing events into a queue; this
    generator drains it. That is what makes the stream LIVE: FastAPI's
    StreamingResponse pulls from here while the pipeline is still working,
    instead of waiting for one big result at the end. All pipeline print()
    output is also captured and streamed as {"type": "log"} events."""
    q: queue.Queue = queue.Queue()
    _DONE = object()

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
            _pipeline(problem, emit)
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


def ndjson_stream(problem: dict):
    """live_run, framed as newline-delimited JSON for a StreamingResponse."""
    for ev in live_run(problem):
        yield json.dumps(ev, default=str) + "\n"