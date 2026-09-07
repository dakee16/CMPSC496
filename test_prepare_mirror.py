"""
test_prepare_mirror.py - watching a preparation run that is already happening.

A teacher uploads a file and waits minutes. Task 2 is being able to click one
problem and watch the pipeline reason about it. The load-bearing decision is
that the watching tab MIRRORS the upload's run rather than starting its own -
a second run would pay for a second set of model calls and race the first to
write the same content-hash entry in the oracle cache.

What is worth testing here:

  * the narration is in main/live_playground.py's event vocabulary, because the
    page that renders it is the playground, unchanged;
  * a failure is narrated, not just returned - a watcher whose stream stops
    cannot tell "failed" from "hung";
  * narration NEVER changes what preparation returns, and a broken watcher
    cannot break an upload;
  * a late subscriber still sees the run from its beginning, and a finished one
    replays in full.

No network, no LLM: the three slow stages are stubbed, because what is under
test is the narration around them, not the pipeline they belong to. The HTTP
route is 15 lines of glue over prepare_bus.subscribe and is not separately
tested; the bus itself is, here and in its own self-check.
"""
import os
import tempfile
import threading
import time

import pytest

# Resolving an entry point WRITES to a regenerable cache in the repo. Redirect
# it before anything imports it: a test run that leaves the working tree dirty,
# with fixture problems that exist nowhere but this file, costs more than it
# pays. Same reason as test_class_problems.py.
import main.identity as _identity                                    # noqa: E402
_identity._RESOLVED_PATH = os.path.join(
    tempfile.mkdtemp(prefix="microtutor-test-"), "resolved_entries.json")

import main.publish as publish
import main.run_phase1 as run_phase1
import tests.sandbox as sandbox
from main.prepare_bus import finish, is_open, key, open_channel, publish as bus_publish
from main.prepare_bus import subscribe
from main.schemas import StepItem

PROBLEM = {"slug": "digit-sum", "title": "digit sum",
           "description": "Sum the digits.",
           "solution": "def digit_sum(n):\n    return sum(int(c) for c in str(n))"}


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Stand in for the three stages that cost money, keeping their contracts."""
    monkeypatch.setattr(sandbox, "get_oracle_tests",
                        lambda p, n=10, emit=None: [{"input": [12], "expected": 3}])
    monkeypatch.setattr(sandbox, "is_oracle_certified", lambda p: True)
    monkeypatch.setattr(run_phase1, "get_chunk_decomposition",
                        lambda p: {"header": "def digit_sum(n):",
                                   "chunks": [StepItem(question_id="q", step_id="Part 1",
                                                       prompt="sum them",
                                                       expected_type="code",
                                                       reference="return 3")]})


def _narrate(problem):
    events = []
    result = publish.prepare_problem(problem, emit=events.append)
    return result, events


def test_a_successful_run_is_narrated_in_the_playgrounds_vocabulary(stub_pipeline):
    result, events = _narrate(dict(PROBLEM))
    assert result["ready"] is True

    kinds = [e["type"] for e in events]
    # Every type here has a handler in frontend/playground.html; an unknown one
    # would render as a raw log line instead of a panel.
    assert set(kinds) <= {"stage", "ground_truth", "entry", "oracle_tests",
                          "verdict", "chunks", "blocked"}, kinds

    stages = [e["name"] for e in events if e["type"] == "stage"]
    assert stages == ["start", "entry", "oracle_gen", "decomposition",
                      "finished"], stages
    assert kinds.index("ground_truth") < kinds.index("entry")
    assert kinds.index("entry") < kinds.index("chunks")

    # The transcript carries the answer key, which is why the route serving it
    # is teacher-gated. Assert it is actually there, so nobody "tidies" the
    # gate away later believing the stream is harmless.
    truth = next(e for e in events if e["type"] == "ground_truth")
    assert "sum(int(c)" in truth["code"]
    chunks = next(e for e in events if e["type"] == "chunks")
    assert chunks["chunks"][0]["reference"] == "return 3"


def test_a_failure_is_narrated_not_just_returned(stub_pipeline):
    result, events = _narrate({**PROBLEM, "solution": ""})
    assert result["ready"] is False and result["stage"] == "parses"
    blocked = [e for e in events if e["type"] == "blocked"]
    assert len(blocked) == 1, events
    assert blocked[0]["at"] == "parses"
    assert blocked[0]["message"] == result["error"], "the row and the transcript must agree"


def test_narration_cannot_change_or_break_preparation(stub_pipeline):
    """`emit` is reporting, never control flow."""
    silent = publish.prepare_problem(dict(PROBLEM))
    watched = publish.prepare_problem(dict(PROBLEM), emit=lambda ev: None)
    assert silent == watched

    # prepare_bus.publish swallows subscriber errors, which is what protects the
    # run. Assert the property at the seam a real upload uses.
    k = key("asg", "digit-sum")
    open_channel(k)

    class Hostile:
        def put_nowait(self, ev):
            raise RuntimeError("a watcher's connection died mid-run")

    import main.prepare_bus as bus
    with bus._lock:
        bus._channels[k].subscribers.append(Hostile())
    hostile = publish.prepare_problem(dict(PROBLEM), emit=lambda ev: bus_publish(k, ev))
    finish(k)
    assert hostile == silent, "a failing watcher changed the result of an upload"


def test_a_late_watcher_still_sees_the_run_from_its_start(stub_pipeline):
    """The teacher clicks BECAUSE it is taking a while - so they always arrive
    late, and a stream of only-future events would open on a blank page."""
    k = key("asg", "digit-sum")
    open_channel(k)

    started = threading.Event()

    def upload():
        publish.prepare_problem(dict(PROBLEM),
                                emit=lambda ev: (bus_publish(k, ev), started.set()))
        time.sleep(0.05)                      # still "running" when we attach
        finish(k)

    t = threading.Thread(target=upload)
    t.start()
    started.wait(timeout=5)

    seen = [e for e in subscribe(k) if e is not None]
    t.join(timeout=5)

    assert [e["type"] for e in seen][0] == "stage"
    assert seen[0]["name"] == "start", seen[0]
    assert any(e["type"] == "chunks" for e in seen), "joined too late to see the end"

    # And once it is over, it replays in full for anyone who opens it after.
    assert is_open(k), "a finished run is still readable - that is the replay"
    again = [e for e in subscribe(k) if e is not None]
    assert [e["type"] for e in again] == [e["type"] for e in seen]


def test_watching_a_run_that_is_not_happening_ends_rather_than_hangs():
    assert not is_open(key("no-such-assignment", "no-such-slug"))
    assert list(subscribe(key("no-such-assignment", "no-such-slug"))) == []
