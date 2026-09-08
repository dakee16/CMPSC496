"""
transcripts.py - keeping the pipeline's narration for problems that did NOT pass.

The live narration (main/prepare_bus.py) is in-memory and dies with the run, so
"watch this problem" only ever worked while the upload was still going. An
instructor who came back an hour later - which is when they actually sit down to
fix things - got a dead page and a button offering to run the whole thing again.

This stores the run so it can be replayed after the fact. Deliberately NOT for
every problem:

    ready          nothing stored. A problem that passed needs no explanation,
                   and _getPostfix alone emits thousands of events - keeping
                   that for the 7 problems that worked is pure cost.
    failed         stored. The instructor has to edit something, and the
                   transcript says what.
    needs_review   stored, and this is the case that motivated the whole thing:
                   the fix is a judgement call, so the evidence IS the feature.

TRIMMING. A transcript is written once and read by a human, so it is trimmed on
the way in rather than paged on the way out:

  * `tally` events are dropped entirely - one per mutant per phase, carrying a
    running count the final verdict already states.
  * `per_test` detail is stripped from mutant_result: for 129 mutants against a
    13-test suite that is 1,677 rows nobody reads, and the killed/crashed flags
    that decide the outcome are kept.
  * whatever survives is capped at MAX_EVENTS, keeping the HEAD and the TAIL.
    The head is where the run is set up and the tail is where it went wrong; the
    middle is the repetitive part, and the gap is marked rather than hidden.

Storage is a JSON file beside the other backend data, not a Supabase column: a
transcript is debugging evidence for one upload, it is never queried across
problems, and it would otherwise be the largest column in the table.
"""
import json
import os
import time

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "transcripts.json")

# Per transcript. _getPostfix's full run is tens of thousands of events; this
# keeps the shape of the run without keeping all of it.
MAX_EVENTS = 1200

# Total transcripts retained. Each upload of a 20-problem file can add up to 20,
# so without a ceiling this grows forever.
MAX_TRANSCRIPTS = 200

# Dropped outright - pure progress-counter noise.
_DROP_TYPES = {"tally"}

# Stripped from the events that carry them: large, and never read once the
# killed/crashed verdict is known.
_DROP_FIELDS = ("per_test", "code")


def path() -> str:
    return os.environ.get("MICROTUTOR_TRANSCRIPTS") or DEFAULT_PATH


def key(assignment_id, slug: str) -> str:
    return f"{assignment_id}:{slug}"


def _load_all() -> dict:
    p = path()
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass                      # a corrupt file is not worth an outage
    return {}


def _save_all(data: dict) -> None:
    p = path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, p)                # atomic: never leave a half-written file


def trim(events: list) -> list:
    """Drop the noise, strip the bulky fields, and cap the length.

    Returns a NEW list; the caller's events are never mutated - they may still
    be on their way to a live watcher."""
    kept = []
    for ev in events:
        if not isinstance(ev, dict) or ev.get("type") in _DROP_TYPES:
            continue
        if any(f in ev for f in _DROP_FIELDS):
            ev = {k: v for k, v in ev.items() if k not in _DROP_FIELDS}
        kept.append(ev)

    if len(kept) <= MAX_EVENTS:
        return kept
    # Head and tail, with the gap stated. Two thirds to the head: that is where
    # the ground truth, the entry point and the oracle suite are established.
    head = (MAX_EVENTS * 2) // 3
    tail = MAX_EVENTS - head - 1
    dropped = len(kept) - head - tail
    return (kept[:head]
            + [{"type": "transcript_truncated", "dropped": dropped,
                "detail": f"{dropped} events omitted - this run was too long to "
                          f"keep in full. The setup above and the outcome below "
                          f"are complete."}]
            + kept[-tail:])


def record(assignment_id, slug: str, events: list, outcome: dict) -> None:
    """Store one problem's run. A READY problem is not stored, and any earlier
    transcript for it is removed - a resolved problem should stop offering an
    explanation of a failure that no longer exists."""
    data = _load_all()
    k = key(assignment_id, slug)

    if outcome.get("ready"):
        if data.pop(k, None) is not None:
            _save_all(data)
        return

    data[k] = {
        "assignment_id": str(assignment_id), "slug": slug,
        "saved_at": time.time(),
        "outcome": {x: outcome.get(x) for x in
                    ("ready", "stage", "error", "needs_review", "undetermined",
                     "kill_rate_lower", "kill_rate_upper", "chunks", "n_tests")},
        "events": trim(events),
    }

    if len(data) > MAX_TRANSCRIPTS:
        for old in sorted(data, key=lambda x: data[x].get("saved_at", 0)
                          )[:len(data) - MAX_TRANSCRIPTS]:
            data.pop(old, None)
    _save_all(data)


def load(assignment_id, slug: str) -> dict | None:
    return _load_all().get(key(assignment_id, slug))


def forget(assignment_id, slug: str) -> None:
    data = _load_all()
    if data.pop(key(assignment_id, slug), None) is not None:
        _save_all(data)


if __name__ == "__main__":
    # Runs against a temp file, so it never touches real data.
    import tempfile
    tmpdir = tempfile.mkdtemp()
    os.environ["MICROTUTOR_TRANSCRIPTS"] = os.path.join(tmpdir, "t.json")

    noisy = ([{"type": "stage", "name": "start"}]
             + [{"type": "tally", "processed": i} for i in range(500)]
             + [{"type": "mutant_result", "index": 1, "killed": True,
                 "per_test": [{"i": i} for i in range(50)]}]
             + [{"type": "blocked", "at": "strength"}])
    t = trim(noisy)
    assert not any(e["type"] == "tally" for e in t), "tally must be dropped"
    assert "per_test" not in t[1], "per_test must be stripped"
    assert t[1]["killed"] is True, "the verdict fields must survive"
    assert [e["type"] for e in t] == ["stage", "mutant_result", "blocked"], t

    # Truncation keeps both ends and says how much went missing.
    long = [{"type": "e", "i": i} for i in range(MAX_EVENTS * 3)]
    cut = trim(long)
    assert len(cut) == MAX_EVENTS, len(cut)
    assert cut[0]["i"] == 0 and cut[-1]["i"] == MAX_EVENTS * 3 - 1
    marker = [e for e in cut if e["type"] == "transcript_truncated"]
    assert len(marker) == 1 and marker[0]["dropped"] == MAX_EVENTS * 3 - MAX_EVENTS + 1

    # A blocked problem is stored; a ready one is not - and becoming ready
    # REMOVES the old explanation rather than leaving a stale one behind.
    record("a1", "stack-pop", noisy, {"ready": False, "stage": "strength",
                                      "needs_review": True, "undetermined": 2})
    got = load("a1", "stack-pop")
    assert got and got["outcome"]["needs_review"] is True
    assert not any(e["type"] == "tally" for e in got["events"])

    record("a1", "stack-push", noisy, {"ready": True})
    assert load("a1", "stack-push") is None, "a passing problem stores nothing"

    record("a1", "stack-pop", noisy, {"ready": True})
    assert load("a1", "stack-pop") is None, "fixed problem drops its transcript"

    # Eviction is by age, and keeps the ceiling exactly.
    for i in range(MAX_TRANSCRIPTS + 25):
        record("a2", f"p{i}", [{"type": "stage"}], {"ready": False})
    assert len(_load_all()) == MAX_TRANSCRIPTS, len(_load_all())
    assert load("a2", f"p{MAX_TRANSCRIPTS + 24}") is not None, "newest survives"

    print("transcripts.py self-check OK")
