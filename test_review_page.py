"""test_review_page.py - a problem waiting on review is explained plainly, and
the fix it offers is proven before it is offered.

THE REPORT (8 Oct, HW4). Three problems sat at "your review" with nothing on
screen but "a few checks came back inconclusive", and the button led to the
Playground, which called them "failed" and "oracle too weak".

AND (13 Oct, HW4). update was READY with `>` -> `>=` still passing its tests -
"strong" is 85% caught - and nothing anywhere said so. Sanan: show the specific
test cases to add, the wrong answers each catches, and one button to add them
all and prepare that problem again.

REAL: the review routes, the stored verdict they read, running the suggested
examples on the teacher's code and on every missed change, inserting them into
the docstring, and the retry route the page sends that to. FAKED: Supabase (in
memory), the model (canned plain words and example STATEMENTS - it is never
asked for an answer), and prepare_problem (minutes of paid model work).
"""
import json

import pytest

from frontend import api_server
from main import mutation, review_help
from main.assignments import parse_assignment_file
from main.identity import content_hash
from main.oracle_store import save_cache
from test_auth_routes import client, register, teacher_client, PW

TANK = '''"""Tanks"""


class Tank:
    """
        >>> t = Tank(10)
        >>> t.fill(4)
        'OK'
    """
    # --- steps: fill ---
    def __init__(self, size):
        self.size = size
        self.level = 0

    def fill(self, amount):
        """Add amount if it fits and return 'OK'; return 'FULL' and change
        nothing if it does not."""
        room = self.size - self.level
        if amount > room:
            return 'FULL'
        self.level = self.level + amount
        return 'OK'
'''
GUARD = "line 5: remove `if amount > room:`"
EDGE = "line 5: > -> >="
LEVEL = "line 7: remove `self.level = self.level + amount`"
WORDS = {GUARD: ("A student could skip the room check", "an amount that does not fit"),
         EDGE: ("A student could use >= instead of >", "an amount that fits exactly"),
         LEVEL: ("A student could forget to raise the level", "reading the level after filling")}
# Statements only - and no exact fit, so the >= change stays unproven. The
# third case catches nothing and must not be offered.
TESTS = [["t = Tank(10)", "t.fill(4)", "t.fill(7)"],
         ["t = Tank(10)", "t.fill(4)", "t.level"],
         ["t = Tank(5)", "t.fill(3)", "t.fill(1)"]]


@pytest.fixture
def review(monkeypatch, tmp_path):
    from main import identity
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(tmp_path / "o.json"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    # The retry route records its run: never into the real data/transcripts.json.
    monkeypatch.setenv("MICROTUTOR_TRANSCRIPTS", str(tmp_path / "tr.json"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    c, sb = teacher_client()
    p = parse_assignment_file(TANK, "tank.py")["problems"][0]
    labels = {m["label"] for m in mutation.generate_mutants(p["solution"])}
    assert {GUARD, EDGE, LEVEL} <= labels, sorted(labels)
    row = {"slug": p["slug"], "title": p["title"], "description": p["description"],
           "solution": p["solution"], "assignment_id": "a-1", "ready": False,
           "prepare_error": "Waiting on your review: a few checks on the generated "
                            "tests came back inconclusive.",
           **api_server._group_columns(p)}
    sb.problems.append(row)
    key = content_hash({"description": row["description"], "solution": row["solution"],
                        **api_server._context_of(row)})
    save_cache({key: {"strong": False, "status": "needs_review",
                      "kill_rate_lower": 0.5, "kill_rate_upper": 1.0, "final_tests": [],
                      "breakdown": {"mutants": [
                          {"label": lb, "status": "undetermined" if lb in (GUARD, EDGE, LEVEL)
                           else "killed"} for lb in sorted(labels)]}}})
    asked = []
    R = type("R", (), {"c": c, "sb": sb, "slug": p["slug"], "asked": asked, "row": row,
                       "tests": TESTS, "url": f"/teacher/review/a-1/{p['slug']}",
                       "total": len(labels)})

    def chat(model, system, messages, **k):
        """Answers by the numbers in the prompt, as the model must."""
        asked.append(k.get("max_tokens"))
        import re
        listed = re.findall(r"^(\d+)\. (line .*)$", messages[0]["content"], re.M)
        return json.dumps({"tests": R.tests, "changes": [
            {"n": int(n), "title": WORDS[lb][0], "missing": WORDS[lb][1]}
            for n, lb in listed]})
    monkeypatch.setattr(review_help, "chat", chat)
    return R


def test_the_free_view_names_every_missed_change_and_its_line(review):
    d = review.c.get(review.url).json()
    assert d["in_review"] and d["help"] is None and review.asked == [], "nothing was paid"
    assert (d["caught"], d["total"]) == (review.total - 3, review.total)
    got = {c["label"]: c for c in d["checks"]}
    assert set(got) == {GUARD, EDGE, LEVEL}
    assert got[GUARD]["code"].strip() == "if amount > room:" and got[GUARD]["line"] == 5
    assert all(c["verdict"] == "unknown" for c in d["checks"])
    assert d["queue"] == [{"slug": review.slug, "title": d["title"]}]
    # Waiting on review is its own state: not also counted as "ready, with gaps".
    (mine,) = [p for p in review.c.get("/teacher/assignments/a-1/problems").json()["problems"]
               if p["slug"] == review.slug]
    assert mine["gaps"] == 0


def test_the_help_is_proven_on_the_teachers_code_and_remembered(review):
    d = review.c.post(review.url + "/explain").json()
    assert review.asked == [review_help._MAX_TOKENS], "one capped call"
    got = {c["label"]: c for c in d["checks"]}
    assert got[GUARD]["title"] == "A student could skip the room check"
    # Proven where an example gives the broken code a different answer...
    assert got[GUARD]["verdict"] == "gap" and got[LEVEL]["verdict"] == "gap"
    # ...and said as unproven where none does - never as harmless for sure.
    assert got[EDGE]["verdict"] == "maybe_harmless"
    h = d["help"]
    assert h["examples_ok"] and h["catches"] == 2
    # The answers in the examples are the TEACHER's: the model gave none.
    assert ">>> t.fill(7)\n'FULL'" in h["doctest"] and ">>> t.level\n4" in h["doctest"]
    assert ">>> t.fill(7)" in h["edited_source"] and "def fill(self, amount):" in h["edited_source"]
    # One card per test case, saying which wrong answers it catches - and the
    # one that catches nothing is not offered.
    assert [t["catches"] for t in h["tests"]] == [
        [WORDS[GUARD][0], WORDS[LEVEL][0]], [WORDS[LEVEL][0]]]
    assert h["tests"][0]["doctest"] == ">>> t = Tank(10)\n>>> t.fill(4)\n'OK'\n>>> t.fill(7)\n'FULL'"
    assert "Tank(5)" not in h["edited_source"]

    again = review.c.post(review.url + "/explain").json()
    assert len(review.asked) == 1, "a reopened page paid again"
    assert again["help"]["doctest"] == h["doctest"]


def test_adding_the_examples_prepares_the_problem_again_with_them(review, monkeypatch):
    """The page's main button: the edited source, sent to the retry route."""
    edited = review.c.post(review.url + "/explain").json()["help"]["edited_source"]
    handed = []
    monkeypatch.setattr("main.publish.prepare_problem",
                        lambda p, emit=None, **_k: handed.append(p) or {
                            "slug": p["slug"], "ready": True, "chunks": 2, "n_tests": 7,
                            "stage": None, "error": None})
    r = review.c.post("/teacher/problems/retry", json={
        "assignment_id": "a-1", "slug": review.slug, "source": edited})
    assert r.status_code == 200 and r.json()["ready"] is True, r.text
    from main.context import doctest_block
    assert ">>> t.fill(7)" in handed[0]["description"]
    assert "t.fill(7)" in (doctest_block(handed[0]) or ""), "the examples are not tests"
    saved = [p for p in review.sb.problems if p["slug"] == review.slug][-1]
    assert content_hash({**saved, **api_server._context_of(saved)}) == content_hash(handed[0])


def test_an_outage_still_shows_everything_that_was_free(review, monkeypatch):
    monkeypatch.setattr(review_help, "chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("OpenAI 503")))
    d = review.c.post(review.url + "/explain").json()
    assert d["help"]["written"] is False and "503" in d["help"]["error"]
    assert len(d["checks"]) == 3 and all(c["code"] for c in d["checks"])


def test_a_student_cannot_open_a_review(review):
    c = client()
    register(c, "abc123@psu.edu", PW)
    assert c.get(review.url).status_code == 403
    assert c.post(review.url + "/explain").status_code == 403


def test_the_help_itself_is_written_once_per_problem_version(review):
    """The route checks first, and so does explain() - two pages opened at
    once must not both pay."""
    p = parse_assignment_file(TANK, "tank.py")["problems"][0]
    review_help.explain(p, "k-1", [GUARD, EDGE, LEVEL])
    review_help.explain(p, "k-1", [GUARD, EDGE, LEVEL])
    assert len(review.asked) == 1
    review_help.explain(p, "k-2", [GUARD])           # a new version is new work
    assert len(review.asked) == 2


# ── ready, with gaps (13 Oct) ────────────────────────────────────────────

def _live(review):
    review.row["ready"], review.row["prepare_error"] = True, None


def test_a_ready_problem_whose_tests_miss_mistakes_says_so_and_offers_the_cases(review):
    _live(review)
    listed = review.c.get("/teacher/assignments/a-1/problems").json()["problems"]
    (mine,) = [p for p in listed if p["slug"] == review.slug]
    assert mine["gaps"] == 3 and "solution" not in mine, "counted, and the answer key not sent"
    d = review.c.get(review.url).json()
    assert (d["in_review"], d["gaps"]) == (False, True)
    assert d["queue"] == [{"slug": review.slug, "title": d["title"]}]
    h = review.c.post(review.url + "/explain").json()["help"]
    assert len(h["tests"]) == 2 and h["edited_source"], "the same help as a problem in review"


def test_a_ready_problem_with_nothing_missed_is_not_flagged(review):
    _live(review)
    entry = __import__("main.oracle_store", fromlist=["x"]).load_cache()
    for e in entry.values():
        for m in e["breakdown"]["mutants"]:
            m["status"] = "killed"
    save_cache(entry)
    (mine,) = [p for p in review.c.get("/teacher/assignments/a-1/problems").json()["problems"]
               if p["slug"] == review.slug]
    assert mine["gaps"] == 0
    d = review.c.get(review.url).json()
    assert d["gaps"] is False and d["queue"] == []
    assert review.c.post(review.url + "/explain").json()["help"] is None and review.asked == []


@pytest.mark.parametrize("comes_out_ready", [True, False])
def test_adding_the_cases_to_a_live_problem_never_hides_it(review, monkeypatch, comes_out_ready):
    """A problem students can see is re-prepared only to add tests. If the new
    version does not come out ready, theirs stays exactly as it was."""
    _live(review)
    before = dict(review.row)
    edited = review.c.post(review.url + "/explain").json()["help"]["edited_source"]
    monkeypatch.setattr("main.publish.prepare_problem", lambda p, emit=None, **_k: {
        "slug": p["slug"], "ready": comes_out_ready, "chunks": 2, "n_tests": 7,
        "stage": None if comes_out_ready else "oracle",
        "error": None if comes_out_ready else "Waiting on your review: still inconclusive."})
    d = review.c.post("/teacher/problems/retry", json={
        "assignment_id": "a-1", "slug": review.slug, "source": edited, "keep_live": True}).json()
    saved = [p for p in review.sb.problems if p["slug"] == review.slug][-1]
    if comes_out_ready:
        assert d["ready"] and ">>> t.fill(7)" in saved["solution"]
    else:
        assert d["ready"] is False and "Nothing changed for students" in d["error"]
        assert (saved["ready"], saved["solution"]) == (True, before["solution"])


def test_test_cases_that_change_each_others_answers_are_not_offered(review):
    """They go into the docstring as one run. A case that patches the class
    would change the next one's answers there, and the docstring would be wrong."""
    p = parse_assignment_file(TANK, "tank.py")["problems"][0]
    review.tests = [["t = Tank(10)", "t.fill(4)", "t.level", "Tank.fill = lambda self, a: 'X'"],
                    ["t = Tank(10)", "t.fill(4)", "t.fill(7)"]]
    out = review_help.explain(p, "k-together", [GUARD, EDGE, LEVEL])
    assert out["tests"] == [] and out["examples"]["ok"] is False
    assert "each other" in out["examples"]["reason"]


# ── marked harmless (14 Oct) ─────────────────────────────────────────────

def test_a_change_marked_harmless_stops_counting_as_a_gap(review):
    """HW4's mruEvict unlinks a `previous` that is always already empty: no
    test can catch leaving it out, because it changes nothing. Sanan: "we can
    just remove that issue showing up manually"."""
    _live(review)
    d = review.c.post(review.url + "/dismiss", json={"label": EDGE}).json()
    assert EDGE not in [c["label"] for c in d["checks"]] and len(d["checks"]) == 2
    assert d["harmless"] == [{"label": EDGE, "line": 5, "phrasing": d["harmless"][0]["phrasing"],
                              "by": "prof@psu.edu"}]
    assert (d["caught"], d["total"]) == (review.total - 3, review.total), "the tests' record is unchanged"
    (mine,) = [p for p in review.c.get("/teacher/assignments/a-1/problems").json()["problems"]
               if p["slug"] == review.slug]
    assert mine["gaps"] == 2
    for lb in (GUARD, LEVEL):
        review.c.post(review.url + "/dismiss", json={"label": lb})
    d = review.c.get(review.url).json()
    assert d["gaps"] is False and d["queue"] == [] and len(d["harmless"]) == 3
    assert review.asked == [], "free"


def test_a_student_cannot_mark_anything_harmless(review):
    c = client()
    register(c, "abc123@psu.edu", PW)
    assert c.post(review.url + "/dismiss", json={"label": EDGE}).status_code == 403
