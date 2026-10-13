"""test_review_page.py - a problem waiting on review is explained plainly, and
the fix it offers is proven before it is offered.

THE REPORT (8 Oct, HW4). Three problems sat at "your review" with nothing on
screen but "a few checks came back inconclusive", and the button led to the
Playground, which called them "failed" and "oracle too weak".

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
# Statements only - and no exact fit, so the >= change stays unproven.
EXAMPLES = ["t = Tank(10)", "t.fill(4)", "t.fill(7)", "t.level"]


@pytest.fixture
def review(monkeypatch, tmp_path):
    from main import identity
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(tmp_path / "o.json"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
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

    def chat(model, system, messages, **k):
        """Answers by the numbers in the prompt, as the model must."""
        asked.append(k.get("max_tokens"))
        import re
        listed = re.findall(r"^(\d+)\. (line .*)$", messages[0]["content"], re.M)
        return json.dumps({"examples": EXAMPLES, "changes": [
            {"n": int(n), "title": WORDS[lb][0], "missing": WORDS[lb][1]}
            for n, lb in listed]})
    monkeypatch.setattr(review_help, "chat", chat)
    return type("R", (), {"c": c, "sb": sb, "slug": p["slug"], "asked": asked,
                          "url": f"/teacher/review/a-1/{p['slug']}", "total": len(labels)})


def test_the_free_view_names_every_missed_change_and_its_line(review):
    d = review.c.get(review.url).json()
    assert d["in_review"] and d["help"] is None and review.asked == [], "nothing was paid"
    assert (d["caught"], d["total"]) == (review.total - 3, review.total)
    got = {c["label"]: c for c in d["checks"]}
    assert set(got) == {GUARD, EDGE, LEVEL}
    assert got[GUARD]["code"].strip() == "if amount > room:" and got[GUARD]["line"] == 5
    assert all(c["verdict"] == "unknown" for c in d["checks"])
    assert d["queue"] == [{"slug": review.slug, "title": d["title"]}]


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

    again = review.c.post(review.url + "/explain").json()
    assert len(review.asked) == 1, "a reopened page paid again"
    assert again["help"]["doctest"] == h["doctest"]


def test_adding_the_examples_prepares_the_problem_again_with_them(review, monkeypatch):
    """The page's main button: the edited source, sent to the retry route."""
    edited = review.c.post(review.url + "/explain").json()["help"]["edited_source"]
    handed = []
    monkeypatch.setattr("main.publish.prepare_problem",
                        lambda p, emit=None: handed.append(p) or {
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
