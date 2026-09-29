"""test_grade_sheet_status.py - the grade sheet tells "sent code" apart from
"only opened a problem".

THE REPORT. The instructor's grade sheet showed a student as "Submitted" with 0
solved, and nobody could tell why: a student who merely OPENED a problem was
counted as submitted (grades.grade_sheet: `bool(mine) or ... in started`). That
rule existed so an opened problem would not read "Missing" - which would be
wrong too. Two states cannot say three things, so there are three:

    submitted - sent code for at least one step
    started   - opened a problem, has not sent any code yet
    missing   - has not opened anything in this assignment
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent


class _Q:
    def __init__(self, rows):
        self.rows, self.filters = rows, []

    def select(self, *_a, **_k):
        return self

    def eq(self, col, val):
        self.filters.append(lambda r: r.get(col) == val)
        return self

    def in_(self, col, vals):
        self.filters.append(lambda r: r.get(col) in vals)
        return self

    def order(self, *_a, **_k):
        return self

    def execute(self):
        return type("R", (), {"data": [r for r in self.rows
                                       if all(f(r) for f in self.filters)]})()


class FakeDB:
    def __init__(self, tables):
        self.tables = tables

    def table(self, name):
        return _Q(self.tables.get(name, []))


def _sheet():
    from main import grades
    db = FakeDB({
        "problems": [{"slug": "p1", "title": "P1", "description": "d", "solution": "def f():\n    return 1",
                      "ready": True, "assignment_id": "a1", "context": None}],
        "students": [{"id": "s-sent", "username": "a@psu.edu", "first_name": "A", "last_name": "Sent", "role": "student"},
                     {"id": "s-open", "username": "b@psu.edu", "first_name": "B", "last_name": "Opened", "role": "student"},
                     {"id": "s-none", "username": "c@psu.edu", "first_name": "C", "last_name": "None", "role": "student"}],
        "mt_sessions": [{"student_id": "s-sent", "slug": "p1", "total_chunks": 2},
                        {"student_id": "s-open", "slug": "p1", "total_chunks": 2}],
        "mt_submissions": [{"student_id": "s-sent", "slug": "p1", "chunk_index": 0, "verdict": "incorrect"}],
    })
    return {r["student_id"]: r for r in grades.grade_sheet(db, "a1")["students"]}


def test_each_student_gets_the_state_they_are_actually_in():
    rows = _sheet()
    assert {k: r["status"] for k, r in rows.items()} == \
        {"s-sent": "submitted", "s-open": "started", "s-none": "missing"}


def test_only_opening_a_problem_is_not_submitting_it():
    rows = _sheet()
    assert rows["s-open"]["submitted"] is False
    assert rows["s-sent"]["submitted"] is True


def test_the_page_shows_all_three_states_and_counts_them():
    js = (ROOT / "frontend" / "grades.js").read_text()
    html = (ROOT / "frontend" / "grades.html").read_text()
    for label in ("Submitted", "Started", "Missing"):
        assert f'"{label}"' in js, f"the grade sheet never labels anyone {label}"
    assert "r.status" in js, "the pill must follow the three-state status"
    labels = re.findall(r'<div class="metric-label">([^<]+)</div>', html)
    assert labels == ["Students", "Submitted", "Started", "Missing"], labels
