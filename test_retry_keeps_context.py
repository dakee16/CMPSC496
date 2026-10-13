"""test_retry_keeps_context.py - preparing one problem again from edited text
saves the same problem that was prepared.

THE BUG (found 9 Oct, before the review page could rely on it). Retry saved
`description` and `solution` but never `context`. A METHOD's docstring also
lives in its context (context_prefix), so adding examples to HW4's `update`
would have saved it half new and half old: its identity (content_hash) no
longer matched what had just been prepared, and students would have opened a
problem with no tests and no roadmap behind it. A plain function re-read from
its own text alone also lost the file's imports and its place in the list.

REAL: the retry route, the parser, the saved row, content_hash. FAKED:
Supabase (in memory) and the pipeline itself - prepare_problem is minutes of
paid model work; here it only records the problem it was handed.
"""
from frontend import api_server
from main.assignments import parse_assignment_file
from main.identity import content_hash
from test_auth_routes import teacher_client

CLASS_FILE = '''"""Shelf"""


class Shelf:
    """
        >>> s = Shelf()
        >>> s.put(3)
        >>> s.count()
        1
    """
    # --- steps: put ---
    def __init__(self):
        self.items = []

    def put(self, value):
        """Add value to the shelf."""
        self.items.append(value)

    def count(self):
        return len(self.items)
'''
EDITED_PUT = '''def put(self, value):
    """Add value to the shelf.

    >>> s = Shelf()
    >>> s.put(1)
    >>> s.put(2)
    >>> s.count()
    2
    """
    self.items.append(value)
'''


def _saved(sb, slug):
    row = [p for p in sb.problems if p["slug"] == slug][-1]
    return {**row, **api_server._context_of(row)}


def _capture(monkeypatch):
    handed = []
    monkeypatch.setattr("main.publish.prepare_problem",
                        lambda p, emit=None: handed.append(dict(p)) or {
                            "slug": p["slug"], "ready": True, "chunks": 2,
                            "n_tests": 5, "stage": None, "error": None})
    return handed


def test_an_edited_methods_docstring_is_saved_with_its_class(monkeypatch):
    c, sb = teacher_client()
    p = parse_assignment_file(CLASS_FILE, "shelf.py")["problems"][0]
    sb.problems.append({"slug": p["slug"], "title": p["title"],
                        "description": p["description"], "solution": p["solution"],
                        "assignment_id": "a-1", "ready": False,
                        "prepare_error": "Waiting on your review.",
                        **api_server._group_columns(p)})
    handed = _capture(monkeypatch)
    r = c.post("/teacher/problems/retry", json={
        "assignment_id": "a-1", "slug": p["slug"], "source": EDITED_PUT})
    assert r.status_code == 200 and r.json()["ready"] is True, r.text
    saved = _saved(sb, p["slug"])
    assert ">>> s.put(2)" in saved["context_prefix"], "the class kept the OLD docstring"
    assert content_hash(saved) == content_hash(handed[0]), \
        "what students load is not what was prepared"


def test_a_plain_function_keeps_its_imports_and_its_place(monkeypatch):
    c, sb = teacher_client()
    sb.problems[:] = [{"slug": "hyp", "title": "Hyp", "assignment_id": "a-1",
                       "description": "Hypotenuse.", "ready": False,
                       "solution": 'def hyp(a, b):\n    """Hypotenuse."""\n'
                                   '    return math.sqrt(a * a + b * b)\n',
                       "prepare_error": "Waiting on your review.",
                       "group_order": 4, "member_order": 0,
                       "context": {"module_preamble": '"""Geometry"""\nimport math',
                                   "entry_hint": "hyp"}}]
    handed = _capture(monkeypatch)
    r = c.post("/teacher/problems/retry", json={
        "assignment_id": "a-1", "slug": "hyp",
        "source": 'def hyp(a, b):\n    """Hypotenuse, rounded."""\n'
                  '    return round(math.sqrt(a * a + b * b), 2)\n'})
    assert r.status_code == 200 and r.json()["ready"] is True, r.text
    assert "import math" in (handed[0].get("module_preamble") or ""), \
        "prepared without the file's imports"
    saved = _saved(sb, "hyp")
    assert "import math" in (saved.get("module_preamble") or "")
    assert saved["group_order"] == 4, "it moved in the student's list"
    assert content_hash(saved) == content_hash(handed[0])
