"""test_crash_messages.py - a student whose code crashes is told WHY.

THE REPORT. A real student (CMPSC 132, calculator-calculate) wrote `self._expr`
where the class stores `self.__expr`. Every case came back as

    you gave: [None, '!AttributeError']

- the exception's NAME and nothing else. Nine attempts later he still had not
found a one-character typo that the message itself names:
"'Calculator' object has no attribute '_expr'".

For a METHOD the test runner records a crash as a value ("!AttributeError"),
because how a method fails is part of its behaviour and has to be compared.
The message now rides along after a NUL separator (context.ERROR_SEP), and the
shared comparison (execution.norm) cuts it off - so what is COMPARED is exactly
what it always was, and only what is SHOWN changes. The last test here pins
that half: remove the cut and a teacher's own code stops matching its tests.

Self-contained on purpose: inline class, inline oracle, no model.
"""
import types

import pytest

# A class whose attribute is name-mangled, exactly like HW3's Calculator.
PREFIX = ("class Box:\n"
          "    def __init__(self):\n"
          "        self.__v = None\n"
          "\n"
          "    def put(self, v):\n"
          "        self.__v = v\n"
          "\n"
          "    def get(self):\n")
SOLUTION = "def get(self):\n    return self.__v\n"
PROBLEM = {"slug": "box-get", "title": "Box.get", "solution": SOLUTION,
           "description": "Return what was put in the box.",
           "context_prefix": PREFIX, "context_suffix": "", "context_indent": 8,
           "entry_hint": "get", "group_title": "Box"}
ORACLE = [{"input": [[["get"]]], "expected": [None]},
          {"input": [[["put", 5], ["get"]]], "expected": [None, 5]}]


@pytest.fixture
def grade(monkeypatch, tmp_path):
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.sqlite3"))
    from main import grading, identity, ollama_client, sessions
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))

    def no_model(*_a, **_k):
        raise AssertionError("showing a crash must never reach a model")
    monkeypatch.setattr(ollama_client, "_openai_chat", no_model)

    def _grade(code, problem=PROBLEM, header="def get(self):", oracle=ORACLE):
        decomp = {"header": header, "chunks": [types.SimpleNamespace(
            step_id="Part 1", expected_type="code", prompt="all of it",
            reference="pass")]}
        sid = sessions.create_session(dict(problem), decomp, "h-crash",
                                      student_id="stu")["session_id"]
        return grading.grade_submission(sessions.load_session(sid), code,
                                        oracle_loader=lambda p: list(oracle))
    return _grade


def test_the_reported_typo_is_named_on_the_call_that_crashed(grade):
    r = grade("return self._v")
    assert r.verdict == "incorrect", r
    shown = "\n".join(r.failing_cases or [])
    assert "no attribute '_v'" in shown, shown
    assert "x.get() crashed: AttributeError" in shown, shown
    assert "!AttributeError" not in shown, "the raw marker must not reach a student"


def test_the_teachers_own_code_still_passes(grade):
    """Same crash type, message attached, compared equal - nothing about
    GRADING moved."""
    assert grade("return self.__v").verdict == "correct"


def test_a_plain_function_crash_reads_as_a_sentence(grade):
    prob = {"slug": "first", "title": "first", "description": "First letter.",
            "solution": "def first(s):\n    return s[0]\n"}
    r = grade("return {}[s]", problem=prob, header="def first(s):",
              oracle=[{"input": ["ab"], "expected": "a"}])
    shown = "\n".join(r.failing_cases or [])
    assert "your code crashed: KeyError: 'ab'" in shown, shown


def test_comparison_ignores_the_message():
    """An expected crash stored WITHOUT a message (every cached oracle) must
    equal the same crash WITH one. This is the line that keeps grading
    unchanged; measured, removing it breaks 6 of the 14 live problems for the
    teacher's own solution."""
    from main.context import ERROR_PREFIX, ERROR_SEP
    from main.execution import _norm
    live = ERROR_PREFIX + "IndexError" + ERROR_SEP + "pop from empty list"
    assert _norm([None, live]) == _norm([None, ERROR_PREFIX + "IndexError"])
    assert _norm(live) != _norm(ERROR_PREFIX + "KeyError")


# A method the tests call with the wrong arguments ON PURPOSE: the teacher's
# code raises TypeError there too, and the oracle expects it.
PEEK = {**PROBLEM, "context_prefix": PREFIX.replace(
    "    def get(self):\n", "    def peek(self, i):\n        return self.__v\n\n"
    "    def get(self):\n")}


def test_an_error_the_teachers_code_raises_too_is_not_blamed_on_the_student(grade):
    """THE AUDIT (29 Sep, _isNumber). A `return True` was shown
    `x._getPostfix() crashed: TypeError: ... missing 1 required positional
    argument` - a call it could not have caused, which the teacher's code
    raises too and the expected list already showed as <raises TypeError>."""
    from main.context import ERROR_PREFIX
    oracle = [{"input": [[["put", 5], ["peek"], ["get"]]],
               "expected": [None, ERROR_PREFIX + "TypeError", 5]}]
    r = grade("return 0", problem=PEEK, oracle=oracle)
    shown = "\n".join(r.failing_cases or [])
    assert r.verdict == "incorrect", r
    assert "expected: [None, <raises TypeError>, 5]" in shown, shown
    assert "you gave: [None, <raises TypeError>, 0]" in shown, shown
    assert "crashed" not in shown, shown

    # A crash of their own on the same run is still named.
    shown = "\n".join(grade("return self._v", problem=PEEK, oracle=oracle).failing_cases)
    assert "x.get() crashed: AttributeError" in shown, shown
    assert "x.peek() crashed" not in shown, shown
