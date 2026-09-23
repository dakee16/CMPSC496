"""test_no_reference_leak.py - nothing we SAY may reveal a reference answer.

THE RULE THIS PINS. ACADIA's whole design rests on the student never learning
that a preferred approach exists. The plan reviewer is deliberately blind to
the teacher's solution so it cannot reject a good-but-different plan; the tutor
is blind so it cannot steer. All of that is undone by one sentence in a message.

An audit on the live site (2026-09-22) found three, and a sweep afterwards
found two more the audit had not reached:

  * "Correct - you named things differently to OUR VERSION, and your step
    works with the rest of the solution."  - and it fired on the ORDINARY
    approach, so it was not even describing an unusual answer;
  * "Correct - your approach DIFFERS FROM OURS, but it works.";
  * "...this takes longer the first time, and longer again if your approach
    DIFFERS FROM OURS." on the loading line, shown before any plan is typed;
  * the explain-button draft, which put "My approach is different from THE ONE
    YOU EXPECTED" into the student's own message box;
  * the button's own label, "My approach is DIFFERENT - let me explain".

Every one was a sentence somebody wrote without thinking about this rule, which
is exactly why it needs a test rather than care. Comments and docstrings are
excluded - this is about what reaches a student, and the reasoning above has to
be writable in the source.
"""
import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parent
# Phrases that assert a reference answer exists, in the second person. "the
# teacher's" in a code comment is fine and necessary; "our version" in a string
# that reaches a student is not.
# \b matters more than it looks: without it "our solution" matches inside
# "Your solution runs but gives the wrong answer", and "our approach" inside
# "what you said your approach would keep track of" - both of which are exactly
# the second-person wording we WANT.
FORBIDDEN = re.compile(
    r"\bour version\b|\bour own version\b|\bdiffers? from ours\b"
    r"|\bdifferent from ours\b|\bthe one you expected\b|\bwe expected\b"
    r"|\bthe expected approach\b|\bour solution\b|\bour approach\b"
    r"|\bthe teacher'?s (answer|solution|approach)\b",
    re.I)


def _python_strings(path):
    """Every string literal in a .py file EXCEPT docstrings."""
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))
    return [(n.lineno, n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstrings]


def _js_without_comments(text):
    """Strip // and /* */ so the reasoning in comments is not mistaken for copy."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(re.sub(r"(^|\s)//.*$", "", line) for line in text.splitlines())


@pytest.mark.parametrize("name", ["grading.py", "tutor.py", "diagnose.py",
                                  "reroute.py", "sessions.py"])
def test_no_python_message_names_a_reference(name):
    path = ROOT / "main" / name
    if not path.exists():
        pytest.skip(f"{name} is not present")
    bad = [(ln, v) for ln, v in _python_strings(path) if FORBIDDEN.search(v)]
    assert not bad, "\n".join(f"  {name}:{ln}: {v!r}" for ln, v in bad)


@pytest.mark.parametrize("name", ["student.js", "workspace.js", "tutorial.js"])
def test_no_page_string_names_a_reference(name):
    path = ROOT / "frontend" / name
    if not path.exists():
        pytest.skip(f"{name} is not present")
    src = _js_without_comments(path.read_text())
    bad = [(i + 1, line.strip())
           for i, line in enumerate(src.splitlines()) if FORBIDDEN.search(line)]
    assert not bad, "\n".join(f"  {name}:{ln}: {t}" for ln, t in bad)


def test_the_guard_itself_catches_the_wording_it_was_written_for():
    """A test that cannot fail guards nothing - these are the real sentences."""
    for sentence in (
            "Correct - you named things differently to our version, and your "
            "step works with the rest of the solution.",
            "Correct - your approach differs from ours, but it works.",
            "this takes longer the first time, and longer again if your "
            "approach differs from ours.",
            "My approach is different from the one you expected."):
        assert FORBIDDEN.search(sentence), sentence
