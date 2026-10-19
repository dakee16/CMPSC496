"""test_handback_after_edit.py - the student's file still builds after ONE
problem of a class was edited on its own.

THE REPORT (18 Oct, GPT's HW4 round 2). "Full file" said "Your file could not
be loaded just now" on every try. Reproduced on the real rows:
HandbackUnsafe "could not blank 'cache-list-put'". Each problem stores the
whole file split at its own method; editing put, update and __contains__ on the
review page re-split only THEIR copies, so their positions no longer matched
the file the builder rebuilds from a sibling - and it refused (rightly) rather
than risk leaving an answer in. The end-of-term download, which does not
refuse, would have silently kept the teacher's put over the student's own.

REAL: parse_assignment_file (as an upload and as the review page's edit
re-parse the class) and build_handback. Synthetic class; no teacher solution.
"""
import pytest

from main.assignments import parse_assignment_file
from main.handback import STUB_MARK, build_handback

SRC = '''"""Shelves"""


class Shelf:
    """
        >>> s = Shelf(2)
        >>> s.add('pen')
        'ADDED'
    """
    # --- steps: add, take ---
    def __init__(self, size):
        self.size = size
        self.items = []

    def add(self, name):
        """Put name on the shelf if there is room."""
        if len(self.items) >= self.size:
            return 'FULL'
        self.items.append(name)
        return 'ADDED'

    def take(self, name):
        """Take name off the shelf."""
        if name not in self.items:
            return 'MISSING'
        self.items.remove(name)
        return 'TAKEN'
'''
# The review page's edit: three example lines added to add's docstring.
EDITED = SRC.replace('        """Put name on the shelf if there is room."""',
                     '        """Put name on the shelf if there is room.\n\n'
                     '        >>> s = Shelf(1)\n        >>> s.add(\'a\')\n        \'ADDED\'\n        """')
# ...and a ONE-line growth: the stored place then still falls inside the file,
# shifted - the case that would cut the wrong lines without a word.
NUDGED = SRC.replace('        """Put name on the shelf if there is room."""',
                     '        """Put name on the shelf\n        if there is room."""')
ADD_BODY = "self.items.append(name)"
TAKE_BODY = "self.items.remove(name)"


def _problems(edited_add):
    base = {p["entry_hint"]: p for p in parse_assignment_file(SRC, "shelf.py")["problems"]}
    if edited_add:
        src = NUDGED if edited_add == "nudged" else EDITED
        base["add"] = next(p for p in parse_assignment_file(src, "shelf.py")["problems"]
                           if p["entry_hint"] == "add")
    out = [base["take"], base["add"]]              # a sibling first, as in HW4
    for i, p in enumerate(out):
        p["order"] = i
    return out


@pytest.mark.parametrize("edited_add", [False, True, "nudged"])
def test_the_blank_file_builds_with_every_answer_taken_out(edited_add):
    text = build_handback(_problems(edited_add), {}, assignment_name="Shelves",
                          blank_unanswered=True)
    assert ADD_BODY not in text and TAKE_BODY not in text, text
    assert "if len(self.items) >= self.size:" not in text, text
    assert sum(ln.strip() == STUB_MARK for ln in text.splitlines()) == 2, text
    compile(text, "<file>", "exec")


def test_a_students_answer_to_the_edited_problem_is_the_one_in_their_file():
    own = "if len(self.items) < self.size:\n    self.items.append(name)\n    return 'ADDED'\nreturn 'FULL'"
    for blank in (True, False):                  # the Full file view, and the download
        text = build_handback(_problems(True), {"shelf-add": own}, assignment_name="Shelves",
                              blank_unanswered=blank)
        assert "if len(self.items) < self.size:" in text, (blank, text)
        assert "if len(self.items) >= self.size:" not in text, (blank, text)
        compile(text, "<file>", "exec")
