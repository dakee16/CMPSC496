"""diagnose.py - what to SAY when execution could not confirm a step.

THE PROBLEM THIS EXISTS FOR. Every tier below `incorrect` is now acquit-only
(see main/grading.py), which is what stops correct code being marked wrong. The
cost is that a student whose step is genuinely wrong gets "we could not confirm
this step" and stays on that step: an indeterminate verdict does not advance the
session, so without something else happening they are stuck repeating a mistake
nobody has named. "No verdict AND no way forward" is worse than the false
conviction it replaced.

So the grader stops deciding and starts ASKING. This module builds the question.

THE DIVISION OF LABOUR IS THE WHOLE DESIGN:

    execution chooses the example      deterministic, real, cannot mislead
    execution reports what THEY got    their own code, their own values
    a model writes the sentence        may be wrong; costs nothing

Nothing here can produce a verdict or spend an attempt. A model that writes a
confusing question has cost the student a confusing question. That is why the
model is allowed anywhere near this at all, and why the counterexample itself -
the part a student will actually act on - is never model-authored.

WHAT IS NEVER SHOWN. The reference's own values pick the example and are then
thrown away. Telling a student "ours holds {'cat': 2} at this point" hands over
the container AND its contents - the two things the design gate held them at to
work out for themselves. They are shown THEIR OWN state, which they could have
printed, and asked what it does not let them do.
"""
import json
import re

from . import bridge
from .ollama_client import TUTOR_MODEL, chat

# How many of the student's own names to show beside the example. Enough to see
# what they built, short of dumping a scope.
MAX_SHOWN_NAMES = 4
# Longest input repr worth putting in front of a student. A 300-character string
# is not something anyone traces by hand, and the point is that they can.
MAX_INPUT_REPR = 120


# Sentence shapes that TELL a student what to change. Every one of these was
# produced live by the real model, with the system prompt above already
# forbidding it in three separate sentences - which is the whole argument for
# checking in code rather than asking nicely. "How can you track the frequency
# of each letter instead of just identifying unique ones?" names the target
# state, and no structure-name guard can see it because it names no structure.
#
# The distinction the patterns draw: "how does THIS OUTPUT ..." points at the
# data in front of them and is the question we want; "how can YOU <verb> ..."
# points at an edit and has already said what the edit achieves.
#
# Deliberately over-inclusive. A false positive costs a fall back to our own
# deterministic question, which is a fine thing to show a student; a false
# negative hands over the answer at the exact moment they were stuck enough to
# take it.
_PRESCRIBES = (
    "how can you", "how could you", "how would you", "how do you",
    "how might you", "what can you do", "what could you change",
    "instead of", "adjust your", "change your", "modify your",
    "update your", "make sure", "ensure that", "you should", "you need to",
    "try using", "consider using", "rewrite",
)


def _prescribes_a_fix(message: str) -> bool:
    """Does this sentence tell them what to change, rather than what to look at?"""
    low = (message or "").lower()
    return any(p in low for p in _PRESCRIBES)


def _degenerate(encoded: str) -> bool:
    """Is this captured value empty, absent, or zero-ish?

    Reads bridge's ENCODED form, which is the repr of a structure like
    "{'t': 'dict', 'v': []}" - so an empty container of any kind shows up as an
    empty 'v'. A value like this makes an example mute: there is nothing in it
    to point at."""
    e = (encoded or "").strip()
    return ("'v': []" in e or "'f': []" in e
            or e in ("''", '""', "None", "0", "0.0", "False",
                     repr(bridge.UNBOUND), f"'{bridge.UNBOUND}'"))


# NOTHING INTERNAL REACHES A SCREEN. Two things did, both measured live off one
# submission: `running is __mt_unbound__` and `tally is <function
# frequency.<locals>.tally at 0x1055189a0>`.
#
# UNBOUND is bridge's marker for "this name does not exist at the boundary" - a
# fact about our probe, not a value anyone can read - and _candidates already
# holds that an absence is not a value, so an absent name is DROPPED here rather
# than rendered. An address is stripped rather than dropped: the object is real
# and worth showing, and `<Node object>` says the same thing while staying the
# same sentence twice, which a 0x address does not.
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


# A DEFINITION IS NOT STATE. `def tally(...)` binds `tally` in the student's own
# scope, so bridge.stores is right to report it - but "after your step, `tally`
# is <function tally>" is not a value anyone traces, and it crowds out the
# variable that is. Same judgement loop_targets already makes about `ch`.
#
# ponytail: matched on the repr, because the capture crosses a process boundary
# as text and the child sends nothing else. Upgrade to a type tag in
# bridge._capture_program if this ever has to tell more kinds apart.
_NOT_STATE = ("<function", "<class", "<bound method", "<built-in",
              "<module", "<method")


def _readable(value: str) -> str | None:
    """A captured value as a student may see it, or None if they may not."""
    v = (value or "").strip()
    if not v or v == bridge.UNBOUND or v.startswith(_NOT_STATE):
        return None
    return _ADDRESS.sub("", v)


def _size(inp) -> int:
    try:
        return len(repr(inp))
    except Exception:
        return 1 << 30


def counterexample(problem: dict, header: str, chunks: list, idx: int,
                   upto: str, tests: list, entry: str, ambient: set) -> dict | None:
    """One concrete oracle input worth tracing by hand, with the student's own
    state after it.

    Chooses the SHORTEST input on which the student's state does not supply
    something the remaining reference needs - shortest because the student has
    to be able to follow it in their head, and a counterexample they cannot
    trace teaches nothing. Returns None when there is nothing honest to show:
    recursive code, an oracle with no usable inputs, or a student whose code did
    not run. A METHOD is fine - bridge.capture records one snapshot per CALL, so
    a single input contributes several signature positions, and `owners` maps
    each position back to the input it came from.

    The reference's values are used to CHOOSE and are never returned."""
    if not tests:
        return None
    refs = [(c.get("reference") or "") for c in chunks]
    if not bridge.is_applicable(problem, entry, [upto] + refs):
        return None

    inputs = [t["input"] for t in tests]
    # Loop and comprehension variables are machinery, not state a student would
    # call theirs. Showing `ch = '3'` back to them beside their real variable is
    # noise at the moment they are least able to filter it.
    stu_names = sorted(bridge.stores(upto) - bridge.loop_targets(upto))
    if not stu_names:
        return None                     # they bound nothing; no state to discuss
    stu, owners = bridge.capture(problem, header, upto, stu_names, inputs,
                                 entry, with_owners=True)
    if stu is None:
        return None

    tail = "\n".join(r for r in refs[idx + 1:] if r.strip())
    needed = sorted(bridge.free_names(tail) - ambient)
    if not needed:
        return None
    ref = bridge.capture(problem, header,
                         "\n".join(r for r in refs[:idx + 1] if r.strip()),
                         needed, inputs, entry)
    if ref is None or not owners:
        return None
    # A position is one (input, call) pair. Both sides replay the identical
    # sequence, so the two signatures line up position by position - but an
    # INPUT can own several positions, and it is inputs a student can be asked
    # to try.
    if len(owners) != len(next(iter(ref.values()), ())):
        return None                     # the two runs did not line up; say nothing

    # An input is INTERESTING when some name the tail needs is matched by
    # nothing the student produced. Per-input rather than over the whole
    # signature, so the example is one the student can actually run.
    positions = {}
    for p, owner in enumerate(owners):
        positions.setdefault(owner, []).append(p)
    interesting = []
    for i, inp in enumerate(inputs):
        if _size(inp) > MAX_INPUT_REPR or i not in positions:
            continue
        for p in positions[i]:
            wanted = {ref[n][p] for n in needed}
            theirs = {stu[s][p] for s in stu_names}
            if wanted - theirs:
                interesting.append(i)
                break
    if not interesting:
        return None

    # AN INPUT THE STEP DOES NOTHING ON DEMONSTRATES NOTHING. On the real
    # frequency oracle the shortest interesting input is '1234567890', which
    # contains no letters at all: the step's own answer there is empty, so the
    # student would be asked what is missing from nothing. Rank degenerate
    # inputs last - on BOTH sides, since either being empty makes the example
    # mute - and only then prefer the shortest, so it stays traceable by hand.
    def _rank(i):
        ps = positions[i]
        dead_ref = all(_degenerate(ref[n][p]) for n in needed for p in ps)
        dead_stu = all(_degenerate(stu[n][p]) for n in stu_names for p in ps)
        return (dead_ref or dead_stu, _size(inputs[i]), i)

    pick = min(interesting, key=_rank)
    # Values are re-read in PLAIN REPR for display. The structural encoding that
    # chose this example exists to make two processes comparable and is
    # unreadable to a person: a student shown {'t': 'dict', 'v': [("'a'", '1')]}
    # has been handed our internals instead of their own {'a': 1}.
    human = bridge.capture(problem, header, upto, stu_names, [inputs[pick]],
                           entry, human=True) or {}
    # FILTERED BEFORE CAPPED, so an unshowable name cannot use up one of the
    # four slots and push a real one off the end. A name with nothing readable
    # behind it is dropped entirely - see _readable - and if that leaves nothing,
    # there is no honest example to build, which is the same answer this
    # function already gives a student who bound nothing.
    shown = []
    for n in stu_names:
        v = _readable((human.get(n) or ("",))[0])
        if v is None:
            continue
        shown.append({"name": n, "value": v})
        if len(shown) >= MAX_SHOWN_NAMES:
            break
    if not shown:
        return None
    return {"input": inputs[pick],
            "input_repr": ", ".join(repr(a) for a in inputs[pick]),
            "student_state": shown}


# The model is told what it may talk about and, more importantly, what it may
# not. The rules mirror main/tutor.py's, because a diagnosis that hands over the
# data structure is the same disclosure as a tutor reply that does - and this
# one arrives at a moment when the student is stuck, which is exactly when the
# temptation to just say it is strongest.
_SYSTEM = """\
A student is part-way through a problem. Their code for THIS STEP ran, but we
could not confirm it does what the step asks. You are writing ONE short message
that helps them see it for themselves.

You are given a concrete input and what THEIR OWN variables hold after their
code runs on it. Both are real - they were measured by running their code.

WRITE: two or three sentences, ending in a question. Point them at the input,
say what their own code produced, and ask what that does not let them do next.

NEVER:
- say whether they are right or wrong - you do not know, and nothing you write
  costs them an attempt
- name a data structure, algorithm or built-in they have not already used or
  that the problem statement does not already name
- say what they should have written, or what the next step should build
- write code, in any form, fenced or not
- guess at a fix

AND NEVER DESCRIBE THE CHANGE THEY SHOULD MAKE, EVEN AS A QUESTION. "How can
you adjust your code so that X and Y are treated the same?" is a fix wearing a
question mark: it names the target state, which is the one thing they were
meant to work out. Measured live - the model wrote exactly that sentence and
every other rule here passed, because naming no data structure is not the same
as giving nothing away.

The test is what your last sentence asks them to DO. "What does this output
show you?" sends them to the data. "How can you make it do Z?" has already
told them Z. Ask only about what is in front of them.

The question must be answerable by running their own code and reading the
output. Return STRICT JSON: {"message": "<your two or three sentences>"}"""


# What a student sees when the model is unavailable or gave the answer away.
# It names the example - which is the useful half, and ours, not the model's -
# and asks the same question in fixed words.
def _fallback(example: dict) -> str:
    lines = [f"Try your code on this input: {example['input_repr']}"]
    for f in example["student_state"]:
        lines.append(f"After your step, `{f['name']}` is {f['value']}.")
    lines.append("Does that give the next step everything it needs? Compare it "
                 "with what you said your approach would keep track of.")
    return " ".join(lines)


def question(problem: dict, chunk: dict, student_code: str,
             example: dict, plan_text: str = "") -> str:
    """The message shown to the student. Never raises, never returns empty.

    A model failure is not a student's problem, so every failure path lands on
    _fallback, which is built from the measured example alone."""
    payload = (f"PROBLEM:\n{(problem.get('description') or '')[:600]}\n\n"
               f"THE STEP THEY ARE ON:\n{chunk.get('prompt', '')}\n\n"
               + (f"WHAT THEY SAID THEIR APPROACH WOULD BE:\n{plan_text[:600]}\n\n"
                  if plan_text.strip() else "")
               + f"THEIR CODE FOR THIS STEP:\n{student_code}\n\n"
               f"AN INPUT: {example['input_repr']}\n"
               f"WHAT THEIR OWN VARIABLES HOLD AFTER IT:\n"
               + "\n".join(f"  {f['name']} = {f['value']}"
                           for f in example["student_state"]))
    try:
        raw = chat(TUTOR_MODEL, _SYSTEM, [{"role": "user", "content": payload}],
                   temperature=0, fmt="json")
        msg = str(json.loads(raw).get("message", "")).strip()
    except Exception:
        return _fallback(example)
    if not msg:
        return _fallback(example)
    # THE SAME GUARD THE TUTOR RUNS, for the same reason. `allowed` is
    # everything already on the student's screen: the statement, the step, their
    # own code. Repeating those is not a disclosure; introducing anything else
    # is.
    from .tutor import _handed_over, _strip_code, _structures
    allowed = _structures(f"{problem.get('description') or ''} "
                          f"{chunk.get('prompt') or ''} {student_code}")
    if (_handed_over(msg, allowed) or _strip_code(msg) != msg
            or _prescribes_a_fix(msg)):
        return _fallback(example)
    return msg


if __name__ == "__main__":
    # Self-check.  python -m main.diagnose
    # Real subprocesses for the measurement; the model is never called.
    ind = lambda c: "\n".join("    " + l if l.strip() else l
                              for l in c.splitlines())
    # The REAL shape of this decomposition in the pool: the counting happens in
    # chunk 0 and chunk 1 returns. A fixture whose chunk 0 only initialises has
    # an empty reference state on every input, so it cannot exercise the
    # degenerate-input ranking below at all.
    H, E = "def frequency(txt):", "frequency"
    REFS = ["counts = {}\nfor ch in txt:\n    if ch.isalpha():\n"
            "        counts[ch] = counts.get(ch, 0) + 1",
            "return counts"]
    CH = [{"prompt": "count the letters", "reference": r} for r in REFS]
    P = {"slug": "frequency", "entry_hint": "frequency",
         "description": "Count how many times each letter appears.",
         "solution": H + "\n" + ind("\n".join(REFS))}
    T = [{"input": ["hello world"], "expected": {}},
         {"input": ["aab"], "expected": {}},
         {"input": ["a much longer piece of text than the others"], "expected": {}}]
    AMB = {"txt", "len", "set", "sorted", "sum", "max", "min", "range"}

    # A student who throws away the counts by using a set.
    ex = counterexample(P, H, CH, 0, "unique = set(txt)", T, E, AMB)
    assert ex is not None, "a set loses information the tail needs"
    # The SHORTEST usable input, so they can trace it by hand.
    assert ex["input"] == ["aab"], ex["input"]
    # Their value is READABLE - plain repr, not the structural encoding that
    # exists only to make two processes comparable.
    assert ex["student_state"][0]["value"] == "{'a', 'b'}", ex["student_state"]

    # A DEGENERATE input is ranked last however short it is. On the real
    # frequency oracle the shortest interesting input is '1234567890', which has
    # no letters: the step's own answer there is empty, so the student would be
    # asked what is missing from nothing.
    _T2 = [{"input": ["12"], "expected": {}},          # shortest, but no letters
           {"input": ["aab"], "expected": {}}]
    _ex2 = counterexample(P, H, CH, 0, "unique = set(txt)", _T2, E, AMB)
    assert _ex2 and _ex2["input"] == ["aab"], (_ex2 or {}).get("input")

    # Loop machinery is not shown back to them - `ch` is not their state.
    _ex3 = counterexample(P, H, CH, 0,
                          "seen = set()\nfor ch in txt:\n    seen.add(ch)",
                          T, E, AMB)
    assert _ex3 and [f["name"] for f in _ex3["student_state"]] == ["seen"], _ex3
    # Their own state, and nothing of ours.
    assert [f["name"] for f in ex["student_state"]] == ["unique"], ex
    assert "counts" not in json.dumps(ex), "the reference's names must not leak"

    # ── NO INTERNAL VALUE REACHES A STUDENT ──────────────────────────────
    # Both of these were on screen, from one submission. `running` and `c` are
    # locals of the student's own HELPER, which the outer frame never binds, so
    # they came back as the probe's UNBOUND marker; `tally` is the helper
    # itself, whose repr carries an address that differs every run.
    _helper = ("def tally(s):\n"
               "    running = 0\n"
               "    for c in s:\n"
               "        running += 1\n"
               "    return running\n"
               "size = tally(txt)")
    _hex = counterexample(P, H, CH, 0, _helper, T, E, AMB)
    assert _hex is not None, "a helper-using submission is still diagnosable"
    _blob = json.dumps(_hex)
    assert bridge.UNBOUND not in _blob, _blob
    assert " at 0x" not in _blob, _blob
    # ...and the helper's locals are not the student's state at all - they
    # belong to another scope (bridge.stores), so they are never offered.
    assert [f["name"] for f in _hex["student_state"]] == ["size"], _hex
    assert bridge.UNBOUND not in _fallback(_hex)

    # An object with no __repr__ keeps its class and loses its address, so the
    # same submission twice produces the same sentence twice.
    assert _readable("<Node object at 0x104e37770>") == "<Node object>"
    assert _readable(bridge.UNBOUND) is None
    # A definition is not state, however readable its repr is made.
    assert _readable("<function f at 0xdeadbeef>") is None
    assert _readable("<class 'Node'>") is None
    assert _readable("") is None and _readable("   ") is None
    # A value that carries no address is untouched - the common case.
    assert _readable("{'a': 2}") == "{'a': 2}"

    # ── a prescribed fix is refused in CODE, not asked against ───────────
    # Every sentence below came back from the real model with the system prompt
    # already forbidding it. That is the reason this is a function and not a
    # paragraph: an instruction to a model is a request it can decline.
    for _bad in (
            "How can you track the frequency of each letter instead of just "
            "identifying unique ones?",
            "How can you adjust your code to ensure that 'T' and 't' are "
            "counted together as the same letter?",
            "You should keep a running total for each letter.",
            "Consider using something that maps letters to numbers.",
            "Try using a different approach here."):
        assert _prescribes_a_fix(_bad), _bad
    # ...and the question we actually want is not caught by it. It points at
    # the output in front of them rather than at an edit.
    for _good in (
            "For the input 'aab', your code produces {'a', 'b'}. How does that "
            "help you keep track of how many times each letter appears?",
            "Your step leaves `seen` holding {'a', 'b'}. What does that tell "
            "you about the text it came from?",
            "After your code runs on 'aab', what is in `unique`?"):
        assert not _prescribes_a_fix(_good), _good

    # The fallback is usable on its own - it is what a model outage falls to.
    fb = _fallback(ex)
    assert "'aab'" in fb and "`unique`" in fb, fb
    assert "dict" not in fb.lower() and "counts" not in fb, fb

    # Nothing to say, said honestly, rather than something invented:
    assert counterexample(P, H, CH, 0, REFS[0], T, E, AMB) is None, \
        "a student whose state DOES supply the tail has no counterexample"
    # ...and a rename is still no counterexample: the values match, only the
    # name differs, which is the bridge's business and not a fault to discuss.
    assert counterexample(P, H, CH, 0, REFS[0].replace("counts", "freq"),
                          T, E, AMB) is None
    assert counterexample(P, H, CH, 0, "pass", T, E, AMB) is None, \
        "a student who bound nothing has no state to discuss"
    assert counterexample(P, H, CH, 0, "unique = set(txt)", [], E, AMB) is None
    # ── A METHOD GETS ONE TOO ────────────────────────────────────────────
    # It used to be declined here for the same reason bridge declined it. One
    # snapshot per CALL, `owners` mapping positions back to inputs, and the rest
    # is identical - a student is still asked about an INPUT, never a call index.
    _mh = "def push(self, value):"
    _mp = {"slug": "stack-push", "group_title": "Stack", "context_indent": 8,
           "context_suffix": "\n", "description": "Add a value to the stack.",
           "context_prefix": "class Node:\n    def __init__(self, v):\n"
                             "        self.value = v\n        self.next = None\n\n"
                             "class Stack:\n    def __init__(self):\n"
                             "        self.top = None\n\n" + "    " + _mh + "\n",
           "solution": "pass"}
    _mch = [{"prompt": "make the new node", "reference":
             "node = Node(value)\nnode.next = self.top"},
            {"prompt": "put it on top", "reference": "self.top = node"}]
    _mt = [{"input": [[["new"], ["push", 2]]], "expected": [None, None]},
           {"input": [[["new"], ["push", 2], ["push", 4]]],
            "expected": [None, None, None]}]
    _mamb = {"self", "value", "Node", "Stack", "calls"}
    # A student who never builds the node the next step puts on top.
    _mex = counterexample(_mp, _mh, _mch, 0, "spare = value", _mt,
                          "_mt_run_calls", _mamb)
    assert _mex is not None, "a method must be diagnosable too"
    assert [f["name"] for f in _mex["student_state"]] == ["spare"], _mex
    # The SHORTEST input, counted as an input and not as a call position.
    assert _mex["input"] == [[["new"], ["push", 2]]], _mex["input"]
    # ...and a student who DOES build it has nothing to be asked about.
    assert counterexample(_mp, _mh, _mch, 0,
                          "node = Node(value)\nnode.next = self.top", _mt,
                          "_mt_run_calls", _mamb) is None
    # A rename is the bridge's business, not a fault to discuss.
    assert counterexample(_mp, _mh, _mch, 0,
                          "n = Node(value)\nn.next = self.top", _mt,
                          "_mt_run_calls", _mamb) is None

    print("diagnose.py self-check OK")
