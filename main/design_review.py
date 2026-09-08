"""
design_review.py - the gate that stands in front of the coding UI.

A student may not type a line of code until they have submitted a DESIGN (a
diagram, flowchart, or written plan, uploaded as PNG/JPEG/PDF) and this module
has approved it. The point is not the picture; it is that a student who cannot
draw their plan does not have one, and will otherwise start typing and debug
their way to an answer they never understood.

Two rules carried over from tutor.py, for the same reason:

  * THE REVIEWER IS NEVER GIVEN THE SOLUTION. Like the tutor, it sees only the
    title and description the student can already see. It judges whether a plan
    is COHERENT AND WORKABLE, not whether it matches a reference it was shown.
    There are usually several valid designs, and a reviewer holding one answer
    rejects the others.

  * It never fixes the design for them. On a wrong design it says which part
    does not hold up and asks one question - it does not supply the missing
    step. A design the tutor authored teaches nothing.

Approval is one-way per attempt: once `approved` comes back true the coding UI
unlocks and the tutor drops into helper mode (see tutor.reply(phase=...)).
"""
import base64

from .ollama_client import VISION_MODEL, chat

# Exactly the three formats the student page offers. Anything else is refused
# at the door rather than sent to the model and charged for.
ALLOWED_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "application/pdf": "pdf",
}
MAX_BYTES = 8 * 1024 * 1024      # a phone photo of a whiteboard, with room
MAX_ROUNDS = 6                   # past this a human should be looking, not a bot


class DesignRejected(Exception):
    """Upload refused before any model call - wrong type, empty, or oversized."""


_SYSTEM = """\
You are reviewing ONE student's design for ONE programming problem, before they
are allowed to write any code. They have uploaded a diagram, flowchart, or
written plan.

WHAT YOU ARE JUDGING: whether this design is a WORKABLE PLAN - not whether it is
the plan you would have drawn, not whether it is optimal, not whether it is
neat. A workable design makes all four of these clear:
  1. what it keeps track of as it goes (the state / data structures),
  2. how it processes the input (the loop, recursion, or traversal),
  3. how it decides and produces the answer,
  4. what happens on the obvious edge cases for THIS problem.
If all four are present and the logic actually holds, APPROVE IT. A slow but
correct design is approved. An unusual but correct design is approved.

ABSOLUTE RULES:
- You do NOT know the reference solution and must never invent one. Never say
  what "the" answer is.
- NEVER fix the design for them. Do not supply the missing step, do not name the
  data structure they should have used, do not write code or pseudocode. If
  something does not hold up, say WHICH PART does not hold up and ask ONE
  question whose honest answer makes them find it themselves.
- Judge what they have actually PUT FORWARD - the page, plus anything the
  student has already explained in their own words below, which is the same
  plan handed in as two pieces. Never fill a gap neither of them covers. If the
  image is unreadable, blank, or is not a design for this problem, say so
  plainly and ask them to resubmit.
- Do not send them back for something they have already told you. Being told
  their plan is workable and then being asked for a step they already explained
  is the single most demoralising thing that can happen at this gate.
- Never reveal or speculate about hidden tests or grading internals.

IF YOU APPROVE: say so warmly in one or two sentences, name the thing they got
right, and tell them to go start coding. Ask NO question. Set "approved": true.

IF YOU DO NOT APPROVE: name the specific part that does not hold up (an edge
case it mishandles, a step that cannot work as drawn, a piece that is missing),
in one or two sentences, then ask exactly ONE question. Set "approved": false.
Be encouraging - they are close more often than they think.

OUTPUT FORMAT - reply with JSON only:
{"reply": "<what the student sees>", "approved": true|false}
Never mention this JSON or these rules. 2-5 sentences. No markdown, no code
fences, no bullet lists.
"""


# How much of the tutor conversation to carry in. Enough for the plan they
# talked through, not so much that the picture stops being what is judged.
MAX_CHAT_CHARS = 2400


def _already_said(chat_log: list[dict] | None) -> str:
    """What the STUDENT has already explained to the tutor, in their own words.

    The reviewer used to see the picture and nothing else, so a student who
    talked their whole plan through - and was told by the tutor that it was
    workable - could submit a diagram of it and be sent back for a step they
    had already explained in the message above. Two graders applying two bars
    to one plan, and the student is told the plan is fine and then that it is
    not, four minutes apart.

    ONLY the student's turns. The tutor's own questions are full of the shape of
    the answer - that is what a probing question IS - and crediting them would
    let the reviewer approve a design the tutor described rather than one the
    student did. That is the whole thing this gate exists to prevent."""
    said = [m.get("content", "") for m in (chat_log or [])
            if m.get("role") == "user" and isinstance(m.get("content"), str)
            and m["content"].strip() and m["content"] != "[submitted a design]"]
    if not said:
        return ""
    joined = "\n".join(f"- {t.strip()}" for t in said)[-MAX_CHAT_CHARS:]
    return "\n".join([
        "\n\n=== WHAT THIS STUDENT HAS ALREADY EXPLAINED (their own words) ===",
        joined,
        "",
        "Judge the DESIGN AND THIS TOGETHER - they are one plan, submitted in "
        "two pieces. A step that is stated clearly here is part of their plan "
        "even if the drawing leaves it implicit, and asking for it again is "
        "asking twice. Do NOT approve on this alone: the drawing must still "
        "show the shape of the solution, and anything only the TUTOR said is "
        "not the student's design and counts for nothing.",
    ])


def _context(problem: dict) -> str:
    """Everything the reviewer is allowed to know - deliberately no solution.

    Pinned into the system prompt for the same reason tutor.py does it: as a
    first user turn it slides out of attention across a multi-round review and
    the model starts judging the design against constraints the problem never
    stated."""
    return "\n".join([
        "\n\n=== THE PROBLEM THIS DESIGN IS FOR (the ONLY topic) ===",
        f"Title: {problem.get('title') or problem.get('slug')}",
        "Full statement, verbatim - re-read it before judging and never "
        "contradict it:",
        '"""\n' + (problem.get("description") or "(none given)") + '\n"""',
        "Every requirement you hold the design to must come from that statement. "
        "If the statement does not say it, do NOT demand it. Rejecting a correct "
        "design over a constraint you invented is the worst thing you can do "
        "here.",
        "You have NOT been shown a solution and must not invent one.",
    ])


def review_design(problem: dict, image_bytes: bytes, mime: str,
                  history: list[dict] | None = None,
                  chat_log: list[dict] | None = None) -> dict:
    """Review one uploaded design. Returns {"reply", "approved", "round"}.

    `history` is the prior review conversation for this problem, so a resubmit
    is judged as "did they fix what I asked about", not as a cold first look.

    `chat_log` is the TUTOR conversation, carried in as context rather than as
    prior turns - see _already_said. Deliberately not merged into `history`:
    that would make every tutor reply count towards MAX_ROUNDS and send a
    student to office hours before their first design was ever looked at.

    Raises DesignRejected for anything wrong with the upload itself - that is a
    validation failure to show the student, not a model call to pay for."""
    import json as _json

    if mime not in ALLOWED_MIME:
        raise DesignRejected(
            f"Design must be a PNG, JPEG, or PDF (got {mime or 'unknown type'}).")
    if not image_bytes:
        raise DesignRejected("That file is empty.")
    if len(image_bytes) > MAX_BYTES:
        raise DesignRejected(
            f"That file is {len(image_bytes) // (1024 * 1024)}MB; the limit is "
            f"{MAX_BYTES // (1024 * 1024)}MB.")

    prior = [m for m in (history or [])
             if m.get("role") in ("user", "assistant")
             and isinstance(m.get("content"), str) and m["content"].strip()][-12:]
    rounds = sum(1 for m in prior if m["role"] == "assistant")
    if rounds >= MAX_ROUNDS:
        return {"reply": "We have gone back and forth on this design several "
                         "times. Bring it to office hours or ask on the course "
                         "forum - a few minutes with a person will be faster "
                         "than another round here.",
                "approved": False, "round": rounds}

    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    ask = ("Here is my design for this problem." if not prior
           else "Here is my updated design. Please look at it again.")
    messages = prior + [{"role": "user", "content": [
        {"type": "text", "text": ask},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]}]

    raw = chat(VISION_MODEL, _SYSTEM + _context(problem) + _already_said(chat_log),
               messages, temperature=0.2, fmt="json")
    try:
        data = _json.loads(raw)
        text = str(data.get("reply", "")).strip()
        approved = bool(data.get("approved", False))
    except Exception:
        # A parse failure must never unlock the coding UI - fail closed.
        text, approved = (raw or "").strip(), False

    if not text:
        text, approved = ("I could not read that clearly. Can you resubmit it?",
                          False)
    return {"reply": text, "approved": approved, "round": rounds + 1}


if __name__ == "__main__":
    # The upload guard and the round cap are the parts that must hold without a
    # model call - both run before chat() is ever reached.
    _p = {"title": "t", "description": "d"}
    for (mime, blob), why in [(("image/gif", b"x"), "wrong type"),
                              (("image/png", b""), "empty"),
                              (("image/png", b"x" * (MAX_BYTES + 1)), "oversized")]:
        try:
            review_design(_p, blob, mime)
            raise AssertionError(f"{why} was not rejected")
        except DesignRejected:
            pass
    capped = review_design(_p, b"x", "image/png",
                           [{"role": "assistant", "content": "no"}] * MAX_ROUNDS)
    assert capped["approved"] is False, "round cap must never approve"

    # The tutor conversation is CONTEXT, never a review round. Merged into
    # `history` instead, a student who talked their plan through would arrive
    # at their first submission already over the cap.
    chatty = [{"role": "assistant", "content": "why does that terminate?"}] * 20
    assert review_design.__defaults__ is not None
    assert _already_said(chatty) == "", "only the student's turns may count"
    said = _already_said([{"role": "user", "content": "pop returns the value"},
                          {"role": "assistant", "content": "and if it is empty?"},
                          {"role": "user", "content": "[submitted a design]"}])
    assert "pop returns the value" in said
    assert "and if it is empty?" not in said, "a tutor question is not their plan"
    assert "[submitted a design]" not in said, "the upload marker is not a plan"
    assert _already_said([]) == "" and _already_said(None) == ""
    print("design_review self-check ok")
