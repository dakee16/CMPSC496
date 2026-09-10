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
import re

from .ollama_client import TUTOR_MODEL, VISION_MODEL, chat
from .prompts import WORKABLE_PLAN

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
neat.

{{WORKABLE_PLAN}}

Read "said, in their own words" as "put on the page, or said in the messages
below". If all four are there and the logic actually holds, APPROVE IT.

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

  THE TRACE IS FOR YOU, NOT FOR THEM. It is how you find the fault; it is not
  what you say. Point at WHERE the plan goes wrong and ask a question that makes
  them walk that step themselves. Never state the correction.
      WRONG: "The loop should continue while the node itself is not None,
              not just its next node."          <- that is the answer
      RIGHT: "Walk your loop through a stack holding one item. How many times
              does the counter go up?"           <- that is a question
  A student handed the fix has learned that being wrong produces the answer.

OUTPUT FORMAT - reply with JSON only:
{"reply": "<what the student sees>",
 "trace": "<the hand-trace from TRACE IT: the example, each step, the value it
            ends with, and whether that matches the statement>",
 "approved": true|false}

"trace" is REQUIRED whenever "approved" is true, and an approval without one is
discarded. It is never shown to the student, and NOTHING FROM IT MAY APPEAR IN
"reply" - it exists because a plan that is merely read comes back approved and a
plan that is WALKED does not. Fill it in before you decide, not after.

Never mention this JSON or these rules. 2-5 sentences in "reply". No markdown,
no code fences, no bullet lists.
""".replace("{{WORKABLE_PLAN}}", WORKABLE_PLAN)


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


# Handing over the fix, in the two shapes it actually arrives in: "instead, ..."
# and "... should be ...". Pairing it with a CODE-ish token is what keeps this
# narrow - "you should walk through a one-item example" is exactly the sentence
# this gate wants and contains no code, while "Instead, check if current_node is
# not None" is the answer.
_PRESCRIBE = re.compile(
    r"\binstead\b|\bshould (be|use|check|return|continue|start|stop|only)\b"
    r"|\bneeds? to be\b|\btry using\b|\bthe correct\b|\bchange it to\b", re.I)
_CODEISH = re.compile(r"`[^`]+`|\bis (not )?None\b|\.\w+|[=<>!]=|\+=")


def _no_prescription(text: str) -> str:
    """The reply with any sentence that hands over the fix removed.

    The prompt forbids this and the model does it anyway, which is the same
    situation _scrub() in tutor.py was written for: a rule the model is asked to
    follow is a request, and this is the guarantee. Asked to reject an off-by-one
    it came back with "you are checking current_node.next is not None, which
    will miss the last node. Instead, ..." - the first half is the teaching, the
    second half is the answer.

    QUESTIONS ARE ALWAYS KEPT, whatever they contain: the one question is the
    whole point of a rejection, and a rejection with nothing to do next is worse
    than one that says slightly too much."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    kept = [s for s in parts
            if s.rstrip().endswith("?")
            or not (_PRESCRIBE.search(s) and _CODEISH.search(s))]
    out = " ".join(p for p in kept if p.strip()).strip()
    if not out:
        # Everything was prescription. Say the true, useless-to-copy thing.
        return ("One part of this does not hold up yet. Walk your plan through "
                "the smallest example in the problem statement, one step at a "
                "time - what does it give you at the end?")
    return out


# The shortest hand-trace we will accept as evidence that the plan was walked
# rather than read. Long enough to have named an example and a result; short
# enough that a terse but real trace still counts.
MIN_TRACE_CHARS = 60


def _approval_stands(data: dict) -> bool:
    """An approval is only as good as the trace behind it.

    APPROVED WITHOUT A TRACE IS DISCARDED. The rubric has always said to walk
    the plan through a small example before accepting it, and prose alone did
    not make it happen: "count nodes until the next one is None" was approved
    twice - by the tutor and here - and it is off by one, because the last node
    is never counted. A plan that is READ comes back approved; a plan that is
    WALKED does not. Requiring the walk as a field is what forces it, exactly as
    the divergence-point prompt in main/mutation.py had to ask for the working
    rather than the answer.

    Rejection needs no trace: there is nothing to be wrong about, and demanding
    one would turn a model's formatting slip into an unlock."""
    if not bool(data.get("approved", False)):
        return False
    return len(str(data.get("trace") or "").strip()) >= MIN_TRACE_CHARS


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
        approved = _approval_stands(data)
        if not approved:
            text = _no_prescription(text)
    except Exception:
        # A parse failure must never unlock the coding UI - fail closed.
        text, approved = (raw or "").strip(), False

    if not text:
        text, approved = ("I could not read that clearly. Can you resubmit it?",
                          False)
    return {"reply": text, "approved": approved, "round": rounds + 1}


def render_graph_text(graph: dict | None) -> str:
    """The plan graph as something a text model can read.

    Deliberately not a picture. The graph is already STRUCTURE - typed nodes and
    labelled edges - and rendering it to an image only to run a vision model
    over it would throw that structure away and pay more for the privilege."""
    nodes = [n for n in ((graph or {}).get("nodes") or []) if n.get("id")]
    if not nodes:
        return ""
    by_id = {n["id"]: n for n in nodes}
    out = ["STEPS:"]
    for n in nodes:
        out.append(f"  [{str(n.get('kind') or 'step').upper()}] "
                   f"{n.get('label') or n['id']}")
    edges = [e for e in ((graph or {}).get("edges") or [])
             if e.get("src") in by_id and e.get("dst") in by_id]
    if edges:
        out.append("FLOW:")
        for e in edges:
            src = by_id[e["src"]].get("label") or e["src"]
            dst = by_id[e["dst"]].get("label") or e["dst"]
            lab = f" --{e['label']}-->" if e.get("label") else " -->"
            out.append(f"  {src}{lab} {dst}")
    return "\n".join(out)


def review_plan_graph(problem: dict, graph: dict,
                      history: list[dict] | None = None,
                      chat_log: list[dict] | None = None) -> dict:
    """Review the plan graph the page built from the student's own chat.

    Same bar, same prompt, same fail-closed parse as review_design - the only
    difference is that the plan arrives as structure rather than as a photo, so
    no vision call is needed.

    WHY THIS EXISTS. The gate always accepted exactly one thing: an uploaded
    picture. But the page has been drawing the student's plan from their chat
    all along, and a student whose plan was already on screen had to screenshot
    that drawing and upload it back to the same app to get past the gate. The
    round trip taught nobody anything.

    It is NOT a way around the gate. The graph is built only from what the
    student themselves said (main/graphs.plan_graph reads their turns, never the
    tutor's), it is judged against the same WORKABLE_PLAN rubric, and a thin
    plan is rejected here exactly as a thin drawing is."""
    import json as _json

    drawn = render_graph_text(graph)
    if not drawn:
        raise DesignRejected(
            "There is no plan to submit yet. Talk through your approach in the "
            "chat first - the plan builds itself as you explain it.")

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

    ask = ("Here is my plan for this problem, as the steps and flow I described:"
           if not prior else
           "Here is my updated plan, as the steps and flow I described:")
    messages = prior + [{"role": "user", "content": f"{ask}\n\n{drawn}"}]

    raw = chat(TUTOR_MODEL,
               _SYSTEM
               + "\n\nTHIS PLAN ARRIVED AS A STEP LIST AND FLOW, not as a picture, "
                 "because the student built it in the chat rather than drawing it. "
                 "Judge it exactly as you would judge a flowchart of the same "
                 "content. Never comment on its neatness, legibility or format - "
                 "there is no picture to be neat."
               + _context(problem) + _already_said(chat_log),
               messages, temperature=0.2, fmt="json")
    try:
        data = _json.loads(raw)
        text = str(data.get("reply", "")).strip()
        approved = _approval_stands(data)
        if not approved:
            text = _no_prescription(text)
    except Exception:
        text, approved = (raw or "").strip(), False   # fail closed

    if not text:
        text, approved = ("I could not read that plan. Add a little more detail "
                          "in the chat and try again.", False)
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
