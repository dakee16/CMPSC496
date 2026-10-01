"""
step_notes.py - two plain lines under every step: what it starts with, and
what it must leave for the next one.

WHY (30 Sep). A student wrote to Dr. Saha: "These kinds of problems seem a
little difficult to know what the problem wants." The question under a step
deliberately says only WHAT to achieve - and for a step that hands work on, it
never said what had to be handed on. Two students answered "prepare everything
needed" with the set-up alone. The page now also says where a step sits in a
loop (sessions.loop_note - derived, free, always right); these notes say what
goes in and what comes out.

ONE model call per roadmap (~1c), at upload (run_phase1.fill_pool) and once for
the roadmaps already saved (main.fix_pools). Checked by gates.check_notes - the
prompt gate's rules: no method, no name only the solution uses. A failure buys
a retry with the gate's own reasons; after the last one the roadmap is kept
WITHOUT notes - notes are help, and a roadmap without them still works.
"""
import json

from .context import header_of
from .gates import check_notes
from .ollama_client import DECOMPOSE_MODEL, chat
from .prompts import STEP_NOTES_SYSTEM


def _ask(problem: dict, chunks: list, rejected: str = "") -> list | None:
    steps = "\n\n".join(
        f"STEP {i + 1}{' (LAST)' if i == len(chunks) - 1 else ''}\n"
        f"QUESTION: {c.prompt}\nCODE:\n{c.reference or ''}"
        for i, c in enumerate(chunks))
    msg = (f"PROBLEM: {problem.get('title') or problem.get('slug')}\n"
           f"{problem.get('description') or ''}\n\nMETHOD: {header_of(problem) or ''}\n\n"
           f"{steps}")
    if rejected:
        msg += f"\n\nYOUR LAST NOTES WERE REJECTED:\n{rejected}\nWrite them again."
    try:
        notes = json.loads(chat(DECOMPOSE_MODEL, STEP_NOTES_SYSTEM,
                                [{"role": "user", "content": msg}],
                                temperature=0.2, fmt="json")).get("notes")
    except (ValueError, AttributeError):
        return None
    if not isinstance(notes, list) or len(notes) != len(chunks) \
            or not all(isinstance(n, dict) for n in notes):
        return None
    return notes


def write_notes(problem: dict, decomposition: dict, tries: int = 3) -> dict:
    """`decomposition` with notes on every step, or unchanged if no try passed
    the gate. The code and the questions are never touched."""
    chunks = decomposition["chunks"]
    rejected = ""
    for _ in range(tries):
        try:
            notes = _ask(problem, chunks, rejected)
        except Exception as e:
            # The provider, not the notes: stop asking, keep the roadmap.
            print(f"  ⚠️  step notes skipped for {problem.get('slug')}: {str(e)[:120]}")
            return decomposition
        if notes is None:
            rejected = "Reply with exactly one {starts_with, leaves} entry per step."
            continue
        noted = [c.model_copy(update={"starts_with": str(n.get("starts_with") or "").strip(),
                                      "leaves": str(n.get("leaves") or "").strip()})
                 for c, n in zip(chunks, notes)]
        verdict = check_notes(noted, problem, decomposition.get("header") or "")
        if verdict["status"] == "pass":
            return {**decomposition, "chunks": noted}
        rejected = verdict["summary"]
    return decomposition
