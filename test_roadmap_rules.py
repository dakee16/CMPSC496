"""test_roadmap_rules.py - what may be built into a roadmap, and served from one.

THE REPORT (13 Oct, HW4, live). One of __contains__'s roadmaps could not be
finished by anyone: when the model split the teacher's method it REWROTE it,
ending `return result if 'result' in locals() else False`; student code may not
use locals(), every step is graded with the later steps behind it, so even the
teacher's own step 1 was refused. 13 of 48 roadmaps had step questions that
failed the wording gate and were served anyway, and one was lopsided.

Sanan's rules (13 Oct): a teacher's roadmap is the teacher's code cut into
pieces, never rewritten; a method too short to cut is ONE step of it; nothing a
student may not use; no unclear questions; three roadmaps a problem.

REAL: gates.roadmap_faults, the builder (fill_pool) and its filter (_usable),
the splitter, the serve gate, the grading route, reroute.build. FAKED: the
model (canned splits and questions) and Supabase.
"""
import json

import pytest

from main import gates, run_phase1, splitter
from test_roadmap_pool_cost import PROBLEM, ROADMAP, SOLUTION, pool  # noqa: F401

LOCALS = {"header": ROADMAP["header"], "chunks": [
    ROADMAP["chunks"][0],
    {**ROADMAP["chunks"][1], "reference": "return counts if 'counts' in locals() else {}"}]}
REWRITE = {"header": ROADMAP["header"], "chunks": [
    {**ROADMAP["chunks"][0], "reference": ROADMAP["chunks"][0]["reference"].replace("ch", "c")},
    ROADMAP["chunks"][1]]}
UNCLEAR = {"header": ROADMAP["header"], "chunks": [
    {**ROADMAP["chunks"][0], "prompt": "Tally the characters."}, ROADMAP["chunks"][1]]}


def _d(r):
    return run_phase1._deserialize(r)


def test_each_fault_is_named_and_a_clean_cut_has_none():
    assert gates.roadmap_faults(PROBLEM, _d(ROADMAP)) == []
    assert gates.roadmap_faults(PROBLEM, _d(REWRITE)) == ["it rewrites the teacher's code"]
    assert any("students may not" in f and "locals" in f
               for f in gates.roadmap_faults(PROBLEM, _d(LOCALS)))
    assert gates.roadmap_faults(PROBLEM, _d(UNCLEAR)) == ["its step questions are unclear"]
    # A comment or a blank line is not a rewrite.
    spaced = {**ROADMAP, "chunks": [{**ROADMAP["chunks"][0], "reference":
              "# start a tally\n" + ROADMAP["chunks"][0]["reference"] + "\n"}, ROADMAP["chunks"][1]]}
    assert gates.roadmap_faults(PROBLEM, _d(spaced)) == []


def test_a_roadmap_students_cannot_finish_is_never_served(pool):
    """Even when it is the only one with clear questions."""
    pool.write([LOCALS, UNCLEAR])
    served = run_phase1._usable(dict(PROBLEM), pool.entries())
    assert served == [UNCLEAR], "the runnable one, unclear or not, beats one nobody can pass"
    pool.write([LOCALS, UNCLEAR, ROADMAP])
    assert run_phase1._usable(dict(PROBLEM), pool.entries()) == [ROADMAP], "clear ones first"
    pool.write([LOCALS])
    with pytest.raises(run_phase1.DecompositionUnavailableError):
        pool.open()


def test_old_rewrites_are_still_served_until_re_prepared(pool):
    """HW3's one-line methods only have rewrites saved; they work, and must
    keep working until re-prepared into one step of the teacher's code."""
    pool.write([REWRITE])
    assert run_phase1._usable(dict(PROBLEM), pool.entries()) == [REWRITE]


def test_a_rewrite_is_never_saved_and_the_teachers_code_is_cut_instead(pool, monkeypatch):
    built = []
    monkeypatch.setattr(run_phase1, "decompose_into_chunks",
                        lambda p, *a, **k: built.append(1) or _d(REWRITE))
    monkeypatch.setattr(splitter, "write_prompts", lambda p, d, tries=3: {**d, "chunks": [
        c.model_copy(update={"prompt": q}) for c, q in zip(d["chunks"], (
            "Tally the characters, keeping the tally for the next step.",
            "Hand back the tally.", "Hand back the tally."))]})
    pool.saved(0)
    n = run_phase1.fill_pool(dict(PROBLEM))
    saved = pool.entries()
    assert built == [1], "a refused build stops the model; it is not retried per roadmap"
    assert 1 <= n == len(saved) <= 3
    assert all(gates.roadmap_faults(PROBLEM, _d(e)) == [] for e in saved)
    assert not any(e["chunks"][0]["reference"] == REWRITE["chunks"][0]["reference"] for e in saved)


ONE_LINER = {"slug": "is-blank", "title": "Is blank", "entry_hint": "is_blank",
             "description": "Return True when txt is empty or only spaces, and False otherwise.",
             "solution": "def is_blank(txt):\n    return txt.strip() == ''\n"}
BLANK_TESTS = [{"input": [x], "expected": x.strip() == ""} for x in ("", "  ", "a", " b ", "\t")]


def test_a_method_too_short_to_cut_is_one_step_of_the_teachers_code(pool, monkeypatch):
    from main import identity
    monkeypatch.setattr(run_phase1, "decompose_into_chunks",
                        lambda p, *a, **k: _d({"header": "def is_blank(txt):", "chunks": [
                            {"step_id": "Part 1", "prompt": "Trim it, keeping the trimmed text.",
                             "expected_type": "code", "reference": "t = txt.strip()"},
                            {"step_id": "Part 2", "prompt": "Say whether nothing is left.",
                             "expected_type": "code", "reference": "return t == ''"}]}))
    run_phase1._add_to_pool(identity.content_hash(ONE_LINER), [], replace=True)
    assert run_phase1.fill_pool(dict(ONE_LINER)) == 1
    (only,) = run_phase1._load_pool()[identity.content_hash(ONE_LINER)]
    assert [c["reference"] for c in only["chunks"]] == ["return txt.strip() == ''"]
    served = run_phase1.get_chunk_decomposition(dict(ONE_LINER))
    assert len(served["chunks"]) == 1, "one step is served, not refused for being one"


def test_a_one_step_roadmap_is_finished_or_refused_through_the_route(monkeypatch, tmp_path):
    """Graded like any final step: the whole answer against the tests."""
    from frontend import api_server
    from main import grading, identity, ollama_client, sessions
    from main.identity import content_hash
    from main.oracle_store import save_cache
    from test_auth_routes import client, register
    from test_plan_gate_outage import Archive
    monkeypatch.setenv("MICROTUTOR_SESSION_DB", str(tmp_path / "s.db"))
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(tmp_path / "o.json"))
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    monkeypatch.setattr(ollama_client, "_openai_chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
    save_cache({content_hash(ONE_LINER): {"strong": True, "status": "strong", "kill_rate": 1.0,
                                          "kill_rate_direct": 1.0, "features": ["calls"],
                                          "final_tests": BLANK_TESTS}})
    c = client(); sb = Archive(); api_server.set_supabase(sb)
    register(c, "stu@psu.edu"); stu = sb.students[0]["id"]
    sb.tables["mt_designs"].append({"student_id": stu, "slug": ONE_LINER["slug"], "approved": True,
                                    "round": 1, "created_at": "2026-10-13T00:00:00+00:00"})
    decomp = splitter.one_step(ONE_LINER)
    for code, want in (("return len(txt.strip()) == 0", "correct"), ("return txt == ''", "incorrect")):
        sid = sessions.create_session(dict(ONE_LINER), decomp, content_hash(ONE_LINER),
                                      student_id=stu)["session_id"]
        grading._VERDICT_MEMO.clear()
        r = c.post("/grade_chunk", json={"session_id": sid, "student_code": code,
                                         "submission_id": f"{sid}:0:{abs(hash(code))}",
                                         "expected_index": 0}).json()
        assert r["verdict"] == want, (code, r)


def test_a_roadmap_in_a_students_own_shape_is_refused_if_it_uses_locals(monkeypatch, tmp_path):
    from main import identity, reroute
    from main.identity import content_hash
    from main.oracle_store import save_cache
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(tmp_path / "o.json"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    import test_roadmap_pool_cost as T
    save_cache({content_hash(PROBLEM): {"strong": True, "status": "strong", "kill_rate": 1.0,
                                        "kill_rate_direct": 1.0, "features": ["calls"],
                                        "final_tests": T.ORACLE}})
    body = "\n".join(c["reference"] for c in LOCALS["chunks"])
    monkeypatch.setattr(reroute, "chat", lambda *a, **k: json.dumps({"body": body}))
    monkeypatch.setattr(run_phase1, "chat", lambda *a, **k: json.dumps({"subproblems": [
        {"prompt": c["prompt"], "reference": c["reference"]} for c in LOCALS["chunks"]]}))
    from main import step_notes
    monkeypatch.setattr(step_notes, "chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("off")))
    reroute._NO_ROUTE.clear()
    plan = {"nodes": [{"id": "s", "kind": "start", "label": "s"}], "edges": []}
    with pytest.raises(reroute.RouteUnavailable, match="students may not"):
        reroute.build(dict(PROBLEM), reroute.effective_header(PROBLEM), plan, max_tries=1)
