"""
graphs.py - the dual-graph artifact.

Two graphs are built for every problem a student works, from two independent
sources, and shown side by side at the end:

  PLAN GRAPH   built incrementally from the CHAT as the student answers each
               subpart. It is what the student SAID they would do. Model-
               extracted, because a plan lives in prose; it never gates
               anything, so a bad extraction costs a confusing picture, not a
               blocked student.

  CODE GRAPH   built from the code they actually SUBMITTED, by walking the AST.
               Deterministic, free, no model call. It is what the student
               actually DID.

The point is the gap between them. A student whose plan says "loop over the
array, and inside the loop check the running total" and whose code checks the
total after the loop has a specific, nameable bug, and seeing the two shapes
next to each other is what makes it nameable. That comparison is compare().

Both graphs share ONE schema so the same renderer draws both and the diff is
structural rather than textual:

    {"nodes": [{"id", "label", "kind", "line"}],
     "edges": [{"src", "dst", "label"}],
     "meta":  {"source", ...}}

kind is drawn from KINDS below. It is the only field compare() aligns on -
labels are prose on one side and code on the other, so they are compared for
similarity but never for equality.
"""
import ast
import difflib
import json
import textwrap

from .indent import dedent_block
from .ollama_client import TUTOR_MODEL, chat

# The shared vocabulary. Both producers emit only these, which is what lets a
# prose plan and a parsed AST be diffed against each other at all.
KINDS = ("start", "step", "branch", "loop", "return", "end")

MAX_LABEL = 60          # a node box, not a paragraph
MAX_NODES = 60          # a graph past this is unreadable; stop drawing
MAX_CHAT_CHARS = 6000   # what the extractor is shown of the conversation


def _empty(source: str) -> dict:
    return {"nodes": [], "edges": [], "meta": {"source": source}}


def _clip(text: str) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= MAX_LABEL else text[:MAX_LABEL - 1] + "…"


# ── code graph: deterministic, from the AST ──────────────────────────────

class _Builder:
    """Accumulates nodes/edges while walking a function body.

    Tracks the set of nodes that "flow out" of the statements walked so far -
    a list rather than a single node because an if/else has two exits that both
    continue into whatever comes next."""

    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self._n = 0

    def add(self, kind: str, label: str, line: int | None = None) -> str:
        nid = f"n{self._n}"
        self._n += 1
        self.nodes.append({"id": nid, "kind": kind, "label": _clip(label),
                           "line": line})
        return nid

    def link(self, srcs: list[str], dst: str, label: str = "") -> None:
        for s in srcs:
            self.edges.append({"src": s, "dst": dst, "label": label})

    def walk(self, stmts: list[ast.stmt], prevs: list[str]) -> list[str]:
        """Walk a statement list; return the nodes control can leave it from.

        An empty return means every path terminated (returned, broke, raised) -
        nothing after this block is reachable, which is exactly what we want to
        show a student who wrote code after an unconditional return."""
        for st in stmts:
            if isinstance(st, ast.If):
                nid = self.add("branch", f"if {_src(st.test)}", st.lineno)
                self.link(prevs, nid)
                yes = self.walk(st.body, [nid])
                # Relabel the first edge out of each side so the picture reads
                # like a flowchart instead of an unlabelled fork.
                _label_from(self.edges, nid, "yes")
                no = self.walk(st.orelse, [nid]) if st.orelse else [nid]
                if st.orelse:
                    _label_from(self.edges, nid, "no", skip_labelled=True)
                prevs = yes + no

            elif isinstance(st, (ast.For, ast.AsyncFor)):
                nid = self.add("loop", f"for {_src(st.target)} in "
                                       f"{_src(st.iter)}", st.lineno)
                self.link(prevs, nid)
                body_out = self.walk(st.body, [nid])
                self.link(body_out, nid, "repeat")   # the back edge
                prevs = [nid]                        # leave when it is exhausted

            elif isinstance(st, ast.While):
                nid = self.add("loop", f"while {_src(st.test)}", st.lineno)
                self.link(prevs, nid)
                body_out = self.walk(st.body, [nid])
                self.link(body_out, nid, "repeat")
                prevs = [nid]

            elif isinstance(st, ast.Return):
                nid = self.add("return",
                               f"return {_src(st.value)}" if st.value else "return",
                               st.lineno)
                self.link(prevs, nid)
                return []                            # nothing flows past a return

            elif isinstance(st, (ast.Break, ast.Continue, ast.Raise)):
                nid = self.add("step", _src(st), st.lineno)
                self.link(prevs, nid)
                return []

            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                # A nested def is not executed where it is written; show it as
                # one step rather than inlining a body that does not run here.
                nid = self.add("step", f"define {st.name}", st.lineno)
                self.link(prevs, nid)
                prevs = [nid]

            elif isinstance(st, (ast.Try, ast.With, ast.AsyncWith)):
                nid = self.add("step", type(st).__name__.lower(), st.lineno)
                self.link(prevs, nid)
                prevs = self.walk(st.body, [nid]) or [nid]

            else:
                nid = self.add("step", _src(st), st.lineno)
                self.link(prevs, nid)
                prevs = [nid]

            if len(self.nodes) >= MAX_NODES:
                break
        return prevs


def _src(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def _label_from(edges: list[dict], src: str, label: str,
                skip_labelled: bool = False) -> None:
    """Put `label` on the first outgoing edge of `src` (optionally the first
    still-unlabelled one) - the yes/no arms of a branch."""
    for e in edges:
        if e["src"] == src and (not skip_labelled or not e["label"]):
            e["label"] = label
            return


def code_graph(code: str, header: str = "") -> dict:
    """Control-flow graph of submitted code. Pure AST - no model call.

    `header` is the function signature the session holds; the chunks are a body
    without a def line, so it is prepended to make the source parseable. A
    syntax error returns a one-node graph rather than raising: a student mid-
    edit should see "this does not parse yet", not a stack trace.

    The body is RE-SEATED under the header rather than concatenated to it. A
    chunk's accepted code is stored at its depth WITHIN the body - column 0 for
    a top-level step - and the function's own four columns are added at assembly
    time by run_phase1.assemble_references() and grading._assemble(). Pasting
    the body straight after `def f(nums):` therefore produced an
    IndentationError for every problem whose first step sits at top level, so a
    finished session drew "does not parse yet" instead of the code graph. That
    was the dual-graph bug: this function was the one assembler that did not add
    the indent its two siblings add."""
    if header.strip():
        body = textwrap.indent(dedent_block(code), "    ")
        # An empty body is not a syntax error worth reporting to a student - a
        # session with nothing accepted yet simply has no code to draw.
        source = header.rstrip() + "\n" + (body if body.strip() else "    pass")
    else:
        source = code
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        g = _empty("code")
        g["nodes"] = [{"id": "n0", "kind": "step",
                       "label": f"does not parse yet (line {e.lineno})",
                       "line": e.lineno}]
        g["meta"]["error"] = "syntax"
        return g

    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    body = fn.body if fn is not None else tree.body

    b = _Builder()
    start = b.add("start", fn.name if fn is not None else "start")
    outs = b.walk(body, [start])
    end = b.add("end", "end")
    b.link(outs, end)
    # Returns are terminal in walk(), so they have no edge out; join them to the
    # end node so the drawing has a single sink.
    for n in b.nodes:
        if n["kind"] == "return":
            b.edges.append({"src": n["id"], "dst": end, "label": ""})
    return {"nodes": b.nodes, "edges": b.edges,
            "meta": {"source": "code", "truncated": len(b.nodes) >= MAX_NODES}}


# ── plan graph: model-extracted, from the chat ───────────────────────────

_PLAN_SYSTEM = """\
You convert a student's stated plan into a flowchart. You are NOT tutoring, NOT
judging, and NOT solving - you are transcribing what they said into a graph.

ABSOLUTE RULES:
- Use ONLY what the student actually said. If they never mentioned a step, it is
  not in the graph. Do NOT complete their plan, do NOT correct it, do NOT add
  the steps a correct solution would need. An incomplete plan MUST produce an
  incomplete graph - that gap is the entire point of this picture.
- If their plan is wrong, transcribe the WRONG plan faithfully.
- You do not know the solution and must not invent one.
- Ignore anything the tutor said. Only the student's own statements count.

NODE KINDS - use exactly these strings:
  "start"  one node, where the work begins
  "step"   a plain action (set something up, compute, update, store)
  "branch" a decision or condition ("if it is even", "when we find a match")
  "loop"   a repetition ("for each item", "keep going until")
  "return" producing the answer
  "end"    one node, where it finishes

EDGES: "src" and "dst" are node ids. Put "yes"/"no" on the two edges out of a
branch, and "repeat" on the edge that goes back to a loop node.

LABELS: the student's own words, shortened to under 60 characters. Plain
language, not code.

OUTPUT - JSON only, no prose:
{"nodes": [{"id": "n0", "kind": "start", "label": "..."}],
 "edges": [{"src": "n0", "dst": "n1", "label": ""}]}
If the student has not described enough to draw anything yet, return
{"nodes": [], "edges": []}.
"""


def _normalise(raw: dict, source: str) -> dict:
    """Force model output into the schema. Anything malformed is dropped rather
    than trusted - a drawing built from junk ids renders as a tangle."""
    nodes, seen = [], set()
    for n in (raw.get("nodes") or [])[:MAX_NODES]:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id") or "").strip()
        kind = str(n.get("kind") or "step").strip().lower()
        if not nid or nid in seen:
            continue
        seen.add(nid)
        nodes.append({"id": nid, "kind": kind if kind in KINDS else "step",
                      "label": _clip(str(n.get("label") or "")), "line": None})
    edges = []
    for e in (raw.get("edges") or []):
        if not isinstance(e, dict):
            continue
        src, dst = str(e.get("src") or ""), str(e.get("dst") or "")
        if src in seen and dst in seen:      # no dangling edges
            edges.append({"src": src, "dst": dst,
                          "label": _clip(str(e.get("label") or ""))})
    return {"nodes": nodes, "edges": edges, "meta": {"source": source}}


def plan_graph(problem: dict, messages: list[dict],
               current: dict | None = None) -> dict:
    """Re-extract the plan graph from the conversation so far.

    Called after each subpart the student answers, so the picture grows as they
    talk. The whole graph is rebuilt rather than patched: a delta protocol gives
    the model a second way to be wrong, and this is cheap enough not to need one.

    Never raises. A visualisation that fails must leave the last good drawing on
    screen, not take the page down - so on any error the previous graph (or an
    empty one) comes back."""
    student_said = "\n".join(
        f"- {m.get('content', '')}" for m in messages
        if m.get("role") == "user" and isinstance(m.get("content"), str))
    if not student_said.strip():
        return current or _empty("plan")

    prompt = (
        f"PROBLEM: {problem.get('title') or problem.get('slug')}\n"
        f"{(problem.get('description') or '')[:1500]}\n\n"
        f"WHAT THE STUDENT HAS SAID (their words only):\n"
        f"{student_said[-MAX_CHAT_CHARS:]}\n\n"
        f"Draw their plan as a flowchart. JSON only.")
    try:
        raw = chat(TUTOR_MODEL, _PLAN_SYSTEM, [{"role": "user", "content": prompt}],
                   temperature=0.1, fmt="json")
        out = _normalise(json.loads(raw), "plan")
    except Exception:
        return current or _empty("plan")
    # An extraction that came back empty should not erase a graph we already
    # had - the student did not un-say what they said.
    if not out["nodes"] and current and current.get("nodes"):
        return current
    return out


def graph_from_design(problem: dict, image_bytes: bytes, mime: str) -> dict:
    """Extract the plan graph from the DESIGN IMAGE itself.

    Needed because plan_graph() reads only what the student TYPED, and a
    student who draws a careful flowchart and types two sentences would
    otherwise get an empty plan graph - punishing exactly the behaviour the
    design gate exists to encourage.

    A separate vision call rather than an extra field on design_review's
    output: that call is judging whether the design holds up, and asking one
    model response to both judge and transcribe is how both get worse. This
    runs once, when a design is approved, so the extra call is per-problem and
    not per-turn.

    Never raises - same contract as plan_graph()."""
    import base64

    from .ollama_client import VISION_MODEL

    if not image_bytes or mime not in ("image/png", "image/jpeg",
                                       "application/pdf"):
        return _empty("plan")
    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    text = (f"PROBLEM: {problem.get('title') or problem.get('slug')}\n"
            f"{(problem.get('description') or '')[:1200]}\n\n"
            f"This is the student's own design. Transcribe exactly what they "
            f"drew into a flowchart. Do not add, complete, or correct any "
            f"step. JSON only.")
    try:
        raw = chat(VISION_MODEL, _PLAN_SYSTEM, [{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url}}]}],
            temperature=0.1, fmt="json")
        # Tagged "design", not "plan", so /plan_graph can tell a drawn plan from
        # a typed one on later turns and not silently replace it.
        return _normalise(json.loads(raw), "design")
    except Exception:
        return _empty("plan")


def merge_plan(from_design: dict, from_chat: dict) -> dict:
    """Pick which plan graph to show when both exist.

    Deliberately a choice, not a union. Merging two independent transcriptions
    of the same plan produces duplicate nodes with no reliable way to tell a
    duplicate from a genuine repeated step, and a graph with phantom steps is
    worse than a thinner honest one. The drawn design wins ties: a student who
    drew a flowchart expressed their plan more precisely there than in chat."""
    d, c = len(from_design.get("nodes", [])), len(from_chat.get("nodes", []))
    if not d:
        return from_chat
    if not c:
        return from_design
    return from_design if d >= c else from_chat


# ── comparison: what the two shapes disagree about ───────────────────────

def _signature(graph: dict) -> list[str]:
    """The control-flow shape as a sequence of kinds, ignoring start/end.

    This is what makes the two comparable at all: the labels are prose on one
    side and Python on the other, but "loop, branch, return" is the same claim
    in both languages."""
    return [n["kind"] for n in graph.get("nodes", [])
            if n["kind"] not in ("start", "end")]


def compare(plan: dict, code: dict) -> dict:
    """Structural diff of plan vs code, in terms a student can act on.

    Aligns the two kind-sequences with difflib and reports what is in one and
    not the other. Deliberately says nothing about correctness - a plan and a
    code graph can differ and both be fine (a student may simplify while
    typing). The output names differences and lets the student judge them."""
    p_sig, c_sig = _signature(plan), _signature(code)
    p_nodes = [n for n in plan.get("nodes", [])
               if n["kind"] not in ("start", "end")]
    c_nodes = [n for n in code.get("nodes", [])
               if n["kind"] not in ("start", "end")]

    aligned, plan_only, code_only = [], [], []
    sm = difflib.SequenceMatcher(a=p_sig, b=c_sig, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                aligned.append({"plan": p_nodes[i1 + k]["label"],
                                "code": c_nodes[j1 + k]["label"],
                                "kind": p_sig[i1 + k]})
        else:
            plan_only += [{"label": n["label"], "kind": n["kind"]}
                          for n in p_nodes[i1:i2]]
            code_only += [{"label": n["label"], "kind": n["kind"],
                           "line": n["line"]} for n in c_nodes[j1:j2]]

    def _count(sig: list[str], kind: str) -> int:
        return sum(1 for k in sig if k == kind)

    notes = []
    for kind, word in (("loop", "loop"), ("branch", "decision")):
        pn, cn = _count(p_sig, kind), _count(c_sig, kind)
        if pn and not cn:
            notes.append(f"Your plan has {pn} {word}(s); your code has none.")
        elif cn and not pn:
            notes.append(f"Your code has {cn} {word}(s) your plan never "
                         f"mentioned.")
        elif pn != cn:
            notes.append(f"Your plan has {pn} {word}(s); your code has {cn}.")
    if not p_sig:
        notes = ["No plan was captured from the chat yet, so there is nothing "
                 "to compare against."]
    elif not notes and not plan_only and not code_only:
        notes = ["Your code follows the plan you described."]

    return {"aligned": aligned, "plan_only": plan_only, "code_only": code_only,
            "notes": notes,
            "similarity": round(sm.ratio(), 3) if (p_sig and c_sig) else 0.0}


def assembled_source(session: dict) -> str:
    """The function as the student actually built it, for code_graph().

    Accepted answers include revealed references (provenance says which), and
    they are part of the submitted program - a graph that silently dropped them
    would not be a graph of the code that ran."""
    return "\n".join(a.get("code", "") for a in session.get("accepted", []))


def build_both(session: dict, plan: dict | None) -> dict:
    """Both graphs plus their diff - the payload the student page draws."""
    code = code_graph(assembled_source(session), session.get("header", ""))
    plan = plan or _empty("plan")
    return {"plan": plan, "code": code, "comparison": compare(plan, code)}


if __name__ == "__main__":
    # code_graph and compare must hold with no model call; plan_graph is the
    # only part that needs one, so it is not exercised here.
    g = code_graph("""
    total = 0
    for x in nums:
        if x > 0:
            total += x
        else:
            total -= x
    return total
""".strip("\n"), "def f(nums):")
    kinds = [n["kind"] for n in g["nodes"]]
    assert kinds[0] == "start" and kinds[-1] == "end", kinds
    assert "loop" in kinds and "branch" in kinds and "return" in kinds, kinds
    assert any(e["label"] == "repeat" for e in g["edges"]), "missing back edge"
    assert any(e["label"] == "yes" for e in g["edges"]), "missing branch label"
    assert all(e["src"] in {n["id"] for n in g["nodes"]} for e in g["edges"])

    bad = code_graph("for x in", "def f(n):")
    assert bad["meta"]["error"] == "syntax", "syntax error must not raise"

    # THE DUAL-GRAPH BUG: chunk bodies are stored flat, the way the session
    # actually holds them, and must still parse under the header.
    flat = build_both({"header": "def f(nums):",
                       "accepted": [{"code": "total = 0"},
                                    {"code": "for x in nums:"},
                                    {"code": "    total += x"},
                                    {"code": "return total"}]}, None)
    assert flat["code"]["meta"].get("error") != "syntax", \
        "a flat-stored body must parse under its header"
    assert [n["kind"] for n in flat["code"]["nodes"]] == \
        ["start", "step", "loop", "step", "return", "end"], \
        [n["kind"] for n in flat["code"]["nodes"]]
    # A session with nothing accepted yet has no code, which is not an error.
    assert code_graph("", "def f():")["meta"].get("error") is None

    # Unreachable code after a return must not be drawn as reachable.
    r = code_graph("    return 1\n    x = 2", "def f():")
    assert [n["kind"] for n in r["nodes"]].count("step") == 0, \
        "statements after a return are unreachable"

    plan = _normalise({"nodes": [{"id": "a", "kind": "loop", "label": "go over each"},
                                 {"id": "b", "kind": "return", "label": "give total"}],
                       "edges": [{"src": "a", "dst": "b"},
                                 {"src": "a", "dst": "ZZZ"}]}, "plan")
    assert len(plan["edges"]) == 1, "dangling edge must be dropped"
    c = compare(plan, g)
    assert c["plan_only"] or c["code_only"] or c["aligned"]
    assert any("decision" in n for n in c["notes"]), c["notes"]
    assert compare(_empty("plan"), g)["notes"][0].startswith("No plan")
    print("graphs self-check ok")
