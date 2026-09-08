"""
context.py - assembling a runnable program for a problem, and naming the entry
point that runs it. Every consumer goes through here.

TWO problems this closes, both of which only appear once a problem can be one
METHOD of a class.

1. A method is not self-contained. A plain function problem is: header + body
   IS the program, and every consumer built one by concatenating those two
   strings. `Stack.push` only means anything with `Node` defined above it, with
   `Stack`'s other methods around it, and at the class's own indent depth - and
   HW3's `Calculator._getPostfix` additionally needs `Stack` itself, a different
   class entirely. So a class-derived problem carries the module it came from,
   split at the point its body begins:

       context_prefix   everything up to and including `def push(self, value):`
       context_indent   the column that body sits at (8 for a method)
       context_suffix   the rest of the class, and the rest of the file

2. A stateful method has no value to compare. The pipeline's oracle contract is
   "call the entry with these args, compare the return value", which is exactly
   right for `_isNumber`, `_getPostfix` and `calculate` - and cannot express
   `push` at all. push(2) returns None; its meaning is the state it leaves
   behind, and pop() only means 6 after three pushes. A fresh instance per call
   can never observe that.

   The fix is NOT a second grading path. It is to make the sequence itself the
   argument: build_program() appends a small driver, `_mt_run_calls(calls)`,
   which replays a recorded sequence against one instance and returns the list
   of results. That is an ordinary pure function taking one argument, so oracle
   generation, mutation testing, the necessity gate and grading all keep using
   the single contract they already have. The recorded sequence in HW3's own
   docstring -

       >>> x=Stack(); x.push(2); x.push(4); x.push(6)
       >>> x.pop()
       6

   - is read straight out of it by calls_from_docstring() and used as the seed.

build_program() is the ONE place assembly happens, so a method cannot be built
one way while being graded another. That is the same reason main/indent.py
exists for the flat case.

Pure: string and AST manipulation only, no I/O, no model calls.
"""
import ast
import doctest

# The synthesized entry point for a method problem. Leading underscore and a
# `mt` prefix so it cannot collide with anything a teacher or student writes.
SEQ_ENTRY = "_mt_run_calls"

# Calls that are not method calls. `len(x)` is how HW3's docstring exercises
# __len__, and there is no other way to reach a dunder from a recorded sequence.
BUILTIN_CALLS = {"len": len, "str": str, "bool": bool}

# ...and the method each of those actually exercises. A recorded sequence can
# only reach __len__ by way of len(x), so asking "does this sequence call
# __len__" has to be asked about `len`.
DUNDER_CALL = {"__len__": "len", "__str__": "str", "__repr__": "str",
               "__bool__": "bool"}

# Marks a call that raised, inside the results list. A string rather than an
# exception object because results cross a JSON boundary - and because HOW a
# method fails is part of its behaviour: a mutant that turns a clean pop into an
# IndexError has to be killed, not silently treated as a crashed harness.
ERROR_PREFIX = "!"


def is_method(problem: dict) -> bool:
    """True when this problem is one method of a class group."""
    return bool((problem or {}).get("context_prefix"))


def entry_name(problem: dict) -> str | None:
    """What the harness should call.

    For a method that is the injected sequence driver, never the method itself:
    calling `push` in isolation observes nothing. See the module docstring."""
    if not problem:
        return None
    if is_method(problem):
        return SEQ_ENTRY
    return problem.get("entry_hint")


def entry_params(problem: dict) -> list[str] | None:
    """Parameter names of the entry point, or None to let the caller resolve
    them from the source. The driver's signature is fixed and known."""
    return ["calls"] if is_method(problem) else None


def _driver(cls: str) -> str:
    """Source of the sequence driver for class `cls`."""
    return f'''

def {SEQ_ENTRY}(calls):
    """Replay a recorded call sequence against one instance (injected).

    Each call is [name, *args]. "new" re-constructs, "len"/"str"/"bool" apply
    that builtin, anything else is an attribute - called with the args when it
    is callable, read when it is a plain field. A call that raises records
    "{ERROR_PREFIX}<ExceptionName>" and the sequence CONTINUES: one failing call
    must not void the observations after it."""
    # A STRING argument is a block of statements rather than a call list. It
    # exists for the observations a flat call list cannot express: catching
    # `node.next = None` needs a reference to the popped node taken BEFORE the
    # pop, and there is nowhere in [["push",2],["pop"]] to put that. The block
    # runs in this module's namespace, statement by statement, recording the
    # value of every expression - so `n = x.top` is a step and `n.next is None`
    # is an observation. Self-contained: it builds its own object.
    if isinstance(calls, str):
        import ast as _ast
        ns = dict(globals())
        out = []
        try:
            body = _ast.parse(calls).body
        except SyntaxError as exc:
            return ["{ERROR_PREFIX}" + type(exc).__name__]
        for stmt in body:
            try:
                if isinstance(stmt, _ast.Expr):
                    out.append(eval(compile(_ast.Expression(stmt.value),
                                            "<block>", "eval"), ns))
                else:
                    exec(compile(_ast.Module([stmt], []), "<block>", "exec"), ns)
                    out.append(None)
            except Exception as exc:
                out.append("{ERROR_PREFIX}" + type(exc).__name__)
        return out

    obj = {cls}()
    out = []
    for call in calls:
        name, args = call[0], list(call[1:])
        try:
            if name == "new":
                obj = {cls}(*args)
                out.append(None)
            elif name == "len":
                out.append(len(obj))
            elif name == "str":
                out.append(str(obj))
            elif name == "bool":
                out.append(bool(obj))
            else:
                attr = getattr(obj, name)
                out.append(attr(*args) if callable(attr) else attr)
        except Exception as exc:
            out.append("{ERROR_PREFIX}" + type(exc).__name__)
    return out
'''


def _indent(body: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + ln if ln.strip() else "" for ln in body.splitlines())


def build_program(problem: dict, body: str, header: str = "") -> str:
    """The full module to execute, with `body` as this problem's implementation.

    `body` is the function/method body WITHOUT its def line and at column 0 -
    exactly what the chunk store holds and what main/indent.py produces.

    For a plain function problem this is `header` with `body` seated four
    columns under it, which is byte-for-byte what grading._assemble and
    run_phase1.assemble_references each used to build on their own. For a method
    it is the whole module, the body dropped into the hole at the class's own
    depth, and the sequence driver appended."""
    if not body.strip():
        body = "pass"                  # an empty body is a syntax error, not a fail
    if not is_method(problem):
        return header.rstrip() + "\n" + _indent(body, 4) if header.strip() else body
    seated = _indent(body, int(problem.get("context_indent") or 8))
    return (problem["context_prefix"].rstrip("\n") + "\n"
            + seated + "\n"
            + (problem.get("context_suffix") or "").rstrip("\n")
            + _driver(problem.get("group_title") or "Solution"))


def header_of(problem: dict) -> str:
    """The def line a STUDENT is working under, at column 0.

    Not the entry point. For a method those are two different things: the
    harness calls the injected driver, but what the student sees, what the
    decomposer splits, and what the code graph draws is `def push(self, value):`
    - the last line of context_prefix, which is exactly where the body hole
    begins. Returns "" for a plain function so the caller keeps building its own
    header from the resolved signature."""
    if not is_method(problem):
        return ""
    for line in reversed((problem.get("context_prefix") or "").splitlines()):
        if line.strip().startswith(("def ", "async def ")):
            return line.strip()
    return ""


def module_with_method(problem: dict, method_src: str) -> str | None:
    """The whole module again, with THIS method replaced by `method_src`.

    `method_src` is a complete method - `def pop(self):`, its docstring and its
    body - dedented to column 0, which is exactly what is stored as `solution`
    and exactly what the teacher edits in the fix panel.

    This exists because retrying a class problem re-PARSES the teacher's edited
    text, and parsing a bare method yields a plain function: no class, no
    context, an entry point of `pop` instead of the sequence driver, and a
    solution that cannot run at all. Splicing it back into its class first means
    the retry re-parses the same shape the upload did.

    Rebuilding rather than patching the stored context is deliberate: the
    docstring lives in context_prefix, so a teacher who edits the STATEMENT -
    the most likely edit of all - would otherwise keep the old one.

    Returns None for a plain function, whose source is already a whole module."""
    if not is_method(problem):
        return None
    prefix = (problem.get("context_prefix") or "").splitlines()
    # Walk back to this method's `def` line; everything above it is the module
    # up to the method, and everything the method itself owns is being replaced.
    cut = None
    for i in range(len(prefix) - 1, -1, -1):
        if prefix[i].strip().startswith(("def ", "async def ")):
            cut = i
            # ...and above its DECORATORS. `method_src` carries its own - the
            # parser's method span starts at the first decorator - so cutting at
            # the `def` line leaves the old `@property` in place and splices a
            # second one under it. `property(property(f))` is not callable, so
            # every retry of Calculator.calculate produced a class that raises
            # TypeError on read, and then saved it.
            while cut and prefix[cut - 1].lstrip().startswith("@"):
                cut -= 1
            break
    if cut is None:
        return None
    # A method sits one level inside its class: the body indent, less one step.
    indent = max(int(problem.get("context_indent") or 8) - 4, 0)
    body = _indent(_dedent(method_src).rstrip(), indent)
    suffix = (problem.get("context_suffix") or "").rstrip("\n")
    return "\n".join(prefix[:cut]) + "\n" + body + "\n" + suffix + "\n"


def surrounding_class(problem: dict) -> str:
    """The module this method lives in, with THIS method's body left blank.

    What the decomposer was missing. It is asked to write a method BODY, and it
    was handed the docstring and the `def` line and nothing else - so for
    `calculateExpressions` it had to guess that the input arrives on
    `self.expressions`, that the state it must reset is `self.states`, that
    `_isVariable` and `_replaceVariables` exist and what they return on bad
    input, and that a `Calculator` is what evaluates an expression. It guessed,
    and the assembled body was then gated against an oracle built from the real
    class - which is a test the guess cannot pass except by luck.

    The siblings' own bodies are included because the chunks RUN against them:
    the assembled program is context_prefix + these chunks + context_suffix, so
    a chunk that misreads what `_replaceVariables` returns is simply wrong. This
    is the same text build_program() assembles and grades with; the one thing
    removed is the body being written.

    Returns "" for a plain function, which is already self-contained."""
    if not is_method(problem):
        return ""
    indent = " " * int(problem.get("context_indent") or 8)
    return "\n".join([
        problem["context_prefix"].rstrip("\n"),
        f"{indent}...        # <- YOUR CHUNKS GO HERE, stacked in order",
        (problem.get("context_suffix") or "").rstrip("\n"),
    ])


def class_properties(problem: dict) -> list[str]:
    """Names on this class that are @property - read, never called.

    `calculate` and `getExpr` look exactly like methods in the source, and a
    generated test that writes `x.calculate()` gets a TypeError rather than the
    value. Every block written for Calculator failed that way: twelve tests that
    observed nothing at all while looking perfectly healthy."""
    if not is_method(problem):
        return []
    cls_name = problem.get("group_title")
    try:
        tree = ast.parse(build_program(problem, solution_body(problem)))
    except SyntaxError:
        return []
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
    if cls is None:
        return []
    out = []
    for f in cls.body:
        if not isinstance(f, ast.FunctionDef):
            continue
        for d in f.decorator_list:
            name = d.id if isinstance(d, ast.Name) else getattr(d, "attr", "")
            if name == "property":
                out.append(f.name)
    return out


def uncall_properties(problem: dict, block: str) -> str:
    """Rewrite `x.calculate()` to `x.calculate` for every @property.

    Belt and braces alongside telling the generator about them: a prompt can be
    ignored, an AST rewrite cannot. Only zero-argument calls are touched, so a
    genuine method is never altered."""
    props = set(class_properties(problem))
    if not props or not isinstance(block, str):
        return block
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return block

    class Fix(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr in props
                    and not node.args and not node.keywords):
                return node.func
            return node

    return ast.unparse(ast.fix_missing_locations(Fix().visit(tree)))


def fixed_internals(problem: dict) -> set[str]:
    """Attribute names a test BLOCK is allowed to look at, derived from the file.

    The worry about observing internals is that it grades implementation
    detail: a student who structures their object differently would fail while
    being correct. That worry does not apply to attributes the GIVEN code
    already depends on. `Stack.__init__` sets `self.top` and `Stack.__str__`
    walks `.next` and `.value` - and students write neither method, so they
    cannot change those names without breaking code they were handed. They are
    part of the class's fixed contract, not private detail.

    So the permitted set is computed, never chosen: take the methods that are
    NOT exercises, and collect the attributes they read. Method CALLS are
    excluded - `out.append(...)` is behaviour on a local list, not state of the
    object under test.

    Returns an empty set for a plain function, and for a class whose given
    methods touch nothing - in both cases no block may reach inside at all."""
    if not is_method(problem):
        return set()
    cls_name = problem.get("group_title")
    try:
        tree = ast.parse(build_program(problem, solution_body(problem)))
    except SyntaxError:
        return set()
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
    if cls is None:
        return set()

    exercise = problem.get("entry_hint")
    # Everything the student is asked to write in this GROUP, not just this
    # problem: a sibling exercise's body is no more fixed than this one's.
    from .assignments import _STEPS_MARK          # the same marker the parser uses
    seg = ast.get_source_segment(
        build_program(problem, solution_body(problem)), cls) or ""
    named = _STEPS_MARK.search(seg)
    exercises = ({w.strip() for w in named.group(1).split(",") if w.strip()}
                 if named else {exercise})

    given = [f for f in cls.body
             if isinstance(f, ast.FunctionDef) and f.name not in exercises]
    called = {n.func.attr for f in given for n in ast.walk(f)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    # `self.__expr` is name-mangled to `_Calculator__expr`, and mangling only
    # happens inside a class body - a block runs at module level, so reading it
    # is AttributeError every single time. Offering it as permitted produced
    # twelve Calculator blocks whose every observation was a constant error: the
    # exact "looks healthy, tests air" failure the permitted set exists to stop.
    # A private name is also the one thing a teacher has said is NOT contract.
    return {n.attr for f in given for n in ast.walk(f)
            if isinstance(n, ast.Attribute) and n.attr not in called
            and not n.attr.startswith("__")}


def block_is_permitted(problem: dict, block: str) -> bool:
    """May this block run? True when it touches only public methods and the
    fixed internals above.

    Checked by PARSING rather than by trusting whoever wrote it, because blocks
    can be model-generated: a block that reaches for an attribute the given code
    never fixed would be grading something the teacher did not specify."""
    if not isinstance(block, str):
        return True                       # an ordinary call list
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return False
    # CALLS are excluded, exactly as fixed_internals excludes them when it
    # derives the set: `sorted(C.states.keys())` is behaviour on a local dict,
    # not state of the object under test. Policing every Attribute node instead
    # rejected any block that used ordinary Python - .keys, .copy, .split,
    # .append - which is why two AdvancedCalculator methods ended up with zero
    # blocks while their permitted set was non-empty.
    allowed = fixed_internals(problem) | set(class_methods(problem))
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return all(n.attr in allowed for n in ast.walk(tree)
               if isinstance(n, ast.Attribute) and id(n) not in called)


def reference_program(problem: dict) -> str:
    """The TEACHER's solution as a runnable module - ground truth.

    Everything that needs to run the reference (oracle generation, mutation
    testing, the necessity gate) reads it through here. For a plain function
    that is just `solution`; for a method, `solution` is a bare `def push(...)`
    dedented to top level, which does not run at all on its own."""
    if is_method(problem):
        return build_program(problem, solution_body(problem))
    return problem.get("solution") or ""


def class_methods(problem: dict) -> list[str]:
    """Every method callable on this problem's class, in definition order.

    Read back out of the assembled module rather than tracked separately, so it
    cannot drift from what the driver can actually reach. Used to tell the input
    generator which calls exist - without it, a generated sequence for `peek`
    is a pile of pushes that never once calls peek."""
    if not is_method(problem):
        return []
    cls_name = problem.get("group_title")
    try:
        tree = ast.parse(build_program(problem, solution_body(problem)))
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            return [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
    return []


def solution_body(problem: dict) -> str:
    """The teacher's own body for this problem, at column 0.

    `problem["solution"]` is the whole `def` - dedented to top level for a
    method - so ground truth is built by feeding THIS back through
    build_program(). Without it the oracle would be generated from a bare method
    with no class around it, which is the shape that cannot run at all."""
    src = problem.get("solution") or ""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None or not fn.body:
        return src
    lines = src.splitlines()
    start = fn.body[0].lineno - 1
    # A docstring is not part of the implementation, and re-emitting it here
    # would duplicate the one already sitting in context_prefix.
    if (isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str) and len(fn.body) > 1):
        start = fn.body[1].lineno - 1
    return _dedent("\n".join(lines[start:fn.end_lineno]))


def _dedent(text: str) -> str:
    """textwrap.dedent, but tolerant of the blank lines it refuses to ignore."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    pad = min((len(ln) - len(ln.lstrip()) for ln in lines), default=0)
    return "\n".join(ln[pad:] if ln.strip() else "" for ln in text.splitlines()).strip("\n")


# ── the recorded sequence a teacher already wrote ────────────────────────

def _as_call(node: ast.AST, cls: str) -> list | None:
    """One doctest statement as [name, *args], or None if it is not a call we
    can replay (a non-literal argument, an unrelated expression)."""
    def literals(args):
        out = []
        for a in args:
            try:
                out.append(ast.literal_eval(a))
            except (ValueError, SyntaxError):
                return None
            if not isinstance(out[-1], (int, float, str, bool, list, dict, tuple, type(None))):
                return None
        return out

    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
        f = node.value.func
        if isinstance(f, ast.Name) and f.id == cls:
            args = literals(node.value.args)
            return None if args is None else ["new", *args]
        return None
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return None
    call = node.value
    if isinstance(call.func, ast.Attribute):
        args = literals(call.args)
        return None if args is None else [call.func.attr, *args]
    if isinstance(call.func, ast.Name) and call.func.id in BUILTIN_CALLS:
        return [call.func.id]
    return None


def doctest_covers(problem: dict) -> bool:
    """Does the teacher's own recorded run actually exercise THIS method?

    The one honest basis for trusting an oracle that is too trivial to
    mutation-test. `return self.count` has no plausible wrong single-point
    implementation, so no mutation operator can ever certify a suite for it -
    but a teacher who wrote

        >>> len(x)
        2

    next to a stack holding two items has stated what the method must do, and
    that statement is in the suite. Note this asks about COVERAGE, not about
    passing: whether the reference agrees with the recorded values is decided by
    running it, the same as every other test.

    False for a plain function, for a class with no examples, and - the case
    that matters - for a method the examples never call."""
    if not is_method(problem):
        return False
    target = problem.get("entry_hint") or ""
    wanted = DUNDER_CALL.get(target, target)
    if not wanted:
        return False
    cls = problem.get("group_title") or "Solution"
    seed = (calls_from_docstring(problem.get("description") or "", cls)
            or calls_from_docstring(problem.get("group_description") or "", cls))
    return any(call and call[0] == wanted for call in seed)


def calls_from_docstring(text: str, cls: str) -> list[list]:
    """The call sequence a teacher recorded in `>>>` examples, in order.

    All the examples in one docstring share one object, so they are ONE
    sequence, not one per line. Returns [] when there are no examples - most
    problems have none, and that is not an error, it just means the oracle is
    grown entirely by main/oracle_gen instead of seeded."""
    calls = []
    for ex in doctest.DocTestParser().get_examples(text or ""):
        try:
            tree = ast.parse(ex.source)
        except SyntaxError:
            continue
        for stmt in tree.body:
            call = _as_call(stmt, cls)
            if call:
                calls.append(call)
    return calls


if __name__ == "__main__":
    # Pure, so it self-checks with no file, no class and no sandbox.
    flat = {"entry_hint": "digit_sum"}
    assert not is_method(flat)
    assert entry_name(flat) == "digit_sum"
    assert entry_params(flat) is None
    # The flat path must produce exactly what _assemble/assemble_references did.
    assert build_program(flat, "return 1", "def f(n):") == "def f(n):\n    return 1"
    assert build_program(flat, "", "def f(n):") == "def f(n):\n    pass"
    assert build_program(flat, "return 1") == "return 1", "no header: caller's own text"

    meth = {"entry_hint": "push", "group_title": "Stack", "context_indent": 8,
            "description": "A stack.\n\n>>> x=Stack(); x.push(2); x.push(4)\n"
                           ">>> x.pop()\n4\n>>> len(x)\n1\n",
            "context_prefix": "class Node:\n    def __init__(self, v):\n"
                              "        self.value, self.next = v, None\n\n"
                              "class Stack:\n    def __init__(self):\n"
                              "        self.top = None\n        self.n = 0\n\n"
                              "    def push(self, value):",
            "context_suffix": "\n    def pop(self):\n"
                              "        v = self.top.value\n"
                              "        self.top = self.top.next\n"
                              "        self.n -= 1\n        return v\n\n"
                              "    def __len__(self):\n        return self.n\n"}
    assert is_method(meth)
    assert entry_name(meth) == SEQ_ENTRY, "a method is graded through the driver"
    assert entry_params(meth) == ["calls"]

    prog = build_program(meth, "node = Node(value)\nnode.next = self.top\n"
                               "self.top = node\nself.n += 1")
    compile(prog, "<t>", "exec")                       # it must be real Python
    assert "        node = Node(value)" in prog, "body not seated at class depth"
    assert "def __len__" in prog, "the rest of the class was dropped"
    # A blank body is a student mid-edit, not a crash.
    compile(build_program(meth, "   \n  "), "<t>", "exec")
    # Internal shape survives seating.
    assert "            self.top = 1" in build_program(meth, "if x:\n    self.top = 1")

    # The recorded sequence, straight out of the teacher's docstring.
    seq = calls_from_docstring(meth["description"], "Stack")
    assert seq == [["new"], ["push", 2], ["push", 4], ["pop"], ["len"]], seq
    assert calls_from_docstring("no examples here", "Stack") == []
    # A call we cannot replay is skipped, not guessed at.
    assert calls_from_docstring(">>> x.push(some_var)\n", "Stack") == []

    # END TO END: the driver actually runs the sequence against the assembled
    # module, and returns what the docstring says it should.
    ns = {}
    exec(compile(prog, "<t>", "exec"), ns)
    assert ns[SEQ_ENTRY](seq) == [None, None, None, 4, 1], ns[SEQ_ENTRY](seq)
    # A raising call is recorded and the sequence continues past it.
    assert ns[SEQ_ENTRY]([["pop"], ["len"]])[0].startswith(ERROR_PREFIX)
    assert ns[SEQ_ENTRY]([["pop"], ["len"]])[1] == 0

    # solution_body strips the def line and the docstring, keeping the body.
    assert solution_body({"solution": 'def push(self, v):\n    """Doc."""\n'
                                      '    self.top = v\n    return None\n'}) \
        == "self.top = v\nreturn None"
    # The reference program is the teacher's own module, and its methods are
    # discoverable from it - that is what the input generator is told about.
    meth_ref = dict(meth, solution="def push(self, value):\n"
                                   "    node = Node(value)\n"
                                   "    node.next = self.top\n"
                                   "    self.top = node\n"
                                   "    self.n += 1\n")
    compile(reference_program(meth_ref), "<t>", "exec")
    assert class_methods(meth_ref) == ["__init__", "push", "pop", "__len__"], \
        class_methods(meth_ref)
    assert class_methods(flat) == [], "a plain function has no class"
    assert reference_program({"solution": "def f(): pass"}) == "def f(): pass"
    # The header a student works under is the METHOD's def line, never the
    # driver's - those are deliberately two different things.
    assert header_of(meth) == "def push(self, value):", header_of(meth)
    assert header_of(flat) == "", "a plain function builds its own header"

    # ── blocks: what a block may touch, and what it may not ──────────────
    # A class shaped like HW3's Calculator: a PRIVATE field, two @property
    # names that look exactly like methods in the source, and one given method
    # whose attribute reads are the class's fixed contract.
    prop = {"entry_hint": "calculate", "group_title": "Calc", "context_indent": 8,
            "description": "A calculator.",
            "solution": "@property\ndef calculate(self):\n    return self.total\n",
            "context_prefix": "class Calc:\n    def __init__(self):\n"
                              "        self.__expr = None\n        self.total = 0\n\n"
                              "    @property\n    def getExpr(self):\n"
                              "        return self.__expr\n\n"
                              "    @property\n    def calculate(self):",
            "context_suffix": ""}
    assert class_properties(prop) == ["getExpr", "calculate"], class_properties(prop)
    # `total` is contract - __init__ sets it and the student cannot rename it.
    # `__expr` is NOT, however often the given code touches it: it is mangled to
    # _Calc__expr, so a block reading it gets AttributeError every single time.
    assert fixed_internals(prop) == {"total"}, fixed_internals(prop)
    # A @property read like a method observes nothing, so the call is rewritten.
    assert uncall_properties(prop, "x.calculate()") == "x.calculate"
    assert uncall_properties(prop, "x.setExpr('1')") == "x.setExpr('1')", \
        "a genuine method must not be stripped of its call"
    # Ordinary Python is not an attempt to reach inside the object under test.
    assert block_is_permitted(prop, "x = Calc()\nsorted([x.total])\nx.total")
    assert block_is_permitted(prop, "x = Calc()\n'a b'.split()\nx.calculate")
    assert not block_is_permitted(prop, "x = Calc()\nx.secret"), \
        "a bare read of an attribute the given code never fixed"
    assert block_is_permitted(prop, [["calculate"]]), "a call list is not a block"

    # ── a decorated method survives the round trip through its module ────
    spliced = module_with_method(prop, prop["solution"])
    assert spliced.count("@property\n    def calculate") == 1, spliced
    ns = {}
    exec(compile(spliced, "<t>", "exec"), ns)          # and it still RUNS:
    assert ns["Calc"]().calculate == 0, "the property must still read as a value"

    # ── the block driver: a program, not a call list ─────────────────────
    ns = {}
    exec(compile(build_program(meth, "node = Node(value)\nnode.next = self.top\n"
                                     "self.top = node\nself.n += 1"),
                 "<t>", "exec"), ns)
    # A reference held ACROSS a call is the whole point: nothing you can call
    # after a pop reveals whether the node that left was unlinked - the popped
    # node is unreachable. This fixture's `pop` does NOT unlink it, and the
    # block is what says so; a call list cannot ask the question at all.
    assert ns[SEQ_ENTRY]("x = Stack()\nx.push(1)\nx.push(2)\n"
                         "n = x.top\nx.pop()\nn.next is None") == \
        [None, None, None, None, 2, False]
    # A raising statement records the error and the block CONTINUES.
    assert ns[SEQ_ENTRY]("x = Stack()\nx.nope()\nlen(x)") == \
        [None, ERROR_PREFIX + "AttributeError", 0]
    assert ns[SEQ_ENTRY]("x = (").pop().startswith(ERROR_PREFIX), "a bad block"

    # The decomposer is shown the class, with the body it must write removed.
    around = surrounding_class(meth)
    assert "def push(self, value):" in around, around
    assert "YOUR CHUNKS GO HERE" in around
    assert "def pop(self):" in around, "the siblings the chunks will run against"
    assert "self.top = node" not in around, "...but never this method's own body"
    compile(around, "<t>", "exec")            # and it is still real Python
    assert surrounding_class(flat) == "", "a plain function is self-contained"

    # A method too trivial to mutate is trusted only when the teacher's own
    # recorded run actually reaches it.
    assert doctest_covers(dict(meth, entry_hint="push"))
    assert doctest_covers(dict(meth, entry_hint="pop"))
    assert doctest_covers(dict(meth, entry_hint="__len__")), "len(x) reaches __len__"
    assert not doctest_covers(dict(meth, entry_hint="peek")), \
        "the examples never call peek - nothing states what it should do"
    assert not doctest_covers(flat), "a plain function is mutation-tested"
    assert not doctest_covers(dict(meth, description="no examples",
                                   group_description=""))
    print("context.py self-check OK")
