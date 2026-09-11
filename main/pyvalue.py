"""
pyvalue.py - moving PYTHON VALUES across the process boundary, losslessly.

THE BUG THIS EXISTS FOR. Test inputs used to cross into the grading subprocess
as JSON, and JSON is a weaker language than Python: its object keys are strings,
and it has no tuples and no sets. So `{2019: {...}}` arrived as
`{'2019': {...}}`, and a correct solution doing `d[year - 1]` raised KeyError on
every single input. Measured on LAB1's employee_update: zero usable oracle
tests, and the only implementation that survived was one that wrote
`d[str(year - 1)]` - wrong Python, passing because the harness was wrong.

Round-tripping fourteen awkward values, JSON corrupts or crashes on nine:

    {2019: {...}}   -> {'2019': {...}}     int keys become strings
    {1.5: 'a'}      -> {'1.5': 'a'}        so do float, bool and None keys
    (1, 2, 3)       -> [1, 2, 3]           tuples become lists, silently
    {(0,1): 'edge'} -> TypeError           tuple keys cannot be encoded at all
    {1, 2, 3}       -> TypeError           nor sets
    {10**30: 1}     -> '1000000...'        big int keys, stringified

SO THE WIRE FORMAT IS A PYTHON LITERAL, read back with ast.literal_eval. That
is what the real judges do - LeetCode ships a typed literal and lets each
language's driver build it natively, rather than serialising objects - and it
means the transport is exactly as expressive as the language being taught.

IT IS SAFE ON UNTRUSTED OUTPUT, which matters because results come back from
the process running student code. literal_eval evaluates literals and nothing
else: `__import__('os').system(...)`, `open(...)`, `[].__class__` and even
`1+1` are all refused, and a deep-nesting bomb is rejected by the parser.

ONE DEFINITION, THREE USES. Both child harnesses run under `python -c` with the
student one isolated (`-I`), which strips the script directory from sys.path -
so a child cannot import this module, and the encoder has to be inlined into
the harness text. Defining it as SOURCE and exec'ing it here gives the parent
the identical functions with nothing to keep in sync.
"""
import ast

# The encoder, as source, so the harnesses can embed it verbatim.
SOURCE = r'''
def mt_lit(v, _depth=0):
    """A Python value as literal source text that ast.literal_eval can read.

    Total by construction: anything without a literal form (a Node, an open
    file, a class) is written as its str() inside a string literal, which is
    what json.dumps(default=str) did and keeps a stray object from taking the
    whole payload down with it."""
    if _depth > 40:
        return repr("<nested too deep>")
    if v is None or v is True or v is False:
        return repr(v)
    if isinstance(v, bool):
        return repr(bool(v))
    if isinstance(v, int):
        return repr(int(v))
    if isinstance(v, str):
        return repr(str(v))
    if isinstance(v, float):
        if v != v:
            return repr("__mt_nan__")      # nan has no literal form
        if v == float("inf"):
            return "1e999"                 # ...but the infinities do
        if v == float("-inf"):
            return "-1e999"
        return repr(float(v))
    if isinstance(v, (list, tuple)):
        inner = ", ".join(mt_lit(x, _depth + 1) for x in v)
        if isinstance(v, tuple):
            return "(" + inner + (",)" if len(v) == 1 else ")")
        return "[" + inner + "]"
    if isinstance(v, (set, frozenset)):
        if not v:
            return "set()"
        return "{" + ", ".join(mt_lit(x, _depth + 1) for x in v) + "}"
    if isinstance(v, dict):
        return "{" + ", ".join(
            mt_lit(k, _depth + 1) + ": " + mt_lit(x, _depth + 1)
            for k, x in v.items()) + "}"
    return repr(str(v))
'''

exec(SOURCE, globals())          # parent-side mt_lit, same definition


def dumps(value) -> str:
    """A Python value as literal text."""
    return mt_lit(value)                                    # noqa: F821


def _restore_nan(v):
    if isinstance(v, str):
        return float("nan") if v == "__mt_nan__" else v
    if isinstance(v, list):
        return [_restore_nan(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_restore_nan(x) for x in v)
    if isinstance(v, set):
        return {_restore_nan(x) for x in v}
    if isinstance(v, dict):
        return {_restore_nan(k): _restore_nan(x) for k, x in v.items()}
    return v


def loads(text: str):
    """Literal text back to a Python value. Raises ValueError on anything that
    is not a literal - which is the security property, not a limitation."""
    return _restore_nan(ast.literal_eval(text))


def json_safe(value) -> bool:
    """Would JSON round-trip this unchanged? Used to decide when the oracle
    cache has to store literal text instead of a JSON structure."""
    import json
    try:
        return json.loads(json.dumps(value)) == value
    except Exception:
        return False


if __name__ == "__main__":
    import json as _json
    import math

    # Every shape JSON loses. These are the reason the module exists.
    for v in ({2019: {"Ann": [1]}, 2020: {}}, {1.5: "a"}, {True: 1, False: 0},
              {None: 1}, (1, 2, 3), (7,), (), {(0, 1): "edge"}, {1, 2, 3},
              set(), frozenset({1}), {1: [(2, 3), {4: {5}}]}, {10 ** 30: 1},
              [], {}, "", 0, -1, 1.25, True, None, "it's \"quoted\"\n",
              {"é": "ü"}, [[[[[1]]]]]):
        back = loads(dumps(v))
        assert back == v, (v, back)
        if not isinstance(v, frozenset):
            assert type(back) is type(v), (v, type(back), type(v))

    assert loads(dumps(float("inf"))) == float("inf")
    assert loads(dumps(float("-inf"))) == float("-inf")
    assert math.isnan(loads(dumps(float("nan"))))

    # JSON's own record, for the comparison the docstring claims.
    lost = [v for v in ({2019: 1}, (1, 2), {1, 2}, {(0, 1): "x"}, {1.5: "a"})
            if not json_safe(v)]
    assert len(lost) == 5, lost

    # An object with no literal form must not take the payload down with it.
    class _Node:
        def __repr__(self):
            return "Node(4)"
    got = loads(dumps({"a": _Node(), "b": [1, _Node()]}))
    assert got == {"a": "Node(4)", "b": [1, "Node(4)"]}, got

    # ...and the security property, which is why this is safe on child output.
    for hostile in ("__import__('os').system('x')", "open('/etc/passwd')",
                    "1+1", "[].__class__", "lambda: 1"):
        try:
            loads(hostile)
            raise AssertionError(f"literal_eval evaluated {hostile!r}")
        except (ValueError, SyntaxError):
            pass

    # The parent's functions and the harness text are the same definition.
    _ns = {}
    exec(SOURCE, _ns)
    for v in ({2019: 1}, (1,), {1, 2}, "x", 1.5, None):
        assert _ns["mt_lit"](v) == dumps(v), v

    print("pyvalue.py self-check OK")
