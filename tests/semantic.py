import ast
import copy

_COMMUTATIVE_BINOP = (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)

_CMP_FLIP = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE,
             ast.GtE: ast.LtE, ast.Eq: ast.Eq, ast.NotEq: ast.NotEq}

_CMP_NEGATE = {ast.Lt: ast.GtE, ast.GtE: ast.Lt, ast.Gt: ast.LtE,
               ast.LtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}


class _Normalizer(ast.NodeTransformer):

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        rhs_target = copy.deepcopy(node.target)
        for n in ast.walk(rhs_target):
            if hasattr(n, "ctx"):
                n.ctx = ast.Load()
        new = ast.Assign(targets=[node.target],
                         value=ast.BinOp(left=rhs_target, op=node.op, right=node.value))
        return self.visit(new)

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, _COMMUTATIVE_BINOP) and \
           ast.dump(node.left) > ast.dump(node.right):
            node.left, node.right = node.right, node.left
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        node.values.sort(key=ast.dump)
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1:
            op = node.ops[0]
            left, right = node.left, node.comparators[0]
            if type(op) in _CMP_FLIP and ast.dump(left) > ast.dump(right):
                node.left, node.comparators[0] = right, left
                node.ops[0] = _CMP_FLIP[type(op)]()
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if (isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Compare)
                and len(node.operand.ops) == 1
                and type(node.operand.ops[0]) in _CMP_NEGATE):
            cmp = node.operand
            cmp.ops[0] = _CMP_NEGATE[type(cmp.ops[0])]()
            return self.visit(cmp)
        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        is_true = lambda n: isinstance(n, ast.Constant) and n.value is True
        is_false = lambda n: isinstance(n, ast.Constant) and n.value is False
        if is_true(node.body) and is_false(node.orelse):
            return node.test
        if is_false(node.body) and is_true(node.orelse):
            return self.visit(ast.UnaryOp(op=ast.Not(), operand=node.test))
        return node


def _normalize(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None
    if line.endswith(":"):
        line = line + "\n    pass"
    try:
        tree = ast.parse(line, mode="exec")
    except SyntaxError:
        return None
    tree = _Normalizer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree)


def ast_equivalent(a: str, b: str) -> bool:
    """True if a and b are logically identical modulo surface form. False if
    either won't parse or they differ — caller falls through to execution/LLM."""
    na = _normalize(a)
    nb = _normalize(b)
    return na is not None and na == nb


# ── self-test: run `python semantic.py` ──
if __name__ == "__main__":
    cases = [
        ("return reversed_num == x", "return True if reversed_num == x else False", True),
        ("return x == reversed_num", "return reversed_num == x", True),
        ("while original > 0:", "while 0 < original:", True),
        ("original //= 10", "original = original // 10", True),
        ("return True if x >= 0 else False", "return False if x < 0 else True", True),
        ("reversed_num = reversed_num * 10 + (original % 10)",
         "reversed_num = 10 * reversed_num + original % 10", True),
        ("total = 0", "count = 0", False),
        ("return x > 0", "return x >= 0", False),
        ("def is_palindrome(x):", "def isPalindrome(x):", False),
    ]
    ok = 0
    for a, b, exp in cases:
        got = ast_equivalent(a, b)
        ok += (got == exp)
        print(f"{'✅' if got == exp else '❌'} got={got!s:5} want={exp!s:5}  {a!r}  ≡?  {b!r}")
    print(f"\n{ok}/{len(cases)} passed")