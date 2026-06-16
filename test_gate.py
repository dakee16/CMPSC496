from schemas import StepItem
from run_phase1 import validate_decomposition

leet = ("class Solution:\n"
        "    def isPalindrome(self, x):\n"
        "        if x < 0:\n            return False\n"
        "        s = str(x)\n        return s == s[::-1]")
problem = {"slug": "palindrome-number", "title": "Palindrome Number",
           "description": "Return True if integer x is a palindrome.", "solution": leet}

def step(n, expected_type, canonical, indent):
    return StepItem(question_id="q", step_id=f"Step {n}", prompt="",
                    expected_type=expected_type, canonical=canonical, indent=indent)

good = [
    step(1, "code", "def is_palindrome(x):", 0),
    step(2, "code", "original = x", 1),
    step(3, "code", "reversed_num = 0", 1),
    step(4, "code", "while original > 0:", 1),
    step(5, "code", "reversed_num = reversed_num * 10 + (original % 10)", 2),
    step(6, "code", "original //= 10", 2),
    step(7, "code", "return reversed_num == x", 1),
]

bad = [  # guaranteed early return → the lines below are dead code (compiles, but wrong)
    step(1, "code", "def is_palindrome(x):", 0),
    step(2, "code", "return False if x < 0 else True", 1),
    step(3, "code", "original = x", 1),
    step(4, "code", "reversed_num = 0", 1),
    step(5, "code", "return reversed_num == x", 1),
]

g = validate_decomposition(good, problem)
print("GOOD decomposition:", g["status"], "—", g["detail"])

r = validate_decomposition(bad, problem)
print("BAD  decomposition:", r["status"], "—", r["detail"])
for f in r["failures"]:
    print("   ", f)