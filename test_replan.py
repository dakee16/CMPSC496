from schemas import StepItem
from run_phase1 import replan_from_prefix, assemble_canonical

leet = ("class Solution:\n"
        "    def isPalindrome(self, x):\n"
        "        if x < 0:\n            return False\n"
        "        s = str(x)\n        return s == s[::-1]")
problem = {"slug": "palindrome-number", "title": "Palindrome Number",
           "description": "Return True if the integer x reads the same forwards and backwards, else False.",
           "solution": leet}

def s(n, c, ind):
    return StepItem(question_id="q", step_id=f"Step {n}", prompt="",
                    expected_type="code", canonical=c, indent=ind)

# Ground truth reverses a STRING. The student is doing it NUMERICALLY.
accepted = [
    s(1, "def is_palindrome(x):", 0),
    s(2, "if x < 0:", 1),
    s(3, "return False", 2),
    s(4, "reversed_num = 0", 1),
    s(5, "original = x", 1),
    s(6, "while original > 0:", 1),
    s(7, "reversed_num = reversed_num * 10 + original % 10", 2),
    s(8, "original //= 10", 2),
]

new_steps = replan_from_prefix(problem, accepted)

print("\nReplanned remaining steps:")
for st in new_steps:
    print(f"  {st.step_id}: indent={st.indent}  canonical={st.canonical!r}")

print("\nFull assembled solution (student's numeric prefix + replan):")
print(assemble_canonical(accepted) + "\n" + assemble_canonical(new_steps))