# A real example of a "different but correct" plan causing friction

This answers the question you're planning to ask Dr. Saha: *"Sometimes a
student might discuss a different plan than the one the backend decomposed,
both work, but they differ in some ways — is that fine?"*

The first example I gave you for this was **invented and wrong** — I tested
it live and it turned out the system handled it cleanly with no issues. This
document is a **replacement**: a real scenario, built through the actual
tutor conversation and graded by the actual grading pipeline, that does show
a real (if subtle) problem. Nothing below is hypothetical — every claim was
run against the live server and the real OpenAI-backed grader.

---

## 1. The problem

HW3's `Calculator._getPostfix(self, txt)` — it converts a math expression
like `"2 * (5 + 3)"` into postfix notation using the classic shunting-yard
algorithm. It's genuinely complex, so the backend splits it into **3
chunks**:

| Chunk | What it does |
|---|---|
| Part 1 | Tokenize `txt` into numbers/operators/parentheses |
| Part 2 | Validate that operands and operators alternate correctly, parens balance |
| Part 3 | Run shunting-yard over the validated tokens, using a `Stack`, to produce the postfix string |

The teacher's reference solution creates **all its variables up front**, in
Part 1 — including `postfixStack = Stack()` and the `precedence` dictionary
— even though neither one is actually *used* until Part 3.

## 2. The divergent (but correct) plan

Most programmers wouldn't do it that way. The natural instinct is: **don't
create something until you're about to use it.** So I planned it that way —
tokenize first, validate second, and only create the stack and precedence
table right at the start of the shunting-yard step, since that's the first
place they're actually needed.

I talked this plan through with the real tutor (not scripted — an actual
back-and-forth). Two things worth noting about how the tutor handled it:

- It asked a deterministic checkpoint question — *"You just mentioned a list
  and a stack. What does each of them start out as?"* — specifically because
  I'd introduced a new structure (the stack). This is the "idea 2" fix we
  built earlier in the session, working as intended.
- When I said *"the stack starts empty, but I don't create it until the
  shunting-yard part, since it isn't needed until then,"* the tutor accepted
  that and moved on. It did not push back or insist I declare it up front.

The plan then went through the **real design review** and was approved on
the first round, with no rejection and no "redundant structure" flag. The
plan graph it extracted correctly shows the stack's creation as one node
sitting right before the shunting-yard loop — not up at the top:

```
"Initialize output list and operator stack"   ← one node, placed right
                                                  before the shunting-yard loop
```

So far, everything is working exactly as designed: the tutor and reviewer
both correctly treat "create it when you need it" as a legitimate choice,
not a mistake.

## 3. Where it actually goes wrong: grading

Here's the mechanism, in plain terms.

When a student submits code for a chunk that **isn't the last one**, the
grader can't just run that chunk by itself — a partial function isn't
callable. So it glues the student's code together with **the teacher's own
reference code** for the remaining chunks, and runs the whole thing against
the real test cases.

That's the problem. The teacher's reference code for Part 3 assumes
`postfixStack` and `precedence` **already exist**, because in the teacher's
version they were created back in Part 1. My student code never created
them at that point — I deferred that to Part 3, which is exactly the point
of my plan. So when the grader glues my Part 1 code to the *teacher's* Part
2 + Part 3, the combined program crashes: `postfixStack` doesn't exist.

This isn't a bug in my code. It's a mismatch between *when* two equally
valid plans decided to create the same variable.

### The safety net — and where it gets shaky

The system has a fallback for exactly this kind of mismatch: a model
(GPT-4o) is asked to **rewrite** the teacher's leftover code so it works
with whatever the student's code actually produced, instead of assuming the
teacher's exact variables exist. If that rewrite can be *proven* to still
behave correctly (it's tested against known-good work before it's trusted),
the student's step is marked correct on solid, deterministic grounds.

I ran this exact scenario **24 times** with the identical student code, to
see how reliable that rescue is. Here's what happened:

- The rewriting model is fully **capable** of fixing this — in every
  successful run, it swapped the `Stack` object out for a plain list and
  wrote working replacement code for Parts 2 and 3.
- But in **12 of the 24 runs**, the model *also* invented a pointless,
  self-referencing instruction alongside its otherwise-correct rewrite — in
  effect, telling the system "before running my new code, first set
  `postfixStack` equal to `postfixStack`" (or, sometimes, equal to
  `tokens`). That instruction makes no sense, and the system correctly
  refuses it (it can only accept instructions that reference something the
  student's own code actually produced). Both of the model's attempts got
  discarded this way, in every one of those 12 runs.
- When both attempts are refused, the system falls back one more level: a
  panel of two independent AI judges look at the code and decide by
  **reading and reasoning about it**, rather than by *running* it against
  test cases.

So for the exact same, entirely correct student code, roughly **half the
time** it got verified the strong way (run against real test cases,
provably correct) and **half the time** it got verified the soft way (an AI
judge's opinion) — purely because of whether the rewriting model happened to
also blurt out a nonsense instruction on that particular attempt.

### The good news

In all 24 runs, and in a full live run through the actual chat → design
review → grading flow (not just the isolated test), **every single verdict
came out "correct."** And the very last chunk of any problem always gets
one more full check — the student's entire assembled solution run against
every test case for real, no shortcuts — so nothing about this ever let a
wrong answer slip through. I confirmed this in the live run too: chunks 1
and 2 landed on the soft AI-judge tier, but chunk 3 passed the hard,
run-it-for-real check and the student was marked as having solved it
independently.

## 4. How to phrase this to Dr. Saha

The honest, precise version:

> "Yes, this happens, and we tested it directly. When a student's plan
> creates a variable at a genuinely different (but equally valid) point than
> the teacher's reference solution does, the grader's first attempt to
> verify that step by *running* it fails — not because the student is
> wrong, but because it borrows the teacher's leftover code, which assumes
> its own variables already exist. There's an automatic repair step for
> this, and it's capable of fixing it, but it isn't reliable: for the exact
> same submission, it succeeds about half the time and falls back to an AI
> judge's opinion the other half, because the repair model sometimes
> generates a nonsense instruction alongside an otherwise-working fix. In
> every case we tested, the final answer given to the student was still
> correct — the last step always gets one fully deterministic check no
> matter what happened before it — but the *confidence level* behind an
> intermediate step's grade is inconsistent for reasons that have nothing to
> do with whether the student is actually right."

## 5. What this is *not* saying

- It is **not** saying divergent plans get graded incorrectly. In every test
  run, the verdict was correct.
- It is **not** the same failure I originally (and wrongly) described for
  the `frequency` problem — that example turned out to hold up fine on
  testing and is not evidence of anything.
- It **is** evidence that the system's grading tiers are not equally
  reliable for every valid plan shape, specifically at the boundary between
  the "run it and prove it" tiers and the "ask an AI to read it" tier — and
  that boundary can be crossed by something as small as *when* a variable
  gets created, for reasons rooted in one model's own inconsistency rather
  than in any real ambiguity about the student's code.
