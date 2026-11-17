"""test_gates_wording.py - the step-question gate refuses what misleads and
accepts plain English (1 Oct, the first live re-word).

Two refusals on that run were the gate being wrong, not the wording:
  * "Start going through each statement..., preparing to handle each one" -
    the right framing for a step that starts the loop - was refused for the
    word "preparing". What misleads is a step FRAMED as preparation ("prepares
    to process each statement"), so what decides is which comes first.
  * "the previous year's records" was refused because the teacher's code has a
    variable called `previous`. Plain English is not the solution's naming;
    code-shaped names (`prev_records`, `postfixStack`) still are.
"""
from main.gates import check_prompts, opens_unsaid
from main.schemas import StepItem

LOOP = [StepItem(question_id="t", step_id="Part 1", prompt="", reference=
                 "report = {}\nfor statement in self.expressions.split(';'):\n"
                 "    statement = statement.strip()"),
        StepItem(question_id="t", step_id="Part 2", prompt="For each statement, ...",
                 reference="    report[statement] = 1"),
        StepItem(question_id="t", step_id="Part 3", prompt="Return the report.",
                 reference="return report")]


def _first(prompt):
    return [LOOP[0].model_copy(update={"prompt": prompt})] + LOOP[1:]


def test_starting_the_loop_first_passes_even_with_preparing_after():
    assert opens_unsaid(_first("Start going through each statement in the expression, "
                               "preparing to handle each one, and keep the report for "
                               "the next step.")) == []


def test_a_step_framed_as_preparation_is_still_refused():
    for framed in ("Write code that prepares to process each statement, and keep it for "
                   "the next step.",
                   "Write code that gets everything ready to work through the statements, "
                   "and keep it for the next step."):
        assert opens_unsaid(_first(framed)) == [0], framed


def test_plain_english_is_not_a_leaked_name_but_code_shaped_names_are():
    steps = [StepItem(question_id="t", step_id="Part 1", reference=
                      "previous = d[year - 1]\nprev_records = dict(previous)",
                      prompt="Write code that finds the previous year's records, and keep "
                             "them for the next step."),
             StepItem(question_id="t", step_id="Part 2", reference="return prev_records",
                      prompt="Write code that returns the records.")]
    problem = {"description": "Update each employee's record for the new year."}
    assert check_prompts(steps, problem)["status"] == "pass"
    steps[0] = steps[0].model_copy(update={"prompt": steps[0].prompt.replace(
        "the previous year's records", "prev_records")})
    assert check_prompts(steps, problem)["status"] == "fail"


# ── the reverse (18 Oct): a set-up step worded as STARTING the loop ──────
SEARCH = [StepItem(question_id="t", step_id="Part 1", prompt="", reference="current = self.head"),
          StepItem(question_id="t", step_id="Part 2", prompt="Go through each node ...",
                   reference="while current is not None:\n    if current.value.cid == cid:\n"
                             "        return True\n    current = current.next"),
          StepItem(question_id="t", step_id="Part 3", prompt="Return False.",
                   reference="return False")]
# The two the live pool held (18 Oct, 2 of 310 steps - both HW4 __contains__).
LIVE_WRONG = ("Write code that starts checking each node in the list, beginning from the "
              "head, and leave the rest of the work on each one for the next step.",
              "Write code that starts going through each node in the CacheList, beginning "
              "with the head, and leave the rest of the work on each one for the next step")


def _setup(prompt):
    return [SEARCH[0].model_copy(update={"prompt": prompt})] + SEARCH[1:]


def test_a_setup_step_worded_as_starting_the_loop_is_refused():
    """GPT's audit of HW4 __contains__ (15 Oct): asked LIVE_WRONG[0] for
    `current = self.head`, a student wrote the loop's first line there - and
    then the next step's loop had nothing under it. The gate made a step that
    STARTS the loop say so, and never checked the reverse."""
    from main.gates import claims_unstarted
    for q in LIVE_WRONG:
        assert claims_unstarted(_setup(q)) == [0], q
        out = check_prompts(_setup(q), {"description": "Search the list for cid."})
        assert out["status"] == "fail" and "Part 1" in out["summary"], out


def test_setup_wording_and_real_loop_starts_are_left_alone():
    """What the live pool already words well (18 Oct): a rule on "each" alone
    would have refused these - and a refused question caps a roadmap below the
    bar."""
    from main.gates import claims_unstarted
    for q in ("Write code that gets ready to go through each node in the CacheList, "
              "starting from the head, and leave the rest of the work on each one for "
              "the next step.",
              "Write code that finds where in the list to begin, and keep it for the next step."):
        assert claims_unstarted(_setup(q)) == [], q
    # A step that really starts the loop is the other rule's, never this one's.
    assert claims_unstarted(_first("Start going through each statement, and keep the "
                                   "report for the next step.")) == []
    # A step with a check and no loop, worded as a check.
    check = [StepItem(question_id="t", step_id="Part 1", reference=
                      "if content.size > self.maxSize:\n    return 'Insertion not allowed'",
                      prompt="Check whether the content item can be inserted into the "
                             "cache. Reject the content if it's too large or is already in "
                             "the list, without executing any further steps."),
             StepItem(question_id="t", step_id="Part 2", prompt="Insert it.",
                      reference="return 'INSERTED'")]
    assert claims_unstarted(check) == []


# ── a step that RETURNS its answer, worded as keeping it (18 Oct) ────────
# GPT's round 2, HW4 __getitem__: the step's code ends `return cache_list.head`
# on a hit, and the question said "Keep the updated node for the next step" -
# the next step being only `return 'Cache miss!'`. A student kept the node,
# and the finished answer said "Cache miss!" for items that were there.
GET = [StepItem(question_id="t", step_id="Part 1", reference=
                "cache_list = self.hierarchy[hash(content)]\n"
                "if content.cid in cache_list:\n    return cache_list.head",
                prompt="Find the cache level based on the content header and check if the "
                       "content id exists in that level, updating the node's position if "
                       "found. Keep the updated node for the next step."),
       StepItem(question_id="t", step_id="Part 2", reference="return 'Cache miss!'",
                prompt="Handle the case where the content id is not found.")]


def test_a_step_that_returns_its_answer_but_says_keep_it_is_refused():
    from main.gates import keeps_what_it_returns
    assert keeps_what_it_returns(GET) == [0]
    out = check_prompts(GET, {"description": "Look the content up in its level."})
    assert out["status"] == "fail" and "Part 1" in out["summary"], out


def test_guards_that_return_a_fixed_value_may_still_keep_the_result():
    """Measured on the live pool (18 Oct, 180 non-final steps): seven guards
    `return False` / `return None` on bad input and keep the result otherwise -
    worded correctly. Only a COMPUTED return is the step's answer."""
    from main.gates import keeps_what_it_returns
    guard = [StepItem(question_id="t", step_id="Part 1",
                      reference="if not word or not word[0].isalpha():\n    return False",
                      prompt="Write code that checks if the word is non-empty and starts with "
                             "a letter, and keep the result for the next step."),
             StepItem(question_id="t", step_id="Part 2", prompt="Check the rest.",
                      reference="return all(ch.isalnum() for ch in word)")]
    assert keeps_what_it_returns(guard) == []
    said = [GET[0].model_copy(update={"prompt": "Find the content's cache level; if the "
                                      "content id is there, return the node at the front. "
                                      "Otherwise leave the miss for the next step."})] + GET[1:]
    assert keeps_what_it_returns(said) == []
