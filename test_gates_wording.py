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
