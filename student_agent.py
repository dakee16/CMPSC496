import re
from ollama_client import chat

# Three models at different ability levels
WEAK_MODEL   = "qwen2.5:0.5b-instruct"
NORMAL_MODEL = "qwen2.5:1.5b-instruct"
STRONG_MODEL = "qwen2.5:7b-instruct"

AGENTS = {
    "weak":   WEAK_MODEL,
    "normal": NORMAL_MODEL,
    "strong": STRONG_MODEL,
}

STUDENT_SYSTEM = """
You are a beginner CS student attempting ONE micro-step of a programming problem.

STRICT RULES — violating any of these will result in a wrong answer:
- Output ONLY the answer to the current step. One line maximum.
- NEVER write a full function, full class, or multiple lines unless the step explicitly asks for multiple lines.
- NEVER wrap your answer in markdown code blocks, backticks, or quotes.
- NEVER prefix your answer with "Answer:", "Here:", or any other label.
- NEVER add explanations, comments, or sentences after your answer.
- If the step says "declare the function signature", output ONLY the def line followed by pass. Nothing else.
- If the step asks for a loop header, output ONLY the loop header line. Nothing else.
- If the step asks for a single statement, output ONLY that statement. Nothing else.
- You are still learning — occasionally make small syntax mistakes or miss edge cases.
- Do NOT copy or repeat code from the prior context. Answer only what is asked NOW.
"""


def clean_answer(text: str) -> str:
    text = re.sub(r'```[\w]*\n?', '', text)
    text = re.sub(r'```', '', text)
    text = re.sub(r'(?i)^answer\s*:\s*', '', text)
    text = re.sub(r'^`([^`]+)`$', r'\1', text.strip())
    return text.strip()


def get_student_answer(step_prompt: str, context: str, agent: str = "normal") -> str:
    model = AGENTS.get(agent, NORMAL_MODEL)

    if agent == "strong":
        temperature = 0.3
    elif agent == "normal":
        temperature = 0.4
    else:
        temperature = 0.6

    context_msg = ""
    if context:
        context_msg = (f"VARIABLES DEFINED SO FAR (for reference only — do NOT repeat them):\n"f"{context}\n\n")

    user_msg = (f"{context_msg}"f"CURRENT STEP (answer THIS only):\n{step_prompt}\n\n""Your one-line answer:")

    raw = chat(model, STUDENT_SYSTEM, [{"role": "user", "content": user_msg}], temperature=temperature)
    return clean_answer(raw)