import re
from main.ollama_client import chat

# Three models at different ability levels
WEAK_MODEL   = "qwen2.5:0.5b-instruct"
NORMAL_MODEL = "qwen2.5:1.5b-instruct"
STRONG_MODEL = "qwen2.5:7b-instruct"

AGENTS = {
    "weak":   WEAK_MODEL,
    "normal": NORMAL_MODEL,
    "strong": STRONG_MODEL,
}

UNIFIED_TEMPERATURE = 0.3   # Same temp across all agents — isolates model size as the variable

STUDENT_SYSTEM = """\
You are a CS student answering ONE sub-question of a larger programming problem.

RULES:
- Write Python code that answers ONLY the current sub-question.
- Do NOT write the function header (def line) — assume it exists above your code.
- Your answer may be multiple lines if the sub-question needs them.
- Do NOT wrap your answer in markdown, backticks, or quotes.
- Do NOT prefix with "Answer:", "Here:", or any label.
- Do NOT add explanations or trailing commentary after the code.
- Build on any variables the previous chunks produced (shown as your accepted code so far).
- You are still learning — sometimes you make mistakes or miss edge cases.
"""


def clean_answer(text: str) -> str:
    text = re.sub(r'```[\w]*\n?', '', text)
    text = re.sub(r'```', '', text)
    text = re.sub(r'(?i)^answer\s*:\s*', '', text)
    text = re.sub(r'^`([^`]+)`$', r'\1', text.strip())
    return text.strip()


def get_student_answer(chunk_prompt: str, accepted_prefix: list[str],
                       agent_level: str, hint: str | None = None) -> str:
    """Get one simulated-student answer for a single chunk.
    accepted_prefix: list of code strings the student has already written for prior chunks.
    hint: optional feedback from a failed prior attempt on THIS chunk."""
    model = AGENTS[agent_level]
    prefix_text = "\n".join(accepted_prefix) if accepted_prefix else "(nothing written yet)"
    hint_text = (f"\n\nYour previous attempt was wrong. Feedback: {hint}\n"
                 "Try a different approach.") if hint else ""

    user = (
        f"SUB-QUESTION:\n{chunk_prompt}\n\n"
        f"Code you have already written and had accepted (build on this):\n{prefix_text}\n"
        f"{hint_text}\n\n"
        "Write the code for this sub-question only."
    )
    raw = chat(model, STUDENT_SYSTEM,
               [{"role": "user", "content": user}], temperature=UNIFIED_TEMPERATURE)
    return raw.replace("```python", "").replace("```", "").strip()