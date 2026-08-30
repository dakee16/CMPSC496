"""
ollama_client.py — the single chat() entry point for every LLM call.

The module name is now a misnomer, kept deliberately: seven modules import
`chat` from here, and renaming buys nothing. What changed is that chat() routes
to TWO backends, chosen by the `model` string it is handed:

  * OpenAI — every SYSTEM-ROLE call (decomposition, oracle-input generation,
    interface adaptation, judgment). These go in front of students on graded
    homework, and local-model flakiness has been a repeated source of bad
    reference solutions and malformed structured output.

  * Ollama — any model given as an Ollama tag ("name:tag", e.g.
    "qwen2.5:0.5b-instruct"). research/student_agent.py simulates weak/normal/
    strong students by MODEL CAPACITY (0.5B/1.5B/7B Qwen) for an already
    submitted paper. It imports this same chat(), so sending its tags to OpenAI
    would silently invalidate that experiment. Routing keeps it on local Qwen
    with no change to that file.

The design principle is unchanged: the model proposes, deterministic validation
(execution, oracle tests, mutation testing) decides. A better model means fewer
mistakes for validation to catch — never more trust in the model.
"""
import os
import time
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# Default system-role model. Oracle test-input generation and the research
# agents read this; the two cost-sensitive paths below override it.
OPENAI_MODEL = os.environ.get("MICROTUTOR_MODEL", "gpt-4o-mini")

# DECOMPOSITION and GRADING are split because they have opposite cost shapes.
#
#   Decomposition runs ONCE per problem, at teacher-upload time, and its result
#   is reused by every student who ever solves that problem. Measured on 20
#   previously undecomposed problems at max_tries=5: gpt-4o-mini succeeded 9/20
#   (45%) at 7.9 calls per success; gpt-4o succeeded 16/20 (80%) at 2.5 calls
#   per success, first-try on 13 of those 16. Per success gpt-4o costs ~5x more
#   ($0.0119 vs $0.0022) — about 10c to prepare a 10-problem assignment — and
#   nearly doubles how much of a teacher's upload is usable.
#
#   Grading runs on EVERY student attempt that reaches Tier 3/4, so its cost
#   scales with class size x attempts. It stays on the cheap model: what makes
#   grading trustworthy is the gates around it (calibration, the anti-bypass
#   knockout, dual-judge agreement), not raw model strength.
DECOMPOSE_MODEL = os.environ.get("MICROTUTOR_DECOMPOSE_MODEL", "gpt-4o")
GRADING_MODEL = os.environ.get("MICROTUTOR_GRADING_MODEL", OPENAI_MODEL)

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

_MAX_RETRIES = 3          # hosted API: rate limits and 5xx are routine
_RETRY_DELAY = 1.5        # seconds, doubled each attempt
_TIMEOUT = 120
# Transient by nature — worth another attempt. Everything else (401 bad key,
# 400 malformed request, 404 unknown model) is a bug, so it surfaces at once.
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _is_ollama_tag(model: str) -> bool:
    """Ollama models are "name:tag" (qwen2.5:7b-instruct); OpenAI model names
    never contain a colon (gpt-4o-mini, o3). That colon is the whole routing
    rule — see the student-simulation note in the module docstring."""
    return ":" in model


def _backoff(attempt: int, retry_after: str | None = None) -> None:
    """Exponential backoff, but honour the server's Retry-After when it sends
    one (429s usually do)."""
    try:
        wait = float(retry_after) if retry_after else _RETRY_DELAY * 2 ** (attempt - 1)
    except ValueError:
        wait = _RETRY_DELAY * 2 ** (attempt - 1)
    time.sleep(min(wait, 30.0))


def _openai_chat(model: str, system: str, messages: List[Dict[str, str]],
                 temperature: float, fmt: Optional[str]) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env next to SUPABASE_URL/"
            "SUPABASE_KEY, or export it in your shell."
        )

    payload: Dict = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": temperature,
    }
    if fmt == "json":
        payload["response_format"] = {"type": "json_object"}
        # OpenAI rejects json_object mode outright unless the word "json"
        # appears in the conversation. Every current call site already says so
        # in its prompt; this guard keeps a future one from 400-ing.
        if "json" not in (system + str(messages)).lower():
            payload["messages"][0]["content"] += "\n\nReturn JSON only."

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_error: Exception = RuntimeError("No attempts made")

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            r = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload,
                              timeout=_TIMEOUT)
            if r.status_code in _RETRY_STATUS:
                last_error = requests.HTTPError(
                    f"OpenAI {r.status_code}: {r.text[:200]}")
                if attempt < _MAX_RETRIES:
                    _backoff(attempt, r.headers.get("Retry-After"))
                    continue
                raise last_error
            r.raise_for_status()        # 4xx: a real bug, fail loudly and now
            return r.json()["choices"][0]["message"]["content"]
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                _backoff(attempt)

    raise RuntimeError(
        f"OpenAI unreachable after {_MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def _ollama_chat(model: str, system: str, messages: List[Dict[str, str]],
                 temperature: float, fmt: Optional[str]) -> str:
    """Unchanged local path — student simulation depends on it."""
    payload: Dict = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "options": {"temperature": temperature},
        "stream": False,
    }
    if fmt is not None:
        payload["format"] = fmt

    last_error: Exception = RuntimeError("No attempts made")
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=_TIMEOUT)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                _backoff(attempt)
        except requests.HTTPError:
            raise  # don't retry HTTP errors (4xx/5xx)

    raise RuntimeError(
        f"Ollama unreachable after {_MAX_RETRIES} attempts. "
        "Make sure Ollama is running: `ollama serve`"
    ) from last_error


def chat(
    model: str,
    system: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    fmt: Optional[str] = None,
) -> str:
    """Send a system+messages exchange and return the assistant's text.

    Signature is unchanged from the Ollama-only version — every existing call
    site works untouched. `fmt="json"` means "guarantee parseable JSON back":
    Ollama's `format` field, OpenAI's response_format={"type":"json_object"}.
    The backend is picked from `model` (see _is_ollama_tag)."""
    if _is_ollama_tag(model):
        return _ollama_chat(model, system, messages, temperature, fmt)
    return _openai_chat(model, system, messages, temperature, fmt)
