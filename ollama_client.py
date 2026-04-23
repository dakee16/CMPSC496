import requests
import time
from typing import Dict, List, Optional

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
_MAX_RETRIES = 2
_RETRY_DELAY = 1.5  # seconds


def chat(
    model: str,
    system: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    fmt: Optional[str] = None,
) -> str:
    """
    Send a chat request to the local Ollama server.

    Args:
        model:       Ollama model tag, e.g. "qwen2.5:7b-instruct"
        system:      System prompt text
        messages:    List of {"role": ..., "content": ...} dicts (user/assistant turns)
        temperature: Sampling temperature (lower = more deterministic)
        fmt:         Optional format hint; pass "json" to request structured JSON output.
                     Only use for decompose/eval calls — NOT for free-text hints.

    Returns:
        Raw string content from the model response.

    Raises:
        requests.HTTPError on non-2xx responses after all retries.
    """
    payload: Dict = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "options": {"temperature": temperature},
        "stream": False,
    }
    if fmt is not None:
        payload["format"] = fmt

    last_error: Exception = RuntimeError("No attempts made")
    for attempt in range(1, _MAX_RETRIES + 2):  # up to _MAX_RETRIES + 1 total tries
        try:
            r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
            if attempt <= _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)
        except requests.HTTPError as e:
            raise  # don't retry HTTP errors (4xx/5xx)

    raise RuntimeError(
        f"Ollama unreachable after {_MAX_RETRIES + 1} attempts. "
        "Make sure Ollama is running: `ollama serve`"
    ) from last_error