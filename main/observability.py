"""
observability.py - every model call, mirrored to Langfuse.

trace.py + spend.py already answer "what did we spend": exact tokens per call,
on our own disk. What they cannot do is SHOW anything - no per-student view, no
per-request grouping, no browsing what was actually asked. This sends the same
calls to Langfuse, which does those things. It replaces nothing: the JSONL
trace stays the record of truth, and this is a view onto the same events.

OFF UNLESS CONFIGURED. Nothing here runs - the SDK is not even imported - until
LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are both set (LANGFUSE_BASE_URL
picks the region or a self-hosted server; the SDK reads all three itself).
MICROTUTOR_LANGFUSE=0 turns it off with the keys still present.

PROMPTS STAY HOME BY DEFAULT. What is sent is the model, the feature that
asked (`purpose`, same label as trace.py), tokens, latency, errors, and the
student's opaque students.id - never their name or email. The prompt and the
reply are NOT sent: they hold student code and chat, and Langfuse is a third
party. MICROTUTOR_LANGFUSE_CONTENT=1 sends them too; images are replaced by a
placeholder either way, because a base64 photo is megabytes of nothing to read.

NEVER RAISES. Same rule as trace.py: a telemetry failure must not turn a good
model reply into an error, so every entry point swallows its own exceptions.
"""
import contextvars
import os
import sys
from contextlib import contextmanager

# Who and what the current HTTP request is, bound once by the middleware in
# frontend/api_server.py - same pattern as account_cap.ACCOUNT. Outside a
# request (scripts, research agents, tests) it is empty and each call simply
# becomes its own trace with no user.
REQUEST: contextvars.ContextVar = contextvars.ContextVar(
    "langfuse_request", default=None)
# The generation the current chat() call is filling in, so ollama_client's
# _record_usage can attach the token counts without threading a handle through
# both backends.
_CURRENT: contextvars.ContextVar = contextvars.ContextVar(
    "langfuse_generation", default=None)

_CLIENT = None
_TRIED = False


def enabled() -> bool:
    return (os.environ.get("MICROTUTOR_LANGFUSE", "1") != "0"
            and bool(os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip())
            and bool(os.environ.get("LANGFUSE_SECRET_KEY", "").strip()))


def _capture_content() -> bool:
    return os.environ.get("MICROTUTOR_LANGFUSE_CONTENT", "0") == "1"


def _client():
    """The Langfuse client, built once, or None. A failure to build it (bad
    install, bad config) disables this for the life of the process rather than
    retrying on every call."""
    global _CLIENT, _TRIED
    if _TRIED:
        return _CLIENT
    _TRIED = True
    if not enabled():
        return None
    try:
        from langfuse import Langfuse
        _CLIENT = Langfuse(environment=os.environ.get("MICROTUTOR_ENV", "dev"))
    except Exception as e:
        print(f"langfuse disabled: {e!r}"[:200], file=sys.stderr)
        _CLIENT = None
    return _CLIENT


def new_trace_id() -> str | None:
    """A trace id for one HTTP request, so every model call it makes - the
    tutor turn, its classifier, the plan pre-review - lands in one trace."""
    c = _client()
    if c is None:
        return None
    try:
        return c.create_trace_id()
    except Exception:
        return None


def _redact(messages: list) -> list:
    """Messages with image parts replaced by a placeholder."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = [p if p.get("type") != "image_url"
                       else {"type": "text", "text": "[image omitted]"}
                       for p in content if isinstance(p, dict)]
        out.append({**m, "content": content})
    return out


class _Generation:
    """What chat() fills in. A no-op shell when Langfuse is off."""

    def __init__(self, obs=None):
        self.obs = obs

    def usage(self, tin, tout, cached=None, served=None) -> None:
        if self.obs is None:
            return
        try:
            details = {}
            if tin is not None:
                # Cached input is billed lower, so it is its own usage type -
                # Langfuse prices it separately instead of overstating cost
                # the way trace.py's list-price estimate does.
                details["input"] = int(tin) - int(cached or 0)
                if cached:
                    details["input_cached_tokens"] = int(cached)
            if tout is not None:
                details["output"] = int(tout)
            self.obs.update(usage_details=details or None,
                            metadata={"served_model": served} if served else None)
        except Exception:
            pass

    def output(self, text: str) -> None:
        if self.obs is None or not _capture_content():
            return
        try:
            self.obs.update(output=text)
        except Exception:
            pass


@contextmanager
def generation(purpose: str, backend: str, model: str, system: str,
               messages: list, temperature: float, fmt, max_tokens):
    """One model call as a Langfuse generation. Yields a _Generation; an
    exception from the call is recorded as an ERROR observation and re-raised
    unchanged."""
    c = _client()
    if c is None:
        yield _Generation()
        return

    gen, token = _Generation(), None
    try:
        from langfuse import propagate_attributes
        req = REQUEST.get() or {}
        params = {"temperature": temperature, "fmt": fmt or "text"}
        if max_tokens:
            params["max_tokens"] = int(max_tokens)
        content = (_redact([{"role": "system", "content": system}] + list(messages))
                   if _capture_content() else None)
        with propagate_attributes(user_id=req.get("user_id"),
                                  trace_name=req.get("route") or purpose,
                                  tags=[backend]):
            obs = c.start_observation(
                trace_context=({"trace_id": req["trace_id"]}
                               if req.get("trace_id") else None),
                name=purpose, as_type="generation", model=model,
                model_parameters=params, input=content,
                metadata={"backend": backend, "route": req.get("route"),
                          "turns": len(messages)})
        gen = _Generation(obs)
        token = _CURRENT.set(gen)
    except Exception:
        gen = _Generation()

    try:
        yield gen
    except Exception as e:
        try:
            if gen.obs is not None:
                gen.obs.update(level="ERROR", status_message=repr(e)[:500])
        except Exception:
            pass
        raise
    finally:
        if token is not None:
            _CURRENT.reset(token)
        try:
            if gen.obs is not None:
                gen.obs.end()
        except Exception:
            pass


def record_usage(tin, tout, cached=None, served=None) -> None:
    """Attach token counts to the generation the current chat() opened."""
    gen = _CURRENT.get()
    if gen is not None:
        gen.usage(tin, tout, cached, served)
