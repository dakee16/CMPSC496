"""account_cap.py - a hard dollar cap on the model spend of named accounts.

WHY. On 29 Sep the site was handed to an outside tester (GPT, driving the
test@test.com account) with real OpenAI calls behind every chat turn, plan
review and hard-to-check step. The OpenAI budget is shared with the real
students, so a site-wide cap would stop THEM too. This caps only the accounts
listed, in the server's .env:

    MICROTUTOR_ACCOUNT_CAPS=test@test.com=1.50

HOW. api_server binds each request's signed-in username (_bind_account); every
OpenAI call made while serving it is priced BEFORE it is sent and refused if it
could take that account past its cap (ollama_client._openai_chat). The price is
a reservation - the prompt, plus a full reply - swapped for the real cost read
off OpenAI's reply. A call that gets no reply keeps its reservation: it may
have been billed, and a cap that undercounts is not a cap.

WHAT A REFUSAL LOOKS LIKE. CapReached is a RuntimeError - what an OpenAI outage
raises - so every feature already has a way out: grading says it could not
check (no try used), the plan gate lets the student through unreviewed, the
tutor says it is unavailable. A capped account sees an outage; nobody else
sees anything.

Spend is kept in MICROTUTOR_ACCOUNT_SPEND (a JSON file on the volume), so a
redeploy mid-test does not refill the budget. Deleting the account's entry
refills it. No accounts listed = nothing is bound, priced or written.
"""
import json
import os
import sys
import threading
from contextvars import ContextVar

# Who the current request is for. Set per request by api_server, only while
# some account is capped; None everywhere else (teacher uploads run in their
# own threads and are never capped).
ACCOUNT: ContextVar[str | None] = ContextVar("acadia_account", default=None)

_CHARS_PER_TOKEN = 4
_REPLY_TOKENS = 1000        # the reservation for the reply: ~1c on gpt-4o
_IMAGE_TOKENS = 1500        # a design picture, high detail, rounded up
_LOCK = threading.Lock()    # one worker (start.sh), so a thread lock is enough
_PARSED: dict = {}


class CapReached(RuntimeError):
    """A RuntimeError on purpose - see the module docstring."""


def caps() -> dict[str, float]:
    """{username: dollars} from MICROTUTOR_ACCOUNT_CAPS. A malformed entry is
    reported and skipped, never guessed at."""
    raw = os.environ.get("MICROTUTOR_ACCOUNT_CAPS", "")
    if raw not in _PARSED:
        out = {}
        for item in filter(None, (s.strip() for s in raw.split(","))):
            name, _, dollars = item.rpartition("=")
            try:
                if not name.strip():
                    raise ValueError(item)
                out[name.strip().lower()] = float(dollars)
            except ValueError:
                print(f"MICROTUTOR_ACCOUNT_CAPS: ignored {item!r} - "
                      f"expected name=dollars", file=sys.stderr)
        _PARSED[raw] = out
    return _PARSED[raw]


def _path() -> str:
    return os.environ.get("MICROTUTOR_ACCOUNT_SPEND", "data/account_spend.json")


def spent() -> dict[str, float]:
    try:
        with open(_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save(totals: dict) -> None:
    tmp = _path() + ".tmp"
    os.makedirs(os.path.dirname(_path()) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(totals, f, indent=1)
    os.replace(tmp, _path())            # never a half-written file


def _prompt_tokens(messages: list) -> float:
    n = 0.0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c) / _CHARS_PER_TOKEN
        else:                           # content parts: text + images
            for part in c or []:
                if part.get("type") == "image_url":
                    n += _IMAGE_TOKENS
                else:
                    n += len(str(part.get("text", ""))) / _CHARS_PER_TOKEN
    return n


def reserve(model: str, messages: list, price: tuple | None):
    """Before a call: None if the current account is not capped, else a hold.
    Raises CapReached if the call could take the account past its cap."""
    who = (ACCOUNT.get() or "").lower()
    cap = caps().get(who)
    if cap is None:
        return None
    if price is None:
        raise CapReached(f"{who}: no price known for {model!r}, so it cannot be capped")
    pin, pout = price[0] / 1e6, price[1] / 1e6
    guess = _prompt_tokens(messages) * pin + _REPLY_TOKENS * pout
    with _LOCK:
        totals = spent()
        if totals.get(who, 0.0) + guess > cap:
            raise CapReached(f"{who} has used ${totals.get(who, 0.0):.2f} of its "
                             f"${cap:.2f} model budget")
        totals[who] = totals.get(who, 0.0) + guess
        _save(totals)
    return who, guess, pin, pout


def settle(hold, usage: dict | None) -> None:
    """After a reply: swap the reservation for what the call really cost."""
    if not hold:
        return
    who, guess, pin, pout = hold
    try:
        actual = usage["prompt_tokens"] * pin + usage["completion_tokens"] * pout
    except (TypeError, KeyError):
        return                          # no usage to read: the estimate stands
    with _LOCK:
        totals = spent()
        totals[who] = max(0.0, totals.get(who, 0.0) - guess + actual)
        _save(totals)
