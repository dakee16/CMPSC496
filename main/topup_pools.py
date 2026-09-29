"""topup_pools.py - ONE-OFF: bring every live problem that has 1-4 saved
roadmaps up to five, the same variety a newly uploaded problem gets.

Since 28 Sep roadmaps are built only at upload (run_phase1.fill_pool) and a
student opening a problem is only ever served a saved one. Problems uploaded
before that were filled by student opens, and some never reached five. This
finishes that once, from the teacher side.

    docker compose exec app python -m main.topup_pools                     # dry run, $0
    docker compose exec app python -m main.topup_pools --apply             # build, $2 cap
    docker compose exec app python -m main.topup_pools --apply --max-dollars 5

Touches only problems that are ready AND already have roadmaps saved under the
key students read. One with none is listed for Reprepare instead: a live problem
with nothing under its key has had its content move since it was prepared, so
its test set is probably missing too. Don't run it during an upload - the pool
lock is per process, and this is a second process.
"""
import sys
from contextlib import contextmanager

from . import ollama_client
from .identity import content_hash
from .run_phase1 import _POOL_TARGET, _load_pool, _sb, _usable, fill_pool
from .sessions import context_of

# Pricing a call BEFORE it is sent. Deliberately high: a token is rarely under
# 3 characters, and a roadmap reply is a few hundred tokens, not 2000.
_CHARS_PER_TOKEN = 3
_REPLY_TOKENS = 2000
# The only features allowed to spend inside the cap: the roadmap builder.
_ALLOWED = {"main.run_phase1.decompose_into_chunks",
            "main.run_phase1.decompose_into_chunks_best",
            "main.splitter.write_prompts"}     # fill_pool's fallback words its steps


class SpendCapReached(BaseException):
    """BaseException, so no `except Exception` on the way can swallow it."""


def _feature() -> str:
    """Which feature asked for the call, past this module and the client."""
    f = sys._getframe(1)
    while f is not None and f.f_globals.get("__name__") in (__name__, ollama_client.__name__):
        f = f.f_back
    return f"{f.f_globals.get('__name__')}.{f.f_code.co_name}" if f else "?"


@contextmanager
def spend_cap(max_dollars: float):
    """A HARD cap on real model spend inside the block. Wraps the client's own
    HTTP call: each request is priced before it is sent and refused if it could
    take the total past `max_dollars`; the real cost is then read off the reply.
    Yields a one-item list holding the running total."""
    real_post = ollama_client.requests.post
    spent = [0.0]

    def post(url, *a, json=None, **k):
        who = _feature()
        if who not in _ALLOWED:
            raise SpendCapReached(f"refused a model call from {who}")
        model = (json or {}).get("model", "")
        if model not in ollama_client.LIST_PRICES:
            raise SpendCapReached(f"refused: no price known for {model!r}")
        pin, pout = (p / 1e6 for p in ollama_client.LIST_PRICES[model])
        guess = (len(str((json or {}).get("messages", ""))) / _CHARS_PER_TOKEN * pin
                 + _REPLY_TOKENS * pout)
        if spent[0] + guess > max_dollars:
            raise SpendCapReached(f"the next call could take spend past "
                                  f"${max_dollars:.2f} (spent ${spent[0]:.3f})")
        r = real_post(url, *a, json=json, **k)
        try:
            u = r.json()["usage"]
            spent[0] += u["prompt_tokens"] * pin + u["completion_tokens"] * pout
        except Exception:
            spent[0] += guess               # no usage to read: count the estimate
        print(f"    model call: ${spent[0]:.3f} spent of ${max_dollars:.2f}")
        return r

    ollama_client.requests.post = post
    try:
        yield spent
    finally:
        ollama_client.requests.post = real_post


def short_problems() -> tuple[list, list]:
    """(short, none): ready problems with 1-4 usable saved roadmaps, as
    (problem, how many), and ready problems with none under their key."""
    rows = _sb().table("problems").select(
        "slug, title, description, solution, context, assignment_id, "
        "group_slug, group_title, group_description").eq("ready", True).execute().data
    pool = _load_pool()
    short, none = [], []
    for row in rows or []:
        if not (row.get("solution") or "").strip():
            continue
        problem = {**row, **context_of(row)}      # the key students read
        have = len(_usable(problem, pool.get(content_hash(problem), [])))
        if have == 0:
            none.append(problem)
        elif have < _POOL_TARGET:
            short.append((problem, have))
    return short, none


def main(argv: list[str]) -> int:
    apply = "--apply" in argv
    cap = float(argv[argv.index("--max-dollars") + 1]) if "--max-dollars" in argv else 2.0
    short, none = short_problems()
    need = sum(_POOL_TARGET - have for _, have in short)
    for p, have in short:
        print(f"  {p['slug']}: {have} saved, needs {_POOL_TARGET - have}")
    for p in none:
        print(f"  {p['slug']}: ready but NOTHING saved under its current content "
              f"- use Reprepare for it, not this")
    print(f"{len(short)} short problem(s), {need} roadmap(s) to build: typically "
          f"~${need * 0.065:.2f} (~2.5 calls x ~2.6c each), at most {need * 5} calls.")
    if not apply:
        print("Dry run - nothing built, $0. Add --apply to build.")
        return 0

    spent, stopped = [0.0], None
    try:
        with spend_cap(cap) as spent:
            for p, have in short:
                try:
                    print(f"  {p['slug']}: {have} -> {fill_pool(p)}")
                except Exception as e:          # this problem's, not the run's
                    print(f"  {p['slug']}: skipped - {e}")
    except SpendCapReached as e:
        stopped = e
    if stopped:
        print(f"STOPPED: {stopped}. Everything built before this is saved.")
    print(f"Spent ${spent[0]:.3f} of the ${cap:.2f} cap.")
    return 1 if stopped else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
