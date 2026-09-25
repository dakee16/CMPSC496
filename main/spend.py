"""
spend.py - what ACADIA's own model calls cost, read back from the trace file.

    docker compose exec app python -m main.spend            # everything on disk
    docker compose exec app python -m main.spend --days 1   # last 24 hours

WHY THIS EXISTS. On 24 Sep 2026 the OpenAI account was billed $325 in one day
and $628 in total, and nothing on the server could say how much of that was
ACADIA - every figure was an estimate built from counting student activity.
ollama_client now writes one `model_usage` event per billed call (tokens, model,
and which feature asked); this reads them back and adds them up.

Reads MICROTUTOR_TRACE_FILE plus its rotated backups (.1, .2, ...), because the
trace rotates and the oldest calls live in the backups. Only calls made AFTER
the recording was deployed are here - there is no history before it.

Costs are OpenAI list prices (ollama_client.LIST_PRICES) applied to exact token
counts, ignoring the cached-input discount, so they are an upper bound. If this
says $40 and the OpenAI dashboard says $300 for the same days, the other $260
was not spent by this server.
"""
import glob
import json
import os
import sys
import time
from collections import defaultdict


def _files(path: str) -> list[str]:
    """The live file and its rotated backups, oldest first."""
    backups = sorted(glob.glob(path + ".*"),
                     key=lambda p: int(p.rsplit(".", 1)[1])
                     if p.rsplit(".", 1)[1].isdigit() else 0, reverse=True)
    return [p for p in backups if p.rsplit(".", 1)[1].isdigit()] + \
        ([path] if os.path.exists(path) else [])


def usage_events(path: str, since: float = 0.0) -> list[dict]:
    out = []
    for f in _files(path):
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue            # a torn last line is not a crash
                if e.get("kind") == "model_usage" and e.get("ts", 0) >= since:
                    out.append(e)
    return out


def summarize(events: list[dict]) -> dict:
    """Totals by day, by feature and by model. Cost None (an unpriced model)
    counts as unpriced, never as zero."""
    def bucket():
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                "cost": 0.0, "unpriced_calls": 0}
    by = {"day": defaultdict(bucket), "purpose": defaultdict(bucket),
          "model": defaultdict(bucket)}
    total = bucket()
    for e in events:
        keys = {"day": time.strftime("%Y-%m-%d", time.gmtime(e.get("ts", 0))),
                "purpose": e.get("purpose") or "?",
                "model": e.get("model") or "?"}
        for b in [total] + [by[k][v] for k, v in keys.items()]:
            b["calls"] += 1
            b["input_tokens"] += e.get("input_tokens") or 0
            b["output_tokens"] += e.get("output_tokens") or 0
            if e.get("estimated_cost_usd") is None:
                b["unpriced_calls"] += 1
            else:
                b["cost"] += e["estimated_cost_usd"]
    return {"total": total, **{k: dict(v) for k, v in by.items()}}


def report(s: dict) -> str:
    lines = []

    def table(title, rows):
        lines.append(f"\n{title}")
        lines.append(f"  {'':44} {'calls':>7} {'tokens in':>12} "
                     f"{'tokens out':>11} {'cost':>9}")
        for name, b in rows:
            lines.append(f"  {name[:44]:44} {b['calls']:>7} "
                         f"{b['input_tokens']:>12,} {b['output_tokens']:>11,} "
                         f"${b['cost']:>8.2f}")

    t = s["total"]
    lines.append(f"TOTAL: {t['calls']} calls, ${t['cost']:.2f} "
                 f"(upper bound at list prices)")
    if t["unpriced_calls"]:
        lines.append(f"  + {t['unpriced_calls']} call(s) on a model with no "
                     f"price in ollama_client.LIST_PRICES - tokens counted, "
                     f"cost not")
    table("BY DAY (UTC)", sorted(s["day"].items()))
    table("BY FEATURE (most expensive first)",
          sorted(s["purpose"].items(), key=lambda kv: -kv[1]["cost"]))
    table("BY MODEL", sorted(s["model"].items(), key=lambda kv: -kv[1]["cost"]))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    path = os.environ.get("MICROTUTOR_TRACE_FILE")
    if not path:
        print("MICROTUTOR_TRACE_FILE is not set - this process records calls "
              "in memory only, so there is nothing on disk to add up.")
        return 1
    since = 0.0
    if "--days" in argv:
        since = time.time() - float(argv[argv.index("--days") + 1]) * 86400
    events = usage_events(path, since)
    if not events:
        print(f"No model calls recorded in {path} yet. Recording starts from "
              f"the deploy that added it.")
        return 0
    print(report(summarize(events)))
    return 0


if __name__ == "__main__" and "--self-check" not in sys.argv and \
        os.environ.get("MICROTUTOR_TRACE_FILE"):
    sys.exit(main(sys.argv[1:]))

if __name__ == "__main__":
    # Self-check: python -m main.spend --self-check (or with no trace file set).
    # Drives the REAL chat() -> trace -> file path with only the HTTP call
    # faked, so a break anywhere between "OpenAI replied" and "the report adds
    # it up" fails here.
    import pathlib
    import tempfile
    from unittest import mock

    from . import ollama_client, trace

    d = pathlib.Path(tempfile.mkdtemp())
    os.environ["MICROTUTOR_TRACE_FILE"] = str(d / "model_trace.jsonl")
    os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

    class _Reply:
        status_code = 200
        headers = {}

        def __init__(self, tin, tout):
            self._b = {"model": "gpt-4o-2024-08-06",
                       "usage": {"prompt_tokens": tin, "completion_tokens": tout,
                                 "prompt_tokens_details": {"cached_tokens": 0}},
                       "choices": [{"message": {"content": "{}"}}]}

        def raise_for_status(self):
            pass

        def json(self):
            return self._b

    def a_feature():
        return ollama_client.chat("gpt-4o", "say json", [
            {"role": "user", "content": "hi"}])

    with mock.patch.object(ollama_client.requests, "post",
                           side_effect=[_Reply(1_000_000, 100_000),
                                        _Reply(2_000_000, 0)]):
        a_feature()
        ollama_client.chat("gpt-4o-mini", "json", [{"role": "user",
                                                    "content": "x"}])

    ev = usage_events(os.environ["MICROTUTOR_TRACE_FILE"])
    assert len(ev) == 2, ev
    # WHICH FEATURE, read off the stack - not the client's own frames.
    assert ev[0]["purpose"] == "__main__.a_feature", ev[0]["purpose"]
    assert ev[0]["served_model"] == "gpt-4o-2024-08-06", ev[0]
    s = summarize(ev)
    # 1M in @ $2.50 + 100k out @ $10 = $3.50;  2M in @ $0.15 = $0.30
    assert abs(s["model"]["gpt-4o"]["cost"] - 3.50) < 1e-9, s["model"]
    assert abs(s["model"]["gpt-4o-mini"]["cost"] - 0.30) < 1e-9, s["model"]
    assert abs(s["total"]["cost"] - 3.80) < 1e-9, s["total"]

    # A REFUSED CALL IS NOT BILLED, so it must not be counted as spend.
    bad = mock.Mock(status_code=429, headers={}, text="no credits")
    with mock.patch.object(ollama_client.requests, "post", return_value=bad), \
            mock.patch.object(ollama_client, "_backoff"):
        try:
            a_feature()
        except Exception:
            pass
    assert len(usage_events(os.environ["MICROTUTOR_TRACE_FILE"])) == 2

    # TELEMETRY CANNOT BREAK A CALL. A trace that raises still returns the reply.
    with mock.patch.object(ollama_client.requests, "post",
                           return_value=_Reply(10, 10)), \
            mock.patch.object(trace, "record", side_effect=OSError("disk full")):
        assert a_feature() == "{}"

    # An unpriced model is counted as unpriced, never as $0.
    s2 = summarize([{"kind": "model_usage", "ts": 0, "model": "gpt-9",
                     "input_tokens": 5, "output_tokens": 5,
                     "estimated_cost_usd": None}])
    assert s2["total"]["unpriced_calls"] == 1 and s2["total"]["cost"] == 0.0

    print(report(s))
    print("\nspend.py self-check OK")
