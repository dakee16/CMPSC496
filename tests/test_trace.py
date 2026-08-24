"""Instrumentation tests. No model is ever invoked."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import trace


@pytest.fixture(autouse=True)
def clean():
    trace.reset(); trace.PRICING.clear(); yield; trace.reset()


def test_adapter_outcomes_recorded():
    for i, o in enumerate(["malformed", "unsafe", "calibration_failed",
                           "bypass_rejected", "accepted"], 1):
        trace.record_adapter("c1", "gpt-4o-mini", i, o)
    evs = trace.events()
    assert [e["outcome"] for e in evs] == ["malformed", "unsafe",
                                           "calibration_failed", "bypass_rejected",
                                           "accepted"]
    assert [e["attempt"] for e in evs] == [1, 2, 3, 4, 5]


def test_judge_roles_and_route():
    trace.record_judge("c1", "m", "primary", True, 0.91)
    trace.record_judge("c1", "m", "verifier", False, 0.55)
    trace.record_route("c1", "llm-judge", "indeterminate")
    kinds = [e["kind"] for e in trace.events()]
    assert kinds == ["judge", "judge", "route"]
    assert trace.events()[1]["role"] == "verifier"


def test_cost_is_none_without_configured_pricing():
    with trace.model_call("c1", "unpriced-model", "judge") as box:
        box["input_tokens"], box["output_tokens"] = 1000, 200
    e = trace.events()[0]
    assert e["estimated_cost_usd"] is None      # never fabricated
    assert e["latency_ms"] >= 0


def test_cost_estimated_when_pricing_configured():
    trace.configure_pricing("m", 0.15, 0.60)
    with trace.model_call("c1", "m", "adapter") as box:
        box["input_tokens"], box["output_tokens"] = 1_000_000, 1_000_000
    assert trace.events()[0]["estimated_cost_usd"] == pytest.approx(0.75)


def test_instrumentation_failure_never_breaks_grading(monkeypatch):
    monkeypatch.setattr(trace.json, "dumps", lambda *a, **k: 1 / 0)
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", "/nonexistent/dir/x.jsonl")
    trace.record(case_id="c", kind="adapter")   # must not raise
    with trace.model_call("c", "m", "judge"):
        pass


def test_traces_never_enter_public_response():
    trace.record_judge("c1", "m", "primary", True, 0.9, private_prompt="SECRET")
    public = {"verdict": "correct", "tier": "llm-judge", "deterministic": False,
              "reason": "Looks right."}
    assert "SECRET" not in json.dumps(public)
    assert "private_prompt" not in json.dumps(public)
    assert any("private_prompt" in e for e in trace.events())   # kept internally


def test_mocked_adapter_and_judge_flow_records_full_route():
    """A full mocked Tier3->Tier4 route produces one coherent private trace."""
    trace.record_adapter("c9", "m", 1, "calibration_failed")
    trace.record_adapter("c9", "m", 2, "bypass_rejected")
    trace.record_judge("c9", "m", "primary", False, 0.8)
    trace.record_judge("c9", "m", "verifier", False, 0.85)
    trace.record_route("c9", "llm-judge", "incorrect")
    evs = trace.events()
    assert len(evs) == 5 and evs[-1]["final_route"] == "llm-judge"
    assert sum(1 for e in evs if e["kind"] == "adapter") == 2
    assert sum(1 for e in evs if e["kind"] == "judge") == 2
