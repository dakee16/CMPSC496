"""Regression: /replan is intentionally disabled and must stay closed.

It previously served ungated material — a solution-less problem dict meant
get_oracle_tests() returned [] and replan_from_prefix() accepted status
"skipped" as success, with neither an oracle-strength nor a necessity check.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_replan_disabled_and_touches_nothing(tmp_path, monkeypatch):
    os.environ["MICROTUTOR_SESSION_DB"] = str(tmp_path / "s.sqlite3")
    import frontend.api_server as srv
    from fastapi.testclient import TestClient

    def boom(*a, **k):
        raise AssertionError("a disabled route must not reach services")

    monkeypatch.setattr(srv, "get_supabase", boom)
    monkeypatch.setattr(srv, "replan_from_prefix", boom, raising=False)
    import tests.sandbox as sb
    monkeypatch.setattr(sb, "get_oracle_tests", boom)
    import main.ollama_client as oc
    monkeypatch.setattr(oc, "chat", boom)

    r = TestClient(srv.app).post("/replan", json={
        "slug": "palindrome-number", "description": "d", "accepted_steps": []})
    assert r.status_code == 410
    assert r.json()["detail"]["reason_code"] == "replan_disabled"
