"""test_reply_cap.py - the paid counterexample search cannot run a reply to
the model's maximum length.

THE REPORT (7 Oct, HW4 upload, live). One problem sat "preparing" for 18
minutes. Its search asked for five test inputs - a few hundred tokens - and 13
replies ran to gpt-4o's 16,384-token ceiling instead: $0.17 each, $2.17 of a
$4.78 upload, every one cut off and unusable. No call set a limit.

REAL: the search's request as it is built and sent by ollama_client. FAKED:
the network - requests.post is replaced, so nothing leaves the machine.
"""
import json

import pytest

from main import mutation, ollama_client

DOUBLE = {"slug": "double", "title": "Double", "entry_hint": "double",
          "description": "Twice n, or 0 for negatives.",
          "solution": "def double(n):\n    if n < 0:\n        return 0\n    return n * 2\n"}


@pytest.fixture
def sent(monkeypatch, tmp_path):
    from main import identity
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(identity, "_RESOLVED_PATH", str(tmp_path / "r.json"))
    payloads = []

    class Reply:
        status_code = 200
        headers = {}
        text = ""

        def raise_for_status(self):
            pass

        def json(self):
            return {"model": "gpt-4o", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "choices": [{"message": {"content": json.dumps({"inputs": [[3], [-1]]})}}]}

    def post(url, headers=None, json=None, timeout=None):
        payloads.append(json)
        return Reply()
    monkeypatch.setattr(ollama_client.requests, "post", post)
    return payloads


def test_the_search_caps_its_reply(sent):
    mutant = DOUBLE["solution"].replace("n < 0", "n <= 0")
    got = mutation._candidate_inputs(DOUBLE, DOUBLE["solution"], mutant)
    assert got == [[3], [-1]], "the reply is still read as before"
    assert len(sent) == 1
    assert sent[0]["max_completion_tokens"] == mutation._CANDIDATE_MAX_TOKENS <= 4000


def test_no_other_call_is_capped(sent):
    ollama_client.chat("gpt-4o", "Return JSON.", [{"role": "user", "content": "hi"}],
                       fmt="json")
    assert "max_completion_tokens" not in sent[0] and "max_tokens" not in sent[0]
