"""
Standalone check for the OpenAI swap in main/ollama_client.py.

Real use case: tests.sandbox.generate_test_inputs for palindrome-number, which
is a live fmt="json" call site and needs ZERO changes of its own.
Run from the repo root:  python test_openai_swap.py
"""
import json
import os

import requests

from main.ollama_client import OPENAI_MODEL, _is_ollama_tag, _openai_chat, chat
from research.student_agent import AGENTS
from tests.sandbox import GEN_MODEL, generate_test_inputs

PROBLEM = {
    "slug": "palindrome-number",
    "title": "Palindrome Number",
    "description": "Given an integer x, return True if x is a palindrome integer.",
    "solution": ("class Solution:\n"
                 "    def isPalindrome(self, x):\n"
                 "        if x < 0:\n"
                 "            return False\n"
                 "        s = str(x)\n"
                 "        return s == s[::-1]"),
}

FAKE_KEY = "sk-test-not-a-real-key"


def check_routing():
    """The student-simulation models MUST still route to Ollama."""
    print("routing (the constraint that protects the paper):")
    for level, tag in AGENTS.items():
        assert _is_ollama_tag(tag), f"{level} student {tag} would go to OpenAI!"
        print(f"  student/{level:<6} {tag:<22} -> Ollama (local, unchanged)")
    assert not _is_ollama_tag(OPENAI_MODEL), "system model routed to Ollama"
    print(f"  system role   {OPENAI_MODEL:<22} -> OpenAI")
    assert GEN_MODEL == OPENAI_MODEL, "sandbox GEN_MODEL not switched"
    print(f"  tests.sandbox.GEN_MODEL == OPENAI_MODEL ({GEN_MODEL})")


def capture(system, fmt):
    """Build the request without sending it."""
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, headers=headers, payload=json)

        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"choices": [{"message": {"content": '{"ok": true}'}}]}
        return R()

    real_post, real_key = requests.post, os.environ.get("OPENAI_API_KEY")
    requests.post = fake_post
    os.environ["OPENAI_API_KEY"] = FAKE_KEY
    try:
        out = _openai_chat(OPENAI_MODEL, system,
                           [{"role": "user", "content": "hi"}], 0.2, fmt)
    finally:
        requests.post = real_post
        os.environ.pop("OPENAI_API_KEY", None)
        if real_key is not None:
            os.environ["OPENAI_API_KEY"] = real_key
    return sent, out


def check_payload():
    sent, out = capture("Return JSON only.", "json")
    print("\nrequest construction (no network):")
    print(f"  url            : {sent['url']}")
    print(f"  auth header    : {'Bearer <set>' if sent['headers']['Authorization'] else 'MISSING'}")
    print(f"  model          : {sent['payload']['model']}")
    print(f"  temperature    : {sent['payload']['temperature']}")
    print(f"  response_format: {sent['payload'].get('response_format')}")
    print(f"  roles in order : {[m['role'] for m in sent['payload']['messages']]}")
    assert sent["payload"]["response_format"] == {"type": "json_object"}
    assert sent["payload"]["messages"][0]["role"] == "system"
    assert json.loads(out) == {"ok": True}
    print("  fmt='json' -> response_format={'type':'json_object'} ✓")

    plain, _ = capture("Return JSON only.", None)
    assert "response_format" not in plain["payload"]
    print("  fmt=None -> no response_format (plain text calls unaffected) ✓")

    guarded, _ = capture("No mention of the magic word.", "json")
    assert "json" in guarded["payload"]["messages"][0]["content"].lower()
    print("  json-mode guard injects the required 'JSON' mention ✓")


def check_live():
    """The real call site, unmodified, through the OpenAI path."""
    print("\nlive call — tests.sandbox.generate_test_inputs (fmt='json'):")
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        print("  SKIPPED: OPENAI_API_KEY is not set in .env or the shell.")
        try:
            chat(OPENAI_MODEL, "s", [{"role": "user", "content": "hi"}])
        except RuntimeError as e:
            print(f"  error surfaced correctly: {e}")
        return False

    inputs = generate_test_inputs(PROBLEM, n=6)
    print(f"  got {len(inputs)} inputs: {inputs}")
    assert inputs, "no inputs returned"
    assert all(isinstance(i, list) for i in inputs), "inputs must be arg-lists"
    print("  JSON parsed into arg-lists by the UNCHANGED call site ✓")
    return True


if __name__ == "__main__":
    check_routing()
    check_payload()
    live = check_live()
    print(f"\n{'=' * 62}")
    print(f"live OpenAI call            : {'CONFIRMED' if live else 'PENDING API KEY'}")
    print("routing + payload + json-mode: CONFIRMED")
    print("=" * 62)
