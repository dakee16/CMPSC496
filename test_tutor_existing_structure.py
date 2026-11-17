"""test_tutor_existing_structure.py - "what does it start out as?" is asked
about a container the student is MAKING, never one the problem already has.

THE REPORT (27 Oct, a TA on HW4's Cache.insert, __getitem__, __setitem__). Each
Cache already holds three CacheLists (`self.hierarchy`). A plan saying "...then
call put on that cache list" was answered "You just mentioned a list. What does
it start out as?", and the TA: "My approach never mentioned accessing a list".
They said it again on all three problems - any "list" set it off.

The canned question still fires where it was built to: a container the student
brings in ("a dictionary", "a new list"), whatever the statement says.

REAL: the /tutor_chat route and tutor.reply. FAKED: Supabase (in memory) and
the model provider - the quick classifier calls every word "proposed", so only
the rule under test can stop the question.
"""
import json
import types

import pytest

from frontend import api_server
from test_auth_routes import client, register
from test_plan_gate_outage import Archive

CANNED = "What does it start out as?"
# The live statements, cut to what matters here: a class method on an object
# the starter built, named by its class (CacheList) and by "that list".
INSERT = ("Insert a ContentItem into the correct level of an initialized Cache. "
          "self has the three CacheLists provided by the starter. Select the "
          "level using hash(content). Call that list's put(content, evictionPolicy) "
          "and return its exact result.")
GETITEM = ("Retrieve a node by content id from the correct level of a Cache. "
           "Select self.hierarchy[hash(content)]. Use that CacheList's membership "
           "operation to find content.cid.")

CONTAINS = ("Search an initialized CacheList for an integer content id. Return True "
            "if a node with value.cid == cid exists, and move that same node to the front.")


@pytest.fixture
def chat(monkeypatch, tmp_path):
    from main import ollama_client
    monkeypatch.setenv("MICROTUTOR_TRACE_FILE", str(tmp_path / "t.jsonl"))
    asked = types.SimpleNamespace(tutor=0)

    def provider(model, system, messages, *a, **k):
        if '"offtrack"' in system:                          # the tutor itself
            asked.tutor += 1
            return json.dumps({"reply": "What happens when the level is full?",
                               "covered": ["state", "processing"], "offtrack": False,
                               "ready": False, "offtrack_reason": ""})
        if "PROPOSE" in (messages[-1].get("content") or ""):  # proposed or rejected
            words = (messages[-1]["content"].split("Words: ", 1)[1]).split("\n", 1)[0]
            return json.dumps({"proposed": [w.strip() for w in words.split(",")],
                               "rejected": []})
        return json.dumps({})
    monkeypatch.setattr(ollama_client, "_openai_chat", provider)
    monkeypatch.setattr(ollama_client, "_ollama_chat", provider)

    c = client()
    sb = Archive()
    api_server.set_supabase(sb)
    register(c, "stu@psu.edu")
    for slug, title, text in (("cache-insert", "insert", INSERT),
                              ("cache-getitem", "__getitem__", GETITEM),
                              ("frequency", "Frequency", "Count letters."),
                              ("cache-list-contains", "__contains__", CONTAINS)):
        sb.problems.append({"slug": slug, "title": title, "description": text})

    def say(slug, said):
        r = c.post("/tutor_chat", json={"slug": slug,
                                        "messages": [{"role": "user", "content": said}]})
        assert r.status_code == 200, r.text
        return r.json()["reply"]
    return types.SimpleNamespace(say=say, asked=asked)


@pytest.mark.parametrize("slug, said", [
    ("cache-insert", "I hash the content to get the level, then call put on that cache list"),
    ("cache-insert", "index into the hierarchy array with hash(content) and call put"),
    ("cache-insert", "pick the right list from self.hierarchy and return what its put gives"),
    ("cache-getitem", "find the level with hash, then check the cid is in that CacheList"),
    ("cache-getitem", "use the cache list at hierarchy[hash(content)] and return its head"),
])
def test_a_list_the_problem_already_has_is_not_asked_about(chat, said, slug):
    reply = chat.say(slug, said)
    assert CANNED not in reply, reply
    assert chat.asked.tutor == 1, "the tutor itself answers instead"


@pytest.mark.parametrize("slug, said, word", [
    # The live case the question was built for: nothing in the statement.
    ("frequency", "I add it to a dictionary, incrementing the count", "dictionary"),
    # A container they BRING IN, even where the statement has one of its kind.
    ("cache-insert", "I'll keep a new list of everything put evicts", "list"),
    ("cache-insert", "I'd store the levels I tried in a list", "list"),
])
def test_a_container_they_bring_in_is_still_asked_about(chat, slug, said, word):
    reply = chat.say(slug, said)
    assert reply == f"You just mentioned a {word}. {CANNED}", reply


# "SET" THE VERB, SAID THE LONG WAY (6 Nov, the class on contains). "set the
# node's .previous and .next to none" was asked "You just mentioned a set. What
# does it start out as?" - twice. The verb check looked for "to" within three
# words of "set"; students name what they set at more length than that.
@pytest.mark.parametrize("said", [
    "you want to set .previous of the node to None for starters",
    "when you find the matching node, set the node's .previous and .next to none",
    "then I set current.next = self.head and move on",
])
def test_set_the_verb_is_not_asked_about(chat, said):
    reply = chat.say("cache-list-contains", said)
    assert CANNED not in reply, reply


def test_a_real_set_is_still_asked_about(chat):
    reply = chat.say("cache-list-contains", "I'll keep a set of the ids I have seen")
    assert reply == f"You just mentioned a set. {CANNED}", reply
