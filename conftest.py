import pytest


@pytest.fixture(autouse=True)
def _fresh_model_memo():
    """Each test starts with an empty chat() answer memo (main/ollama_client.
    _MEMO). In production a repeated identical request is meant to be answered
    from it; in a test suite it would let one test's fake reply answer the next
    test's request, so a test counting model calls would count too few."""
    from main import ollama_client
    ollama_client._MEMO.clear()
    yield
