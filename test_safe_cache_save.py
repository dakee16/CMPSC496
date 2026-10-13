"""test_safe_cache_save.py - saving the test suites can never leave half a file.

THE RISK (9 Oct, during the HW4 upload). save_cache dumped straight into
data/oracles/tests_cache.json, so a restart in the middle of a save - a deploy
while an upload runs - could truncate the one file holding every saved suite,
HW3's certified ones included. Now written beside it and swapped in whole.

REAL: oracle_store.save_cache / load_cache on a real file. The crash is
simulated by making the encoder fail halfway through writing.
"""
import json

from main import oracle_store


def test_a_save_that_dies_halfway_leaves_the_old_file_whole(monkeypatch, tmp_path):
    path = tmp_path / "tests_cache.json"
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(path))
    good = {"hw3-key": {"strong": True, "status": "strong", "final_tests": [
        {"input": [1], "expected": 2}]}}
    oracle_store.save_cache(good)
    before = path.read_text()

    real_dump = json.dump

    def dies_halfway(obj, f, **kw):
        f.write('{"hw3-key": {"strong": tr')         # a partial write...
        raise OSError("killed mid-save")              # ...then the process dies
    monkeypatch.setattr(oracle_store.json, "dump", dies_halfway)
    oracle_store.save_cache({**good, "new-key": {"strong": False}})
    monkeypatch.setattr(oracle_store.json, "dump", real_dump)

    assert path.read_text() == before, "the saved suites were damaged"
    assert json.loads(path.read_text())["hw3-key"]["status"] == "strong"
    assert not list(tmp_path.glob("*.tmp")), "a half-written temp file was left behind"


def test_a_normal_save_still_saves(monkeypatch, tmp_path):
    path = tmp_path / "tests_cache.json"
    monkeypatch.setenv("MICROTUTOR_ORACLE_CACHE", str(path))
    oracle_store.save_cache({"k": {"strong": True, "status": "strong", "final_tests": []}})
    assert oracle_store.load_cache()["k"]["status"] == "strong"
