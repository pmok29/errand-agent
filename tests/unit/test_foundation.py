import pytest

from errand.clock import FrozenClock
from errand.config import llm_settings
from errand.db import DB


def test_frozen_clock_advances_without_sleeping():
    clock = FrozenClock()
    start = clock.now()
    clock.sleep(600)
    assert (clock.now() - start).total_seconds() == 600


def test_tx_rolls_back_on_error(tmp_path):
    db = DB(tmp_path / "agent.db")
    with pytest.raises(RuntimeError):
        with db.tx() as c:
            c.execute("INSERT INTO sessions VALUES ('s1','u1','t')")
            raise RuntimeError("boom")
    assert db.one("SELECT * FROM sessions") is None


def test_llm_presets(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gk")
    s = llm_settings("groq")
    assert s.base_url.startswith("https://api.groq.com") and s.api_key == "gk"
    assert s.model == "openai/gpt-oss-120b"
    o = llm_settings("ollama")
    assert o.base_url == "http://localhost:11434/v1" and o.model == "qwen3:4b"
    monkeypatch.setenv("LLM_MODEL", "custom")
    assert llm_settings("groq").model == "custom"
