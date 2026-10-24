from __future__ import annotations

import pytest
from langchain_ollama import ChatOllama

from parallel_agents.config import Settings
from parallel_agents.fake import FakeDeskModel
from parallel_agents.llm import ModelConfigError
from parallel_agents.llm import is_loopback_url
from parallel_agents.llm import make_model


def test_defaults_are_local() -> None:
    s = Settings.from_env({})
    assert s.provider == "ollama"
    assert s.model == "qwen3:4b"
    assert s.ollama_host == "http://127.0.0.1:11434"
    assert s.max_concurrency == 4
    assert s.run_config() == {"recursion_limit": 50, "max_concurrency": 4}


def test_env_overrides() -> None:
    s = Settings.from_env(
        {
            "PARALLEL_AGENTS_PROVIDER": "Anthropic",
            "PARALLEL_AGENTS_MAX_CONCURRENCY": "0",
            "PARALLEL_AGENTS_MAX_BRANCHES": "7",
            "PARALLEL_AGENTS_CALL_TIMEOUT": "1.5",
            "PARALLEL_AGENTS_CORPUS_DIR": "/data",
            "PARALLEL_AGENTS_ALLOW_REMOTE_OLLAMA": "1",
            "PARALLEL_AGENTS_RETRIES": "",
        }
    )
    assert s.provider == "anthropic" and s.model == "claude-sonnet-5-5"
    assert s.max_concurrency is None and "max_concurrency" not in s.run_config()
    assert s.max_branches == 7 and s.call_timeout == 1.5 and s.corpus_dir == "/data"
    assert s.allow_remote_ollama is True and s.retries == 3


@pytest.mark.parametrize(
    "env",
    [
        {"PARALLEL_AGENTS_PROVIDER": "openai"},
        {"PARALLEL_AGENTS_MAX_BRANCHES": "0"},
        {"PARALLEL_AGENTS_CALL_TIMEOUT": "-1"},
        {"PARALLEL_AGENTS_SAMPLES": "many"},
    ],
)
def test_bad_env_values_raise(env: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        Settings.from_env(env)


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("http://127.0.0.1:11434", True),
        ("http://localhost:11434", True),
        ("http://[::1]:11434", True),
        ("127.0.0.1:11434", True),
        ("http://10.0.0.5:11434", False),
        ("https://ollama.example.com", False),
        ("http://127.0.0.1.example.com", False),
    ],
)
def test_is_loopback_url(url: str, ok: bool) -> None:
    assert is_loopback_url(url) is ok


def test_make_model_builds_each_local_provider() -> None:
    assert isinstance(make_model(Settings(provider="fake", fake_latency=0.1)), FakeDeskModel)
    ollama = make_model(Settings(provider="ollama", model="llama3.2"))
    assert isinstance(ollama, ChatOllama) and ollama.model == "llama3.2"
    remote = make_model(
        Settings(provider="ollama", ollama_host="http://10.0.0.5:11434", allow_remote_ollama=True)
    )
    assert isinstance(remote, ChatOllama)
    with pytest.raises(ModelConfigError):
        make_model(Settings(provider="ollama", ollama_host="http://10.0.0.5:11434"))
