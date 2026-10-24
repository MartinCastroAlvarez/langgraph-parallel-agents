from __future__ import annotations

import io

import pytest

from parallel_agents.cli import build_graph
from parallel_agents.cli import main
from parallel_agents.config import Settings
from parallel_agents.fake import FakeDeskModel


def _run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), out=out)
    return code, out.getvalue()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("PARALLEL_AGENTS_PROVIDER", "PARALLEL_AGENTS_MODEL", "OLLAMA_HOST"):
        monkeypatch.delenv(key, raising=False)


def test_research_desk_end_to_end_offline() -> None:
    code, out = _run(
        "--provider", "fake", "What powers does the Constitution give to Congress and the courts?"
    )
    assert code == 0
    assert "branch 0" in out and "branch 1" in out and "plan" in out
    assert "Sources:\n  [1] us-constitution:article-1-section-1" in out


def test_quiet_hides_progress() -> None:
    code, out = _run(
        "--provider",
        "fake",
        "--quiet",
        "--branches",
        "1",
        "What does the corpus say about spies and fire?",
    )
    assert code == 0
    assert "branch 0" not in out
    assert "art-of-war:" in out


def test_sectioning_and_voting_patterns() -> None:
    code, out = _run(
        "--provider",
        "fake",
        "--pattern",
        "sectioning",
        "--quiet",
        "We the People of the United States.",
    )
    assert code == 0 and '"sentiment"' in out
    code, out = _run(
        "--provider",
        "fake",
        "--pattern",
        "voting",
        "--samples",
        "3",
        "--max-concurrency",
        "0",
        "How many?",
    )
    assert code == 0 and "Answer: '2'" in out and "Votes: {'2': 3}" in out


def test_failures_are_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    from parallel_agents import cli

    monkeypatch.setattr(
        cli, "make_model", lambda s: FakeDeskModel(permanent_failures=("Sample 0",))
    )
    code, out = _run("--pattern", "voting", "--samples", "2", "--quiet", "How many?")
    assert code == 0
    assert "! branch 0 failed: RuntimeError" in out


@pytest.mark.parametrize("pattern", ["research", "researcher", "sectioning", "voting"])
def test_mermaid_comes_from_the_real_graph(pattern: str) -> None:
    code, out = _run("--mermaid", pattern)
    assert code == 0
    assert out.startswith("---") and "graph TD;" in out and "__start__" in out


def test_task_is_required_and_bounded() -> None:
    with pytest.raises(SystemExit):
        _run("--provider", "fake")
    with pytest.raises(SystemExit):
        _run("--provider", "fake", "x" * 20_001)


def test_remote_ollama_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://10.0.0.5:11434")
    code, out = _run("--provider", "ollama", "anything?")
    assert code == 2
    assert "not loopback" in out


def test_bad_env_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARALLEL_AGENTS_PROVIDER", "openai")
    with pytest.raises(SystemExit):
        _run("--provider", "fake", "q?")


def test_provider_flag_picks_that_providers_default_model() -> None:
    settings = Settings.from_env({"PARALLEL_AGENTS_PROVIDER": "fake"})
    assert settings.model == "fake-desk"


def test_build_graph_rejects_unknown_patterns() -> None:
    with pytest.raises(ValueError):
        build_graph("nope", FakeDeskModel(), Settings(provider="fake"))
