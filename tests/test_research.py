"""The research desk: fan-out counts, reducer fan-in, ordering, bounds and isolation."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import random

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage

from parallel_agents.config import Settings
from parallel_agents.corpus import Corpus
from parallel_agents.fake import FakeDeskModel
from parallel_agents.research import build_research_desk
from parallel_agents.research import build_researcher
from parallel_agents.research import make_search_tool
from parallel_agents.research import parse_plan
from tests.conftest import QUESTION
from tests.conftest import collect
from tests.conftest import run

SUBS = [
    "What powers does the Constitution give to Congress?",
    "What powers does the Constitution give to the President?",
    "What powers does the Constitution give to the courts?",
    "What powers does the Constitution give to the states?",
]


def test_planner_fans_out_one_researcher_per_sub_question(
    corpus: Corpus, settings: Settings
) -> None:
    model = FakeDeskModel()
    result = run(build_research_desk(model, corpus, settings), {"question": QUESTION})
    assert result["sub_questions"] == SUBS
    # 1 plan + 2 calls per branch (tool call, then answer) + 1 synthesis.
    assert model.calls.count("researcher") == 2 * len(SUBS)
    assert model.calls[0] == "planner" and model.calls[-1] == "synthesizer"
    assert [f["index"] for f in result["findings"]] == [0, 1, 2, 3]
    assert result["failures"] == []


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_max_branches_caps_the_fan_out(corpus: Corpus, settings: Settings, limit: int) -> None:
    model = FakeDeskModel()
    result = run(
        build_research_desk(model, corpus, settings), {"question": QUESTION, "max_branches": limit}
    )
    assert len(result["sub_questions"]) == limit
    assert len(result["findings"]) == limit
    assert model.calls.count("researcher") == 2 * limit


def test_reducer_merges_every_branch_and_sources_follow_findings(
    corpus: Corpus, settings: Settings
) -> None:
    result = run(build_research_desk(FakeDeskModel(), corpus, settings), {"question": QUESTION})
    findings = result["findings"]
    assert [f["sub_question"] for f in findings] == SUBS
    assert findings[0]["citations"] == ["us-constitution:article-1-section-1"]
    assert findings[2]["citations"] == ["us-constitution:article-3-section-1"]
    expected_sources = list(dict.fromkeys(c for f in findings for c in f["citations"]))
    assert result["sources"] == expected_sources
    for source in result["sources"]:
        assert f"[{source}]" in result["answer"]


def test_output_is_identical_whatever_order_the_branches_finish(
    corpus: Corpus, settings: Settings
) -> None:
    """Random per-call latency reorders completions; the reduced state must not change."""
    outputs = []
    for seed in range(4):
        rng = random.Random(seed)
        model = FakeDeskModel(latency_for=lambda role, prompt, rng=rng: rng.uniform(0, 0.03))
        outputs.append(run(build_research_desk(model, corpus, settings), {"question": QUESTION}))
    first = outputs[0]
    for other in outputs[1:]:
        assert other["findings"] == first["findings"]
        assert other["answer"] == first["answer"]
        assert other["sources"] == first["sources"]


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_max_concurrency_bounds_the_branches_in_flight(
    corpus: Corpus, settings: Settings, limit: int
) -> None:
    model = FakeDeskModel(latency=0.02)
    graph = build_research_desk(model, corpus, settings)
    result = run(graph, {"question": QUESTION}, {"max_concurrency": limit})
    assert model.peak_concurrency == limit
    assert len(result["findings"]) == 4


def test_unbounded_fan_out_runs_all_branches_at_once(corpus: Corpus, settings: Settings) -> None:
    model = FakeDeskModel(latency=0.02)
    run(build_research_desk(model, corpus, settings), {"question": QUESTION})
    assert model.peak_concurrency == 4


def test_settings_run_config_carries_max_concurrency(corpus: Corpus, settings: Settings) -> None:
    model = FakeDeskModel(latency=0.02)
    bounded = dataclasses.replace(settings, max_concurrency=2)
    run(
        build_research_desk(model, corpus, bounded),
        {"question": QUESTION},
        dict(bounded.run_config()),
    )
    assert model.peak_concurrency == 2


def test_a_failing_branch_does_not_kill_the_run(corpus: Corpus, settings: Settings) -> None:
    model = FakeDeskModel(permanent_failures=("the President",))
    result = run(build_research_desk(model, corpus, settings), {"question": QUESTION})
    assert [f["index"] for f in result["findings"]] == [0, 2, 3]
    [failure] = result["failures"]
    assert failure["index"] == 1
    assert failure["error"] == "RuntimeError"
    assert failure["task"] == SUBS[1]
    assert "Not researched" in result["answer"] and SUBS[1] in result["answer"]


def test_non_retryable_errors_are_not_retried(corpus: Corpus, settings: Settings) -> None:
    model = FakeDeskModel(permanent_failures=("the courts",))
    run(
        build_research_desk(model, corpus, dataclasses.replace(settings, retries=3)),
        {"question": QUESTION},
    )
    # 3 healthy branches x 2 calls + exactly 1 attempt for the RuntimeError branch.
    assert model.calls.count("researcher") == 7


def test_transient_errors_are_retried_by_the_retry_policy(
    corpus: Corpus, settings: Settings
) -> None:
    model = FakeDeskModel(transient_failures={"the states": 1})
    result = run(build_research_desk(model, corpus, settings), {"question": QUESTION})
    assert result["failures"] == []
    assert len(result["findings"]) == 4
    assert model.calls.count("researcher") == 2 * 4 + 1


def test_retries_give_up_after_max_attempts(corpus: Corpus, settings: Settings) -> None:
    model = FakeDeskModel(transient_failures={"Congress": 99})
    result = run(build_research_desk(model, corpus, settings), {"question": QUESTION})
    [failure] = result["failures"]
    assert failure == {**failure, "index": 0, "error": "ConnectionError"}
    assert model.calls.count("researcher") == 2 * 3 + settings.retries


def test_a_stalled_model_call_times_out_and_is_isolated(corpus: Corpus, settings: Settings) -> None:
    model = FakeDeskModel(stalls={"the courts": 30.0})
    tight = dataclasses.replace(settings, call_timeout=0.2, retries=2)
    result = run(build_research_desk(model, corpus, tight), {"question": QUESTION})
    [failure] = result["failures"]
    assert failure["index"] == 2
    assert failure["error"] == "NodeTimeoutError"
    assert "timed out" in failure["message"]
    assert len(result["findings"]) == 3
    assert model.calls.count("researcher") == 2 * 3 + 2  # the timeout was retried once


def test_the_branch_budget_caps_a_whole_branch(corpus: Corpus, settings: Settings) -> None:
    model = FakeDeskModel(stalls={"the states": 30.0})
    budget = dataclasses.replace(settings, call_timeout=0, branch_timeout=0.3)
    result = run(build_research_desk(model, corpus, budget), {"question": QUESTION})
    [failure] = result["failures"]
    assert failure == {**failure, "index": 3, "error": "BranchTimeout"}


def test_every_branch_failing_still_returns_an_answer(corpus: Corpus, settings: Settings) -> None:
    model = FakeDeskModel(permanent_failures=("What",))
    result = run(build_research_desk(model, corpus, settings), {"question": QUESTION})
    assert result["findings"] == []
    assert len(result["failures"]) == 4
    assert result["answer"].startswith("No sub-question could be researched")
    assert result["sources"] == []


def test_streaming_reports_progress_for_every_branch(corpus: Corpus, settings: Settings) -> None:
    graph = build_research_desk(FakeDeskModel(), corpus, settings)
    chunks = asyncio.run(collect(graph, {"question": QUESTION}))
    custom = [data for _, mode, data in chunks if mode == "custom"]
    assert custom[0] == {"event": "plan", "sub_questions": SUBS}
    for event in ("start", "tool", "done"):
        assert sorted(e["branch"] for e in custom if e.get("event") == event) == [0, 1, 2, 3]
    # Updates from inside the researcher subgraphs come with their namespace.
    nested = {ns[0].split(":")[0] for ns, mode, _ in chunks if mode == "updates" and ns}
    assert nested == {"research"}


class _LyingModel(FakeDeskModel):
    """Cites a passage it never retrieved."""

    def _respond(self, role, prompt, messages):  # type: ignore[no-untyped-def]
        reply = super()._respond(role, prompt, messages)
        if role == "researcher" and reply.text:
            return AIMessage(content=reply.text + " Also see [us-constitution:article-7].")
        return reply


def test_citations_to_passages_never_retrieved_are_removed(
    corpus: Corpus, settings: Settings
) -> None:
    result = run(
        build_research_desk(_LyingModel(), corpus, settings),
        {"question": QUESTION, "max_branches": 1},
    )
    [finding] = result["findings"]
    assert "us-constitution:article-7" not in finding["citations"]
    assert "[us-constitution:article-7]" not in finding["summary"]
    assert "[us-constitution:article-7]" not in result["answer"]


class _ToolHappyModel(FakeDeskModel):
    """Never stops calling the tool."""

    def _respond(self, role, prompt, messages):  # type: ignore[no-untyped-def]
        if role == "researcher":
            call = {"name": "search_corpus", "args": {"query": "spies"}, "id": f"c{len(messages)}"}
            return AIMessage(content="", tool_calls=[call])
        return super()._respond(role, prompt, messages)


def test_the_tool_loop_is_bounded(corpus: Corpus, settings: Settings) -> None:
    model = _ToolHappyModel()
    researcher = build_researcher(model, corpus, dataclasses.replace(settings, max_tool_steps=2))
    out = asyncio.run(researcher.ainvoke({"index": 0, "sub_question": "spies?"}))
    assert model.calls.count("researcher") == 3
    assert out["findings"][0]["summary"] == "No answer."
    assert out["findings"][0]["citations"] == []


def test_search_tool_clamps_untrusted_arguments(corpus: Corpus) -> None:
    tool = make_search_tool(corpus)
    hits = json.loads(tool.invoke({"query": "spies " * 150, "k": 5}))  # cut to 300 chars
    assert 1 <= len(hits) <= 5
    assert set(hits[0]) == {"id", "title", "text"}
    with pytest.raises(ValueError):
        tool.invoke({"query": "spies", "k": 50})
    with pytest.raises(ValueError):
        tool.invoke({"query": "x" * 1001})


def test_tool_messages_with_garbage_are_ignored_when_collecting_citations() -> None:
    from parallel_agents.research import _retrieved_ids

    messages = [
        ToolMessage("not json", tool_call_id="1"),
        ToolMessage('[{"id": "a:b"}]', tool_call_id="2"),
    ]
    assert _retrieved_ids(messages) == ["a:b"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('["a?", "b?"]', ["a?", "b?"]),
        ('Sure! ["a?", "a?", "  b?  "] hope that helps', ["a?", "b?"]),
        ("1. first?\n2) second?\n- third?", ["first?", "second?", "third?"]),
        ('["x?", 3, null, "y?"]', ["x?", "y?"]),
        ("[not json", ["[not json"]),
        ("", ["fallback"]),
        ('{"a": 1}', ['{"a": 1}']),
    ],
)
def test_parse_plan(text: str, expected: list[str]) -> None:
    assert parse_plan(text, 5, "fallback") == expected


def test_parse_plan_caps_count_and_length() -> None:
    subs = parse_plan(json.dumps([f"q{i} " + "x" * 400 for i in range(10)]), 3, "f")
    assert len(subs) == 3
    assert all(len(s) <= 300 for s in subs)
