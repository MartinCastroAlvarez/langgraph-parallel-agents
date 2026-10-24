"""Sectioning: static fan-out to three specialists, deferred aggregator, isolation."""

from __future__ import annotations

import asyncio

import pytest

from parallel_agents.config import Settings
from parallel_agents.fake import FakeDeskModel
from parallel_agents.sectioning import build_sectioning
from parallel_agents.sectioning import parse_entities
from parallel_agents.sectioning import parse_sentiment
from tests.conftest import collect
from tests.conftest import run

TEXT = (
    "We the People of the United States, in Order to form a more perfect Union, establish "
    "Justice, insure domestic Tranquility and secure the Blessings of Liberty."
)


def _aggregations(chunks: list[tuple]) -> int:  # type: ignore[type-arg]
    return sum(
        1 for ns, mode, data in chunks if mode == "updates" and not ns and "aggregate" in data
    )


def test_three_specialists_run_in_parallel_and_report(settings: Settings) -> None:
    model = FakeDeskModel(latency=0.05)
    result = run(build_sectioning(model, settings), {"text": TEXT})
    assert sorted(model.calls) == ["entities", "sentiment", "summary"]
    assert model.peak_concurrency == 3
    report = result["report"]
    assert report["sentiment"] == "positive"
    assert report["entities"] == sorted(report["entities"], key=str.casefold)
    assert {"United States", "Union", "Justice"} <= set(report["entities"])
    assert report["summary"].startswith("We the People")
    assert report["failed"] == []


def test_deferred_aggregator_runs_once_after_the_long_branch(settings: Settings) -> None:
    chunks = asyncio.run(collect(build_sectioning(FakeDeskModel(), settings), {"text": TEXT}))
    assert _aggregations(chunks) == 1
    order = [next(iter(data)) for ns, mode, data in chunks if mode == "updates" and not ns]
    assert order.index("aggregate") > order.index("normalize")


def test_without_defer_the_aggregator_fires_twice(settings: Settings) -> None:
    chunks = asyncio.run(
        collect(build_sectioning(FakeDeskModel(), settings, defer=False), {"text": TEXT})
    )
    assert _aggregations(chunks) == 2


def test_a_failing_specialist_leaves_a_hole_not_a_crash(settings: Settings) -> None:
    model = FakeDeskModel(permanent_failures=("",), fault_roles=("entities",))
    result = run(build_sectioning(model, settings), {"text": TEXT})
    report = result["report"]
    assert report["failed"] == ["entities"]
    assert report["entities"] == []
    assert report["sentiment"] == "positive" and report["summary"]
    assert result["failures"][0]["error"] == "RuntimeError"


def test_a_transient_specialist_failure_is_retried(settings: Settings) -> None:
    model = FakeDeskModel(transient_failures={"": 1}, fault_roles=("summary",))
    result = run(build_sectioning(model, settings), {"text": TEXT})
    assert result["report"]["failed"] == []
    assert model.calls.count("summary") == 2


def test_streams_progress_per_section(settings: Settings) -> None:
    chunks = asyncio.run(collect(build_sectioning(FakeDeskModel(), settings), {"text": TEXT}))
    done = sorted(d["branch"] for _, mode, d in chunks if mode == "custom" and d["event"] == "done")
    assert done == [0, 1, 2]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("Positive.", "positive"), ("The tone is NEGATIVE", "negative"), ("unclear", "neutral")],
)
def test_parse_sentiment(reply: str, expected: str) -> None:
    assert parse_sentiment(reply) == expected


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('["Congress", "Senate"]', ["Congress", "Senate"]),
        ('Here: ["A", 1, null, {"x": 1}]', ["A", "1"]),
        ("no list", []),
        ("[broken", []),
        ('[{"a": 1}]', []),
    ],
)
def test_parse_entities(reply: str, expected: list[str]) -> None:
    assert parse_entities(reply) == expected
