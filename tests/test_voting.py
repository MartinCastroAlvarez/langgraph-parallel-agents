"""Voting: N samples in parallel, normalised majority, deterministic tie-break."""

from __future__ import annotations

import random

import pytest

from parallel_agents.config import Settings
from parallel_agents.fake import FakeDeskModel
from parallel_agents.voting import Vote
from parallel_agents.voting import build_voting
from parallel_agents.voting import normalize_answer
from parallel_agents.voting import tally_votes
from tests.conftest import run

Q = {"question": "How many senators does each state have?"}


def test_majority_wins_after_normalisation(settings: Settings) -> None:
    model = FakeDeskModel()  # votes: "2", "two", "Two.", "3", "2"
    result = run(build_voting(model, settings), {**Q, "samples": 5})
    assert model.calls == ["voter"] * 5
    assert result["answer"] == "2"
    assert result["counts"] == {"2": 4, "3": 1}
    assert result["agreement"] == 0.8
    assert result["consensus"] is True
    assert [v["index"] for v in result["votes"]] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("samples", [1, 3, 9])
def test_fan_out_count_matches_samples(settings: Settings, samples: int) -> None:
    model = FakeDeskModel()
    result = run(build_voting(model, settings), {**Q, "samples": samples})
    assert len(model.calls) == samples
    assert len(result["votes"]) == samples


def test_samples_default_comes_from_settings(settings: Settings) -> None:
    model = FakeDeskModel()
    run(build_voting(model, settings), Q)
    assert len(model.calls) == settings.samples


def test_result_does_not_depend_on_completion_order(settings: Settings) -> None:
    results = []
    for seed in range(3):
        rng = random.Random(seed)
        model = FakeDeskModel(
            votes=("x", "y", "y", "x", "z"), latency_for=lambda r, p, rng=rng: rng.uniform(0, 0.03)
        )
        results.append(run(build_voting(model, settings), {**Q, "samples": 5}))
    assert all(r["votes"] == results[0]["votes"] for r in results)
    # x and y tie 2-2: the tie goes to the answer seen first (sample 0 voted "x").
    assert {r["answer"] for r in results} == {"x"}
    assert results[0]["consensus"] is False


def test_failed_samples_do_not_vote(settings: Settings) -> None:
    model = FakeDeskModel(votes=("2", "3", "3"), permanent_failures=("Sample 1 of",))
    result = run(build_voting(model, settings), {**Q, "samples": 3})
    assert [f["index"] for f in result["failures"]] == [1]
    assert result["counts"] == {"2": 1, "3": 1}
    assert result["answer"] == "2"


def test_every_sample_failing_gives_no_answer(settings: Settings) -> None:
    model = FakeDeskModel(permanent_failures=("Sample",))
    result = run(build_voting(model, settings), {**Q, "samples": 3})
    assert result["answer"] == ""
    assert result["counts"] == {}
    assert result["agreement"] == 0.0
    assert result["consensus"] is False


def test_max_concurrency_bounds_the_samples(settings: Settings) -> None:
    model = FakeDeskModel(latency=0.02)
    run(build_voting(model, settings), {**Q, "samples": 6}, {"max_concurrency": 2})
    assert model.peak_concurrency == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Two.", "2"),
        ("  TWO senators\nbecause...", "2 senators"),
        ("The Senate", "senate"),
        ("", ""),
        ("!!!", ""),
    ],
)
def test_normalize_answer(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


def test_tally_ignores_empty_answers_and_orders_counts() -> None:
    votes = [
        Vote(index=2, raw="b", answer="b"),
        Vote(index=0, raw="", answer=""),
        Vote(index=1, raw="a", answer="a"),
        Vote(index=3, raw="b", answer="b"),
    ]
    assert tally_votes(votes) == ("b", {"b": 2, "a": 1})
    assert tally_votes([]) == ("", {})
