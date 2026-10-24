"""Shared fixtures: the bundled corpus, fake-provider settings and a run helper."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from parallel_agents.config import Settings
from parallel_agents.corpus import Corpus

QUESTION = (
    "What powers does the Constitution give to Congress, the President, the courts and the states?"
)


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return Corpus.from_dir()


@pytest.fixture
def settings() -> Settings:
    return Settings(provider="fake", retries=2, call_timeout=5.0, branch_timeout=10.0)


def run(graph: Any, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = asyncio.run(graph.ainvoke(inputs, config or {}))
    return result


async def collect(
    graph: Any, inputs: dict[str, Any], config: dict[str, Any] | None = None
) -> list[tuple[Any, ...]]:
    return [
        chunk
        async for chunk in graph.astream(
            inputs, config or {}, stream_mode=["updates", "custom"], subgraphs=True
        )
    ]
