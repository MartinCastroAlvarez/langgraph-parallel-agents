"""Pins a langgraph behaviour this repo works around (see parallel_agents/isolation.py).

``add_node(..., error_handler=...)`` recovers a failing node when it runs alone, but in
langgraph 1.2.17 the handled exception is re-raised when parallel siblings share the
superstep. If an upgrade fixes that, the xfail turns into an XPASS and fails the suite:
time to revisit the isolation wrapper.
"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated
from typing import Any
from typing import TypedDict

import pytest
from langgraph.errors import NodeError
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph


class S(TypedDict, total=False):
    errors: Annotated[list[str], operator.add]
    ok: Annotated[list[str], operator.add]


async def _fails(state: S) -> dict[str, Any]:
    raise RuntimeError("boom")


async def _works(state: S) -> dict[str, Any]:
    await asyncio.sleep(0.01)
    return {"ok": ["b"]}


def _handler(state: S, error: NodeError) -> dict[str, Any]:
    return {"errors": [error.node]}


def _graph(parallel: bool) -> Any:
    g = StateGraph(S)
    g.add_node("a", _fails, error_handler=_handler)
    g.add_edge(START, "a")
    g.add_edge("a", END)
    if parallel:
        g.add_node("b", _works)
        g.add_edge(START, "b")
        g.add_edge("b", END)
    return g.compile()


def test_error_handler_recovers_a_lone_node() -> None:
    assert asyncio.run(_graph(parallel=False).ainvoke({}))["errors"] == ["a"]


@pytest.mark.xfail(
    strict=True, raises=RuntimeError, reason="langgraph 1.2.17 re-raises with parallel siblings"
)
def test_error_handler_with_parallel_siblings() -> None:
    assert asyncio.run(_graph(parallel=True).ainvoke({}))["errors"] == ["a"]
