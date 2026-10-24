"""Stream per-branch progress while a graph runs, then hand back its final state.

Nodes call :func:`~parallel_agents.isolation.emit` (LangGraph's ``get_stream_writer``)
with events such as ``{"branch": 2, "event": "tool", "query": ...}``. Streaming with
``stream_mode=["custom", "values"]`` and ``subgraphs=True`` surfaces the events from
inside every researcher subgraph as they happen, tagged with the time since the start.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

__all__ = ["Progress", "format_progress", "stream_run"]


@dataclass(frozen=True, slots=True)
class Progress:
    seconds: float
    event: dict[str, Any]
    namespace: tuple[str, ...] = ()


async def stream_run(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    inputs: dict[str, Any],
    config: RunnableConfig | None = None,
) -> AsyncIterator[Progress | dict[str, Any]]:
    """Yield a :class:`Progress` per custom event, then the final state (a ``dict``)."""
    started = time.perf_counter()
    final: dict[str, Any] = {}
    async for namespace, mode, data in graph.astream(
        inputs, config, stream_mode=["custom", "values"], subgraphs=True
    ):
        if mode == "custom" and isinstance(data, dict):
            yield Progress(round(time.perf_counter() - started, 3), data, tuple(namespace))
        elif mode == "values" and not namespace and isinstance(data, dict):
            final = data
    yield final


def format_progress(p: Progress) -> str:
    event = dict(p.event)
    branch = event.pop("branch", None)
    kind = event.pop("event", "?")
    who = f"branch {branch}" if branch is not None else "graph"
    detail = " ".join(f"{k}={v!r}" for k, v in event.items())
    return f"[{p.seconds:7.3f}s] {who:<16} {kind:<6} {detail}".rstrip()
