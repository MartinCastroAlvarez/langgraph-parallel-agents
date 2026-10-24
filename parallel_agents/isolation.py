"""Per-branch fault isolation for parallel fan-outs.

LangGraph retries a failing node with its ``RetryPolicy`` and offers
``add_node(..., error_handler=...)`` for what is left. In langgraph 1.2.17 that handler
only saves the run when the failing node is the only task in its superstep: with
siblings running in parallel (a ``Send`` fan-out or a static one) the background
executor re-raises the handled exception when the superstep closes, and the whole run
fails (``tests/test_upstream.py`` pins this). So every branch here runs as a compiled
subgraph whose nodes carry the ``RetryPolicy`` and ``TimeoutPolicy``, and
:func:`run_isolated` - called by the node the fan-out targets - turns whatever is left
after the retries into a ``BranchFailure`` record and gives the branch an overall time
budget. Siblings never see the exception.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any
from typing import TypedDict

from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.errors import GraphBubbleUp
from langgraph.errors import NodeTimeoutError
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
from langgraph.types import TimeoutPolicy

__all__ = ["BranchFailure", "branch_retry_policy", "emit", "run_isolated", "single_node_graph"]

MAX_ERROR_CHARS = 200


class BranchFailure(TypedDict):
    """A branch that gave up; it stays in the state so the reducer can report it."""

    index: int
    task: str
    error: str
    message: str


def branch_retry_policy(attempts: int) -> RetryPolicy:
    """Retry transient faults (connection errors, 5xx, node timeouts) with fast backoff."""
    return RetryPolicy(max_attempts=attempts, initial_interval=0.05, max_interval=2.0)


def single_node_graph(
    schema: type[Any],
    name: str,
    fn: Callable[..., Awaitable[dict[str, Any]]],
    *,
    retry: RetryPolicy,
    timeout: TimeoutPolicy | None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Wrap one async step in its own graph so retries + timeout happen inside the branch."""
    graph = StateGraph(schema)
    graph.add_node(name, fn, retry_policy=retry, timeout=timeout)
    graph.add_edge(START, name)
    graph.add_edge(name, END)
    return graph.compile(name=name)


def emit(event: dict[str, Any]) -> None:
    """Send a ``custom`` stream event; a no-op when the run is not being streamed."""
    try:
        writer = get_stream_writer()
    except RuntimeError:  # called outside a runnable context
        return
    writer(event)


def _describe(exc: BaseException) -> str:
    if isinstance(exc, NodeTimeoutError):
        return f"{exc.node} timed out after {exc.elapsed:.2f}s ({exc.kind} timeout)"
    return str(exc)[:MAX_ERROR_CHARS] or type(exc).__name__


async def run_isolated(
    branch: Runnable[Any, Any],
    payload: dict[str, Any],
    config: RunnableConfig,
    *,
    index: int,
    task: str,
    budget: float,
) -> tuple[dict[str, Any] | None, BranchFailure | None]:
    """Run one branch; return ``(output, None)`` or ``(None, failure)``, never raise.

    Interrupts and other ``GraphBubbleUp`` control-flow signals still propagate.
    """
    started = time.perf_counter()
    emit({"branch": index, "event": "start", "task": task})
    try:
        async with asyncio.timeout(budget or None):
            output = await branch.ainvoke(payload, config)
    except GraphBubbleUp:
        raise
    except TimeoutError:
        failure = BranchFailure(
            index=index, task=task, error="BranchTimeout", message=f"budget of {budget}s spent"
        )
    except Exception as exc:  # isolation boundary: any failure stays in this branch
        failure = BranchFailure(
            index=index, task=task, error=type(exc).__name__, message=_describe(exc)
        )
    else:
        elapsed = round(time.perf_counter() - started, 3)
        emit({"branch": index, "event": "done", "seconds": elapsed})
        return output, None
    emit({"branch": index, "event": "failed", "error": failure["error"]})
    return None, failure
