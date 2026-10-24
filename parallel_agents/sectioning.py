"""Sectioning: one input, three specialists in parallel, one deferred aggregator.

A static fan-out - plain edges from ``START`` to ``sentiment``, ``entities`` and
``summary`` - runs the three in the same superstep. The branches are uneven on
purpose: ``entities`` is two steps (``entities`` -> ``normalize``). Without
``defer=True`` the aggregator would fire as soon as the short branches finish and
again after the long one; deferred, it runs exactly once, after everything else.

Each specialist's model call runs inside its own one-node subgraph with a
``RetryPolicy`` and a ``TimeoutPolicy``; whatever survives the retries becomes a
``BranchFailure`` (see :mod:`parallel_agents.isolation`) and the report says which
section is missing, while the other specialists carry on.
"""

from __future__ import annotations

import json
import operator
import re
from typing import Annotated
from typing import Any
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import TimeoutPolicy

from parallel_agents import prompts
from parallel_agents.config import Settings
from parallel_agents.isolation import BranchFailure
from parallel_agents.isolation import branch_retry_policy
from parallel_agents.isolation import run_isolated
from parallel_agents.isolation import single_node_graph

__all__ = ["SECTIONS", "SectionState", "build_sectioning"]

SECTIONS = ("sentiment", "entities", "summary")
_SENTIMENTS = ("positive", "negative", "neutral")
MAX_ENTITIES = 25


class SectionState(TypedDict, total=False):
    text: str
    sentiment: str
    raw_entities: list[str]
    entities: list[str]
    summary: str
    failures: Annotated[list[BranchFailure], operator.add]
    report: dict[str, Any]


def parse_sentiment(text: str) -> str:
    words = re.findall(r"[a-z]+", text.lower())
    return next((w for w in words if w in _SENTIMENTS), "neutral")


def parse_entities(text: str) -> list[str]:
    match = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        items = []
    if not isinstance(items, list):
        return []
    return [str(i) for i in items if isinstance(i, str | int | float)]


def build_sectioning(
    model: BaseChatModel, settings: Settings, *, defer: bool = True
) -> CompiledStateGraph[Any, Any, Any, Any]:
    retry = branch_retry_policy(settings.retries)
    timeout = TimeoutPolicy(run_timeout=settings.call_timeout or None)

    async def ask(role: str, text: str) -> str:
        reply = await model.ainvoke(
            [
                SystemMessage(prompts.SECTION_PROMPTS[role]),
                HumanMessage(prompts.data_block("text", text)),
            ]
        )
        return reply.text.strip()

    async def sentiment(state: SectionState) -> dict[str, Any]:
        return {"sentiment": parse_sentiment(await ask("sentiment", state["text"]))}

    async def entities(state: SectionState) -> dict[str, Any]:
        return {"raw_entities": parse_entities(await ask("entities", state["text"]))}

    async def summary(state: SectionState) -> dict[str, Any]:
        return {"summary": " ".join((await ask("summary", state["text"])).split())}

    def specialist(name: str, writes: str, fn: Any) -> Any:
        inner = single_node_graph(SectionState, name, fn, retry=retry, timeout=timeout)

        async def node(state: SectionState, config: RunnableConfig) -> dict[str, Any]:
            output, failure = await run_isolated(
                inner,
                {"text": state["text"]},
                config,
                index=SECTIONS.index(name),
                task=name,
                budget=settings.branch_timeout,
            )
            if output is None:
                return {"failures": [failure] if failure else []}
            return {writes: output.get(writes)}

        node.__name__ = name
        return node

    def normalize(state: SectionState) -> dict[str, Any]:
        """Second step of the long branch: trim, de-duplicate case-insensitively, sort."""
        unique: dict[str, str] = {}
        for item in state.get("raw_entities") or []:
            clean = " ".join(item.split())[:80]
            if clean:
                unique.setdefault(clean.casefold(), clean)
        return {"entities": sorted(unique.values(), key=str.casefold)[:MAX_ENTITIES]}

    def aggregate(state: SectionState) -> dict[str, Any]:
        failed = sorted(f["task"] for f in state.get("failures", []))
        return {
            "report": {
                "sentiment": state.get("sentiment"),
                "entities": state.get("entities", []),
                "summary": state.get("summary"),
                "failed": failed,
            }
        }

    graph = StateGraph(SectionState)
    for name, writes, fn in (
        ("sentiment", "sentiment", sentiment),
        ("entities", "raw_entities", entities),
        ("summary", "summary", summary),
    ):
        graph.add_node(name, specialist(name, writes, fn))
        graph.add_edge(START, name)
    graph.add_node("normalize", normalize)
    graph.add_node("aggregate", aggregate, defer=defer)
    graph.add_edge("entities", "normalize")
    for name in ("sentiment", "normalize", "summary"):
        graph.add_edge(name, "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile(name="sectioning")
