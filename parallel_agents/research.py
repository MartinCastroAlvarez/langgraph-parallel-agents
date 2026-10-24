"""The research desk: plan -> ``Send`` N researcher subgraphs -> reduce -> cited answer.

``planner`` splits the question into sub-questions; a conditional edge returns one
``Send("research", ...)`` per sub-question, so LangGraph runs the branches in the same
superstep, at most ``max_concurrency`` at a time. Each branch is a small tool-using
agent (its own compiled ``StateGraph``: model <-> ``search_corpus`` tool) wrapped by
:func:`~parallel_agents.isolation.run_isolated`. Branch results land in
``findings: Annotated[list, operator.add]`` - the reducer that makes the fan-in safe -
and ``synthesizer`` writes the answer once every branch has reported.
"""

from __future__ import annotations

import json
import operator
import re
from collections.abc import Sequence
from typing import Annotated
from typing import Any
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import AnyMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send
from langgraph.types import TimeoutPolicy
from pydantic import BaseModel
from pydantic import Field

from parallel_agents import prompts
from parallel_agents.config import Settings
from parallel_agents.corpus import MAX_QUERY_CHARS
from parallel_agents.corpus import Corpus
from parallel_agents.isolation import BranchFailure
from parallel_agents.isolation import branch_retry_policy
from parallel_agents.isolation import emit
from parallel_agents.isolation import run_isolated

__all__ = [
    "DeskState",
    "Finding",
    "build_research_desk",
    "build_researcher",
    "make_search_tool",
    "parse_plan",
]

MAX_SUB_QUESTION_CHARS = 300
_CITATION = re.compile(r"\[([a-z0-9][a-z0-9:-]*)\]")


class Finding(TypedDict):
    """What one researcher branch reports back."""

    index: int
    sub_question: str
    summary: str
    citations: list[str]


class DeskState(TypedDict, total=False):
    question: str
    max_branches: int
    sub_questions: list[str]
    # The fan-in: every branch appends; the reducer concatenates in task order.
    findings: Annotated[list[Finding], operator.add]
    failures: Annotated[list[BranchFailure], operator.add]
    answer: str
    sources: list[str]


class BranchInput(TypedDict):
    index: int
    sub_question: str


class ResearcherState(TypedDict, total=False):
    index: int
    sub_question: str
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    findings: list[Finding]


class ResearcherOutput(TypedDict):
    findings: list[Finding]


class SearchArgs(BaseModel):
    query: str = Field(description="What to look for in the documents.", max_length=1000)
    k: int = Field(default=3, ge=1, le=5, description="How many passages to return.")


def make_search_tool(corpus: Corpus, default_k: int = 3) -> BaseTool:
    """``search_corpus``: BM25 over the local corpus. Input is untrusted and clamped."""

    def search_corpus(query: str, k: int = default_k) -> str:
        hits = corpus.search(query[:MAX_QUERY_CHARS], k=max(1, min(k, 5)))
        return json.dumps(
            [{"id": h.passage.id, "title": h.passage.title, "text": h.passage.text} for h in hits]
        )

    return StructuredTool.from_function(
        func=search_corpus,
        name="search_corpus",
        description="Search the local document collection. Returns passages with ids to cite.",
        args_schema=SearchArgs,
    )


def parse_plan(text: str, limit: int, fallback: str) -> list[str]:
    """Sub-questions from a planner reply: a JSON array, else one per line. Never empty."""
    candidates: list[Any]
    match = re.search(r"\[.*\]", text, re.S)
    try:
        candidates = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        candidates = []
    if not isinstance(candidates, list) or not candidates:
        candidates = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line) for line in text.splitlines()]
    seen: dict[str, None] = {}
    for item in candidates:
        if isinstance(item, str) and (clean := " ".join(item.split())[:MAX_SUB_QUESTION_CHARS]):
            seen.setdefault(clean, None)
    return list(seen)[:limit] or [fallback[:MAX_SUB_QUESTION_CHARS]]


def _retrieved_ids(messages: Sequence[AnyMessage]) -> list[str]:
    ids: dict[str, None] = {}
    for message in messages:
        if isinstance(message, ToolMessage):
            try:
                for hit in json.loads(message.text):
                    ids.setdefault(str(hit["id"]), None)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
    return list(ids)


def build_researcher(
    model: BaseChatModel, corpus: Corpus, settings: Settings
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """One map branch: a tool-calling loop over ``search_corpus``, then a cited report."""
    tool = make_search_tool(corpus, settings.top_k)
    llm = model.bind_tools([tool])
    retry = branch_retry_policy(settings.retries)
    timeout = TimeoutPolicy(run_timeout=settings.call_timeout or None)

    async def agent(state: ResearcherState) -> dict[str, Any]:
        messages = state.get("messages") or [
            SystemMessage(prompts.RESEARCHER),
            HumanMessage(prompts.data_block("sub_question", state["sub_question"])),
        ]
        reply = await llm.ainvoke(messages)
        new = [] if state.get("messages") else messages
        for call in reply.tool_calls if isinstance(reply, AIMessage) else []:
            emit({"branch": state["index"], "event": "tool", "query": call["args"].get("query")})
        return {"messages": [*new, reply], "steps": state.get("steps", 0) + 1}

    def route(state: ResearcherState) -> str:
        last = state["messages"][-1]
        wants_tool = isinstance(last, AIMessage) and bool(last.tool_calls)
        return (
            "tools" if wants_tool and state.get("steps", 0) <= settings.max_tool_steps else "report"
        )

    def report(state: ResearcherState) -> ResearcherOutput:
        messages = state.get("messages", [])
        retrieved = _retrieved_ids(messages)
        last = messages[-1] if messages else None
        summary = last.text.strip() if isinstance(last, AIMessage) and not last.tool_calls else ""
        cited = [c for c in dict.fromkeys(_CITATION.findall(summary)) if c in retrieved]
        # A citation the model made up is removed from the text, not just from the list.
        summary = _CITATION.sub(lambda m: m.group(0) if m.group(1) in cited else "", summary)
        finding = Finding(
            index=state["index"],
            sub_question=state["sub_question"],
            summary=" ".join(summary.split()) or "No answer.",
            citations=cited,
        )
        return {"findings": [finding]}

    graph = StateGraph(ResearcherState, input_schema=BranchInput, output_schema=ResearcherOutput)
    graph.add_node("agent", agent, retry_policy=retry, timeout=timeout)
    graph.add_node("tools", ToolNode([tool]))
    graph.add_node("report", report)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", "report"])
    graph.add_edge("tools", "agent")
    graph.add_edge("report", END)
    return graph.compile(name="researcher")


def build_research_desk(
    model: BaseChatModel, corpus: Corpus, settings: Settings
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """The map-reduce graph: ``planner -> Send x N -> research -> synthesizer``."""
    researcher = build_researcher(model, corpus, settings)
    retry = branch_retry_policy(settings.retries)
    timeout = TimeoutPolicy(run_timeout=settings.call_timeout or None)

    async def planner(state: DeskState) -> dict[str, Any]:
        limit = state.get("max_branches") or settings.max_branches
        reply = await model.ainvoke(
            [
                SystemMessage(prompts.PLANNER),
                HumanMessage(
                    prompts.data_block("question", state["question"])
                    + f"\nReturn at most {limit} sub-questions."
                ),
            ]
        )
        subs = parse_plan(reply.text, limit, state["question"])
        emit({"event": "plan", "sub_questions": subs})
        return {"sub_questions": subs}

    def fan_out(state: DeskState) -> list[Send]:
        return [
            Send("research", BranchInput(index=i, sub_question=q))
            for i, q in enumerate(state["sub_questions"])
        ]

    async def research(state: BranchInput, config: RunnableConfig) -> dict[str, Any]:
        output, failure = await run_isolated(
            researcher,
            dict(state),
            config,
            index=state["index"],
            task=state["sub_question"],
            budget=settings.branch_timeout,
        )
        if failure is not None or output is None:
            return {"failures": [failure] if failure else []}
        return {"findings": output["findings"]}

    async def synthesizer(state: DeskState) -> dict[str, Any]:
        findings = sorted(state.get("findings", []), key=lambda f: f["index"])
        failures = sorted(state.get("failures", []), key=lambda f: f["index"])
        sources = list(dict.fromkeys(c for f in findings for c in f["citations"]))
        if not findings:
            gaps = "; ".join(f["task"] for f in failures)
            return {"answer": f"No sub-question could be researched ({gaps}).", "sources": []}
        lines = [f"[{f['index'] + 1}] {f['sub_question']}: {f['summary']}" for f in findings]
        content = prompts.data_block("question", state["question"])
        content += "\n" + prompts.data_block("findings", "\n".join(lines))
        if failures:
            content += "\n" + prompts.data_block("gaps", "\n".join(f["task"] for f in failures))
        reply = await model.ainvoke([SystemMessage(prompts.SYNTHESIZER), HumanMessage(content)])
        answer = _CITATION.sub(lambda m: m.group(0) if m.group(1) in sources else "", reply.text)
        return {"answer": " ".join(answer.split()), "sources": sources}

    graph = StateGraph(DeskState)
    graph.add_node("planner", planner, retry_policy=retry, timeout=timeout)
    graph.add_node("research", research)
    graph.add_node("synthesizer", synthesizer, retry_policy=retry, timeout=timeout)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", fan_out, ["research"])
    graph.add_edge("research", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile(name="research_desk")
