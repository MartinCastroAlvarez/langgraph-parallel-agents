"""Voting: ask the same question N times in parallel, return the majority answer.

``Send`` fans out N ``sample`` tasks (self-consistency). Each vote lands in
``votes: Annotated[list, operator.add]``; ``tally`` normalises the answers ("Two.",
"two" and "2" are one answer), counts them and picks the winner. Ties go to the answer
whose first vote has the lowest sample index, so the result never depends on which
branch finished first. Failed samples are reported and simply do not vote.
"""

from __future__ import annotations

import operator
import re
from collections import Counter
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
from langgraph.types import Send
from langgraph.types import TimeoutPolicy

from parallel_agents import prompts
from parallel_agents.config import Settings
from parallel_agents.isolation import BranchFailure
from parallel_agents.isolation import branch_retry_policy
from parallel_agents.isolation import run_isolated
from parallel_agents.isolation import single_node_graph

__all__ = ["Vote", "VotingState", "build_voting", "normalize_answer", "tally_votes"]

MAX_ANSWER_CHARS = 120
_NUMBERS = {
    w: str(i)
    for i, w in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
        ]
    )
}


class Vote(TypedDict):
    index: int
    raw: str
    answer: str


class VotingState(TypedDict, total=False):
    question: str
    samples: int
    votes: Annotated[list[Vote], operator.add]
    failures: Annotated[list[BranchFailure], operator.add]
    answer: str
    counts: dict[str, int]
    agreement: float
    consensus: bool


class SampleState(TypedDict, total=False):
    index: int
    total: int
    question: str
    raw: str


def normalize_answer(text: str) -> str:
    """Case-, punctuation- and number-word-insensitive form of a short answer."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    words = re.findall(r"[a-z0-9]+", first.lower())[:12]
    words = [_NUMBERS.get(w, w) for w in words if w not in {"the", "a", "an"}]
    return " ".join(words)[:MAX_ANSWER_CHARS]


def tally_votes(votes: list[Vote]) -> tuple[str, dict[str, int]]:
    """Majority answer and the counts; ties break on the earliest sample index."""
    ordered = sorted((v for v in votes if v["answer"]), key=lambda v: v["index"])
    counts = Counter(v["answer"] for v in ordered)
    if not counts:
        return "", {}
    first_seen: dict[str, int] = {}
    for vote in ordered:
        first_seen.setdefault(vote["answer"], vote["index"])
    winner = min(counts, key=lambda a: (-counts[a], first_seen[a]))
    return winner, dict(sorted(counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]])))


def build_voting(
    model: BaseChatModel, settings: Settings
) -> CompiledStateGraph[Any, Any, Any, Any]:
    retry = branch_retry_policy(settings.retries)
    timeout = TimeoutPolicy(run_timeout=settings.call_timeout or None)

    async def ask(state: SampleState) -> dict[str, Any]:
        reply = await model.ainvoke(
            [
                SystemMessage(prompts.VOTER),
                HumanMessage(
                    prompts.data_block("question", state["question"])
                    + f"\nSample {state['index']} of {state['total']}."
                ),
            ]
        )
        return {"raw": reply.text.strip()[:MAX_ANSWER_CHARS]}

    sampler = single_node_graph(SampleState, "ask", ask, retry=retry, timeout=timeout)

    def fan_out(state: VotingState) -> list[Send]:
        n = state.get("samples") or settings.samples
        return [
            Send("sample", SampleState(index=i, total=n, question=state["question"]))
            for i in range(n)
        ]

    async def sample(state: SampleState, config: RunnableConfig) -> dict[str, Any]:
        output, failure = await run_isolated(
            sampler,
            dict(state),
            config,
            index=state["index"],
            task=f"sample {state['index']}",
            budget=settings.branch_timeout,
        )
        if output is None:
            return {"failures": [failure] if failure else []}
        raw = str(output.get("raw", ""))
        return {"votes": [Vote(index=state["index"], raw=raw, answer=normalize_answer(raw))]}

    def tally(state: VotingState) -> dict[str, Any]:
        votes = state.get("votes", [])
        winner, counts = tally_votes(votes)
        cast = sum(counts.values())
        top = counts.get(winner, 0)
        return {
            "answer": winner,
            "counts": counts,
            "agreement": round(top / cast, 3) if cast else 0.0,
            "consensus": top * 2 > cast,
        }

    graph = StateGraph(VotingState)
    graph.add_node("sample", sample)
    graph.add_node("tally", tally)
    graph.add_conditional_edges(START, fan_out, ["sample"])
    graph.add_edge("sample", "tally")
    graph.add_edge("tally", END)
    return graph.compile(name="voting")
