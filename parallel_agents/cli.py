"""``python -m parallel_agents "<task>"``: run one of the three parallel graphs.

python -m parallel_agents "What powers does the Constitution give to Congress and the courts?"
python -m parallel_agents --pattern sectioning "We the People of the United States..."
python -m parallel_agents --pattern voting "How many senators does each state have?"
python -m parallel_agents --provider fake --pattern research "..."   # offline, no model
python -m parallel_agents --mermaid research                          # print the graph
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from collections.abc import Sequence
from typing import Any
from typing import TextIO

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from parallel_agents.config import PROVIDERS
from parallel_agents.config import Settings
from parallel_agents.corpus import Corpus
from parallel_agents.llm import ModelConfigError
from parallel_agents.llm import make_model
from parallel_agents.progress import Progress
from parallel_agents.progress import format_progress
from parallel_agents.progress import stream_run
from parallel_agents.research import build_research_desk
from parallel_agents.research import build_researcher
from parallel_agents.sectioning import build_sectioning
from parallel_agents.voting import build_voting

__all__ = ["build_graph", "main"]

PATTERNS = ("research", "sectioning", "voting")
MAX_TASK_CHARS = 20_000


def build_graph(
    pattern: str, model: BaseChatModel, settings: Settings
) -> CompiledStateGraph[Any, Any, Any, Any]:
    if pattern == "research":
        return build_research_desk(model, Corpus.from_dir(settings.corpus_dir), settings)
    if pattern == "researcher":
        return build_researcher(model, Corpus.from_dir(settings.corpus_dir), settings)
    if pattern == "sectioning":
        return build_sectioning(model, settings)
    if pattern == "voting":
        return build_voting(model, settings)
    raise ValueError(f"unknown pattern {pattern!r}")


def _inputs(pattern: str, task: str, settings: Settings) -> dict[str, Any]:
    if pattern == "research":
        return {"question": task, "max_branches": settings.max_branches}
    if pattern == "sectioning":
        return {"text": task}
    return {"question": task, "samples": settings.samples}


def _render(pattern: str, state: dict[str, Any], out: TextIO) -> None:
    if pattern == "research":
        out.write(f"\n{state.get('answer', '')}\n\nSources:\n")
        for i, source in enumerate(state.get("sources", []), 1):
            out.write(f"  [{i}] {source}\n")
    elif pattern == "sectioning":
        out.write("\n" + json.dumps(state.get("report", {}), indent=2, ensure_ascii=False) + "\n")
    else:
        out.write(
            f"\nAnswer: {state.get('answer')!r}  agreement={state.get('agreement')} "
            f"consensus={state.get('consensus')}\nVotes: {state.get('counts')}\n"
        )
    for failure in state.get("failures", []):
        out.write(
            f"  ! branch {failure['index']} failed: {failure['error']}: {failure['message']}\n"
        )


async def _run(pattern: str, task: str, settings: Settings, out: TextIO, quiet: bool) -> int:
    model = make_model(settings)
    graph = build_graph(pattern, model, settings)
    config = settings.run_config()
    final: dict[str, Any] = {}
    async for item in stream_run(graph, _inputs(pattern, task, settings), config):
        if isinstance(item, Progress):
            if not quiet:
                out.write(format_progress(item) + "\n")
        else:
            final = item
    _render(pattern, final, out)
    return 0


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    stream = out or sys.stdout
    parser = argparse.ArgumentParser(
        prog="python -m parallel_agents",
        description="Parallel LangGraph agents: research desk (map-reduce), sectioning, voting.",
    )
    parser.add_argument("task", nargs="?", help="the question (or, for sectioning, the text)")
    parser.add_argument("--pattern", choices=PATTERNS, default="research")
    parser.add_argument("--provider", choices=PROVIDERS)
    parser.add_argument("--model")
    parser.add_argument("--branches", type=int, help="max sub-questions (research)")
    parser.add_argument("--samples", type=int, help="number of votes (voting)")
    parser.add_argument("--max-concurrency", type=int, help="0 = unbounded")
    parser.add_argument("--latency", type=float, help="fake provider: seconds per call")
    parser.add_argument("--quiet", action="store_true", help="no progress events")
    parser.add_argument(
        "--mermaid",
        choices=(*PATTERNS, "researcher"),
        help="print the graph as Mermaid and exit",
    )
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_env()
    except ValueError as exc:
        parser.error(str(exc))
    changes: dict[str, Any] = {}
    if args.provider:
        changes["provider"] = args.provider
        if not args.model and args.provider != settings.provider:
            changes["model"] = Settings.from_env({"PARALLEL_AGENTS_PROVIDER": args.provider}).model
    if args.model:
        changes["model"] = args.model
    if args.branches is not None:
        changes["max_branches"] = max(1, args.branches)
    if args.samples is not None:
        changes["samples"] = max(1, args.samples)
    if args.max_concurrency is not None:
        changes["max_concurrency"] = args.max_concurrency or None
    if args.latency is not None:
        changes["fake_latency"] = max(0.0, args.latency)
    if args.pattern == "voting" and settings.temperature == 0.0:
        changes["temperature"] = 0.7  # votes need diversity
    settings = dataclasses.replace(settings, **changes)

    if args.mermaid:
        from parallel_agents.fake import FakeDeskModel

        graph = build_graph(args.mermaid, FakeDeskModel(), settings)
        stream.write(graph.get_graph().draw_mermaid())
        return 0
    if not args.task:
        parser.error("a task is required (or use --mermaid)")
    if len(args.task) > MAX_TASK_CHARS:
        parser.error(f"the task is longer than {MAX_TASK_CHARS} characters")
    try:
        return asyncio.run(_run(args.pattern, args.task, settings, stream, args.quiet))
    except ModelConfigError as exc:
        stream.write(f"error: {exc}\n")
        return 2
