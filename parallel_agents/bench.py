"""Wall-clock benchmark: the same graphs run sequentially, bounded and unbounded.

Every model call goes to :class:`~parallel_agents.fake.FakeDeskModel`, which sleeps a
fixed ``latency`` (default 0.25 s) per call, the way a hosted or local LLM keeps you
waiting. The graph, the reducers and the corpus search are the real ones; only the
model is simulated. "Sequential" is the same graph with ``max_concurrency=1``.

    python -m parallel_agents.bench --latency 0.25 --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from parallel_agents.config import Settings
from parallel_agents.corpus import Corpus
from parallel_agents.fake import FakeDeskModel
from parallel_agents.research import build_research_desk
from parallel_agents.sectioning import build_sectioning
from parallel_agents.voting import build_voting

__all__ = ["Row", "main", "research_question", "run_benchmark"]

TOPICS = (
    "Congress",
    "the President",
    "the courts",
    "the states",
    "treaties",
    "taxes",
    "impeachment",
    "amendments",
    "spies",
    "fire",
    "terrain",
    "energy",
    "the general",
    "supplies",
    "deception",
    "marching",
)

SECTION_TEXT = (
    "We the People of the United States, in Order to form a more perfect Union, establish "
    "Justice, insure domestic Tranquility, provide for the common defence, promote the "
    "general Welfare, and secure the Blessings of Liberty to ourselves and our Posterity."
)

MODES: dict[str, int | None] = {"sequential": 1, "bounded (4)": 4, "parallel": None}


@dataclass(frozen=True, slots=True)
class Row:
    pattern: str
    branches: int
    mode: str
    max_concurrency: int | None
    seconds: float
    model_calls: int
    peak_in_flight: int


def research_question(n: int) -> str:
    topics = list(TOPICS[:n])
    if n == 1:
        return f"What does the corpus say about {topics[0]}?"
    return f"What does the corpus say about {', '.join(topics[:-1])} and {topics[-1]}?"


async def _time(
    graph: Any, inputs: dict[str, Any], config: RunnableConfig, model: FakeDeskModel, repeats: int
) -> tuple[float, int, int]:
    samples: list[float] = []
    for _ in range(repeats):
        model.reset()
        started = time.perf_counter()
        await graph.ainvoke(inputs, config)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), len(model.calls), model.peak_concurrency


async def run_benchmark(
    *,
    latency: float = 0.25,
    repeats: int = 3,
    branch_counts: Sequence[int] = (2, 4, 8, 16),
    vote_counts: Sequence[int] = (5, 15),
) -> list[Row]:
    corpus = Corpus.from_dir()
    rows: list[Row] = []

    def settings(n: int) -> Settings:
        return Settings(provider="fake", max_branches=n, max_concurrency=None, samples=n)

    for n in branch_counts:
        model = FakeDeskModel(latency=latency)
        graph = build_research_desk(model, corpus, settings(n))
        for mode, limit in MODES.items():
            config: RunnableConfig = {"max_concurrency": limit} if limit else {}
            inputs = {"question": research_question(n), "max_branches": n}
            secs, calls, peak = await _time(graph, inputs, config, model, repeats)
            rows.append(Row("research desk", n, mode, limit, round(secs, 3), calls, peak))

    model = FakeDeskModel(latency=latency)
    sections = build_sectioning(model, settings(3))
    for mode, limit in MODES.items():
        config = {"max_concurrency": limit} if limit else {}
        secs, calls, peak = await _time(sections, {"text": SECTION_TEXT}, config, model, repeats)
        rows.append(Row("sectioning", 3, mode, limit, round(secs, 3), calls, peak))

    for n in vote_counts:
        model = FakeDeskModel(latency=latency)
        voting = build_voting(model, settings(n))
        for mode, limit in MODES.items():
            config = {"max_concurrency": limit} if limit else {}
            inputs = {"question": "How many senators does each state have?", "samples": n}
            secs, calls, peak = await _time(voting, inputs, config, model, repeats)
            rows.append(Row("voting", n, mode, limit, round(secs, 3), calls, peak))
    return rows


def to_markdown(rows: Sequence[Row]) -> str:
    sequential = {(r.pattern, r.branches): r.seconds for r in rows if r.max_concurrency == 1}
    lines = [
        "| Pattern | Branches | Mode | Wall clock (s) | Speed-up | Model calls | Peak in flight |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        speedup = sequential[(r.pattern, r.branches)] / r.seconds if r.seconds else 0.0
        lines.append(
            f"| {r.pattern} | {r.branches} | {r.mode} | {r.seconds:.3f} | {speedup:.1f}x "
            f"| {r.model_calls} | {r.peak_in_flight} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--latency", type=float, default=0.25, help="seconds per model call")
    parser.add_argument("--repeats", type=int, default=3, help="runs per cell (median)")
    parser.add_argument("--branches", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--votes", type=int, nargs="+", default=[5, 15])
    parser.add_argument("--json", type=Path, help="also write the rows to this JSON file")
    args = parser.parse_args(argv)
    if any(n < 1 or n > len(TOPICS) for n in args.branches):
        parser.error(f"--branches must be between 1 and {len(TOPICS)}")
    rows = asyncio.run(
        run_benchmark(
            latency=args.latency,
            repeats=args.repeats,
            branch_counts=args.branches,
            vote_counts=args.votes,
        )
    )
    print(f"Simulated model latency: {args.latency}s per call, median of {args.repeats} runs.\n")
    print(to_markdown(rows))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "latency": args.latency,
            "repeats": args.repeats,
            "rows": [asdict(r) for r in rows],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
