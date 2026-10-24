"""All three parallel patterns, offline: the fake model stands in for ollama.

    python -m examples.quickstart

Swap ``make_model(Settings(provider="fake", ...))`` for ``make_model(Settings())`` to
run the same graphs on a local ollama model.
"""

from __future__ import annotations

import asyncio
import time

from parallel_agents import Corpus
from parallel_agents import Settings
from parallel_agents import build_research_desk
from parallel_agents import build_sectioning
from parallel_agents import build_voting
from parallel_agents import make_model
from parallel_agents.progress import Progress
from parallel_agents.progress import format_progress
from parallel_agents.progress import stream_run

QUESTION = "What powers does the Constitution give to Congress, the President and the courts?"
TEXT = (
    "We the People of the United States, in Order to form a more perfect Union, establish "
    "Justice, insure domestic Tranquility, and secure the Blessings of Liberty."
)


async def main() -> None:
    settings = Settings(provider="fake", fake_latency=0.2, max_concurrency=4)
    model = make_model(settings)
    config = settings.run_config()

    print("== research desk: planner -> Send x N researchers -> synthesizer ==")
    desk = build_research_desk(model, Corpus.from_dir(), settings)
    started = time.perf_counter()
    async for item in stream_run(desk, {"question": QUESTION}, config):
        if isinstance(item, Progress):
            print(format_progress(item))
        else:
            print(f"\n{item['answer']}\nsources: {item['sources']}")
    print(f"({time.perf_counter() - started:.2f}s with 0.2s per model call)\n")

    print("== sectioning: sentiment | entities -> normalize | summary -> aggregate ==")
    report = await build_sectioning(model, settings).ainvoke({"text": TEXT}, config)
    print(report["report"], "\n")

    print("== voting: 5 samples in parallel -> majority ==")
    vote = await build_voting(model, settings).ainvoke(
        {"question": "How many senators does each state have?", "samples": 5}, config
    )
    print(f"answer={vote['answer']!r} counts={vote['counts']} consensus={vote['consensus']}")


if __name__ == "__main__":
    asyncio.run(main())
