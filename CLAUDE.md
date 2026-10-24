# CLAUDE.md

Contract between this repository and any AI agent contributing to it.

## Required reading

1. This file. 2. [`README.md`](README.md). 3. [`ARCHITECTURE.md`](ARCHITECTURE.md).
4. [`SECURITY.md`](SECURITY.md).

## Mission

`langgraph-parallel-agents` shows the parallelization / map-reduce agent pattern in
current LangGraph (`Send`, reducers, `defer`, `max_concurrency`, `RetryPolicy`,
`TimeoutPolicy`, subgraphs, streaming) as small, correct and tested code.

## The rules

1. **Local by default.** ollama on loopback is the default; hosted providers stay
   optional adapters. No telemetry, no LangSmith upload, no web search.
2. **Model output and tool input are untrusted.** Wrap them with `prompts.data_block`,
   validate citations against retrieved ids, and clamp tool arguments.
3. **A branch never takes down the run.** Every fan-out target goes through
   `isolation.run_isolated`; retries and timeouts live inside the branch subgraph.
4. **Deterministic output.** Reduced lists are ordered by branch index; ties break on
   index, never on completion order.
5. **Offline tests.** `FakeDeskModel` only. Never a real model or the network. Tests
   check control flow (fan-out counts, merges, bounds, isolation), not prose.
6. **README numbers come from real runs.** Rerun `python -m parallel_agents.bench
   --json docs/benchmark/results.json` and paste its table. Never edit a number by hand.
7. Never add `# noqa`, `# nosec` or `# type: ignore` to the package, skip a test, or
   loosen lint/type/coverage config. Fix the code.

## Done means

`poetry run pytest` (coverage >= 85%), `ruff check`, `ruff format --check`, `mypy`
(strict), `bandit` and `pip-audit` all green; `python -m examples.quickstart` runs;
README and CHANGELOG are updated.

## Git

Small commits, imperative subjects, never force-push `main`, never commit secrets or `.env`.
