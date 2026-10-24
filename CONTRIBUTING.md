# Contributing to langgraph-parallel-agents

## TL;DR

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install poetry
poetry install                       # add -E anthropic for the optional adapter
poetry run pytest
poetry run ruff check parallel_agents tests examples && poetry run ruff format --check parallel_agents tests examples
poetry run mypy parallel_agents
poetry run bandit -c pyproject.toml -q -r parallel_agents
poetry run pip-audit
poetry run python -m examples.quickstart
```

Open a PR against `main` from a `feat/...`, `fix/...`, `docs/...` or `chore/...` branch.

## What this project will not accept

- Tests that call a real model or touch the network.
- Hosted providers, web search or tracing turned on by default.
- A fan-out branch that can raise into the parent graph.
- Hand-edited benchmark numbers.

## What it does accept

- New parallel patterns or variants (e.g. map-reduce with a checkpointer, nested fan-out),
  each with tests for fan-out count, merge and failure isolation.
- Better planners, tools or corpora (public domain only, recorded in
  `parallel_agents/data/README.md`).

## Updating the benchmark table

```bash
poetry run python -m parallel_agents.bench --json docs/benchmark/results.json
```

Paste the printed table into the README together with the machine, versions and date.

## House rules

1. Tests alongside features; security-relevant changes get a hostile case.
2. README, ARCHITECTURE/SECURITY (when relevant) and CHANGELOG change with the code.
3. No secrets in commits; `.env.example` holds defaults only.
