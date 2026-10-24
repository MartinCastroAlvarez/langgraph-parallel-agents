# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-24

### Added

- Research desk (map-reduce): a planner, a `Send` fan-out to tool-using researcher
  subgraphs (model ⇄ `search_corpus` over a bundled public-domain corpus with BM25),
  fan-in through `Annotated[list, operator.add]`, and a synthesizer that writes a
  cited answer and names the sub-questions it couldn't research.
- Sectioning: a static fan-out to sentiment, entities (two steps) and summary
  specialists, joined by a `defer=True` aggregator that runs exactly once.
- Voting: N samples in parallel, normalised majority vote, index-based tie-break,
  agreement and consensus.
- Per-branch `RetryPolicy` and `TimeoutPolicy` inside branch subgraphs, a per-branch
  time budget, and failure isolation (`BranchFailure` records instead of a crashed run).
- Bounded concurrency through `max_concurrency` in the run config.
- Per-branch progress streaming (`get_stream_writer` + `subgraphs=True`).
- `FakeDeskModel`: a role-aware fake chat model with simulated latency, injectable
  faults and a peak-concurrency gauge.
- `python -m parallel_agents` CLI (ollama by default, Anthropic optional, fake offline),
  `--mermaid` diagrams from the real graphs, and `python -m parallel_agents.bench`,
  the wall-clock benchmark.
