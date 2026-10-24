# Architecture

```
parallel_agents/
  config.py      Settings from PARALLEL_AGENTS_* env vars; run_config() -> {"max_concurrency": k}
  llm.py         make_model(): ChatOllama (loopback-only) | ChatAnthropic (optional) | FakeDeskModel
  fake.py        FakeDeskModel: role-aware fake BaseChatModel, latency, faults, peak-concurrency gauge
  prompts.py     system prompts ("Role: <name>." first) and data_block() for untrusted text
  corpus.py      Markdown -> heading-bounded passages, in-memory BM25, confined loader
  isolation.py   BranchFailure, branch_retry_policy(), single_node_graph(), run_isolated(), emit()
  research.py    researcher subgraph (agent <-> ToolNode -> report) + the research desk graph
  sectioning.py  static fan-out to 3 specialists + deferred aggregator
  voting.py      Send fan-out of N samples + normalised majority tally
  progress.py    stream_run(): custom events from every subgraph, then the final state
  bench.py       sequential vs bounded vs parallel wall-clock benchmark
  cli.py         python -m parallel_agents "<task>" [--pattern ...] [--mermaid ...]
  data/corpus/   public-domain corpus (US Constitution, The Art of War)
```

## Layers of a branch

Every parallel branch, in all three patterns, has the same three layers:

```
parent graph superstep            (max_concurrency caps how many of these run at once)
└─ fan-out target node            research / sample / sentiment|entities|summary
   └─ run_isolated(...)           asyncio.timeout(branch_timeout); exception -> BranchFailure
      └─ branch subgraph          compiled StateGraph
         └─ model-calling node    RetryPolicy(max_attempts) + TimeoutPolicy(run_timeout)
```

- **Inside the branch**, LangGraph does the retrying. `RetryPolicy` covers connection
  errors, 5xx responses and `NodeTimeoutError` (a per-attempt `TimeoutPolicy`), and
  leaves programming errors alone.
- **At the branch boundary**, `run_isolated` converts whatever survived the retries into
  a `BranchFailure` written to the `failures` reducer. It re-raises `GraphBubbleUp` so
  interrupts still work. Siblings never see the exception.
- **In the parent**, `max_concurrency` from the run config bounds the tasks in each
  superstep. It is inherited by the branch subgraphs, where it is harmless: each branch
  runs one node at a time.

Why not `add_node(..., error_handler=...)`? In langgraph 1.2.17 it recovers a node that
fails alone, but when siblings run in the same superstep the background executor
re-raises the handled exception when the superstep closes (see `tests/test_upstream.py`,
a strict xfail that flags the upgrade that fixes it).

## Fan-in and ordering

Parallel writes to one key need a reducer. `findings`, `failures` and `votes` are
`Annotated[list[...], operator.add]`. LangGraph applies a superstep's writes in task
order, not completion order, and every consumer still sorts by the branch `index` it
was given at fan-out time. Ties in voting go to the answer first seen at the lowest
index. Result: identical output under any interleaving (tested with random latencies).

## Joins: one superstep vs `defer`

In the desk and in voting, all `Send` tasks belong to one superstep, so the edge to the
synthesizer/tally fires once. In sectioning the branches have different lengths
(`entities -> normalize`), so a plain join would fire twice. `defer=True` on
`aggregate` postpones it until nothing else is pending.

## Streaming

Nodes call `emit()`, a thin wrapper over `langgraph.config.get_stream_writer()`.
`stream_run` uses `astream(stream_mode=["custom", "values"], subgraphs=True)`, so
events raised inside a researcher subgraph (`tool`) reach the caller next to the
wrapper's `start` / `done` / `failed` events, each tagged with its branch index.

## Data flow of the research desk

1. `planner`: question -> JSON array of sub-questions -> `parse_plan` (dedupe, cap
   count and length, fallback to the question).
2. `fan_out`: `[Send("research", {"index": i, "sub_question": q}) ...]`.
3. `research` (xN): `run_isolated(researcher)`. The model calls `search_corpus`; the
   `ToolNode` returns JSON passages; the model answers with `[id]` citations; `report`
   keeps only ids that were actually retrieved.
4. `synthesizer`: findings sorted by index, wrapped in `<data>`, gaps listed. The model's
   answer is filtered to known citations, and `sources` are ordered by first appearance.
