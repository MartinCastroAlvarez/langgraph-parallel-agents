# Security

langgraph-parallel-agents runs LLM agents that read a document collection through a
tool and write answers that cite it. People install it and point it at their own
folders and models. Design goals: nothing leaves the machine unless you opt in, text in
the corpus or the question can't take over the agents or the host, and one misbehaving
branch can't take down the run or exhaust the machine.

## Reporting a vulnerability

Open a private security advisory at
<https://github.com/MartinCastroAlvarez/langgraph-parallel-agents/security/advisories/new>.
Do **not** open a public issue for a suspected vulnerability. We acknowledge within 5
business days and aim to ship a fix within 30 days. You are credited in the advisory
unless you ask otherwise.

Supported versions: the latest minor release.

## 1. Threat model

**Assets.** The documents in the corpus folder, the questions you ask, the local
machine and filesystem, and your model endpoint (and its key, if you use a hosted one).

| Actor | Controls | Must not be able to |
| --- | --- | --- |
| A document author | the text of a file in the corpus folder | give the agents instructions, forge citations, make the loader read files outside the folder |
| Whoever writes the question | the task text | escape the prompt structure, run code, exhaust the host |
| The model | plans, tool arguments, answers | invent sources, read arbitrary files through the tool, fan out without limit |
| Whoever controls `OLLAMA_HOST` | where prompts are sent | receive your documents without your explicit opt-in |

| Threat | Control |
| --- | --- |
| Prompt injection from documents, questions or model output | Every system prompt says that `<data>` content is untrusted and never instructions. Tool results, questions, findings and gaps are wrapped in `<data name=...>` after escaping `& < >`, so content can't close the element (tested with a `</data>` payload). Escaping reduces injection but can't remove it: treat answers about untrusted corpora as untrusted too. |
| Forged citations | The researcher keeps only `[id]`s that appeared in its own `search_corpus` results and deletes the rest from the text. The synthesizer keeps only ids that appear in a finding. |
| Tool abuse by the model | `search_corpus` takes a pydantic-validated `query` (≤ 1000 chars, cut to 300) and `k` in 1–5. It searches an in-memory index and has no filesystem or network access. |
| Path traversal / symlink escape in the corpus loader | The root is resolved first. Only regular `.md`/`.txt` files are read. Symlinks and hidden entries (`.git`, `.env`) are skipped, every file must resolve inside the root, and each is capped at 2 MB. Tested with a symlink pointing outside. |
| Runaway fan-out or loops | The plan is capped at `max_branches` sub-questions (300 chars each). `max_concurrency` bounds tasks in flight. The tool loop stops after `max_tool_steps`. `recursion_limit` is 50, and every model call and every branch has a timeout. The CLI refuses tasks over 20,000 characters. |
| One branch failing the run | Retries inside the branch subgraph; `run_isolated` turns leftover exceptions into a `BranchFailure`. Error messages are truncated to 200 chars and contain no prompts. |
| Documents leaving the machine | `OLLAMA_HOST` must be loopback unless `PARALLEL_AGENTS_ALLOW_REMOTE_OLLAMA=1`. Hosted providers are opt-in and need your own key. There is no web search, and LangSmith tracing stays off unless you set `LANGSMITH_TRACING` yourself. |
| Secrets | No key is read except `ANTHROPIC_API_KEY`, and only by the optional adapter. Keys never enter prompts, state, stream events or logs. `.env` is git-ignored. |
| Code execution | No `eval`, `exec`, `pickle`, `marshal`, `yaml` or `subprocess` in the package. An AST test enforces this, and also that no `noqa`/`nosec`/`type: ignore` hides a finding. |

## 2. Dependencies

CI runs `bandit` on the package and `pip-audit` on the locked environment. Dependabot
alerts, security updates and secret scanning are enabled on the repository.

## 3. Running it safely

- Keep ollama on `127.0.0.1`. If you point `OLLAMA_HOST` elsewhere, use TLS and trust
  that host with your documents.
- Set `max_concurrency` to what your model server can actually serve. Unbounded
  fan-out against a rate-limited API turns into retries and 429s.
- Treat answers as drafts: check the cited passages, especially when the corpus comes
  from untrusted sources.
