"""A deterministic chat model with simulated latency, faults and a concurrency gauge.

``FakeDeskModel`` is a real LangChain ``BaseChatModel``: the graphs call it exactly as
they call ``ChatOllama``. It reads the ``Role: <name>.`` line of the system prompt and
answers like that node would (a JSON plan, a ``search_corpus`` tool call, a cited
summary, a vote...), so every graph runs end to end offline. It also

* sleeps ``latency`` seconds per call (``asyncio.sleep`` on the async path), which is
  what makes the wall-clock benchmark meaningful;
* raises ``ConnectionError`` (retryable) for prompts matching ``transient_failures``
  a set number of times, ``RuntimeError`` (not retryable) for ``permanent_failures``,
  and sleeps extra for ``stalls`` - to exercise retries, isolation and timeouts;
* records every call and the peak number of calls in flight, which is how the tests
  prove ``max_concurrency`` bounds the fan-out.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field
from pydantic import PrivateAttr

from parallel_agents.prompts import role_of

__all__ = ["FakeDeskModel"]

_DATA = re.compile(r'<data name="(?P<name>[^"]*)">\n(?P<body>.*?)\n</data>', re.S)
_LIST_STEM = re.compile(r"^(?P<stem>.*\b(?:to|between|among|of|for|about|on|by))\s+(?P<items>.+)$")
_SPLIT_ITEMS = re.compile(r",\s*(?:and\s+)?|\s+and\s+")
_ENTITY = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+(?:of\s+)?[A-Z][a-zA-Z]+)*")
_NOT_ENTITIES = frozenset({"The", "A", "An", "No", "All", "Each", "Every", "If", "In", "It", "We"})
_POSITIVE = frozenset(
    [
        "good",
        "great",
        "secure",
        "secured",
        "justice",
        "liberty",
        "blessings",
        "welfare",
        "peace",
        "win",
    ]
)
_NEGATIVE = frozenset(["bad", "war", "ruin", "defeat", "death", "danger", "enemy", "loss", "fail"])


def _blocks(text: str) -> dict[str, str]:
    return {m["name"]: html.unescape(m["body"]) for m in _DATA.finditer(text)}


def _first_sentence(text: str, max_words: int = 40) -> str:
    flat = " ".join(text.split())
    sentence = re.split(r"(?<=[.;:])\s", flat, maxsplit=1)[0]
    words = sentence.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]).rstrip(",;:") + "..."
    return sentence.rstrip(",;:") + ("" if sentence.endswith(".") else ".")


class FakeDeskModel(BaseChatModel):
    """Role-aware, deterministic fake; see the module docstring."""

    latency: float = 0.0
    stalls: dict[str, float] = Field(default_factory=dict)
    transient_failures: dict[str, int] = Field(default_factory=dict)
    permanent_failures: tuple[str, ...] = ()
    votes: tuple[str, ...] = ("2", "two", "Two.", "3", "2")
    latency_for: Callable[[str, str], float] | None = None
    fault_roles: tuple[str, ...] = ("researcher", "sentiment", "entities", "summary", "voter")
    """Faults and stalls only fire for these roles, so the plan/synthesis steps stay healthy."""

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _in_flight: int = PrivateAttr(default=0)
    _peak: int = PrivateAttr(default=0)
    _calls: list[str] = PrivateAttr(default_factory=list)
    _failed: Counter[str] = PrivateAttr(default_factory=Counter)

    @property
    def _llm_type(self) -> str:
        return "fake-desk"

    # -- introspection used by tests and the benchmark ----------------------------------
    @property
    def peak_concurrency(self) -> int:
        return self._peak

    @property
    def calls(self) -> list[str]:
        """The role of every call, in start order."""
        return list(self._calls)

    def reset(self) -> None:
        with self._lock:
            self._in_flight = self._peak = 0
            self._calls.clear()
            self._failed.clear()

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        # The fake knows its only tool (search_corpus) and decides when to call it.
        return self

    # -- LangChain hooks -------------------------------------------------------------------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        role, prompt, delay = self._enter(messages)
        try:
            time.sleep(delay)
            return self._result(role, prompt, messages)
        finally:
            self._exit()

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        role, prompt, delay = self._enter(messages)
        try:
            await asyncio.sleep(delay)
            return self._result(role, prompt, messages)
        finally:
            self._exit()

    # -- internals ---------------------------------------------------------------------------
    def _enter(self, messages: list[BaseMessage]) -> tuple[str, str, float]:
        system = next((m.text for m in messages if isinstance(m, SystemMessage)), "")
        role = role_of(system)
        prompt = "\n".join(m.text for m in messages if isinstance(m, HumanMessage))
        delay = self.latency_for(role, prompt) if self.latency_for else self.latency
        if role in self.fault_roles:
            delay += sum(s for key, s in self.stalls.items() if key in prompt)
        with self._lock:
            self._calls.append(role)
            self._in_flight += 1
            self._peak = max(self._peak, self._in_flight)
        return role, prompt, delay

    def _exit(self) -> None:
        with self._lock:
            self._in_flight -= 1

    def _result(self, role: str, prompt: str, messages: list[BaseMessage]) -> ChatResult:
        if role in self.fault_roles:
            self._inject_faults(prompt)
        message = self._respond(role, prompt, messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _inject_faults(self, prompt: str) -> None:
        for key in self.permanent_failures:
            if key in prompt:
                raise RuntimeError(f"fake model refuses prompts containing {key!r}")
        for key, times in self.transient_failures.items():
            if key in prompt:
                with self._lock:
                    fail = self._failed[key] < times
                    self._failed[key] += 1
                if fail:
                    raise ConnectionError(f"fake transient failure for {key!r}")

    def _respond(self, role: str, prompt: str, messages: list[BaseMessage]) -> AIMessage:
        data = _blocks(prompt)
        if role == "planner":
            return AIMessage(content=json.dumps(self._plan(data.get("question", prompt), prompt)))
        if role == "researcher":
            return self._research(data.get("sub_question", prompt), messages)
        if role == "synthesizer":
            return AIMessage(content=self._synthesize(data))
        if role == "sentiment":
            return AIMessage(content=self._sentiment(data.get("text", prompt)))
        if role == "entities":
            return AIMessage(content=json.dumps(self._entities(data.get("text", prompt))))
        if role == "summary":
            return AIMessage(content=_first_sentence(data.get("text", prompt), 30))
        if role == "voter":
            match = re.search(r"Sample (\d+) of", prompt)
            index = int(match.group(1)) if match else 0
            return AIMessage(content=self.votes[index % len(self.votes)])
        return AIMessage(content="ok")

    @staticmethod
    def _plan(question: str, prompt: str) -> list[str]:
        """'What X does Y give to A, B and C?' -> one sub-question per list item."""
        limit_match = re.search(r"at most (\d+)", prompt)
        limit = int(limit_match.group(1)) if limit_match else 4
        body = question.strip().rstrip("?").strip()
        head, _, _ = body.partition(",")
        stem_match = _LIST_STEM.match(head)
        if stem_match:
            stem = stem_match["stem"]
            items = [i.strip() for i in _SPLIT_ITEMS.split(body[len(stem) :]) if i.strip()]
            if len(items) > 1:
                return [f"{stem} {item}?" for item in items][:limit]
        return [question.strip()]

    @staticmethod
    def _research(sub_question: str, messages: list[BaseMessage]) -> AIMessage:
        tool_results = [m for m in messages if isinstance(m, ToolMessage)]
        if not tool_results:
            call = {"name": "search_corpus", "args": {"query": sub_question}, "id": "call-1"}
            return AIMessage(content="", tool_calls=[call])
        try:
            hits = json.loads(tool_results[-1].text)
        except json.JSONDecodeError:
            hits = []
        if not hits:
            return AIMessage(content="The corpus has no passage about this.")
        top = hits[0]
        return AIMessage(content=f"{_first_sentence(top['text'])} [{top['id']}]")

    @staticmethod
    def _synthesize(data: dict[str, str]) -> str:
        findings = [
            line.split(": ", 1)[1]
            for line in data.get("findings", "").splitlines()
            if line.startswith("[") and ": " in line
        ]
        answer = " ".join(findings) or "None of the sub-questions could be researched."
        gaps = data.get("gaps", "").strip()
        if gaps:
            answer += f" Not researched: {'; '.join(gaps.splitlines())}."
        return answer

    @staticmethod
    def _sentiment(text: str) -> str:
        words = re.findall(r"[a-z]+", text.lower())
        score = sum(w in _POSITIVE for w in words) - sum(w in _NEGATIVE for w in words)
        return "positive" if score > 0 else "negative" if score < 0 else "neutral"

    @staticmethod
    def _entities(text: str) -> list[str]:
        seen: dict[str, None] = {}
        for match in _ENTITY.findall(text):
            if match not in _NOT_ENTITIES:
                seen.setdefault(match, None)
        return list(seen)
