"""Settings, read from ``PARALLEL_AGENTS_*`` environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from langchain_core.runnables import RunnableConfig

__all__ = ["PROVIDERS", "Provider", "Settings"]

Provider = Literal["ollama", "anthropic", "fake"]
PROVIDERS: tuple[Provider, ...] = ("ollama", "anthropic", "fake")

DEFAULT_MODELS: dict[Provider, str] = {
    "ollama": "qwen3:4b",
    "anthropic": "claude-sonnet-5-5",
    "fake": "fake-desk",
}


def _int(env: Mapping[str, str], key: str, default: int, *, minimum: int = 0) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value}")
    return value


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    value = float(raw)
    if value < 0:
        raise ValueError(f"{key} must be >= 0, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the three graphs and the CLI can be tuned with."""

    provider: Provider = "ollama"
    model: str = DEFAULT_MODELS["ollama"]
    ollama_host: str = "http://127.0.0.1:11434"
    allow_remote_ollama: bool = False
    temperature: float = 0.0
    max_branches: int = 4
    max_concurrency: int | None = 4
    retries: int = 3
    call_timeout: float = 120.0
    branch_timeout: float = 300.0
    max_tool_steps: int = 2
    top_k: int = 3
    samples: int = 5
    corpus_dir: str | None = None
    fake_latency: float = 0.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        provider = e.get("PARALLEL_AGENTS_PROVIDER", "ollama").strip().lower()
        if provider not in PROVIDERS:
            raise ValueError(f"PARALLEL_AGENTS_PROVIDER must be one of {PROVIDERS}")
        p: Provider = provider
        concurrency = _int(e, "PARALLEL_AGENTS_MAX_CONCURRENCY", 4)
        return cls(
            provider=p,
            model=e.get("PARALLEL_AGENTS_MODEL") or DEFAULT_MODELS[p],
            ollama_host=e.get("OLLAMA_HOST") or "http://127.0.0.1:11434",
            allow_remote_ollama=e.get("PARALLEL_AGENTS_ALLOW_REMOTE_OLLAMA", "") == "1",
            temperature=_float(e, "PARALLEL_AGENTS_TEMPERATURE", 0.0),
            max_branches=_int(e, "PARALLEL_AGENTS_MAX_BRANCHES", 4, minimum=1),
            max_concurrency=concurrency or None,
            retries=_int(e, "PARALLEL_AGENTS_RETRIES", 3, minimum=1),
            call_timeout=_float(e, "PARALLEL_AGENTS_CALL_TIMEOUT", 120.0),
            branch_timeout=_float(e, "PARALLEL_AGENTS_BRANCH_TIMEOUT", 300.0),
            max_tool_steps=_int(e, "PARALLEL_AGENTS_MAX_TOOL_STEPS", 2, minimum=1),
            top_k=_int(e, "PARALLEL_AGENTS_TOP_K", 3, minimum=1),
            samples=_int(e, "PARALLEL_AGENTS_SAMPLES", 5, minimum=1),
            corpus_dir=e.get("PARALLEL_AGENTS_CORPUS_DIR") or None,
            fake_latency=_float(e, "PARALLEL_AGENTS_FAKE_LATENCY", 0.0),
        )

    def run_config(self) -> RunnableConfig:
        """The LangGraph ``config`` for a run: bounded fan-out via ``max_concurrency``."""
        config: RunnableConfig = {"recursion_limit": 50}
        if self.max_concurrency:
            config["max_concurrency"] = self.max_concurrency
        return config
