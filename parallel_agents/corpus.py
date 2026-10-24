"""A tiny local corpus with BM25 search: the researchers' only source of evidence.

Markdown files are split along their headings into passages; long sections are split
again into paragraph windows. Each passage gets a stable id (``<file>:<heading-slug>``)
that the agents cite as ``[id]``. Loading is confined to one directory: no symlinks,
no hidden files, only ``.md``/``.txt``, and a size cap per file.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

__all__ = ["Corpus", "CorpusError", "Hit", "Passage", "bundled_corpus_dir"]

ALLOWED_SUFFIXES = frozenset({".md", ".txt"})
MAX_FILE_BYTES = 2_000_000
MAX_PASSAGE_WORDS = 160
MAX_QUERY_CHARS = 300

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "shall",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
    ]
)


class CorpusError(ValueError):
    """The corpus directory or one of its files is unusable."""


@dataclass(frozen=True, slots=True)
class Passage:
    """One citable chunk of a document."""

    id: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class Hit:
    """A passage returned by :meth:`Corpus.search`, with its BM25 score."""

    passage: Passage
    score: float


def bundled_corpus_dir() -> Path:
    """The public-domain corpus shipped inside the package."""
    return Path(str(resources.files("parallel_agents") / "data" / "corpus"))


def tokenize(text: str) -> list[str]:
    """Lower-case, accent-folded word tokens without stopwords."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return [t for t in _TOKEN.findall(folded) if t not in _STOPWORDS]


def _slug(text: str) -> str:
    return "-".join(tokenize(text)) or "section"


def _split_words(text: str, limit: int) -> list[str]:
    """Group paragraphs into windows of at most ``limit`` words (a paragraph is never cut)."""
    windows: list[str] = []
    current: list[str] = []
    count = 0
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        words = len(para.split())
        if current and count + words > limit:
            windows.append("\n\n".join(current))
            current, count = [], 0
        current.append(para)
        count += words
    if current:
        windows.append("\n\n".join(current))
    return windows


def split_markdown(stem: str, text: str) -> list[Passage]:
    """Split one Markdown document into heading-bounded passages."""
    sections: list[tuple[list[str], list[str]]] = []
    trail: list[str] = []
    body: list[str] = []

    def flush() -> None:
        if any(line.strip() for line in body):
            sections.append((list(trail), list(body)))

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            trail[:] = [*trail[: level - 1], match.group(2)]
            body = []
        else:
            body.append(line)
    flush()

    passages: list[Passage] = []
    seen: Counter[str] = Counter()
    for headings, lines in sections:
        title = " > ".join(headings) if headings else stem
        slug = _slug(" ".join(headings[1:] or headings)) if headings else "body"
        for window in _split_words("\n".join(lines), MAX_PASSAGE_WORDS):
            seen[slug] += 1
            suffix = "" if seen[slug] == 1 else f"-p{seen[slug]}"
            passages.append(Passage(id=f"{stem}:{slug}{suffix}", title=title, text=window))
    return passages


def _safe_files(root: Path) -> Iterable[Path]:
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise CorpusError(f"corpus root is not a directory: {root}")
    for path in sorted(resolved_root.rglob("*")):
        rel = path.relative_to(resolved_root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        if not path.resolve().is_relative_to(resolved_root):  # pragma: no cover - defensive
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            raise CorpusError(f"{rel} is larger than {MAX_FILE_BYTES} bytes")
        yield path


class Corpus:
    """An in-memory BM25 index over a list of passages."""

    def __init__(self, passages: Iterable[Passage], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.passages = list(passages)
        if not self.passages:
            raise CorpusError("the corpus has no passages")
        ids = [p.id for p in self.passages]
        if len(set(ids)) != len(ids):
            raise CorpusError("passage ids must be unique")
        self.k1, self.b = k1, b
        self._by_id = {p.id: p for p in self.passages}
        self._tf = [Counter(tokenize(f"{p.title} {p.text}")) for p in self.passages]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / len(self._len)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.passages)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    @classmethod
    def from_dir(cls, root: str | Path | None = None) -> Corpus:
        """Load every ``.md``/``.txt`` file under ``root`` (default: the bundled corpus)."""
        base = Path(root) if root is not None else bundled_corpus_dir()
        passages: list[Passage] = []
        for path in _safe_files(base):
            text = path.read_text(encoding="utf-8", errors="replace")
            passages.extend(split_markdown(path.stem, text))
        return cls(passages)

    def __len__(self) -> int:
        return len(self.passages)

    def __contains__(self, passage_id: object) -> bool:
        return passage_id in self._by_id

    def get(self, passage_id: str) -> Passage | None:
        return self._by_id.get(passage_id)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Top ``k`` passages for ``query``; ties break on corpus order, so results are stable."""
        terms = tokenize(query[:MAX_QUERY_CHARS])
        if not terms or k <= 0:
            return []
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._len[i] / self._avg)
            for term in terms:
                freq = tf.get(term)
                if freq:
                    score += self._idf[term] * freq * (self.k1 + 1) / (freq + norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [Hit(self.passages[i], round(s, 4)) for s, i in scored[:k]]
