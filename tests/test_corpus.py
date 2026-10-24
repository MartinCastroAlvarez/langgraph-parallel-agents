from __future__ import annotations

import os
from pathlib import Path

import pytest

from parallel_agents.corpus import MAX_FILE_BYTES
from parallel_agents.corpus import Corpus
from parallel_agents.corpus import CorpusError
from parallel_agents.corpus import Passage
from parallel_agents.corpus import split_markdown
from parallel_agents.corpus import tokenize


def test_bundled_corpus_has_stable_heading_ids(corpus: Corpus) -> None:
    assert len(corpus) > 100
    assert "us-constitution:article-1-section-8" in corpus
    assert "art-of-war:xiii-use-spies" in corpus
    passage = corpus.get("us-constitution:article-3-section-1")
    assert passage is not None
    assert "judicial Power" in passage.text
    assert passage.title.endswith("Article 3 > Section 1")


def test_search_ranks_the_obvious_passage_first(corpus: Corpus) -> None:
    hits = corpus.search("judicial power supreme court inferior courts")
    assert hits[0].passage.id == "us-constitution:article-3-section-1"
    assert hits == corpus.search("judicial power supreme court inferior courts")  # deterministic
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_edge_cases(corpus: Corpus) -> None:
    assert corpus.search("") == []
    assert corpus.search("the of and") == []  # only stopwords
    assert corpus.search("spies", k=0) == []
    assert len(corpus.search("spies " * 500, k=2)) == 2  # long queries are truncated, not refused


def test_tokenize_folds_accents_and_drops_stopwords() -> None:
    assert tokenize("The Tzŭ of MANŒUVERING") == ["tzu", "manuvering"]


def test_long_sections_are_split_without_cutting_paragraphs() -> None:
    para = " ".join(["word"] * 100)
    passages = split_markdown("doc", f"# Doc\n\n## Part\n\n{para}\n\n{para}\n\n{para}\n")
    assert [p.id for p in passages] == ["doc:part", "doc:part-p2", "doc:part-p3"]
    assert all(p.text == para for p in passages)


def test_text_without_headings_is_one_section() -> None:
    passages = split_markdown("notes", "just text\n\nmore text")
    assert passages == [Passage(id="notes:body", title="notes", text="just text\n\nmore text")]


def test_loader_stays_inside_the_root(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "ok.md").write_text("# Ok\n\nvisible text", encoding="utf-8")
    (root / "notes.txt").write_text("plain text file", encoding="utf-8")
    (root / ".hidden.md").write_text("# Secret\n\nhidden", encoding="utf-8")
    (root / "image.png").write_bytes(b"\x89PNG")
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside\n\nmust not load", encoding="utf-8")
    os.symlink(outside, root / "link.md")
    corpus = Corpus.from_dir(root)
    ids = sorted(p.id for p in corpus.passages)
    assert ids == ["notes:body", "ok:ok"]


def test_loader_rejects_bad_roots(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Corpus.from_dir(tmp_path / "missing")
    file = tmp_path / "file.md"
    file.write_text("x", encoding="utf-8")
    with pytest.raises(CorpusError, match="not a directory"):
        Corpus.from_dir(file)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CorpusError, match="no passages"):
        Corpus.from_dir(empty)
    big = tmp_path / "big"
    big.mkdir()
    (big / "huge.md").write_bytes(b"a" * (MAX_FILE_BYTES + 1))
    with pytest.raises(CorpusError, match="larger than"):
        Corpus.from_dir(big)


def test_duplicate_ids_are_refused() -> None:
    with pytest.raises(CorpusError, match="unique"):
        Corpus([Passage("a", "A", "x"), Passage("a", "A", "y")])
