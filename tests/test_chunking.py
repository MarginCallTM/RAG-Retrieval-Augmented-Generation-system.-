"""Tests for src/chunking.py and src/ingest.py."""

from pathlib import Path
from typing import List

import pytest

from src.chunking import chunk_file, hard_split
from src.ingest import list_corpus_files, read_text, to_corpus_path
from src.models import Chunk

RAW_DIR = Path("data/raw")

MARKDOWN = """# Title

Intro paragraph.

## Install

```bash
# this is a shell comment, not a heading
pip install vllm
```

## Usage

Run it.
"""


def assert_exact(chunks: List[Chunk], text: str, max_size: int) -> None:
    """Check the invariants every chunker must hold."""
    previous_last = 0
    for chunk in chunks:
        first = chunk.first_character_index
        last = chunk.last_character_index
        assert text[first:last] == chunk.text
        assert 0 < last - first <= max_size
        assert first >= previous_last, "chunks overlap or are unordered"
        # Nothing but whitespace is left out between two chunks.
        assert text[previous_last:first].strip() == ""
        previous_last = last
    assert text[previous_last:].strip() == ""


def test_markdown_splits_on_headings_only_outside_fences() -> None:
    """The # line inside the code fence does not start a new chunk."""
    chunks = chunk_file("doc.md", MARKDOWN, 2000)
    starts = [chunk.text.splitlines()[0] for chunk in chunks]
    assert starts == ["# Title", "## Install", "## Usage"]
    assert_exact(chunks, MARKDOWN, 2000)


def test_oversized_section_is_split_under_the_limit() -> None:
    """A section above the limit is cut on paragraphs, then on lines."""
    text = "## Big\n\n" + "word " * 300 + "\n\n" + "line\n" * 200
    chunks = chunk_file("doc.md", text, 500)
    assert len(chunks) > 1
    assert_exact(chunks, text, 500)


def test_single_long_line_is_cut_mid_line() -> None:
    """A line with no newline at all still ends up under the limit."""
    text = "x" * 4500
    assert hard_split(text, (0, 4500), 2000) == [
        (0, 2000), (2000, 4000), (4000, 4500),
    ]


def test_txt_ignores_hash_comments() -> None:
    """In CMakeLists.txt, # is a comment: split on blank lines only."""
    text = "# comment\nset(A 1)\n\n# other\nset(B 2)\n"
    chunks = chunk_file("CMakeLists.txt", text, 2000)
    assert len(chunks) == 1
    assert_exact(chunks, text, 2000)


def test_whitespace_only_file_gives_no_chunk() -> None:
    """Empty or blank files produce nothing, and do not crash."""
    assert chunk_file("empty.md", "", 2000) == []
    assert chunk_file("blank.txt", "\n\n  \n", 2000) == []


def test_non_positive_size_is_refused() -> None:
    """max_chunk_size 0 or negative is a user error, not a crash."""
    with pytest.raises(ValueError):
        chunk_file("doc.md", MARKDOWN, 0)


def test_lora_reference_is_exactly_one_chunk() -> None:
    """The LoRA docs reference [4695:6098] is one of our chunks."""
    path = RAW_DIR / "vllm-0.10.1/docs/features/lora.md"
    if not path.is_file():
        pytest.skip("corpus not present")
    text = read_text(path)
    assert text is not None
    spans = [
        (c.first_character_index, c.last_character_index)
        for c in chunk_file(to_corpus_path(path), text, 2000)
    ]
    assert (4695, 6098) in spans


def test_exclusion_is_top_level_only() -> None:
    """tests/ is skipped, vllm/benchmarks/ is kept (it holds sources)."""
    if not RAW_DIR.is_dir():
        pytest.skip("corpus not present")
    paths = [to_corpus_path(p) for p in list_corpus_files(RAW_DIR)]
    assert len(paths) == 1205
    assert not any(p.startswith("data/raw/vllm-0.10.1/tests/") for p in paths)
    assert any("/vllm/benchmarks/" in p for p in paths)
    assert all(p.startswith("data/raw/vllm-0.10.1/") for p in paths)


@pytest.mark.parametrize("max_size", [2000, 500])
def test_whole_corpus_invariants(max_size: int) -> None:
    """Every chunk of every file holds the invariants, at two sizes."""
    if not RAW_DIR.is_dir():
        pytest.skip("corpus not present")
    for path in list_corpus_files(RAW_DIR):
        text = read_text(path)
        assert text is not None
        chunks = chunk_file(to_corpus_path(path), text, max_size)
        assert_exact(chunks, text, max_size)


PYTHON = '''import os
import sys

# Maximum value of an FP8 number.
FP8_MAX = 448.0


class Engine:
    """A toy engine."""

    def __init__(self) -> None:
        self.steps = 0

    # Advance by one step.
    @staticmethod
    def step(x: int) -> int:
        y = x * 2
        return y
'''


def test_python_small_file_is_one_chunk() -> None:
    """A file under the limit is packed into a single chunk."""
    chunks = chunk_file("mod.py", PYTHON, 2000)
    assert len(chunks) == 1
    assert_exact(chunks, PYTHON, 2000)


def test_python_big_class_splits_on_methods() -> None:
    """Above the limit, a class is cut between its methods, and a method
    keeps its decorator and the comment right above it."""
    chunks = chunk_file("mod.py", PYTHON, 120)
    assert_exact(chunks, PYTHON, 120)
    starts = [chunk.text.splitlines()[0] for chunk in chunks]
    assert "# Advance by one step." in starts
    assert not any(s.startswith("@staticmethod") for s in starts)


def test_python_syntax_error_falls_back_to_text() -> None:
    """A file that does not parse is still chunked, never a crash."""
    broken = "def f(:\n    pass\n\n\nx = 1\n"
    chunks = chunk_file("broken.py", broken, 2000)
    assert_exact(chunks, broken, 2000)
    assert chunks
