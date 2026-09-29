"""Tests for src/indexer.py and the index command."""

from pathlib import Path

import pytest

from src.cli import CliError, RagCli
from src.indexer import (
    INDEX_FILENAME,
    build_chunks,
    check_chunks,
    load_index,
    save_index,
)
from src.models import Chunk, ChunkIndex


@pytest.fixture
def tiny_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project directory with a two-file corpus, used as the cwd."""
    repo = tmp_path / "data" / "raw" / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "docs" / "guide.md").write_text("# Guide\n\nHello.\n")
    (repo / "main.py").write_text("def f() -> int:\n    return 1\n")
    (repo / "tests" / "test_main.py").write_text("def test() -> None:\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_build_chunks_uses_dataset_style_paths(tiny_project: Path) -> None:
    """Paths are relative to the project and tests/ is skipped."""
    chunks = build_chunks(Path("data/raw"), 2000)
    assert sorted(c.file_path for c in chunks) == [
        "data/raw/repo/docs/guide.md",
        "data/raw/repo/main.py",
    ]


def test_save_then_load_round_trip(tiny_project: Path) -> None:
    """What is saved is exactly what is loaded, and no temp file stays."""
    index = ChunkIndex(
        max_chunk_size=2000, chunks=build_chunks(Path("data/raw"), 2000)
    )
    target = save_index(index, Path("data/processed"))
    assert target == Path("data/processed") / INDEX_FILENAME
    assert load_index(Path("data/processed")) == index
    assert list(Path("data/processed").iterdir()) == [target]


def test_check_chunks_refuses_oversized_chunk() -> None:
    """A chunk above the limit is refused before anything is written."""
    chunk = Chunk(
        file_path="data/raw/x.md",
        first_character_index=0,
        last_character_index=11,
        text="hello world",
    )
    check_chunks([chunk], 11)
    with pytest.raises(ValueError):
        check_chunks([chunk], 10)


def test_index_command_writes_the_index(tiny_project: Path) -> None:
    """The CLI command runs end to end on the tiny corpus."""
    RagCli().index(max_chunk_size=500)
    index = load_index(Path("data/processed"))
    assert index.max_chunk_size == 500
    assert len(index.chunks) == 2


@pytest.mark.parametrize("size", ["abc", 2.5, True, 0, 99, 2001])
def test_index_command_rejects_bad_sizes(
    tiny_project: Path, size: object
) -> None:
    """Every invalid --max_chunk_size is a CliError, not a traceback."""
    with pytest.raises(CliError):
        RagCli().index(max_chunk_size=size)  # type: ignore[arg-type]


@pytest.mark.parametrize("raw_dir", ["nowhere", "/"])
def test_index_command_rejects_bad_raw_dir(
    tiny_project: Path, raw_dir: str
) -> None:
    """A missing raw_dir, or one outside the project, is refused."""
    with pytest.raises(CliError):
        RagCli().index(raw_dir=raw_dir)


def test_index_command_rejects_empty_corpus(tiny_project: Path) -> None:
    """A raw_dir with nothing to index is an error, not an empty index."""
    Path("empty").mkdir()
    with pytest.raises(CliError):
        RagCli().index(raw_dir="empty")
