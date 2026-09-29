"""Command definitions exposed through Python fire.

Every public method of ``RagCli`` becomes a sub-command of
``uv run python -m src``. User errors are raised as ``CliError`` and turned
into a clean message by ``src/__main__.py``, never into a traceback.
"""

import time
from pathlib import Path

from pydantic import ValidationError

from src.indexer import build_chunks, check_chunks, save_index
from src.models import ChunkIndex

# The moulinette rejects any source longer than this (max_context_length).
MAX_CONTEXT_LENGTH = 2000
# Below this, a chunk holds a few words and memory grows with the chunk
# count: 650 MB at 50 characters, over 10 GB at 1.
MIN_CHUNK_SIZE = 100


class CliError(Exception):
    """A user error: bad argument, missing file. Printed without traceback."""


def parse_chunk_size(value: object) -> int:
    """Validate ``--max_chunk_size`` as given by fire.

    fire turns ``--max_chunk_size abc`` into a string and ``2.5`` into a
    float, so the type is checked here, not trusted from the annotation.

    Raises:
        CliError: If the value is not an integer in 100..2000.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise CliError(f"--max_chunk_size must be an integer, got {value!r}")
    if not MIN_CHUNK_SIZE <= value <= MAX_CONTEXT_LENGTH:
        raise CliError(
            f"--max_chunk_size must be between {MIN_CHUNK_SIZE} and "
            f"{MAX_CONTEXT_LENGTH} (the moulinette rejects longer sources),"
            f" got {value}"
        )
    return value


class RagCli:
    """Commands of the RAG system, one method per CLI command."""

    def index(
        self,
        max_chunk_size: int = MAX_CONTEXT_LENGTH,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        """Chunk the corpus under ``raw_dir`` and save it to ``processed_dir``.

        Args:
            max_chunk_size: Maximum size of a chunk, in characters.
            raw_dir: Directory holding the raw repositories.
            processed_dir: Directory where the index is written.

        Raises:
            CliError: On an invalid size, a missing or misplaced raw_dir,
                or an empty corpus.
        """
        size = parse_chunk_size(max_chunk_size)
        raw = Path(raw_dir)
        if not raw.is_dir():
            raise CliError(f"raw directory not found: {raw_dir}")
        if not raw.resolve().is_relative_to(Path.cwd().resolve()):
            raise CliError(
                f"{raw_dir} must be inside the current directory, so that "
                "chunk paths start with data/raw/ like the dataset paths"
            )

        started = time.perf_counter()
        chunks = build_chunks(raw, size)
        if not chunks:
            raise CliError(f"no .py, .md or .txt file to index in {raw_dir}")
        try:
            check_chunks(chunks, size)
            target = save_index(
                ChunkIndex(max_chunk_size=size, chunks=chunks),
                Path(processed_dir),
            )
        except (ValueError, ValidationError, OSError) as error:
            raise CliError(f"could not write the index: {error}") from error

        elapsed = time.perf_counter() - started
        files = len({chunk.file_path for chunk in chunks})
        print(
            f"Indexed {len(chunks)} chunks from {files} files "
            f"(max_chunk_size={size}) in {elapsed:.1f}s -> {target}"
        )

    def search(self, query: str, k: int = 5) -> None:
        """Return the top-k sources for a single query.

        Args:
            query: The question to search for.
            k: Number of sources to return.
        """
        print(f"search: not implemented (query={query!r}, k={k})")

    def search_dataset(
        self, dataset_path: str, k: int = 10, save_directory: str = "."
    ) -> None:
        """Run search over a whole dataset and write a results JSON file.

        Args:
            dataset_path: Path to a dataset JSON file.
            k: number of sources to return per question.
            save_directory: Directory where the results file is written.
            """
        print(
            "search_dataset: not implemented "
            f"(dataset_path={dataset_path!r}, k={k},"
            f"save_directory={save_directory!r})"
        )

    def answer(self, query: str, k: int = 5) -> None:
        """Answer a single query using the retrieved context.

        Args:
            query: The question to answer.
            k: Number of sources to retrieve before answering.
        """
        print(f"answer: not implemented (query= {query!r}, k={k})")

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str = "."
    ) -> None:
        """Generate answers for every question of a search results file.

        Args:
            student_search_results_path: Path to a search results JSON file.
            save_directory: Directory where the answers file is written.
        """
        print(
            "answer_dataset: not implemented "
            f"(student_search_result_path={student_search_results_path!r}, "
            f"save_directory={save_directory!r})"
        )

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Report recall@k of a search results file against a dataset.

        Args:
                student_search_results_path: Path to a search result JSON file.
                dataset_path: Path to the ground-truth dataset JSON file.
        """
        print(
            "evaluate: not implemented"
            f"(student_search_results_path={student_search_results_path!r}, "
            f"dataset_path={dataset_path!r})"
        )
