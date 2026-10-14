"""Command definitions exposed through Python fire.

Every public method of ``RagCli`` becomes a sub-command of
``uv run python -m src``. User errors are raised as ``CliError`` and turned
into a clean message by ``src/__main__.py``, never into a traceback.
"""

import time
from collections import Counter
from pathlib import Path
from typing import List

from pydantic import ValidationError
from tqdm import tqdm

from src.datasets import (
    MAX_CONTEXT_LENGTH, load_chunk, read_model, save_model,
    save_search_results,
)
from src.evaluation import MISS_HINTS, evaluate_results, list_misses
from src.generator import DEFAULT_MODEL, Generator
from src.indexer import build_chunks, check_chunks, load_index, save_index
from src.models import (
    AnsweredDataset, ChunkIndex, MinimalAnswer, MinimalSearchResults,
    MinimalSource, RagDataset, StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retriever import Retriever

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


def parse_k(value: object, option: str = "--k") -> int:
    """Validate a count option as given by fire: a non-negative integer.

    ``k=0`` is allowed and returns nothing, like a search with no match.

    Args:
        value: The raw value from fire.
        option: The option name, for the error message.

    Raises:
        CliError: if the value is not an integer or is negative.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise CliError(f"{option} must be an integer, got {value!r}")
    if value < 0:
        raise CliError(f"{option} must be >= 0, got {value}")
    return value


def load_retriever(processed_dir: str) -> Retriever:
    """Load the index written by ``index`` and build a retriever on it.

    Raises:
        CliError: If the index is missing or cannot be parsed.
    """
    try:
        index = load_index(Path(processed_dir))
    except FileNotFoundError:
        raise CliError(
            f"no index in {processed_dir}: run `index` first"
        ) from None
    except ValidationError as error:
        # str(error) is a multi-line dump ending with a pydantic URL.
        raise CliError(
            f"the index in {processed_dir} is not valid "
            f"({error.error_count()} validation error(s)): run `index` again"
        ) from error
    except (ValueError, OSError) as error:
        raise CliError(
            f"could not load the index in {processed_dir}, run `index` "
            f"again: {error}"
        ) from error
    return Retriever(index)


def load_generator(model_name: str) -> Generator:
    """Load the generation model, with a clean error when it cannot be.

    Raises:
        CliError: If the model is not found, not downloadable (offline),
            or does not fit in memory.
    """
    print(f"Loading {model_name}...", flush=True)
    try:
        return Generator(model_name)
    except (OSError, ValueError, RuntimeError) as error:
        raise CliError(
            f"could not load the model {model_name!r}: {error}\n"
            "The first run downloads it from Hugging Face (about 1.5 GB) "
            "and needs the network."
        ) from error


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

    def search(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
        render_path: str = "",
    ) -> None:
        """Print the top-k sources for a single query.

        Args:
            query: The question to search for.
            k: Number of sources to return.
            processed_dir: Directory holding the index written by ``index``.
            render_path: If given, also write a JSON file there with the
                text of each result and the score share of each query term.

        Raises:
            CliError: On an invalid k, a missing index, or an unwritable
                render path.
        """
        top = parse_k(k)
        # fire turns `search 42` into an int: the query is always a string.
        text = str(query)
        retriever = load_retriever(processed_dir)
        results = retriever.search(text, top)
        if not results:
            print("no result")
        for rank, result in enumerate(results, start=1):
            chunk = result.chunk
            print(
                f"{rank:2d}. {chunk.file_path} "
                f"[{chunk.first_character_index}:"
                f"{chunk.last_character_index}] score={result.score:.2f}"
            )
        if render_path:
            target = Path(str(render_path))
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "w", encoding="utf-8") as handle:
                    handle.write(
                        retriever.explain(text, top).model_dump_json(indent=2)
                    )
            except OSError as error:
                raise CliError(f"could not write {target}: {error}") from error
            print(f"rendered to {target}")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        processed_dir: str = "data/processed",
    ) -> None:
        """Run search over a whole dataset and write a results JSON file.

        The output is named after the dataset file, so scope
        ``save_directory`` by dataset: the public datasets share names.

        Args:
            dataset_path: Path to a dataset JSON file.
            k: Number of sources to return per question.
            save_directory: Directory where the results file is written.
            processed_dir: Directory holding the index written by ``index``.

        Raises:
            CliError: On an invalid k, an unreadable dataset, a missing
                index, or an output the grader would reject.
        """
        top = parse_k(k)
        source = Path(str(dataset_path))
        try:
            dataset = read_model(source, RagDataset)
        except (OSError, ValueError) as error:
            raise CliError(f"could not read the dataset: {error}") from error
        retriever = load_retriever(processed_dir)

        started = time.perf_counter()
        results = [
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[
                    MinimalSource(
                        file_path=hit.chunk.file_path,
                        first_character_index=hit.chunk.first_character_index,
                        last_character_index=hit.chunk.last_character_index,
                    )
                    for hit in retriever.search(question.question, top)
                ],
            )
            for question in tqdm(
                dataset.rag_questions, desc="Searching", unit="question"
            )
        ]
        elapsed = time.perf_counter() - started

        student = StudentSearchResults(search_results=results, k=top)
        try:
            target = save_search_results(
                student, Path(str(save_directory)) / source.name
            )
        except (ValueError, OSError) as error:
            raise CliError(f"could not write the results: {error}") from error
        print(
            f"Searched {len(results)} questions (k={top}) "
            f"in {elapsed:.1f}s -> {target}"
        )

    def answer(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        """Answer a single query from the top-k retrieved chunks.

        Args:
            query: The question to answer.
            k: Number of sources to retrieve before answering.
            processed_dir: Directory holding the index written by ``index``.
            model_name: Hugging Face model id or local directory.

        Raises:
            CliError: On an invalid k, a missing index, or a model that
                cannot be loaded.
        """
        top = parse_k(k)
        text = str(query)
        # Retrieve before loading the model: a missing index fails in
        # 0.4 s instead of after a 1.5 GB download.
        retriever = load_retriever(processed_dir)
        results = retriever.search(text, top)
        generator = load_generator(str(model_name))
        chunks = generator.select([result.chunk for result in results])

        started = time.perf_counter()
        answer = generator.generate(text, chunks)
        elapsed = time.perf_counter() - started
        print(answer)
        print(f"\nSources ({len(chunks)} of {len(results)} retrieved, "
              f"generated in {elapsed:.1f}s):")
        for number, chunk in enumerate(chunks, start=1):
            print(
                f"[{number}] {chunk.file_path} "
                f"[{chunk.first_character_index}:"
                f"{chunk.last_character_index}]"
            )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        model_name: str = DEFAULT_MODEL,
        limit: int = 0,
    ) -> None:
        """Generate an answer for every question of a search results file.

        The output keeps the retrieved sources and adds the answer, under
        the same file name, so scope ``save_directory`` by dataset.

        Args:
            student_search_results_path: Path to a search results JSON
                file written by ``search_dataset``.
            save_directory: Directory where the answers file is written.
            model_name: Hugging Face model id or local directory.
            limit: If positive, only answer the first ``limit`` questions
                (for prompt experiments: a full dataset takes over an
                hour on CPU).

        Raises:
            CliError: On an unreadable results file, a source that cannot
                be read from the corpus, or a model that cannot be loaded.
        """
        source = Path(str(student_search_results_path))
        try:
            student = read_model(source, StudentSearchResults)
        except (OSError, ValueError) as error:
            raise CliError(f"could not read the results: {error}") from error
        results = student.search_results[:parse_k(limit, "--limit") or None]
        generator = load_generator(str(model_name))

        started = time.perf_counter()
        answers: List[MinimalAnswer] = []
        for result in tqdm(results, desc="Answering", unit="question"):
            try:
                chunks = [load_chunk(s) for s in result.retrieved_sources]
            except (OSError, ValueError) as error:
                raise CliError(
                    f"could not read a source of question "
                    f"{result.question_id}: {error} (run from the "
                    "repository root, with data/raw/ in place)"
                ) from error
            text = generator.generate(
                result.question, generator.select(chunks)
            )
            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=text,
            ))
        elapsed = time.perf_counter() - started

        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=student.k
        )
        try:
            target = save_model(
                output, Path(str(save_directory)) / source.name
            )
        except OSError as error:
            raise CliError(f"could not write the answers: {error}") from error
        print(
            f"Answered {len(answers)} questions in {elapsed:.0f}s "
            f"({elapsed / max(len(answers), 1):.1f}s each) -> {target}"
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        show_misses: bool = False,
        misses_at: int = 5,
    ) -> None:
        """Report recall@k of a search results file against a dataset.

        Args:
            student_search_results_path: Path to a search results JSON
                file written by ``search_dataset``.
            dataset_path: Path to the ground-truth dataset JSON file,
                where every question carries its sources.
            show_misses: If set, list every source missed at ``misses_at``
                with the closest result, grouped by cause.
            misses_at: The k at which misses are listed.

        Raises:
            CliError: If a file is missing or does not match its model, or
                on an invalid misses_at.
        """
        # Validate every argument before any work, so a typo in the last
        # option fails at once rather than after the report.
        at = parse_k(misses_at, "--misses_at")
        try:
            student = read_model(
                Path(str(student_search_results_path)), StudentSearchResults
            )
            truth = read_model(Path(str(dataset_path)), AnsweredDataset)
        except (OSError, ValueError) as error:
            raise CliError(f"could not read the inputs: {error}") from error

        report = evaluate_results(truth.rag_questions, student.search_results)
        if report.missing_ids:
            print(
                f"warning: {len(report.missing_ids)} question(s) have no "
                "results in the file, counted as misses"
            )
        if report.unknown_ids:
            print(
                f"warning: {len(report.unknown_ids)} result(s) match no "
                "question of the dataset, ignored"
            )
        largest = max(report.recall)
        if student.k < largest:
            print(
                f"warning: the file holds at most k={student.k} sources per "
                f"question, recall@{largest} is capped by that"
            )
        print(f"Questions evaluated: {report.evaluated}")
        for k, value in report.recall.items():
            print(f"recall@{k}: {value:.3f} ({value:.1%})")

        if not show_misses:
            return
        misses = list_misses(truth.rag_questions, student.search_results, at)
        counts: Counter[str] = Counter(miss.kind for miss in misses)
        print(f"\nMisses at k={at}: {len(misses)} / {report.evaluated}")
        for kind, hint in MISS_HINTS.items():
            print(f"  {kind:<10} {counts.get(kind, 0):3d}  {hint}")
        for miss in misses:
            where = ""
            if miss.rank is not None:
                where = f" rank={miss.rank} iou={miss.best_iou:.3f}"
            expected = miss.expected
            print(
                f"\n[{miss.kind}{where}] {miss.question}\n"
                f"  expected {expected.file_path} "
                f"[{expected.first_character_index}:"
                f"{expected.last_character_index}]"
            )
