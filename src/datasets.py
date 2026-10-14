"""Read the question datasets and write the search results, as JSON.

These files are the contract with the moulinette: this module is the only
place that reads or writes them, and it refuses an output the grader
would reject.
"""

from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from src.models import Chunk, MinimalSource, StudentSearchResults

# Every corpus path starts with this, verbatim, like the dataset paths.
CORPUS_PREFIX = "data/raw/"
# The moulinette rejects any source longer than this (max_context_length).
MAX_CONTEXT_LENGTH = 2000

Model = TypeVar("Model", bound=BaseModel)


def read_model(path: Path, model: Type[Model]) -> Model:
    """Load a JSON file into ``model``.

    Args:
        path: The JSON file.
        model: The pydantic model it must match.

    Returns:
        The validated model.

    Raises:
        OSError: If the file cannot be read (missing, permissions).
        ValueError: If it is not JSON or does not match the model.
    """
    with open(path, encoding="utf-8") as handle:
        content = handle.read()
    try:
        return model.model_validate_json(content)
    except ValidationError as error:
        raise ValueError(
            f"{path} does not match {model.__name__}: "
            f"{error.error_count()} validation error(s)"
        ) from error


def check_search_results(
    student: StudentSearchResults, max_length: int = MAX_CONTEXT_LENGTH
) -> None:
    """Refuse a results file the moulinette would reject as a whole.

    Raises:
        ValueError: On the first question with more than ``k`` sources, a
            path outside the corpus, or a source longer than ``max_length``.
    """
    for result in student.search_results:
        where = f"question {result.question_id}"
        if len(result.retrieved_sources) > student.k:
            raise ValueError(f"{where}: more than k={student.k} sources")
        for source in result.retrieved_sources:
            length = (
                source.last_character_index - source.first_character_index
            )
            if not source.file_path.startswith(CORPUS_PREFIX):
                raise ValueError(
                    f"{where}: path {source.file_path!r} does not start "
                    f"with {CORPUS_PREFIX!r}"
                )
            if length > max_length:
                raise ValueError(
                    f"{where}: source of {length} characters in "
                    f"{source.file_path} exceeds {max_length}"
                )


def save_model(model: BaseModel, target: Path) -> Path:
    """Write ``model`` as indented JSON to ``target``, atomically.

    The file is written under a temporary name and renamed at the end,
    so a crash mid-write never leaves a half JSON file behind.

    Raises:
        OSError: If the file cannot be written.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(model.model_dump_json(indent=2))
    temporary.replace(target)
    return target


def save_search_results(student: StudentSearchResults, target: Path) -> Path:
    """Check ``student`` and write it to ``target``, atomically.

    Raises:
        ValueError: If the results break an invariant of the grader.
        OSError: If the file cannot be written.
    """
    check_search_results(student)
    return save_model(student, target)


def load_chunk(source: MinimalSource) -> Chunk:
    """Re-read the text of ``source`` from the corpus on disk.

    The results files only carry paths and offsets; generation needs the
    text. The path is relative to the repository root, like every corpus
    path, so this runs from there.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the offsets fall outside the file.
    """
    with open(source.file_path, encoding="utf-8") as handle:
        content = handle.read()
    if source.last_character_index > len(content):
        raise ValueError(
            f"{source.file_path}: span [{source.first_character_index}:"
            f"{source.last_character_index}] exceeds the file "
            f"({len(content)} characters)"
        )
    return Chunk(
        file_path=source.file_path,
        first_character_index=source.first_character_index,
        last_character_index=source.last_character_index,
        text=content[
            source.first_character_index:source.last_character_index
        ],
    )
