"""Build the chunks of the whole corpus and persist them to disk."""

from pathlib import Path
from typing import List

from tqdm import tqdm

from src.chunking import chunk_file
from src.ingest import list_corpus_files, read_text, to_corpus_path
from src.models import Chunk, ChunkIndex

INDEX_FILENAME = "chunks.json"


def build_chunks(raw_dir: Path, max_chunk_size: int) -> List[Chunk]:
    """Chunk every indexable file under ``raw_dir``.

    Args:
        raw_dir: Directory holding the raw repositories.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        The chunks of all files, in sorted file order.

    Raises:
        ValueError: If ``raw_dir`` is outside the current directory, since
            the corpus paths would then not match the dataset paths.
    """
    chunks: List[Chunk] = []
    files = list_corpus_files(raw_dir)
    for path in tqdm(files, desc="Chunking", unit="file"):
        text = read_text(path)
        if text is None:
            continue
        chunks.extend(chunk_file(to_corpus_path(path), text, max_chunk_size))
    return chunks


def check_chunks(chunks: List[Chunk], max_chunk_size: int) -> None:
    """Refuse to persist a chunk the moulinette would reject.

    A single source longer than the limit makes the whole output file
    invalid, so the check runs on every chunk before anything is written.

    Raises:
        ValueError: On the first chunk that is empty, too long, or whose
            text does not match its indices.
    """
    for chunk in chunks:
        length = chunk.last_character_index - chunk.first_character_index
        if not 0 < length <= max_chunk_size or len(chunk.text) != length:
            raise ValueError(
                f"invalid chunk {chunk.file_path} "
                f"[{chunk.first_character_index}:"
                f"{chunk.last_character_index}] (limit {max_chunk_size})"
            )


def save_index(index: ChunkIndex, processed_dir: Path) -> Path:
    """Write the index as JSON, atomically, and return its path.

    The file is written under a temporary name then renamed, so an
    interrupted run never leaves a half-written index behind.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    target = processed_dir / INDEX_FILENAME
    temporary = processed_dir / (INDEX_FILENAME + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(index.model_dump_json())
    temporary.replace(target)
    return target


def load_index(processed_dir: Path) -> ChunkIndex:
    """Read the index written by ``save_index``.

    Raises:
        FileNotFoundError: If ``index`` has not been run yet.
        pydantic.ValidationError: If the file is corrupted.
    """
    path = processed_dir / INDEX_FILENAME
    with open(path, encoding="utf-8") as handle:
        return ChunkIndex.model_validate_json(handle.read())
