"""Find the corpus files to index and read them without shifting offsets."""

import sys
from pathlib import Path
from typing import List, Optional, Sequence

DEFAULT_EXTENSIONS: Sequence[str] = (".py", ".md", ".txt")
# Excluded only at the top of a repository (vllm-0.10.1/tests), never
# deeper: vllm/benchmarks/ holds 2 reference sources of the code dataset.
DEFAULT_EXCLUDED_DIRS: Sequence[str] = ("tests", "benchmarks")


def list_corpus_files(
    raw_dir: Path,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    excluded_dirs: Sequence[str] = DEFAULT_EXCLUDED_DIRS,
) -> List[Path]:
    """Return the files to index under ``raw_dir``, sorted.

    ``raw_dir`` holds one folder per repository (``data/raw/vllm-0.10.1``).
    A file is kept when its extension is in ``extensions`` and it does not
    sit in a folder of ``excluded_dirs`` located at the top of its
    repository.

    Args:
        raw_dir: Directory holding the raw repositories.
        extensions: File extensions to keep, with their leading dot.
        excluded_dirs: Top-level repository folders to skip.

    Returns:
        The kept files, sorted so that indexing is deterministic.
    """
    kept: List[Path] = []
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.suffix not in extensions:
            continue
        parts = path.relative_to(raw_dir).parts
        # parts = ("vllm-0.10.1", "tests", ...): parts[1] is the top folder.
        if len(parts) > 2 and parts[1] in excluded_dirs:
            continue
        kept.append(path)
    return sorted(kept)


def to_corpus_path(path: Path) -> str:
    """Return ``path`` as the grader expects it: relative, with ``/``.

    The grader compares paths verbatim with the dataset, which uses
    ``data/raw/vllm-0.10.1/...``. The path is made relative to the current
    directory (the repo root), whatever form ``raw_dir`` was given in.

    Args:
        path: A file under the current directory.

    Returns:
        The POSIX relative path, e.g. ``data/raw/vllm-0.10.1/README.md``.

    Raises:
        ValueError: If ``path`` is not under the current directory.
    """
    return path.resolve().relative_to(Path.cwd().resolve()).as_posix()


def read_text(path: Path) -> Optional[str]:
    """Read a file as strict UTF-8 text, or return None if impossible.

    Text mode with strict decoding matches how the dataset offsets were
    computed. A file that cannot be decoded is skipped with a warning:
    replacing bad bytes would shift every offset after them.

    Args:
        path: File to read.

    Returns:
        The file content, or None if it is unreadable.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except (UnicodeDecodeError, OSError) as error:
        print(f"warning: skipping {path}: {error}", file=sys.stderr)
        return None
