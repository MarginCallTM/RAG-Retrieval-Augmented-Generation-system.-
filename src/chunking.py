"""Split a file into chunks with exact character offsets.

Every function here works on positions in the original text and never
modifies it, so that ``text[first:last]`` is always the chunk text.
A span is a pair ``(first, last)`` with ``last`` exclusive.
"""

import re
from typing import List, Tuple

from src.models import Chunk

Span = Tuple[int, int]

# [ \t] and not \s: \s also matches "\n", which would let a blank line
# followed by a fence count as a fence, and flip the in-code state twice.
HEADING = re.compile(r"#{1,6}[ \t]")
FENCE = re.compile(r"[ \t]*(```|~~~)")


def line_starts(text: str) -> List[int]:
    """Return the offset of the first character of every line."""
    starts: List[int] = [0]
    for index, char in enumerate(text):
        if char == "\n" and index + 1 < len(text):
            starts.append(index + 1)
    return starts


def paragraph_blocks(text: str, start: int, end: int) -> List[Span]:
    """Split ``text[start:end]`` into contiguous blocks at blank lines.

    A block is a run of non-blank lines followed by the blank lines after
    it, so the blocks tile the range with no gap and no overlap.
    """
    blocks: List[Span] = []
    block_start = start
    previous_blank = False
    for line_start in line_starts(text[start:end]):
        position = start + line_start
        line_end = text.find("\n", position, end)
        line = text[position:end if line_end == -1 else line_end]
        blank = line.strip() == ""
        if previous_blank and not blank and position > block_start:
            blocks.append((block_start, position))
            block_start = position
        previous_blank = blank
    if block_start < end:
        blocks.append((block_start, end))
    return blocks


def hard_split(text: str, span: Span, max_chunk_size: int) -> List[Span]:
    """Cut an oversized span into pieces of at most ``max_chunk_size``.

    Each piece ends after the last newline that fits, so lines stay whole
    when possible. A single line longer than the limit is cut mid-line.
    """
    pieces: List[Span] = []
    start, end = span
    while end - start > max_chunk_size:
        cut = text.rfind("\n", start, start + max_chunk_size) + 1
        if cut <= start:
            cut = start + max_chunk_size
        pieces.append((start, cut))
        start = cut
    pieces.append((start, end))
    return pieces


def pack_blocks(
    text: str, blocks: List[Span], max_chunk_size: int
) -> List[Span]:
    """Merge adjacent blocks greedily into spans of at most the limit.

    Blocks must be contiguous. A block alone above the limit is hard-split.
    """
    spans: List[Span] = []
    if not blocks:
        return spans
    current_start = blocks[0][0]
    current_end = current_start
    for block_start, block_end in blocks:
        if block_end - current_start <= max_chunk_size:
            current_end = block_end
            continue
        if current_end > current_start:
            spans.append((current_start, current_end))
        if block_end - block_start > max_chunk_size:
            spans.extend(
                hard_split(text, (block_start, block_end), max_chunk_size)
            )
            current_start = current_end = block_end
        else:
            current_start, current_end = block_start, block_end
    if current_end > current_start:
        spans.append((current_start, current_end))
    return spans


def split_text(text: str, max_chunk_size: int) -> List[Span]:
    """Split plain text on blank lines, packing paragraphs up to the limit.

    Used for ``.txt`` files, where ``#`` starts a comment, not a heading.
    """
    blocks = paragraph_blocks(text, 0, len(text))
    return pack_blocks(text, blocks, max_chunk_size)


def markdown_sections(text: str) -> List[Span]:
    """Return one span per Markdown section, from a heading to the next.

    Lines starting with ``#`` inside a fenced code block are code comments,
    not headings: 137 such lines exist in the vLLM docs.
    """
    boundaries: List[int] = [0]
    in_fence = False
    for start in line_starts(text):
        if FENCE.match(text, start):
            in_fence = not in_fence
        elif not in_fence and start > 0 and HEADING.match(text, start):
            boundaries.append(start)
    boundaries.append(len(text))
    return [
        (first, last)
        for first, last in zip(boundaries, boundaries[1:])
        if last > first
    ]


def split_markdown(text: str, max_chunk_size: int) -> List[Span]:
    """Split Markdown on headings; split oversized sections on paragraphs.

    One chunk per section when it fits, because 79 of the 100 docs
    references start on a heading. Sections are never merged together.
    """
    spans: List[Span] = []
    for first, last in markdown_sections(text):
        if last - first <= max_chunk_size:
            spans.append((first, last))
        else:
            blocks = paragraph_blocks(text, first, last)
            spans.extend(pack_blocks(text, blocks, max_chunk_size))
    return spans


def trim_span(text: str, span: Span) -> Span:
    """Move the span bounds inward past leading and trailing whitespace.

    Only the indices move, the text is untouched. The docs references never
    start or end with whitespace (0/100), so trimmed chunks line up with
    them exactly. A whitespace-only span comes back empty.
    """
    first, last = span
    while first < last and text[first].isspace():
        first += 1
    while last > first and text[last - 1].isspace():
        last -= 1
    return first, last


def chunk_file(
    file_path: str, text: str, max_chunk_size: int
) -> List[Chunk]:
    """Split one file into chunks, choosing the splitter by extension.

    Spans are trimmed of surrounding whitespace, and whitespace-only spans
    are dropped: they carry no words to match.

    Args:
        file_path: Corpus path of the file, stored verbatim in each chunk.
        text: Full content of the file, unmodified.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        The chunks in file order.

    Raises:
        ValueError: If ``max_chunk_size`` is not positive.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")
    if file_path.endswith(".md"):
        spans = split_markdown(text, max_chunk_size)
    else:
        # .txt, and .py until the Python splitter exists.
        spans = split_text(text, max_chunk_size)
    chunks: List[Chunk] = []
    for span in spans:
        first, last = trim_span(text, span)
        if last > first:
            chunks.append(Chunk(
                file_path=file_path,
                first_character_index=first,
                last_character_index=last,
                text=text[first:last],
            ))
    return chunks
