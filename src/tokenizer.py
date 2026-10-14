"""Turn a text into the list of terms the lexical index works on.

The same function runs on every chunk at index time and on the question at
query time. Two vocabularies built by two different functions would never
match, so there is exactly one tokeniser in the project.
"""

import re
from typing import List

# A word is a run of letters, digits and underscores: this keeps a Python
# identifier such as ''get_model_config'' in one piece. Everything else
# (punctuation, space, dots in ''vllm.engine.arg_utils´') seperates words.
WORD = re.compile(r"[A-Za-z0-9_]+")
# Piece of a CamelCase word: "LLMEngine"-> "LLM", "Engine";
# "Qwen3" -> "Qwen3". An acronym ends before the last capital that
# starts a capitalised word.
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z0-9])|[A-Z]?[a-z0-9]+|[A-Z]+")
# English function words, dropped at index and query time. On this mixed
# corpus they are not neutral: "the" is rare in code chunks, so it marks
# "this is prose" and pushed Markdown chunks up for every question
# (stage 6 error analysis, 2026-10-13).
STOPWORDS = frozenset("""
a an the of in on at to for from by with as is are was were be been being
it its this that these those and or not no but if then than so such what
which who whom whose when where why how do does did done can could should
would may might must will shall have has had having i you he she we they
me him her us them my your his our their there here very also any all each
some more most other into over under about between through during before
after
""".split())


# Pieces shorter than this are dropped: "vLLM" -> "v", "MoE" -> "e",
# "vLLM's" -> "s" are rare terms with a high IDF and no meaning.
MIN_PIECE = 2


def split_identifier(word: str) -> List[str]:
    """Split ``word`` on underscores and CamelCase, lowercased.

    ``get_model_config`` -> ``['get', 'model', 'config']``;
    ``LLMEngine`` -> ``['llm', 'engine']``; ``lora`` -> ``['lora']``.

    Args:
        word: A single word as matched by ``WORD``.

    Returns:
        The lowercased pieces, in order, never empty for a non-empty word.
    """
    pieces: List[str] = []
    for part in word.split("_"):
        pieces.extend(piece.lower() for piece in CAMEL.findall(part))
    return pieces


def tokenize(text: str) -> List[str]:
    """Return the terms of ``text``, with identifiers whole and split.

    Each word yields its lowercased form, and, when it is a compound
    identifier, its pieces too. Keeping both lets a question that quotes
    ``max_model_len`` verbatim match exactly, while a question that
    paraphrases it as "the maximum model length" still matches the pieces.

    Args:
        text: Any text: a chunk or a question.

    Returns:
        The terms in text order, with repetitions kept (the scorer counts
        them). Empty for a text without any word.
    """
    terms: List[str] = []
    for match in WORD.finditer(text):
        word = match.group()
        lowered = word.lower()
        if lowered in STOPWORDS:
            continue
        terms.append(lowered)
        pieces = split_identifier(word)
        if pieces != [lowered]:
            terms.extend(
                p for p in pieces
                if p not in STOPWORDS and len(p) >= MIN_PIECE
            )
    return terms
