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
        terms.append(lowered)
        pieces = split_identifier(word)
        if pieces != [lowered]:
            terms.extend(pieces)
    return terms
