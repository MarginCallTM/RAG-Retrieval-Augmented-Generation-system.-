"""Tests for src/generator.py, without loading the model."""

from src.generator import NO_ANSWER, SYSTEM_PROMPT, build_messages, fit_sources
from src.models import Chunk


def chunk(text: str, path: str = "data/raw/repo/a.md") -> Chunk:
    return Chunk(
        file_path=path, first_character_index=0,
        last_character_index=len(text), text=text,
    )


# A character is a token: keeps the arithmetic readable.
def count(text: str) -> int:
    return len(text)


def cut(text: str, tokens: int) -> str:
    return text[:tokens]


def test_fit_keeps_best_ranked_chunks_within_budget() -> None:
    chunks = [chunk("a" * 40), chunk("b" * 40), chunk("c" * 40)]
    kept = fit_sources(chunks, 100, count, cut)
    assert [c.text[0] for c in kept] == ["a", "b"]


def test_fit_stops_at_the_first_chunk_that_does_not_fit() -> None:
    # The third chunk would fit, but it is ranked after one that does not.
    chunks = [chunk("a" * 40), chunk("b" * 70), chunk("c" * 10)]
    kept = fit_sources(chunks, 100, count, cut)
    assert [c.text[0] for c in kept] == ["a"]


def test_fit_truncates_the_first_chunk_rather_than_dropping_it() -> None:
    kept = fit_sources([chunk("a" * 500)], 100, count, cut)
    assert len(kept) == 1
    assert kept[0].text == "a" * 100
    # The offsets still point at the original span.
    assert kept[0].last_character_index == 500


def test_fit_with_no_budget_or_no_chunk() -> None:
    assert fit_sources([chunk("abc")], 0, count, cut) == []
    assert fit_sources([], 100, count, cut) == []


def test_messages_number_and_label_the_sources() -> None:
    chunks = [chunk("first", "data/raw/repo/a.md"), chunk("second")]
    system, user = build_messages("Why?", chunks)
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert user["role"] == "user"
    assert "[1] data/raw/repo/a.md\nfirst" in user["content"]
    assert "[2] data/raw/repo/a.md\nsecond" in user["content"]
    assert user["content"].endswith("Question: Why?")


def test_messages_without_source_still_ask_the_question() -> None:
    _, user = build_messages("Why?", [])
    assert "(no source)" in user["content"]
    assert NO_ANSWER in SYSTEM_PROMPT
