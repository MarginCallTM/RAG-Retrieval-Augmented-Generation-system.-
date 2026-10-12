"""Tests for src/retriever.py."""

from typing import List

from src.models import Chunk, ChunkIndex
from src.retriever import Retriever

TEXTS = [
    "LoRA adapters can be loaded at runtime with enable_lora.",
    "The scheduler decides which requests run in the next step.",
    "The scheduler budget is set by max_num_seqs and max_num_batched_tokens.",
    "Quantization with fp8 reduces memory usage.",
]


def make_index() -> ChunkIndex:
    chunks: List[Chunk] = [
        Chunk(
            file_path=f"data/raw/repo/file{i}.md",
            first_character_index=0,
            last_character_index=len(text),
            text=text,
        )
        for i, text in enumerate(TEXTS)
    ]
    return ChunkIndex(max_chunk_size=2000, chunks=chunks)


def test_best_match_comes_first() -> None:
    results = Retriever(make_index()).search("how to enable lora", k=4)
    assert results[0].chunk.file_path == "data/raw/repo/file0.md"
    assert results[0].score > 0


def test_results_are_capped_at_k() -> None:
    results = Retriever(make_index()).search("quantization fp8 lora", k=1)
    assert len(results) == 1
    assert results[0].chunk.file_path == "data/raw/repo/file3.md"


def test_term_in_half_the_corpus_has_zero_idf() -> None:
    # "scheduler" is in 2 chunks out of 4: IDF = ln((4-2+0.5)/(2+0.5)) = 0.
    # BM25 gives it no weight, so alone it retrieves nothing. On the real
    # corpus this is what neutralises "the" or "vllm" without stopwords.
    assert Retriever(make_index()).search("scheduler", k=5) == []


def test_scores_are_decreasing() -> None:
    results = Retriever(make_index()).search("scheduler max_num_seqs", k=4)
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].chunk.file_path == "data/raw/repo/file2.md"


def test_degenerate_queries_return_nothing() -> None:
    retriever = Retriever(make_index())
    assert retriever.search("", k=5) == []
    assert retriever.search("   ", k=5) == []
    assert retriever.search("xqzv plorp", k=5) == []
    assert retriever.search("scheduler", k=0) == []
    assert retriever.search("scheduler", k=-1) == []


def test_k_larger_than_corpus_is_fine() -> None:
    results = Retriever(make_index()).search("scheduler lora fp8", k=100)
    assert 0 < len(results) <= len(TEXTS)


def test_identifier_split_matches_paraphrase() -> None:
    # "batched tokens" is only in the chunk via max_num_batched_tokens.
    results = Retriever(make_index()).search("batched tokens", k=1)
    assert results[0].chunk.file_path == "data/raw/repo/file2.md"


def test_explain_contributions_sum_to_the_score() -> None:
    explained = Retriever(make_index()).explain("enable lora lora", k=2)
    assert explained.terms == ["enable", "lora", "lora"]
    for result in explained.results:
        assert set(result.contributions) <= set(explained.terms)
        assert abs(sum(result.contributions.values()) - result.score) < 0.01
