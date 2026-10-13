"""Tests for src/datasets.py."""

from pathlib import Path

import pytest

from src.datasets import check_search_results, read_model, save_search_results
from src.models import (
    MinimalSearchResults, MinimalSource, RagDataset, StudentSearchResults,
)


def result(path: str, first: int, last: int) -> MinimalSearchResults:
    return MinimalSearchResults(
        question_id="q1",
        question="?",
        retrieved_sources=[MinimalSource(
            file_path=path, first_character_index=first,
            last_character_index=last,
        )],
    )


def test_valid_results_pass() -> None:
    student = StudentSearchResults(
        search_results=[result("data/raw/repo/a.md", 0, 2000)], k=1
    )
    check_search_results(student)


def test_wrong_prefix_is_rejected() -> None:
    student = StudentSearchResults(
        search_results=[result("/abs/data/raw/repo/a.md", 0, 10)], k=1
    )
    with pytest.raises(ValueError, match="does not start with"):
        check_search_results(student)


def test_too_long_source_is_rejected() -> None:
    student = StudentSearchResults(
        search_results=[result("data/raw/repo/a.md", 0, 2001)], k=1
    )
    with pytest.raises(ValueError, match="exceeds 2000"):
        check_search_results(student)


def test_more_than_k_sources_is_rejected() -> None:
    too_many = result("data/raw/repo/a.md", 0, 10)
    too_many.retrieved_sources *= 3
    student = StudentSearchResults(search_results=[too_many], k=2)
    with pytest.raises(ValueError, match="more than k=2"):
        check_search_results(student)


def test_save_then_read_round_trip(tmp_path: Path) -> None:
    student = StudentSearchResults(
        search_results=[result("data/raw/repo/a.md", 5, 50)], k=1
    )
    target = save_search_results(student, tmp_path / "out" / "r.json")
    assert read_model(target, StudentSearchResults) == student
    assert not (tmp_path / "out" / "r.json.tmp").exists()


def test_read_model_errors(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        read_model(tmp_path / "missing.json", RagDataset)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match RagDataset"):
        read_model(bad_json, RagDataset)
    wrong_schema = tmp_path / "wrong.json"
    wrong_schema.write_text('{"questions": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match RagDataset"):
        read_model(wrong_schema, RagDataset)


def test_real_datasets_load() -> None:
    for scope in ("AnsweredQuestions", "UnansweredQuestions"):
        path = Path("data/datasets") / scope / "dataset_docs_public.json"
        if path.exists():
            assert len(read_model(path, RagDataset).rag_questions) == 100
