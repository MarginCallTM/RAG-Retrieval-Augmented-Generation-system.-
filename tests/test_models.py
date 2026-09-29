"""Tests for the pydantic models of src/models.py."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel, ValidationError

from src.models import (
    AnsweredDataset,
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)

DATASETS_DIR = Path("data/datasets")
DATASET_NAMES: List[str] = [
    "dataset_docs_public.json",
    "dataset_code_public.json",
]

# Field names copied from section VI.4 of the subject. The moulinette reads
# these exact names: a typo here is a silent zero.
SUBJECT_FIELDS: Dict[type[BaseModel], List[str]] = {
    MinimalSource: [
        "file_path", "first_character_index", "last_character_index",
    ],
    UnansweredQuestion: ["question_id", "question"],
    AnsweredQuestion: ["question_id", "question", "sources", "answer"],
    RagDataset: ["rag_questions"],
    MinimalSearchResults: ["question_id", "question", "retrieved_sources"],
    MinimalAnswer: [
        "question_id", "question", "retrieved_sources", "answer",
    ],
    StudentSearchResults: ["search_results", "k"],
    StudentSearchResultsAndAnswer: ["search_results", "k"],
}


def load_raw(folder: str, name: str) -> str:
    """Return the text of a dataset file, or skip if data is absent."""
    path = DATASETS_DIR / folder / name
    if not path.is_file():
        pytest.skip(f"{path} not found (data is not committed)")
    return path.read_text(encoding="utf-8")


def keep_subject_fields(question: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the dataset-only keys (difficulty, is_valid) from a question."""
    fields = SUBJECT_FIELDS[AnsweredQuestion]
    return {key: value for key, value in question.items() if key in fields}


@pytest.mark.parametrize("model, fields", SUBJECT_FIELDS.items())
def test_field_names_match_subject(
    model: type[BaseModel], fields: List[str]
) -> None:
    """Every model has exactly the field names of the subject."""
    assert list(model.model_fields) == fields


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_answered_round_trip(name: str) -> None:
    """An answered dataset loads, and dumps back to the same content."""
    raw = load_raw("AnsweredQuestions", name)
    dataset = RagDataset.model_validate_json(raw)

    assert all(
        isinstance(q, AnsweredQuestion) for q in dataset.rag_questions
    )
    questions = json.loads(raw)["rag_questions"]
    original = [keep_subject_fields(q) for q in questions]
    dumped = json.loads(dataset.model_dump_json())["rag_questions"]
    assert dumped == original


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_unanswered_picks_unanswered_variant(name: str) -> None:
    """Questions without sources are parsed as UnansweredQuestion."""
    raw = load_raw("UnansweredQuestions", name)
    dataset = RagDataset.model_validate_json(raw)

    assert all(
        type(q) is UnansweredQuestion for q in dataset.rag_questions
    )
    assert dataset.rag_questions[0].question_id


def test_question_ids_are_unique_by_default() -> None:
    """The default_factory gives each question its own id."""
    first = UnansweredQuestion(question="a")
    second = UnansweredQuestion(question="b")
    assert first.question_id != second.question_id


@pytest.mark.parametrize(
    "first, last",
    [(10, 10), (20, 10), (-1, 5)],
)
def test_source_rejects_bad_range(first: int, last: int) -> None:
    """Empty, reversed or negative ranges are refused."""
    with pytest.raises(ValidationError):
        MinimalSource(
            file_path="data/raw/x.md",
            first_character_index=first,
            last_character_index=last,
        )


BROKEN_ANSWERED = json.dumps({
    "rag_questions": [{
        "question": "q",
        "answer": "a",
        "sources": [{
            "file_path": "data/raw/x.md",
            "first_character_index": "abc",
            "last_character_index": 5,
        }],
    }],
})


def test_rag_dataset_silently_drops_broken_sources() -> None:
    """Document the union fallback: a broken answer becomes Unanswered."""
    dataset = RagDataset.model_validate_json(BROKEN_ANSWERED)
    assert type(dataset.rag_questions[0]) is UnansweredQuestion


def test_answered_dataset_refuses_broken_sources() -> None:
    """The ground-truth model raises instead of dropping the sources."""
    with pytest.raises(ValidationError):
        AnsweredDataset.model_validate_json(BROKEN_ANSWERED)


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_answered_dataset_loads_public_ground_truth(name: str) -> None:
    """The public answered datasets pass the strict model."""
    raw = load_raw("AnsweredQuestions", name)
    assert AnsweredDataset.model_validate_json(raw).rag_questions
