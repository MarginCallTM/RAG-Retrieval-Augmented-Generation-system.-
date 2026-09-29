"""Pydantic models for the data exchanged between the stages of the pipeline.

The eight models below follow section VI.4 of the subject, field for field.
They are the contract with the datasets (input) and with the moulinette
(output): renaming a field breaks that contract.
"""

import uuid
from typing import List

from pydantic import BaseModel, Field, model_validator


class MinimalSource(BaseModel):
    """A character range inside one file of the corpus.

    Attributes:
        file_path: Path of the file, verbatim from the repo root,
            e.g. ``data/raw/vllm-0.10.1/docs/features/lora.md``.
        first_character_index: Index of the first character (inclusive).
        last_character_index: Index after the last character (exclusive),
            so the text is ``content[first:last]``, Python slice style.
    """

    file_path: str
    first_character_index: int
    last_character_index: int

    @model_validator(mode="after")
    def check_range(self) -> "MinimalSource":
        """Reject a range that is empty, reversed or negative."""
        if self.first_character_index < 0:
            raise ValueError("first_character_index must be >= 0")
        if self.first_character_index >= self.last_character_index:
            raise ValueError(
                "first_character_index must be < last_character_index"
            )
        return self


class Chunk(MinimalSource):
    """A slice of a corpus file, produced by the chunkers.

    Not in the subject. It is a MinimalSource plus the text of the slice,
    so that ``text == file_content[first:last]`` always holds and a chunk
    can be returned wherever a MinimalSource is expected.

    Attributes:
        text: The exact characters of the file between the two indices.
    """

    text: str


class ChunkIndex(BaseModel):
    """What index writes to data/processed/: the chunks and their settings.

    Not in the subject. ``max_chunk_size`` is stored so that a later
    command can tell which setting produced the chunks it loads.

    Attributes:
        max_chunk_size: The limit every chunk was cut under.
        chunks: All chunks of the corpus, file by file, in file order.
    """

    max_chunk_size: int
    chunks: List[Chunk]


class UnansweredQuestion(BaseModel):
    """A question without its ground truth, as given to search_dataset."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference sources and expected answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset file: a list of questions, answered or not."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class AnsweredDataset(BaseModel):
    """A ground-truth dataset: every question must carry its sources.

    Not in the subject. RagDataset accepts both variants, so an answered
    question with a broken source silently falls back to UnansweredQuestion
    and loses its sources. evaluate loads the ground truth with this model
    instead, so that such a file is refused rather than miscounted.
    """

    rag_questions: List[AnsweredQuestion]


class MinimalSearchResults(BaseModel):
    """The top-k sources retrieved for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results for one question, plus the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Output of search_dataset: results for every question, and k."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of answer_dataset: results and answers for every question."""

    search_results: List[MinimalAnswer]
    k: int
