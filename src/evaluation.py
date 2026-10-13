"""Recall@k of a search results file, with the grader's matching rule.

A reference source counts as found when one of the retrieved sources is
in the same file and overlaps it with IoU above a threshold. Written here
rather than imported from the moulinette: this is the measurement loop
of stage 6, and it must be able to explain a miss, not only count it.
"""

from typing import Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel

from src.models import AnsweredQuestion, MinimalSearchResults, MinimalSource

# The grader counts a source as found strictly above this overlap.
DEFAULT_IOU_THRESHOLD = 0.05
# The cut-offs the grader reports.
DEFAULT_KS = (1, 3, 5, 10)

MissKind = Literal["late", "near", "same_file", "wrong_file"]

# What each kind of miss points at, for the stage 6 error analysis.
MISS_HINTS: Dict[str, str] = {
    "wrong_file": "no result in the expected file: vocabulary or ranking",
    "same_file": "results in the file, none overlapping: chunk boundaries",
    "near": "overlap at or below the threshold: chunk too big",
    "late": "found after rank k: ranking",
}


class Miss(BaseModel):
    """One expected source not found at k, with the closest result."""

    question_id: str
    question: str
    expected: MinimalSource
    kind: MissKind
    rank: Optional[int]
    best_iou: float


class RecallReport(BaseModel):
    """Recall@k over a dataset, plus the matching problems met on the way."""

    evaluated: int
    recall: Dict[int, float]
    missing_ids: List[str]
    unknown_ids: List[str]


def iou(a: MinimalSource, b: MinimalSource) -> float:
    """Intersection over union of two character spans.

    Spans in different files never overlap. ``last_character_index`` is
    exclusive, so touching spans ([0:10] and [10:20]) do not intersect.

    Returns:
        A value in [0, 1]: 0 when disjoint, 1 when identical.
    """
    if a.file_path != b.file_path:
        return 0.0
    overlap = (
        min(a.last_character_index, b.last_character_index)
        - max(a.first_character_index, b.first_character_index)
    )
    if overlap <= 0:
        return 0.0
    length_a = a.last_character_index - a.first_character_index
    length_b = b.last_character_index - b.first_character_index
    return overlap / (length_a + length_b - overlap)


def is_found(
    source: MinimalSource,
    retrieved: Sequence[MinimalSource],
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> bool:
    """Whether one of ``retrieved`` overlaps ``source`` enough to count."""
    return any(iou(source, hit) > threshold for hit in retrieved)


def question_recall(
    expected: Sequence[MinimalSource],
    retrieved: Sequence[MinimalSource],
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> float:
    """Share of the expected sources found among ``retrieved``.

    With one expected source per question, as in the public datasets,
    this is 0 or 1.
    """
    if not expected:
        return 0.0
    found = sum(is_found(source, retrieved, threshold) for source in expected)
    return found / len(expected)


def evaluate_results(
    truth: Sequence[AnsweredQuestion],
    results: Sequence[MinimalSearchResults],
    ks: Sequence[int] = DEFAULT_KS,
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> RecallReport:
    """Average the per-question recall over the dataset, for each k.

    Questions are matched by ``question_id``. A ground-truth question
    with no result counts as a miss; a result with no ground-truth
    question is ignored. Both are reported, not raised: a partial file
    still gives an honest number.
    """
    retrieved = {
        result.question_id: result.retrieved_sources for result in results
    }
    truth_ids = {question.question_id for question in truth}
    totals = {k: 0.0 for k in ks}
    evaluated = 0
    for question in truth:
        if not question.sources:
            continue
        evaluated += 1
        hits = retrieved.get(question.question_id, [])
        for k in ks:
            totals[k] += question_recall(question.sources, hits[:k], threshold)
    return RecallReport(
        evaluated=evaluated,
        recall={k: totals[k] / evaluated if evaluated else 0.0 for k in ks},
        missing_ids=[
            question.question_id for question in truth
            if question.question_id not in retrieved
        ],
        unknown_ids=sorted(set(retrieved) - truth_ids),
    )


def diagnose(
    question: AnsweredQuestion,
    retrieved: Sequence[MinimalSource],
    k: int,
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> List[Miss]:
    """Explain why each expected source of ``question`` is missed at k.

    Looks at the whole result list, not only the first k, to tell a
    ranking problem (found later) from a chunking or vocabulary one.
    """
    misses: List[Miss] = []
    for expected in question.sources:
        if is_found(expected, retrieved[:k], threshold):
            continue
        in_file = [
            (rank, iou(expected, hit))
            for rank, hit in enumerate(retrieved, start=1)
            if hit.file_path == expected.file_path
        ]
        late = [item for item in in_file if item[1] > threshold]
        kind: MissKind
        rank: Optional[int]
        if late:
            kind, (rank, best) = "late", late[0]
        elif in_file:
            rank, best = max(in_file, key=lambda item: item[1])
            kind = "near" if best > 0 else "same_file"
        else:
            kind, rank, best = "wrong_file", None, 0.0
        misses.append(Miss(
            question_id=question.question_id, question=question.question,
            expected=expected, kind=kind, rank=rank, best_iou=best,
        ))
    return misses


def list_misses(
    truth: Sequence[AnsweredQuestion],
    results: Sequence[MinimalSearchResults],
    k: int,
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> List[Miss]:
    """Every expected source of ``truth`` missed at k, in dataset order."""
    retrieved = {
        result.question_id: result.retrieved_sources for result in results
    }
    misses: List[Miss] = []
    for question in truth:
        hits = retrieved.get(question.question_id, [])
        misses.extend(diagnose(question, hits, k, threshold))
    return misses
