"""Tests for src/evaluation.py."""

from src.evaluation import evaluate_results, iou, question_recall
from src.models import AnsweredQuestion, MinimalSearchResults, MinimalSource

FILE = "data/raw/repo/a.md"


def span(first: int, last: int, path: str = FILE) -> MinimalSource:
    return MinimalSource(
        file_path=path, first_character_index=first, last_character_index=last
    )


def answered(question_id: str, *sources: MinimalSource) -> AnsweredQuestion:
    return AnsweredQuestion(
        question_id=question_id, question="?", sources=list(sources),
        answer="",
    )


def retrieved(
    question_id: str, *sources: MinimalSource
) -> MinimalSearchResults:
    return MinimalSearchResults(
        question_id=question_id, question="?", retrieved_sources=list(sources)
    )


def test_iou_identical_spans() -> None:
    assert iou(span(0, 100), span(0, 100)) == 1.0


def test_iou_disjoint_spans() -> None:
    assert iou(span(0, 100), span(200, 300)) == 0.0


def test_iou_touching_spans() -> None:
    # last is exclusive: [0:100] ends before [100:200] starts.
    assert iou(span(0, 100), span(100, 200)) == 0.0


def test_iou_containing_span_is_symmetric() -> None:
    # A 100-char reference inside a 2000-char chunk: 100 / 2000.
    assert iou(span(500, 600), span(0, 2000)) == 0.05
    assert iou(span(0, 2000), span(500, 600)) == 0.05


def test_iou_partial_overlap() -> None:
    # LoRA reference [4695:6098] vs a chunk [5000:7000]:
    # overlap 1098, union 2305.
    expected = 1098 / 2305
    assert abs(iou(span(4695, 6098), span(5000, 7000)) - expected) < 1e-9


def test_iou_different_files_is_zero() -> None:
    other = span(0, 100, "data/raw/repo/b.md")
    assert iou(span(0, 100), other) == 0.0


def test_threshold_is_strict() -> None:
    # IoU exactly 0.05 is not enough for the grader; 101/2000 is.
    assert question_recall([span(500, 600)], [span(0, 2000)]) == 0.0
    assert question_recall([span(500, 601)], [span(0, 2000)]) == 1.0


def test_recall_at_k_only_looks_at_the_first_k() -> None:
    truth = [answered("q1", span(0, 100))]
    results = [retrieved(
        "q1", span(500, 600), span(600, 700), span(0, 100)
    )]
    report = evaluate_results(truth, results, ks=(1, 3, 5))
    assert report.evaluated == 1
    assert report.recall == {1: 0.0, 3: 1.0, 5: 1.0}


def test_recall_is_averaged_over_the_ground_truth() -> None:
    truth = [answered(f"q{i}", span(0, 100)) for i in range(4)]
    results = [
        retrieved("q0", span(0, 100)),      # found
        retrieved("q1", span(900, 999)),    # wrong span
        retrieved("q2"),                    # no result at all
    ]                                        # q3 missing from the file
    report = evaluate_results(truth, results, ks=(1,))
    assert report.evaluated == 4
    assert report.recall == {1: 0.25}
    assert report.missing_ids == ["q3"]


def test_unknown_ids_are_ignored_and_reported() -> None:
    truth = [answered("q1", span(0, 100))]
    results = [retrieved("q1", span(0, 100)), retrieved("zz", span(0, 100))]
    report = evaluate_results(truth, results, ks=(1,))
    assert report.recall == {1: 1.0}
    assert report.unknown_ids == ["zz"]


def test_several_expected_sources_give_a_fraction() -> None:
    truth = [answered("q1", span(0, 100), span(5000, 5100))]
    results = [retrieved("q1", span(0, 100))]
    report = evaluate_results(truth, results, ks=(1,))
    assert report.recall == {1: 0.5}


def test_empty_inputs_give_zero_without_crashing() -> None:
    report = evaluate_results([], [], ks=(1, 5))
    assert report.evaluated == 0
    assert report.recall == {1: 0.0, 5: 0.0}
