"""Rank the chunks of an index against a query with BM25."""

from collections import Counter
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.models import (
    ChunkIndex, ExplainedChunk, ExplainedSearch, ScoredChunk,
)
from src.tokenizer import tokenize

# k1 is the rank_bm25 default. b was swept in stage 6 (2026-10-13): 0.75
# -> 0.6 -> 0.5 -> 0.3 gave code recall@5 71.7 / 73.7 / 75.8 / 73.7 with
# docs unchanged at 85.0. Less length normalisation helps because the
# reference is often a long chunk (a whole function or section) that lost
# to a short chunk of the same file.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.5


class Retriever:
    """BM25 scorer over the chunks of a ``ChunkIndex``.

    Tokenising the whole corpus happens once, in the constructor, so one
    instance serves every question of a dataset.
    """

    def __init__(
        self,
        index: ChunkIndex,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        """Tokenise every chunk and build the BM25 statistics.

        Args:
            index: The chunks written by the ``index`` command.
            k1: Term-frequency saturation parameter.
            b: Length normalisation parameter, between 0 and 1.
        """
        self.chunks = index.chunks
        corpus = [
            tokenize(chunk.text)
            for chunk in tqdm(self.chunks, desc="Tokenising", unit="chunk")
        ]
        self.bm25 = BM25Okapi(corpus, k1=k1, b=b)

    def _top(self, terms: List[str], k: int) -> List[Tuple[int, float]]:
        """Return ``(chunk position, score)`` of the k best chunks.

        Shared by ``search`` and ``explain`` so both rank the same way.
        """
        if k <= 0 or not terms:
            return []
        scores = np.asarray(self.bm25.get_scores(terms), dtype=float)
        # Stable sort on -score keeps index order among equal scores.
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def search(self, query: str, k: int) -> List[ScoredChunk]:
        """Return the ``k`` best chunks for ``query``, best first.

        Only chunks with a positive score are returned: a chunk that shares
        no term with the query is not a result, so a query made of unknown
        words returns nothing rather than arbitrary chunks.

        Args:
            query: The question, as typed by the user.
            k: Maximum number of chunks to return.

        Returns:
            At most ``k`` scored chunks, ties broken by index order so
            that the output is deterministic.
        """
        return [
            ScoredChunk(chunk=self.chunks[i], score=score)
            for i, score in self._top(tokenize(query), k)
        ]

    def explain(self, query: str, k: int) -> ExplainedSearch:
        """Like ``search``, with each score split per query term.

        BM25 is a sum over the query terms, so scoring each distinct term
        alone gives its share, times how often it occurs in the query.
        Costs one pass over the corpus per term: fine for inspection, not
        used by the dataset commands.
        """
        terms = tokenize(query)
        counts = Counter(terms)
        per_term = {
            term: np.asarray(self.bm25.get_scores([term]), dtype=float)
            for term in counts
        }
        results: List[ExplainedChunk] = []
        for i, score in self._top(terms, k):
            shares = {
                term: round(float(count * per_term[term][i]), 3)
                for term, count in counts.items()
                if per_term[term][i] > 0
            }
            results.append(ExplainedChunk(
                chunk=self.chunks[i],
                score=score,
                contributions=dict(
                    sorted(shares.items(), key=lambda item: -item[1])
                ),
            ))
        return ExplainedSearch(query=query, terms=terms, results=results)
