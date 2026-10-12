"""Rank the chunks of an index against a query with BM25."""

from typing import List

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.models import ChunkIndex, ScoredChunk
from src.tokenizer import tokenize

# rank_bm25 defaults, the usual starting point in the literature. Exposed
# so that stage 6 can sweep them.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


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
        terms = tokenize(query)
        if k <= 0 or not terms:
            return []
        scores = np.asarray(self.bm25.get_scores(terms), dtype=float)
        # Stable sort on -score keeps index order among equal scores.
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            ScoredChunk(chunk=self.chunks[i], score=float(scores[i]))
            for i in order
            if scores[i] > 0
        ]
