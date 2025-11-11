import os

from search.keyword_search import InvertedIndex
from search.chunked_semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_command(numbers: list):

    norms = []
    if len(numbers) == 0:
        return norms

    len_numbers = len(numbers)

    min_score = min(numbers)
    max_score = max(numbers)

    if min_score == max_score:
        return [1.0] * len_numbers

    for i in range(len_numbers):
        score = (numbers[i] - min_score) / (max_score - min_score)
        norms.append(score)

    return norms