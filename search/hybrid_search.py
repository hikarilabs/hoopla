import os

from search.keyword_search import InvertedIndex
from search.chunked_semantic_search import ChunkedSemanticSearch
from search.search_utils import DEFAULT_SEARCH_LIMIT, DEFAULT_ALPHA, MOVIES_DATA_PATH, load_movies, gemini_client


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5) -> list[dict]:

        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        results = combine_bm25_semantic_search(bm25_results, semantic_results, alpha)

        return results[:limit]

    def rrf_search(self, query, k, limit=10) -> list[dict]:

        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        results = combine_rrf_search(bm25_results, semantic_results, limit)

        return results[:limit]


def normalize_scores(scores: list[float]) -> list[float]:

    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    norm_scores = []

    for score in scores:
        norm_scores.append(
            (score - min_score) / (max_score - min_score)
        )

    return norm_scores


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores = []
    for result in results:
        scores.append(result["score"])

    normalised = normalize_scores(scores)

    for idx, result in enumerate(results):
        result["normalized_score"] = normalised[idx]

    return results

def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k = 60) -> float:
    return 1 / (k + rank)


def combine_bm25_semantic_search(bm25_results: list[dict], semantic_results: list[dict], alpha: float) -> list[dict]:

    normalized_bm25 = normalize_search_results(bm25_results)
    normalised_semantic = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in normalized_bm25:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    for result in normalised_semantic:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = {
            "id": doc_id,
            "title": data["title"],
            "document": data["document"],
            "score": score_value,
            "bm25_score": data["bm25_score"],
            "semantic_score": data["semantic_score"]
        }
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def combine_rrf_search(bm25_results: list[dict], semantic_results: list[dict], alpha: float) -> list[dict]:
    combined_scores = {}

    for idx, result in enumerate(bm25_results):
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_rank": idx,
                "score": rrf_score(idx)
            }

    for idx, result in enumerate(semantic_results):
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "semantic_rank": idx,
                "score": rrf_score(idx)
            }
        else:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_rank": combined_scores[doc_id]["bm25_rank"],
                "semantic_rank": idx,
                "score": rrf_score(combined_scores[doc_id]["bm25_rank"]) + rrf_score(idx)
            }

    rrf_results = []

    for doc_id, data in combined_scores.items():
        result = {
            "id": doc_id,
            "title": data["title"],
            "document": data["document"],
            "score": data["score"],
            "bm25_rank": data.get("bm25_rank", None),
            "semantic_rank": data.get("bm25_rank", None)
        }
        rrf_results.append(result)

    return sorted(rrf_results, key=lambda x: x["score"], reverse=True)



def normalize_command(scores: list):
    results = normalize_scores(scores)

    return results


def weighted_search_command(query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:

    movies = load_movies(MOVIES_DATA_PATH)

    search = HybridSearch(movies)

    results = search.weighted_search(query, alpha, limit)

    return {
        "original_query": query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }


def rrf_search_command(query: str, enhance: str, k: int, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:

    movies = load_movies(MOVIES_DATA_PATH)

    search = HybridSearch(movies)

    if enhance is not None:
        enhanced_query = gemini_client(query, enhance)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")

        results = search.rrf_search(enhanced_query, k, limit)

    else:
        results = search.rrf_search(query, k, limit)

    return {
        "original_query": query,
        "query": query,
        "results": results,
    }