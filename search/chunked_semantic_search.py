import json

import numpy as np

from search.impl.semantic_search_impl import SemanticSearch, cosine_similarity
from search.impl.search_utils_impl import PROJECT_ROOT, MOVIES_DATA_PATH, load_movies, SCORE_PRECISION, \
    DEFAULT_SEARCH_LIMIT, _semantic_chunk_text


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None


    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents

        self.document_map = {}

        for document in documents:
            self.document_map[document["id"]] = document

        chunks: list[str] = []
        metadata: list[dict] = []

        for idx, document in enumerate(documents):
            document_description = document.get("description", "")
            if not document_description.strip():
                continue

            chunked_descriptions = _semantic_chunk_text(document_description,
                                                       chunk_size=4,
                                                       overlap=1)
            total_chunks = len(chunked_descriptions)

            for chunk_id, chunked_description in enumerate(chunked_descriptions):
                chunks.append(chunked_description)

                metadata.append({
                    "movie_idx": idx,
                    "chunk_idx": chunk_id,
                    "total_chunks": total_chunks
                })

        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = metadata

        chunk_embeddings_file = PROJECT_ROOT / "cache" / "chunk_embeddings.npy"
        np.save(chunk_embeddings_file, self.chunk_embeddings)

        chunk_metadata_file = PROJECT_ROOT / "cache" / "chunk_metadata.json"
        with open(chunk_metadata_file, 'w') as f:
            json.dump(
                {"chunks": metadata,
                 "total_chunks": len(chunks)},
                f,
                indent=2)

        return self.chunk_embeddings


    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        self.document_map = {}

        for document in documents:
            self.document_map[document["id"]] = document

        chunk_embeddings_file = PROJECT_ROOT / "cache" / "chunk_embeddings.npy"
        chunk_metadata_file = PROJECT_ROOT / "cache" / "chunk_metadata.json"

        if chunk_embeddings_file.is_file() and chunk_metadata_file.is_file():
            self.chunk_embeddings = np.load(chunk_embeddings_file)
            with open(chunk_metadata_file, 'r') as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]

            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:

        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("No chunk embeddings loaded. Run load or create chunk embeddings first")

        query_embed = self.generate_embedding(query)

        chunk_scores: list[dict] = []

        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embed, chunk_embedding)
            chunk_scores.append({
                "chunk_idx": i,
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": similarity
            })

        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (
                movie_idx not in movie_scores or
                chunk_score["score"] > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = chunk_score["score"]

        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for movie_idx, score in sorted_movies[:limit]:
            doc = self.documents[movie_idx]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "document": doc["description"],
                    "score": round(score, SCORE_PRECISION)
                }
            )

        return results


def embed_chunks_command():
    documents = load_movies(MOVIES_DATA_PATH)

    chunked_semantic_search = ChunkedSemanticSearch()

    return chunked_semantic_search.load_or_create_chunk_embeddings(documents)


def search_chunked_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    documents = load_movies(MOVIES_DATA_PATH)

    chunked_semantic_search = ChunkedSemanticSearch()
    chunked_semantic_search.load_or_create_chunk_embeddings(documents)

    results = chunked_semantic_search.search_chunks(query, limit)

    return {
        "query": query,
        "results": results
    }
