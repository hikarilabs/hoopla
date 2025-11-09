import re
from typing import Any

import numpy as np
from numpy import ndarray, dtype

from sentence_transformers import SentenceTransformer

from search.search_utils import PROJECT_ROOT, load_movies, _semantic_chunk_text


class SemanticSearch:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def search(self, query: str, limit: int):

        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        results = []

        for idx, embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, embedding)
            results.append(
                (similarity, self.documents[idx])
            )

        results.sort(key=lambda x: x[0], reverse=True)

        return results[: limit]

    def build_embeddings(self, documents: list[dict]) -> ndarray[tuple[Any, ...], dtype[Any]]:
        self.documents = documents

        movies = []
        for document in documents:
            doc_id = document["id"]
            doc_description = document["description"]
            doc_title = document["title"]
            self.document_map[doc_id] = {
                "title": doc_title,
                "description": doc_description
            }

            movie = f"{document['title']}: {document['description']}"
            movies.append(movie)

        movies_embeddings_file = PROJECT_ROOT / "cache" / "movie_embeddings.npy"

        movies_embeddings = self.model.encode(movies)
        np.save(movies_embeddings_file, movies_embeddings)

        self.embeddings = movies_embeddings

        return self.embeddings


    def load_or_create_embeddings(self, documents: list[dict]) -> ndarray[tuple[Any, ...], dtype[Any]] | None | Any:
        self.documents = documents

        movie_embeddings_file = PROJECT_ROOT / "cache" / "movie_embeddings.npy"

        if movie_embeddings_file.is_file():
            self.embeddings = np.load(movie_embeddings_file)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        else:
            return self.build_embeddings(documents)


    def generate_embedding(self, text):
        """
        Generate a text embedding using the initialised model
        :param text:
        :return:
        """
        if text == "" or text == " ":
            raise ValueError("The text should not be an empty string")

        embedding = self.model.encode(text)

        return embedding

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def verify_model():
    semantic_search = SemanticSearch('all-MiniLM-L6-v2')

    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embedd_text(text):
    semantic_search = SemanticSearch('all-MiniLM-L6-v2')

    embedding = semantic_search.generate_embedding(text)
    print(embedding)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_search = SemanticSearch('all-MiniLM-L6-v2')

    movies = load_movies()

    semantic_search.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in "
          f"{semantic_search.embeddings.shape[1]} dimensions")

def embed_query_text(query):
    semantic_search = SemanticSearch('all-MiniLM-L6-v2')

    query_embed = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {query_embed[:5]}")
    print(f"Shape: {query_embed.shape}")


def search_query(query: str, limit: int) -> None:
    semantic_search = SemanticSearch('all-MiniLM-L6-v2')

    movies = load_movies()

    semantic_search.load_or_create_embeddings(movies)

    results = semantic_search.search(query, limit)

    for idx, result in enumerate(results, 1):
        print(f"{idx}. {result[1]["title"]} (score: {result[0]:.4f})")
        print(f"{result[1]["description"]}\n")


def _chunk_text(text_split, chunk_size, overlap):
    chunks = []

    if overlap == 0:
        for i in range(0, len(text_split), chunk_size):
            chunks.append(text_split[i: i + chunk_size])
    else:
        for i in range(0, len(text_split), chunk_size):
            if i == 0:
                chunks.append(text_split[i: i + chunk_size])
            else:
                chunks.append(text_split[i - overlap: i + chunk_size])

    return chunks

def chunk_text_command(text: str, chunk_size: int, overlap: int) -> None:

    text_split = text.split(" ")
    total_characters = len(text)

    chunks = _chunk_text(text_split, chunk_size, overlap)

    print(f"Chunking {total_characters} characters")
    for idx, chunk in enumerate(chunks, 1):
        print(f"{idx}. {" ".join(chunk)}")


def semantic_chunk_command(text: str, chunk_size: int, overlap: int) -> None:

    total_characters = len(text)

    chunks = _semantic_chunk_text(text, chunk_size, overlap)

    print(f" Semantically chunking {total_characters} characters")
    for idx, chunk in enumerate(chunks, 1):
        print(f"{idx}. {chunk}")