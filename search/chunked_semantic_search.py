import json
import re

import numpy as np

from search.semantic_search import SemanticSearch
from search.search_utils import PROJECT_ROOT, load_movies


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None


    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents

        chunks: list[str] = []
        metadata: list[dict] = []

        for document in documents:
            doc_id = document["id"]
            doc_description = document["description"]
            doc_title = document["title"]
            self.document_map[doc_id] = {
                "title": doc_title,
                "description": doc_description
            }

        for document in documents:
            document_description = document["description"]
            if document_description == "":
                continue

            chunked_descriptions = _semantic_chunk_text(document_description,
                                                       chunk_size=4,
                                                       overlap=1)
            total_chunks = len(chunked_descriptions)

            for idx, chunked_description in enumerate(chunked_descriptions):
                chunks.append(chunked_description)

                metadata.append({
                    "movie_idx": document["id"],
                    "chunk_idx": idx,
                    "total_chunks": total_chunks
                })

        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = metadata

        chunk_embeddings_file = PROJECT_ROOT / "cache" / "chunk_embeddings.npy"
        np.save(chunk_embeddings_file, self.chunk_embeddings)

        chunk_metadata_file = PROJECT_ROOT / "cache" / "chunk_metadata.json"
        with open(chunk_metadata_file, 'w') as f:
            json.dump(
                {"chunks": self.chunk_metadata,"total_chunks": len(chunks)},
                f,
                indent=2)

        return self.chunk_embeddings


    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for document in documents:
            doc_id = document["id"]
            doc_description = document["description"]
            doc_title = document["title"]
            self.document_map[doc_id] = {
                "title": doc_title,
                "description": doc_description
            }

        chunk_embeddings_file = PROJECT_ROOT / "cache" / "chunk_embeddings.npy"
        chunk_metadata_file = PROJECT_ROOT / "cache" / "chunk_metadata.json"

        if chunk_embeddings_file.is_file() and chunk_metadata_file.is_file():
            self.chunk_embeddings = np.load(chunk_embeddings_file)
            with open(chunk_metadata_file, 'r') as f:
                self.chunk_metadata = json.load(f)

            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)


def _semantic_chunk_text(text, chunk_size, overlap):

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []

    i = 0
    n_sentences = len(sentences)

    while i < n_sentences:
        chunk_sentences = sentences[i: i + chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += chunk_size - overlap

    return chunks


def embed_chunks_command():
    documents = load_movies()

    chunked_semantic_search = ChunkedSemanticSearch()
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(documents)

    print(f"Generated {len(embeddings)} chunked embeddings")
