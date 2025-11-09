import json

from pathlib import Path
from typing import Any


DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3
DOCUMENT_PREVIEW_LENGTH = 100
PROJECT_ROOT = Path(__file__).parent.parent
MOVIES_DATA_PATH = PROJECT_ROOT / "data" / "movies.json"

BM25_K1 = 1.5
BM25_B = 0.7


def load_movies(data_path) -> list[dict[int, str]]:
    with open(data_path, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stop_words() -> list:
    data_path = PROJECT_ROOT / "data" / "stopwords.txt"
    with open(data_path, "r") as f:
        data = f.read().splitlines()

    return data

def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create a standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of a search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }