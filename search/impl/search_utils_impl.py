import json
import re
import os

from pathlib import Path
from typing import Any
from google import genai
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_ALPHA = 0.5
RRF_K = 60
SEARCH_MULTIPLIER = 5
SCORE_PRECISION = 3
DOCUMENT_PREVIEW_LENGTH = 100
PROJECT_ROOT = Path(__file__).parent.parent
MOVIES_DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
EVAL_DATA_PATH = PROJECT_ROOT / "data" / "golden_dataset.json"

BM25_K1 = 1.5
BM25_B = 0.7


def load_movies(data_path) -> list[dict[int, str]]:
    with open(data_path, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_golden_dataset(data_path) -> list[dict[int, str]]:
    with open(data_path, "r") as f:
        data = json.load(f)
    return data["test_cases"]


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

def _semantic_chunk_text(text, chunk_size, overlap) -> list:

    strip_text = text.strip()
    if len(strip_text) == 0:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", strip_text)
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


def gemini_client(query: str, enhance: str):

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = ""

    match enhance:
        case "spell":
            prompt = f"""Fix any spelling errors in this movie search query.
        
                    Only correct obvious typos. Don't change correctly spelled words.
        
                    Query: "{query}"
        
                    If no errors, return the original query.
                    
                    Corrected:"""
        case "rewrite":
            prompt = f"""Rewrite this movie search query to be more specific and searchable.
    
                    Original: "{query}"
            
                    Consider:
                    - Common movie knowledge (famous actors, popular films)
                    - Genre conventions (horror = scary, animation = cartoon)
                    - Keep it concise (under 10 words)
                    - It should be a google style search query that's very specific
                    - Don't use boolean logic
            
                    Examples:
            
                    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"
        
                    Rewritten query:"""
        case "expand":
            prompt = f"""Expand this movie search query with related terms.
            
                    Add synonyms and related concepts that might appear in movie descriptions.
                    Keep expansions relevant and focused.
                    This will be appended to the original query.
                    
                    Examples:
                    
                    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                    - "action movie with bear" -> "action thriller bear chase fight adventure"
                    - "comedy with bear" -> "comedy funny bear humor lighthearted"
                    
                    Query: "{query}"        
                    """
        case _:
            raise NotImplementedError("The value for query enhancement is not supported. Valid values: spell, rewrite")

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt
    )

    return response.text


def gemini_client_document(query:str, rerank_method: str, document: dict, document_list = None):
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = ""

    match rerank_method:
        case "individual":
            prompt = f"""Rate how well this movie matches the search query.

                    Query: "{query}"
                    Movie: {document.get("title", "")} - {document.get("document", "")}
                    
                    Consider:
                    - Direct relevance to query
                    - User intent (what they're looking for)
                    - Content appropriateness
                    
                    Rate 0-10 (10 = perfect match).
                    Give me ONLY the number in your response, no other text or explanation.
                    
                    Score:"""
        case "batch":
            prompt = f"""Rank these movies by relevance to the search query.
            
                    Query: "{query}"
                
                    Movies:
                    {document_list}
                
                    Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:
                
                    [75, 12, 34, 2, 1]
                    """
        case _:
            raise NotImplementedError("The value for rerank method is not supported. Valid values: individual")


    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt
    )

    return response.text