import json
from typing import List
from pathlib import Path

from search.utils import remove_punctuation, tokenize_text

DEFAULT_SEARCH_LIMIT = 5
PROJECT_ROOT = Path(__file__).parent.parent

def load_movies() -> List[dict]:
    data_path = PROJECT_ROOT / "data" / "movies.json"
    with open(data_path, "r") as f:
        data = json.load(f)

    return data["movies"]

def keyword_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> List:

    movies = load_movies()
    results = []

    for movie in movies:

        # process - remove text punctuation
        query_punctuation = remove_punctuation(query)
        movie_title_punctuation = remove_punctuation(movie["title"])

        # process - tokenize text
        query_tokens = tokenize_text(query_punctuation)
        movie_title_tokens = tokenize_text(movie_title_punctuation)

        for query_token in query_tokens:
            for movie_token in movie_title_tokens:

                if movie_token.find(query_token) != -1:
                    results.append(movie)

                    if len(results) >= limit:
                        break

    return results