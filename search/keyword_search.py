from typing import List

from search.inverted_index import InvertedIndex
from search.search_utils import DEFAULT_SEARCH_LIMIT, load_movies
from search.text_processor import process_text, has_matching_token


def keyword_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> List:
    movies = load_movies()
    results = []

    for movie in movies:
        title = movie["title"]

        query_tokens = process_text(query)
        movie_title_tokens = process_text(title)

        if has_matching_token(query_tokens, movie_title_tokens):
            results.append(movie)

            if len(results) >= limit:
                break

    return results


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")
