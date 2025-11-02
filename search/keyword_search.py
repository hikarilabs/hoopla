from typing import List

from search.search_utils import DEFAULT_SEARCH_LIMIT, load_movies
from search.text_processor import (
    text_lowercase,
    text_remove_punctuation,
    text_tokenize,
    has_matching_token,
    text_remove_stop_words,
    text_stem
)


def keyword_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> List:
    movies = load_movies()
    results = []

    for movie in movies:
        # process lower case
        query_lower = text_lowercase(query)
        movie_title_lower = text_lowercase(movie["title"])

        # process - remove text punctuation
        query_tokens = text_remove_punctuation(query_lower)
        movie_title_tokens = text_remove_punctuation(movie_title_lower)

        # process - tokenize text
        query_tokens = text_tokenize(query_tokens)
        movie_title_tokens = text_tokenize(movie_title_tokens)

        # process - remove stop words
        query_tokens = text_remove_stop_words(query_tokens)
        movie_title_tokens = text_remove_stop_words(movie_title_tokens)

        # process - stemming
        query_tokens = text_stem(query_tokens)
        movie_title_tokens = text_stem(movie_title_tokens)

        if has_matching_token(query_tokens, movie_title_tokens):
            results.append(movie)

            if len(results) >= limit:
                break

    return results
