from typing import List

from search.inverted_index import InvertedIndex
from search.search_utils import DEFAULT_SEARCH_LIMIT
from search.text_processor import process_text, has_matching_token


def keyword_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> List:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the 'build' command first to create the search index.")
        return []

    index_tokens = idx.index

    if index_tokens is None or len(index_tokens) == 0:
        print("Index is empty. Please run the 'build' command first.")
        return []

    query_results = []
    query_tokens = process_text(query)

    for token in query_tokens:

        doc_ids = idx.get_documents(token)
        if doc_ids:
            for doc_id in doc_ids:
                query_results.append((doc_id, idx.docmap[doc_id]))

            if len(query_results) >= limit:
                break

    return query_results


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
