from search.inverted_index import InvertedIndex
from search.impl.search_utils_impl import DEFAULT_SEARCH_LIMIT, BM25_K1
from search.text_processor import process_text


def keyword_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list:
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


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


def idf_calculator_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    return idx.get_idf(term)

def tf_idf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    return idx.get_tf_idf(doc_id, term)

def bm25idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    return idx.get_bm25_idf(term)

def bm25tf_command(doc_id: int, term: str, k1 = BM25_K1) -> float:
    idx = InvertedIndex()
    idx.load()

    return idx.get_bm25_tf(doc_id, term, k1)

def bm25search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)