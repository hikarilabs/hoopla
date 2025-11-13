import pickle
import math
from collections import defaultdict, Counter


from search.text_processor import process_text
from search.search_utils import PROJECT_ROOT, BM25_K1, BM25_B, load_movies, DEFAULT_SEARCH_LIMIT, format_search_result, \
    SCORE_PRECISION, MOVIES_DATA_PATH


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}
        self.index_path = PROJECT_ROOT / "cache" / "index.pkl"
        self.docmap_path = PROJECT_ROOT / "cache" / "docmap.pkl"
        self.term_frequencies_path = PROJECT_ROOT / "cache" / "term_frequencies.pkl"
        self.doc_lengths_path = PROJECT_ROOT / "cache" / "doc_lengths.pkl"

    def build(self):
        movies = load_movies(MOVIES_DATA_PATH)
        for movie in movies:
            doc_id = movie["id"]
            doc_description = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_description)

    def save(self):
        with open(self.index_path, "wb") as i:
            pickle.dump(self.index, i)
        with open(self.docmap_path, "wb") as d:
            pickle.dump(self.docmap, d)
        with open(self.term_frequencies_path, "wb") as tf:
            pickle.dump(self.term_frequencies, tf)
        with open(self.doc_lengths_path, "wb") as dlp:
            pickle.dump(self.doc_lengths, dlp)

    def load(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.term_frequencies_path, "rb") as tf:
                self.term_frequencies = pickle.load(tf)
            with open(self.doc_lengths_path, "rb") as dlp:
                self.doc_lengths = pickle.load(dlp)

        except FileNotFoundError as e:
            print(e)
            raise FileNotFoundError("Index File des not exists")

    def get_documents(self, query) -> list:
        doc_ids = self.index.get(query, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        token = self._tokenize_term(term)

        return self.term_frequencies[doc_id].get(token, 0)

    def get_idf(self, term: str) -> float:
        token = self._tokenize_term(term)

        doc_count = len(self.docmap)
        term_doc_count = len(self.index.get(token, []))

        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:

        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)

        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        token = self._tokenize_term(term)

        doc_count = len(self.docmap)
        term_doc_count = len(self.index.get(token, []))

        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:

        tf = self.get_tf(doc_id, term)

        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()

        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        tf_component = self.get_bm25_tf(doc_id, term)
        idf_component = self.get_bm25_idf(term)
        return tf_component * idf_component


    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        query_tokens = process_text(query)

        scores = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = {
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"],
                "score": round(score, SCORE_PRECISION)
            }
            results.append(formatted_result)

        return results

    def _tokenize_term(self, term):
        tokens = process_text(term)

        if len(tokens) != 1:
            raise ValueError("term must be a single token")

        token = tokens[0]

        return token

    def __add_document(self, doc_id: int, doc_description: str) -> None:
        tokens: list = process_text(doc_description)
        for token in set(tokens):
            self.index[token].add(doc_id)

        self.doc_lengths[doc_id] = len(tokens)
        self.term_frequencies[doc_id].update(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        total_length = 0
        for length in self.doc_lengths.values():
            total_length += length
        return total_length / len(self.doc_lengths)
