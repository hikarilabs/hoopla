import pickle
import math
from collections import defaultdict, Counter

from search.text_processor import process_text
from search.search_utils import PROJECT_ROOT, load_movies


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, str] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.index_path = PROJECT_ROOT / "cache" / "index.pkl"
        self.docmap_path = PROJECT_ROOT / "cache" / "docmap.pkl"
        self.term_frequencies_path = PROJECT_ROOT / "cache" / "term_frequencies.pkl"

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id: int = movie["id"]
            doc_description = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = doc_description
            self.__add_document(doc_id, doc_description)

    def save(self):
        with open(self.index_path, "wb") as i:
            pickle.dump(self.index, i)
        with open(self.docmap_path, "wb") as d:
            pickle.dump(self.docmap, d)
        with open(self.term_frequencies_path, "wb") as tf:
            pickle.dump(self.term_frequencies, tf)

    def load(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.term_frequencies_path, "rb") as tf:
                self.term_frequencies = pickle.load(tf)

        except FileNotFoundError as e:
            print(e)
            raise FileNotFoundError("Index File des not exists")

    def get_documents(self, query) -> list:
        doc_ids = self.index.get(query, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        token = self._process_term(term)

        return self.term_frequencies[doc_id].get(token)

    def get_idf(self, term: str) -> float:
        token = self._process_term(term)

        doc_count = len(self.docmap)
        term_doc_count = len(self.index.get(token))

        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:

        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)

        return tf * idf


    def _process_term(self, term):
        tokens = process_text(term)

        if len(tokens) != 1:
            raise ValueError("term must be a single token")

        token = tokens[0]

        return token

    def __add_document(self, doc_id: int, doc_description: str) -> None:
        tokens = process_text(doc_description)
        for token in set(tokens):
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(tokens)

