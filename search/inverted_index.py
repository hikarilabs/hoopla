import pickle
from collections import defaultdict, Counter
from typing import List

from search.text_processor import process_text
from search.search_utils import PROJECT_ROOT, load_movies


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, str] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.index_path = PROJECT_ROOT / "cache" / "index.pkl"
        self.docmap_path = PROJECT_ROOT / "cache" / "docmap.pkl"
        self.term_frequencies_path = PROJECT_ROOT / "cache" / "term_frequencies.pkl"

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            doc_description = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = doc_description
            self.__add_document(doc_id, doc_description)

    def save(self):
        with open(self.index_path, "wb") as i:
            pickle.dump(self.index, i)
        with open(self.docmap_path, "wb") as d:
            pickle.dump(self.docmap, d)

    def load(self):
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)

        except FileNotFoundError as e:
            print(e)
            raise FileNotFoundError("Index File des not exists")

    def get_documents(self, query) -> List:
        doc_ids = self.index.get(query, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, doc_description: str) -> None:
        tokens = process_text(doc_description)
        for token in set(tokens):
            self.index[token].add(doc_id)
