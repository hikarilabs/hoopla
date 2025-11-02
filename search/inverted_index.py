import pickle
from collections import defaultdict

from search.text_processor import process_text
from search.search_utils import PROJECT_ROOT, load_movies


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = PROJECT_ROOT / "cache" / "index.pkl"
        self.docmap_path = PROJECT_ROOT / "cache" / "docmap.pkl"

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
            pickle.dump(self.docmap_path, d)

    def get_documents(self, query):
        doc_ids = self.index.get(query, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, doc_description: str) -> None:
        tokens = process_text(doc_description)
        for token in set(tokens):
            self.index[token].add(doc_id)
