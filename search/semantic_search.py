from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text):
        if text == "" or text == " ":
            raise ValueError("The text should not be an empty string")

        embedding = self.model.encode(text)

        return embedding




def verify_model():
    semantic_search = SemanticSearch()

    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embedd_text(text):
    semantic_search = SemanticSearch()

    embedding = semantic_search.generate_embedding(text)
    print(embedding)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")