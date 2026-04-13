import numpy as np
import os
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, load_movies

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents: list[dict] = None
        self.document_map: dict[int, dict] = {}
        self.embeddings = None
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("text can not be an empty string")
        return self.model.encode([text])[0]
    
    def build_embeddings(self):
        documents_strings: list[str] = []
        for doc in self.documents:
            documents_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(documents_strings, show_progress_bar=True)

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings 
    
    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
            
        return self.build_embeddings()


def verify_model() -> None:
        ss = SemanticSearch()
        print(f"Model loaded: {ss.model}")
        print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    ss = SemanticSearch()
    movies = load_movies()
    embeddings = ss.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
