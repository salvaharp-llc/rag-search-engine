import os

from .search_utils import (
    load_movies,
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
)

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        keyword_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        norm_keyword_results = normalize_search_results(keyword_results)
        norm_semantic_results = normalize_search_results(semantic_results)

        results: dict[int, dict] = {}
        for result in norm_keyword_results:
            doc_id = result["id"]
            results[doc_id] = {
                "id": doc_id,
                "title": result["title"],
                "description": result["description"],
                "bm25_score": result["normalized_score"],
                "semantic_score": 0.0,
            }

        for result in norm_semantic_results:
            doc_id = result["id"]
            if doc_id in results:
                results[doc_id]["semantic_score"] = result["normalized_score"]
            else:
                results[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "description": result["description"],
                    "bm25_score": 0.0,
                    "semantic_score": result["normalized_score"],
                }
        
        for doc_id, doc in results.items():
            results[doc_id]["score"] = hybrid_score(doc["bm25_score"], doc["semantic_score"], alpha)

        return sorted(results.values(), key=lambda x: x["score"], reverse=True)[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
def normalize_scores(scores: list[float]) -> list[float]:
    n_scores = len(scores)
    if n_scores == 0:
        return []
    
    max_score, min_score = max(scores), min(scores)
    score_range = max_score - min_score

    normalized_scores = [1.0] * n_scores
    if score_range == 0:
        return normalized_scores
    
    for i in range(n_scores):
        normalized_scores[i] = (scores[i] - min_score) / score_range

    return normalized_scores

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results
    
def normalize_command(scores: list[float]) -> None:
    normalized_scores = normalize_scores(scores)
    for score in normalized_scores:
        print(f"* {score:.4f}")

def weighted_search_command(query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.weighted_search(query, alpha, limit)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result["title"]}")
        print(f"   Hybrid Score: {result["score"]:.3f}")
        print(f"   BM25: {result["bm25_score"]:.3f}, Semantic: {result["semantic_score"]:.3f}")
        print(f"   {result["description"]}...")


