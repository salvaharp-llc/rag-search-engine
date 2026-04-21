from .hybrid_search import HybridSearch

from .search_utils import (
    load_movies, 
    load_test_cases,
    DEFAULT_EVALUATION_LIMIT,
)


def evaluate_command(limit: int = DEFAULT_EVALUATION_LIMIT) -> None:
    movies = load_movies()
    test_cases = load_test_cases()

    hs = HybridSearch(movies)

    for test_case in test_cases:
        query: str = test_case["query"]
        relevant_docs: set[str] = set(test_case["relevant_docs"])
        search_results = hs.rrf_search(query, limit=limit)

        retrieved_docs = []
        for result in search_results:
            title = result.get("title", "")
            if title:
                retrieved_docs.append(title)

        relevant_count = 0
        for doc in retrieved_docs:
            if doc in relevant_docs:
                relevant_count += 1
        precision = relevant_count / limit
        recall = relevant_count / len(relevant_docs)

        print(f"- Query: {query}")
        print(f"    - Precision@{limit}: {precision:.4f}")
        print(f"    - Recall@{limit}: {recall:.4f}")
        print(f"    - Retreived: {', '.join(retrieved_docs)}")
        print(f"    - Relevant: {', '.join(relevant_docs)}")
        print()