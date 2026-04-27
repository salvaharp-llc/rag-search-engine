from google import genai

from .gen_utils import API_KEY, GEN_MODEL
from .search_utils import load_movies
from .hybrid_search import HybridSearch

def rag_command(query: str) -> dict:
    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.rrf_search(query)

    if not results:
        return {
            "search_results": [],
            "error": "No results found",
        }

    formatted_results = []
    for result in results:
        formatted_results.append(f"{result['title']}: {result['description']}")

    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {'\n\n'.join(formatted_results)}

    Answer:"""

    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    answer = (response.text or "").strip().strip('"')

    return {
        "search_results": results,
        "answer": answer,
    }
