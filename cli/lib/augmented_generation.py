from google import genai

from .gen_utils import API_KEY, GEN_MODEL
from .search_utils import load_movies, DEFAULT_SEARCH_LIMIT
from .hybrid_search import HybridSearch


def rag_command(query: str) -> dict:
    return generate_answer(query, "search")

def summarize_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    return generate_answer(query, "summary", limit=limit)

def format_search_results(results: list[dict]) -> str:
    formatted_results = []
    for result in results:
        formatted_results.append(f"{result['title']}: {result['description']}")
    return '\n\n'.join(formatted_results)

def generate_answer(query: str, answer_type: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    hs = HybridSearch(movies)
    results = hs.rrf_search(query, limit=limit)

    if not results:
        return {
            "search_results": [],
            "error": "No results found",
        }
    
    match answer_type:
        case "search":
            prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
            Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
            Provide a comprehensive answer that addresses the user's query.

            Query: {query}

            Documents:
            {format_search_results(results)}

            Answer:"""
        case "summary":
            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Query: {query}

            Search results:
            {format_search_results(results)}

            Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
        case _:
            return {
                "search_results": results,
                "error": f"{answer_type} geneation is not supported",
            }

    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    answer = (response.text or "").strip().strip('"')

    return {
        "search_results": results,
        "answer": answer,
    }