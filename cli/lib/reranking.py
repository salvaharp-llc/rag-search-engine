from google import genai

from .gen_utils import API_KEY, GEN_MODEL
from .search_utils import DEFAULT_SEARCH_LIMIT

def rerank_individual(query: str, results: list[dict], limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    for result in results:
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {result.get("title", "")} - {result.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:"""

        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(model=GEN_MODEL, contents=prompt)
        corrected = (response.text or "").strip().strip('"')

        result["rerank_score"] = float(corrected)
    
    return sorted(results, key = lambda x: x["rerank_score"], reverse=True)[:limit]

def rerank_results(query: str, results: list[dict], option: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    match option:
        case "individual":
            return rerank_individual(query, results, limit)
        case _:
            return results[:limit]