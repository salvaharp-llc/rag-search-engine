import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

GEN_MODEL = "gemma-3-27b-it"

ENHANCEMENT_PROMPTS = {
    "spell": """Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
"""
}

def enhance_query(query: str, option: str) -> str:
    if option not in ENHANCEMENT_PROMPTS:
        return query
    
    prompt = f"""{ENHANCEMENT_PROMPTS[option]}
User query: "{query}"
"""

    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query
