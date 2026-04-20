import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

GEN_MODEL = "gemma-3-27b-it"
