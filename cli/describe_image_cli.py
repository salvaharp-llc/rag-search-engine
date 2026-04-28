import os
import argparse
import mimetypes
from google import genai

from lib.gen_utils import API_KEY, GEN_MODEL

system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Description CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to image file to describe")
    parser.add_argument("--query", type=str, required=True, help="Text query")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        img = f.read()

    parts = [
        system_prompt,
        genai.types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]

    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(model=GEN_MODEL, contents=parts)
    if response.text is None:
        raise RuntimeError("No text in Gemini API response")

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

if __name__ == "__main__":
    main()