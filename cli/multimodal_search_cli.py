import argparse

from lib.multimodal_search import (
    verify_image_embedding, 
    image_search_command,
)
from lib.search_utils import DOCUMENT_PREVIEW_LENGTH


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify dimensions of computed image embedding"
    )
    verify_image_parser.add_argument("image", type=str, help="Path to the image file")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Perform semantic search using an image"
    )
    image_search_parser.add_argument("image", type=str, help="Path to the image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            results = image_search_command(args.image)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (similarity: {result['similarity_score']:0.3f})")
                print(f"   {result["description"][:DOCUMENT_PREVIEW_LENGTH]}...")
                print()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()