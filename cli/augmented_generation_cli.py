import argparse
import sys

from lib.augmented_generation import (
    rag_command, 
    summarize_command, 
    citations_command,
)
from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform search + document summarization"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Limit number of search results to summarize")

    citations_parser = subparsers.add_parser(
        "citations", help="Perform search + document summarization with citations in the summary"
    )
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Limit number of search results to summarize")

    args = parser.parse_args()

    match args.command:
        case "rag":
            print_result(rag_command(args.query), "RAG Response")
        case "summarize":
            print_result(summarize_command(args.query, args.limit), "LLM Summary")
        case "citations":
            print_result(citations_command(args.query, args.limit), "LLM Answer")
        case _:
            parser.print_help()

def print_result(result: dict, answer_type: str) -> None:
    if "error" in result:
        print(f"Error: {result["error"]}")
        sys.exit()

    print("Search Results:")
    for search_result in result["search_results"]:
        print(f"- {search_result["title"]}")
    print()
    print(f"{answer_type}:")
    print(result["answer"])

if __name__ == "__main__":
    main()