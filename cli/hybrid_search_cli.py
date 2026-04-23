import argparse

from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT, 
    DEFAULT_ALPHA, DEFAULT_K, 
    DOCUMENT_PREVIEW_LENGTH,
)

from lib.hybrid_search import normalize_command, weighted_search_command, rrf_search_command
from lib.evaluation import evaluate_with_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a given list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="List of scores to normalize")

    weighted_search_subparser = subparsers.add_parser("weighted-search", help="Perform weighted search")
    weighted_search_subparser.add_argument("query", type=str, help="Query to search")
    weighted_search_subparser.add_argument("--alpha", type=float, nargs='?', default=DEFAULT_ALPHA, help="Weight to value keyword over semantic search")
    weighted_search_subparser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit number of search results")

    rrf_search_subparser = subparsers.add_parser("rrf-search", help="Perform rrf search")
    rrf_search_subparser.add_argument("query", type=str, help="Query to search")
    rrf_search_subparser.add_argument("-k", type=int, nargs='?', default=DEFAULT_K, help="Weight to emphasize the ranking of the results")
    rrf_search_subparser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit number of search results")
    rrf_search_subparser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_subparser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Search reranking method")
    rrf_search_subparser.add_argument("--evaluate", action="store_true", help="Evaluate the results using an LLM")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            results = rrf_search_command(args.query, args.k, args.limit, args.enhance, args.rerank_method)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result["title"]}")
                match args.rerank_method:
                    case "individual":
                        print(f"   Re-rank Score: {result.get("rerank_score", -1):.3f}/10")
                    case "batch":
                        print(f"   Re-rank Rank: {result.get("rerank_rank", -1)}")
                    case "cross_encoder":
                        print(f"   Cross Encoder Score: {result.get("cross_encoder_score", -1):.3f}")
                print(f"   RRF Score: {result["score"]:.3f}")
                print(f"   BM25 Rank: {result.get("bm25_rank", -1)}, Semantic: {result.get("semantic_rank", -1)}")
                print(f"   {result["description"][:DOCUMENT_PREVIEW_LENGTH]}...")
            print()

            print("LLM Evaluation (0-3 relevance scale):")
            if args.evaluate:
                evaluations = evaluate_with_llm(args.query, results)
                for i, eval in enumerate(evaluations, 1):
                    print(f"{i}. {eval["title"]}: {eval["rating"]}/3")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()