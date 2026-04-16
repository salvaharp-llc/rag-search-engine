import argparse

from lib.search_utils import DEFAULT_SEARCH_LIMIT, DEFAULT_ALPHA, DEFAULT_K

from lib.hybrid_search import normalize_command, weighted_search_command, rrf_search_command

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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()