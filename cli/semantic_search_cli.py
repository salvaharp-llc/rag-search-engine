#!/usr/bin/env python3

import argparse

from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT, 
    DEFAULT_CHUNK_SIZE, 
    DEFAULT_CHUNK_OVERLAP, 
    DEFAULT_SEMANTIC_CHUNK_SIZE,
)

from lib.semantic_search import (
    verify_model, 
    embed_text, 
    verify_embeddings, 
    embed_query_text,
    search_command,
    chunk_command,
    semantic_chunk_command,
    embed_chunks_command,
    search_chunked_command,
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _ = subparsers.add_parser("verify", help="Verify semantic model")

    embed_subparser = subparsers.add_parser("embed_text", help="Generate text embedding")
    embed_subparser.add_argument("text", type=str, help="Text to embed")

    _ = subparsers.add_parser("verify_embeddings", help="Verify embeddings for the movie dataset")

    embedquery_subparser = subparsers.add_parser("embedquery", help="Generate query embedding")
    embedquery_subparser.add_argument("query", type=str, help="Query to embed")

    search_subparser = subparsers.add_parser("search", help="Perform semantic search")
    search_subparser.add_argument("query", type=str, help="Query to search")
    search_subparser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit number of search results")

    chunk_subparser = subparsers.add_parser("chunk", help="Divide text into chunks")
    chunk_subparser.add_argument("text", type=str, help="Text to divide")
    chunk_subparser.add_argument("--chunk-size", type=int, nargs='?', default=DEFAULT_CHUNK_SIZE, help="Size of the chunks to divide the text into")
    chunk_subparser.add_argument("--overlap", type=int, nargs='?', default=DEFAULT_CHUNK_OVERLAP, help="Number of words to overlap between chunks")

    semantic_chunk_subparser = subparsers.add_parser("semantic_chunk", help="Divide text into chunks based on semantic boundaries")
    semantic_chunk_subparser.add_argument("text", type=str, help="Text to divide")
    semantic_chunk_subparser.add_argument("--max-chunk-size", type=int, nargs='?', default=DEFAULT_SEMANTIC_CHUNK_SIZE, help="Size of the chunks to divide the text into")
    semantic_chunk_subparser.add_argument("--overlap", type=int, nargs='?', default=DEFAULT_CHUNK_OVERLAP, help="Number of words to overlap between chunks")

    embed_chunks_subparser = subparsers.add_parser("embed_chunks", help="Generate chunks embeddings")

    search_chunked_subparser = subparsers.add_parser("search_chunked", help="Perform semantic chunked search")
    search_chunked_subparser.add_argument("query", type=str, help="Query to search")
    search_chunked_subparser.add_argument("--limit", type=int, nargs='?', default=DEFAULT_SEARCH_LIMIT, help="Limit number of search results")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            movies = search_command(args.query, args.limit)
            for i, movie in enumerate(movies, 1):
                print(f"{i}. {movie['title']} (score: {movie["score"]:.4f})")
                print(f"{movie["description"][:50]}...")
                print()
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks_command()
        case "search_chunked":
            movies = search_chunked_command(args.query, args.limit)
            for i, movie in enumerate(movies, 1):
                print(f"\n{i}. {movie["title"]} (score: {movie["score"]:.4f})")
                print(f"   {movie["document"]}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()