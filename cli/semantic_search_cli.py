#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add the parent directory to Python path to find the search module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from search.semantic_search import (verify_model, embedd_text,
                                    verify_embeddings,
                                    embed_query_text,
                                    search_query,
                                    chunk_text_command,
                                    semantic_chunk_command)
from search.chunked_semantic_search import embed_chunks_command, search_chunked_command


def main():

    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Check which embedding model is used")
    subparsers.add_parser("verify_embeddings", help="Check which embedding model is used")

    embedd_parser = subparsers.add_parser("embed_text", help="Generate a text embedd")
    embedd_parser.add_argument("text", type=str, help="Text to create its embeddings")

    query_parser = subparsers.add_parser("embedquery", help="Generate a query embedd")
    query_parser.add_argument("query", type=str, help="Query embedding")

    user_search_query = subparsers.add_parser("search", help="Search for a user query")
    user_search_query.add_argument("query", type=str, help="User query")
    user_search_query.add_argument("--limit", type=int, nargs='?', default=5, help="Tunable result list")

    chunk_text = subparsers.add_parser("chunk", help="Search for a user query")
    chunk_text.add_argument("text", type=str, help="Text to chunk")
    chunk_text.add_argument("--chunk-size", type=int, default=200, help="Tunable result list")
    chunk_text.add_argument("--overlap", type=int, default=0, help="Tunable result list")

    semantic_chunk = subparsers.add_parser("semantic_chunk", help="Generate chunks using semantic splits")
    semantic_chunk.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk.add_argument("--max-chunk-size", type=int, default=4, help="Tunable result list")
    semantic_chunk.add_argument("--overlap", type=int, default=0, help="Tunable result list")

    embed_chunks = subparsers.add_parser("embed_chunks", help="Generate embeddings for chunked documents")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for a user query")
    search_chunked_parser.add_argument("query", type=str, help="User query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Tunable result list")

    args = parser.parse_args()

    match args.command:

        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embedd_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_query(args.query, args.limit)
        case "chunk":
            chunk_text_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embeddings = embed_chunks_command()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            results = search_chunked_command(args.query, args.limit)
            print(f"Query: {results['query']}")
            print("Results")
            for i, result in enumerate(results["results"], 1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f}")
                print(f"    {result['document']} ...")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()