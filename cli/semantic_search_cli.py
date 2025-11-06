#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add the parent directory to Python path to find the search module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from search.semantic_search import verify_model, embedd_text

def main():

    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Check which embedding model is used")

    embedd_parser = subparsers.add_parser("embed_text", help="Generate a text embedd")
    embedd_parser.add_argument("text", type=str, help="Text to create its embeddings")



    args = parser.parse_args()

    match args.command:

        case "verify":
            verify_model()
        case "embed_text":
            embedd_text(args.text)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()