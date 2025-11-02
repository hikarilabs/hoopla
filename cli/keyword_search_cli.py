import argparse
import sys
from pathlib import Path

# Add the parent directory to Python path to find the search module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from search.keyword_search import keyword_search_command, build_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index ...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            # print the search query
            print(f"Searching for: {args.query}")

            search_result = keyword_search_command(args.query)
            for idx, result in search_result:
                print(f"{idx}. {result}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
