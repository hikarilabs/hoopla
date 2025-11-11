import argparse
import sys
from pathlib import Path


# Add the parent directory to Python path to find the search module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from search.hybrid_search import normalize_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize = subparsers.add_parser("normalize", help="Compute score normalization")
    normalize.add_argument("numbers", type=float, nargs="+")


    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize_command(args.numbers)
            for score in scores:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()