import argparse
import sys
from pathlib import Path

# Add the parent directory to Python path to find the search module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from search.search_utils import BM25_K1, BM25_B
from search.keyword_search import (keyword_search_command,
                                   build_command,
                                   tf_command,
                                   idf_calculator_command,
                                   tf_idf_command,
                                   bm25idf_command,
                                   bm25tf_command,
                                   bm25search_command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a given document id and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get it frequency in the document")

    idf_parser = subparsers.add_parser("idf", help="Get the inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to calculate the IDF frequency")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TD-IDF for a given document id and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get it frequency in the document")

    bm25_idf_parser = subparsers.add_parser(
        'bm25idf', help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")

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
        case "tf":
            term_frequency = tf_command(args.doc_id, args.term)
            print(f"Term frequency for '{args.term}' in document '{args.doc_id}' : {term_frequency}")
        case "idf":
            idf = idf_calculator_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tf_idf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25idf = bm25idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25tf_command(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            print("Searching for:", args.query)
            results = bm25search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']} - Score: {res['score']:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
