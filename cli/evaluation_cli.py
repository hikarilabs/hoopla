import argparse
import json
import sys

from pathlib import Path


# Add the parent directory to Python path to find the search module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from search.search_utils import EVAL_DATA_PATH, RRF_K
from search.hybrid_search import rrf_search_command


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open(EVAL_DATA_PATH, "r") as f:
        eval_data = json.load(f)
        evals = eval_data["test_cases"]

    for eval in evals:

        results = rrf_search_command(eval["query"], RRF_K, enhance = None, rerank_method = None, limit = limit)

        if results:
            retrieved_movies = []

            for result in results["results"]:
                retrieved_movies.append(result["title"])

            relevant_movies = eval["relevant_docs"]

            relevant_retrieved = 0

            for movie in retrieved_movies:
                if movie in relevant_movies:
                    relevant_retrieved += 1

            precision = relevant_retrieved / len(retrieved_movies)

            print(f"- Query: {eval['query']}")
            print(f"    - Precision@{limit}: {precision:.4f}")
            print(f"    - Retrieved: {', '.join(retrieved_movies)}")
            print(f"    - Relevant: {', '.join(relevant_movies)}")
            print()


if __name__ == "__main__":
    main()