import argparse
import sys
from pathlib import Path


# Add the parent directory to Python path to find the search module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from search.hybrid_search import normalize_command, weighted_search_command, rrf_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize = subparsers.add_parser("normalize", help="Compute score normalization")
    normalize.add_argument("numbers", type=float, nargs="+")

    weighted_search = subparsers.add_parser("weighted-search", help="Search using a weighted combination")
    weighted_search.add_argument("query", type=str, help="User search query")
    weighted_search.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 vs semantic")
    weighted_search.add_argument("--limit", type=int, default=5, help="Number of results to return")

    rrf_search = subparsers.add_parser("rrf-search", help="Search using reciprocal rank fusion")
    rrf_search.add_argument("query", type=str, help="User search query")
    rrf_search.add_argument("--k", type=int, default=60, help="Weight for low-rank vs high rank results")
    rrf_search.add_argument("--limit", type=int, default=5, help="Number of results to return")
    rrf_search.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="LLM movie re-ranking")


    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize_command(args.numbers)
            for score in scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.limit)

            print(
                f"Weighted Hybrid Search Results for '{results['query']}' (alpha={results['alpha']}):"
            )
            print(
                f"  Alpha {results['alpha']}: {int(results['alpha'] * 100)}% Keyword, {int((1 - results['alpha']) * 100)}% Semantic"
            )
            for i, res in enumerate(results["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()
        case "rrf-search":
            result = rrf_search_command(
                args.query, args.k, args.enhance, args.rerank_method, args.limit
            )

            if result["enhanced_query"]:
                print(
                    f"Enhanced query ({result['enhance_method']}): '{result['original_query']}' -> '{result['enhanced_query']}'\n"
                )

            if result["reranked"]:
                print(
                    f"Reranking top {len(result['results'])} results using {result['rerank_method']} method...\n"
                )

            print(
                f"Reciprocal Rank Fusion Results for '{result['query']}' (k={result['k']}):"
            )

            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                if "individual_score" in res:
                    print(f"   Rerank Score: {res.get('individual_score', 0):.3f}/10")
                if "batch_rank" in res:
                    print(f"   Rerank Rank: {res.get('batch_rank', 0)}")
                print(f"   RRF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                ranks = []
                if metadata.get("bm25_rank"):
                    ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
                if metadata.get("semantic_rank"):
                    ranks.append(f"Semantic Rank: {metadata['semantic_rank']}")
                if ranks:
                    print(f"   {', '.join(ranks)}")
                print(f"   {res['document'][:100]}...")
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()