import json

from pathlib import Path


DEFAULT_SEARCH_LIMIT = 5
PROJECT_ROOT = Path(__file__).parent.parent


def load_movies() -> list[dict[int, str]]:
    data_path = PROJECT_ROOT / "data" / "movies.json"
    with open(data_path, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stop_words() -> list:
    data_path = PROJECT_ROOT / "data" / "stopwords.txt"
    with open(data_path, "r") as f:
        data = f.read().splitlines()

    return data
