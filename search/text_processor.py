import string
from typing import List, Tuple
from search.search_utils import load_stop_words

from nltk.stem import PorterStemmer


def text_lowercase(text: str) -> str:
    return text.lower()


def text_remove_punctuation(text: str) -> str:
    translation_table = str.maketrans("", "", string.punctuation)
    text = text.translate(translation_table)
    return text


def text_tokenize(text: str, delimiter: str = " ") -> List:
    """
    Utility function to tokenize a given text
    :param text: The text
    :param delimiter: Delimiter used to separate the words in the given text
    :return: the input text as a list of tokens
    """
    tokens = text.split(delimiter)

    valid_tokens = []

    for token in tokens:
        if token:
            valid_tokens.append(token)

    return valid_tokens


def has_matching_token(query_tokens: List[str], movie_title_tokens: List[str]) -> bool:
    """

    :param query_tokens:
    :param movie_title_tokens:
    :return:
    """
    for query_token in query_tokens:
        for title_token in movie_title_tokens:
            if query_token in title_token:
                return True
    return False


def text_remove_stop_words(tokens: List[str]) -> List[str]:
    stop_words = load_stop_words()

    valid_tokens = []

    for token in tokens:
        if token in stop_words:
            continue
        else:
            valid_tokens.append(token)

    return valid_tokens


def text_stem(tokens: List[str]) -> List[str]:
    stemmer = PorterStemmer()

    stem_tokens = []

    for token in tokens:
        stem_tokens.append(stemmer.stem(token))

    return stem_tokens


def process_text(text: str) -> List[str]:
    # process lower case
    lower = text_lowercase(text)

    # process - remove text punctuation
    punctuation = text_remove_punctuation(lower)

    # process - tokenize text
    tokens = text_tokenize(punctuation)

    # process - remove stop words
    clean_stop_words = text_remove_stop_words(tokens)

    # process - stemming
    stem_words = text_stem(clean_stop_words)

    return stem_words
