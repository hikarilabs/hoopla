from search.text_processor import text_lowercase, text_tokenize, text_remove_punctuation


def test_should_return_a_lowercase_word():
    given = "Matrix"
    expected = "matrix"
    assert expected == text_lowercase(given)


def test_should_remove_punctuation():
    given = "Boots the bear!"
    expected = "Boots the bear"
    assert expected == text_remove_punctuation(given)


def test_should_return_a_list_of_tokens_for_a_non_empty_string():
    given = "the matrix is a great movie"
    expected = ["the", "matrix", "is", "a", "great", "movie"]

    assert expected == text_tokenize(given)
