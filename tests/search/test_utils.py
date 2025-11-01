from search.utils import tokenize_text, remove_punctuation

def test_should_return_empty_list_when_empty_string():
    given = tokenize_text("")
    expected = []
    assert given == expected

def test_should_return_a_list_of_strings_given_a_non_empty_string():
    text = remove_punctuation("The Matrix is a great movie!")
    given = tokenize_text(text)
    expected = ["the", "matrix", "is", "a", "great", "movie"]

    assert given == expected