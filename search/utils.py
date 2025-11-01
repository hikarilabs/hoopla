import string

def remove_punctuation(text: str) -> str:
    translation_table = str.maketrans("", "", string.punctuation)
    text = text.lower()
    text = text.translate(translation_table)
    return text

def tokenize_text(text: str):
    if text == "":
        return []
    tokens = text.split(" ")

    result = []

    for token in tokens:
        if token == "":
            continue
        else:
            result.append(token)

    return result