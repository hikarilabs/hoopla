import string

def remove_punctuation(text: str) -> str:
    translation_table = str.maketrans("", "", string.punctuation)
    text = text.lower()
    text = text.translate(translation_table)
    return text
