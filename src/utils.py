import pickle

from .config import VECTORIZER_PATH


def load_vectorizer():
    with open(VECTORIZER_PATH, "rb") as f:
        vect = pickle.load(f)
    return vect


def text_with_title(text):
    res = text["title"] + " " + text["text"]
    return res
