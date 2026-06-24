import pickle

from .config import MODEL_PATH, CLASSES
from .utils import text_with_title, load_vectorizer


class detection:
    def __init__(self):
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        print("\033[33mModel Loaded!\033[00m")

        self.vect = load_vectorizer()

        self.classes = CLASSES

    def print_predicted(self, texts, predicted):
        for i, p in enumerate(predicted):
            if len(texts[i]) > 30:
                t = texts[i][:30] + "..."
            else:
                t = texts[i]

            if p:
                color = "\033[31;01m"
            else:
                color = "\033[32;01m"

            print(f"{i+1}. {t:^33} : {color}{self.classes[p]}\033[00m")

    def detect_one_text(self, text, with_title, verbose=True):
        text = [text]
        if with_title:
            text = text_with_title(text)

        vectorized = self.vect.transform(text)

        predicted = self.model.predict(vectorized)
        if verbose:
            self.print_predicted(text, predicted)
        return predicted

    def detect(self, texts, with_title, verbose=False):
        if with_title:
            texts = text_with_title(texts)

        vectorized = self.vect.transform(texts)

        predicted = self.model.predict(vectorized)
        if verbose:
            self.print_predicted(texts, predicted)
        return predicted
