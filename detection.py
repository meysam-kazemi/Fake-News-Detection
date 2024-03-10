import pickle
from utils import text_with_title, load_vectorizer

class detection:
    def __init__(self):
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        print('\033[33mModel Loaded!\033[00m')

        self.vect = load_vectorizer()
        
    def detect(self,text, with_title):
        if with_title:
            text = text_with_title(text)

        vectorized = self.vect.transform(text)

        predicted = self.model.predict(vectorized)
        return predicted


# test
det = detection()
det.detect("I write a test text",False)
