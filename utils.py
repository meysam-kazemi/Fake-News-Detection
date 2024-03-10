import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score , confusion_matrix , ConfusionMatrixDisplay
import pickle

def load_vectorizer():
    with open("Vectorizer.pkl","rb") as f:
                vect = pickle.load(f)
    return vect

