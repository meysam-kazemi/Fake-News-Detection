import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score , confusion_matrix , ConfusionMatrixDisplay
import pickle

data_dir = "/run/media/meysam/PROGRAM/0.data/Fake News Detection Dataset/News _dataset"
files = os.listdir(data_dir)
files = list(map(lambda a:data_dir+"/"+a , files))

fakes = pd.read_csv(files[0])
trues = pd.read_csv(files[1])

vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(features)

# Load the model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open("Vectorizer.pkl","rb") as f:
    vect = pickle.load(f)