import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score , confusion_matrix , ConfusionMatrixDisplay
import pickle


class detection:
    def __init__(self):
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        print('\033[33mModel Loaded!')
        
    def 
