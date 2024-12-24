#!pip install datasets stanza

from datasets import load_dataset

dataset = load_dataset("Overfit-GM/turkish-toxic-language")

dataset

# ignore unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# numpy and pandas for process and analyze
import pandas as pd
import numpy as np

# stanza for NLP preprocessing
import stanza

# sklearn libraries for models and evaulation metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# matpilotlib and seaborn libraries for visualize
import matplotlib.pyplot as plt
import seaborn as sns

# progress bar for data processing
from tqdm import tqdm

# install and download stanza's turkish model
stanza.download("tr", verbose=False)
print("Stanza Turkish model downloaded!")

nlp = stanza.Pipeline("tr", use_gpu=False)

print("Libraries and NLTK datasets loaded!")

# Preprocess the dataset
def preprocess_text(text):
    """
    Preprocess a single text: tokenization, stopword removal, and lemmatization.
    """
    doc = nlp(text)
    tokens = [
        word.lemma for sent in doc.sentences for word in sent.words if word.upos != "PUNCT"
    ]
    return " ".join(tokens)

# Convert dataset to pandas DataFrame
train_data = dataset['train']
df = pd.DataFrame(train_data)

# Preprocess the text column
print("Preprocessing text data...")
df['processed_text'] = df['text'].apply(preprocess_text)

# Splitting the dataset into train and test sets
print("Splitting dataset into train and test sets...")
X = df['processed_text']
y = df['label']  # Adjust 'label' to your actual label column name if different
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transforming text into numerical features using TF-IDF
print("Applying TF-IDF vectorization...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Data preprocessing completed!")
