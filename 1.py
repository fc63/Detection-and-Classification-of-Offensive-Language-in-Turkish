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

from nltk.corpus import stopwords
import nltk
import re

# stanza for NLP preprocessing
!pip install stanza
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

nlp = stanza.Pipeline("tr", use_gpu=True)

print("Libraries and NLTK datasets loaded!")

nltk.download("stopwords")
stop_words = set(stopwords.words("turkish"))

# function: remove special char
def remove_special_characters(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\U00010000-\U0010FFFF]', '', text)
    return text

# preprocessing function
def preprocess_text_stanza(text):
    text = remove_special_characters(text.lower())

    doc = nlp(text)
    lemmatized_tokens = [
        word.lemma if word.lemma is not None else word.text
        for sentence in doc.sentences for word in sentence.words
        if word.text.isalpha() and word.text not in stop_words
    ]
    return " ".join(lemmatized_tokens)

tqdm.pandas()

train = dataset['train']
train = train.to_pandas()
train["cleaned_text"] = train["text"].progress_apply(preprocess_text_stanza)

train
