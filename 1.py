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
