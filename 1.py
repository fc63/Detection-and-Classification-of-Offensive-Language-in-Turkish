#!pip install datasets stanza
#!pip install transformers
#!pip install fasttext

from datasets import load_dataset

dataset = load_dataset("Overfit-GM/turkish-toxic-language")

dataset

# ignore unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# numpy and pandas for process and analyze
import pandas as pd
import numpy as np

# nltk for stopwords and stemmer
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer

#re for clean special char
import re

# stanza for NLP preprocessing
!pip install stanza
import stanza

# sklearn libraries for models and evaulation metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# transformers libraries for NLP model training and evaluation
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# pytorch library for data handling, model training, and loss computation
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn import CrossEntropyLoss

# matpilotlib and seaborn libraries for visualize
import matplotlib.pyplot as plt
import seaborn as sns

# progress bar for data processing
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

#fasttext for model training
import fasttext

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
    processed_tokens = []

    for sentence in doc.sentences:
        for word in sentence.words:
            if word.text.isalpha() and word.text not in stop_words:
                if word.lemma is not None:
                    processed_tokens.append(word.lemma)
                else:
                    processed_tokens.append(word.text[:5] if len(word.text) > 5 else word.text)

    return " ".join(processed_tokens)

print("Preprocessing dataset...")
tqdm.pandas()

train = dataset['train']
train = train.to_pandas()
train["cleaned_text"] = train["text"].progress_apply(preprocess_text_stanza)

train

output_file = "/content/drive/My Drive/cleaned_dataset.csv"

train.to_csv(output_file, index=False, encoding="utf-8")

print(f"Cleaned dataset saved to Google Drive at {output_file}")

input_file = "/content/drive/My Drive/cleaned_dataset.csv"

train = pd.read_csv(input_file, encoding="utf-8")

train

train = train.dropna(subset=["cleaned_text"])

def convert_turkish_to_english(text):
    turkish_to_english = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    return text.translate(turkish_to_english)

print("Expanding dataset with Turkish and English character versions...")

train_turkish = train.copy()
train_english = train.copy()
train_english.loc[:, "cleaned_text"] = train_english["cleaned_text"].progress_apply(convert_turkish_to_english)

train_expanded = pd.concat([train_turkish, train_english], ignore_index=True)

print(f"Dataset expanded. Original size: {len(train_turkish)}, Expanded size: {len(train_expanded)}")

train

# Custom tokenizer
def fasttext_tokenizer(text):
    turkish_to_english = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    text = text.translate(turkish_to_english)
    text = text.lower()
    return text

train["fasttext_format"] = "__label__" + train["is_toxic"].astype(str) + " " + train["cleaned_text"].apply(fasttext_tokenizer)

# Save training and test data
train_file = "/content/drive/My Drive/fasttext_train.txt"
test_file = "/content/drive/My Drive/fasttext_test.txt"

train_data = train.sample(frac=0.8, random_state=42).copy()
test_data = train.drop(train_data.index).copy()

train_data["fasttext_format"].to_csv(train_file, index=False, header=False)
test_data["fasttext_format"].to_csv(test_file, index=False, header=False)

print("FastText dataset prepared and saved.")


model = fasttext.train_supervised(
    input=train_file,
    lr=0.1,
    epoch=25,
    wordNgrams=3,
    bucket=200000,
    dim=100,
    loss="softmax"
)

model_file = "/content/drive/My Drive/toxicity_model.bin"
model.save_model(model_file)

# classification report

model_file = "/content/drive/My Drive/toxicity_model.bin"
model = fasttext.load_model(model_file)

test_sentences = test_data["cleaned_text"].tolist()
test_labels = test_data["is_toxic"].tolist()

predictions = [int(model.predict(sentence)[0][0].replace("__label__", "")) for sentence in test_sentences]

print("Classification Report:")
print(classification_report(test_labels, predictions, target_names=["Non-Toxic", "Toxic"]))

# confusion matrix
conf_matrix = confusion_matrix(test_labels, predictions)

# visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Toxic", "Toxic"], yticklabels=["Non-Toxic", "Toxic"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for FastText Model")
plt.show()


# test sırasında ve modele text verirken aynı tokenizerı kullanmak zorunludur
def fasttext_tokenizer(text):
    turkish_to_english = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    text = text.translate(turkish_to_english)
    text = text.lower()
    return text

def test_fasttext_model(model, sentences, tokenizer):
    predictions = []
    for sentence in sentences:
        processed_sentence = tokenizer(sentence)
        prediction = model.predict(processed_sentence)[0][0].replace("__label__", "")
        predictions.append((sentence, prediction))
    return predictions

# testing
test_sentences = [
    "Sen bir aptalsın!",
    "sen bir aptalsin",
    "Sen bir aptalsin!",
    "bu tamamen zararsiz bir cümle.",
    "Bugün hava çok güzel.",
    "Kes lan sesini",
    "Naber lan",
    "naber kanka",
    "nasılsın",
    "aga bu modeli nasıl eğiteceğiz ya",
    "senin anan toxic",
    "güzel günler mazide kaldı",
    "kanka ve lan her türlü toxic oluyor",
    "dataset ile ilgili galiba",
    "neyse olduğu kadar"
]

model_file = "/content/drive/My Drive/toxicity_model.bin"
model = fasttext.load_model(model_file)

print("Testing FastText Model:")
results = test_fasttext_model(model, test_sentences, fasttext_tokenizer)

for sentence, prediction in results:
    label = "Toxic" if prediction == "1" else "Non-Toxic"
    print(f"Sentence: {sentence}\nPrediction: {label}\n")

def prepare_category_fasttext_format(row):
    label = f"__label__{row['target']}"
    return f"{label} {row['cleaned_text']}"

print("Preparing category FastText format...")
train_expanded.loc[:, "fasttext_category_format"] = train_expanded.apply(prepare_category_fasttext_format, axis=1)

category_train_file = "/content/drive/My Drive/fasttext_category_train.txt"
category_test_file = "/content/drive/My Drive/fasttext_category_test.txt"

train_category_data = train_expanded.sample(frac=0.8, random_state=42).copy()
test_category_data = train_expanded.drop(train_category_data.index).copy()

train_category_data["fasttext_category_format"].to_csv(category_train_file, index=False, header=False)
test_category_data["fasttext_category_format"].to_csv(category_test_file, index=False, header=False)

print("Category FastText dataset prepared and saved.")

print("Training FastText category model...")
category_model = fasttext.train_supervised(
    input=category_train_file,
    lr=0.1,
    epoch=25,
    wordNgrams=3,
    bucket=200000,
    dim=100,
    loss="softmax"
)

category_model_file = "/content/drive/My Drive/category_model.bin"
category_model.save_model(category_model_file)
print(f"Category model saved to {category_model_file}")


print("Evaluating category model...")

category_test_sentences = test_category_data["cleaned_text"].tolist()
category_test_labels = test_category_data["target"].tolist()

predicted_labels = [category_model.predict(sentence)[0][0].replace("__label__", "") for sentence in category_test_sentences]

print("Classification Report for Category Model:")
print(classification_report(category_test_labels, predicted_labels, target_names=list(train_expanded["target"].unique())))

conf_matrix = confusion_matrix(category_test_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=list(train_expanded["target"].unique()), yticklabels=list(train_expanded["target"].unique()))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Category Model")
plt.show()

# Test sırasında ve modele text verirken aynı tokenizer kullanılmalıdır
def fasttext_tokenizer(text):
    turkish_to_english = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    text = text.translate(turkish_to_english)
    text = text.lower()
    return text

def test_fasttext_category_model(model, sentences, tokenizer):
    predictions = []
    for sentence in sentences:
        processed_sentence = tokenizer(sentence)
        pred = model.predict(processed_sentence)
        print(f"Raw prediction: {pred}")
        prediction = pred[0][0]
        predictions.append((sentence, prediction))
    return predictions

# testing
test_sentences_category = [
    "Sen bir aptalsın!",
    "sen bir aptalsin",
    "Sen bir aptalsin!",
    "bu tamamen zararsiz bir cümle.",
    "Bugün hava çok güzel.",
    "Kes lan sesini",
    "Naber lan",
    "naber kanka",
    "nasılsın",
    "aga bu modeli nasıl eğiteceğiz ya",
    "senin anan toxic",
    "güzel günler mazide kaldı",
    "kanka ve lan her türlü toxic oluyor",
    "dataset ile ilgili galiba",
    "neyse olduğu kadar",
    "BÖYLE MODELİN AMK",
    "lana hakaret diyen kafanı sikeyim senin"
]

category_model_file = "/content/drive/My Drive/category_model.bin"
category_model = fasttext.load_model(category_model_file)

print("Testing FastText Category Model:")
results_category = test_fasttext_category_model(category_model, test_sentences_category, fasttext_tokenizer)

for sentence, prediction in results_category:
    category = {
        "__label__INSULT": "INSULT",
        "__label__PROFANITY": "PROFANITY",
        "__label__RACIST": "RACIST",
        "__label__SEXIST": "SEXIST",
        "__label__OTHER": "OTHER"
    }.get(prediction, "Unknown")
    print(f"Sentence: {sentence}\nPredicted Category: {category}\n")
