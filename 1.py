#!pip3 install datasets stanza sinkaf TurkishStemmer transformers JPype1

#!apt-get install -y openjdk-11-jdk-headless -qq > /dev/null

from datasets import load_dataset

dataset = load_dataset("Overfit-GM/turkish-toxic-language")

dataset

# ignore unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd

# numpy and pandas for process and analyze
import pandas as pd
import numpy as np

# nltk for stopwords
from nltk.corpus import stopwords
import nltk

#re for clean special char
import re

# stanza for NLP preprocessing
!pip install stanza
import stanza

import torch.nn as nn

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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BertModel, DataCollatorWithPadding

# pytorch library for data handling, model training, and loss computation
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn import CrossEntropyLoss

# matpilotlib and seaborn libraries for visualize
import matplotlib.pyplot as plt
import seaborn as sns

# progress bar for data processing
from tqdm import tqdm

# Turkish Stemmer
from TurkishStemmer import TurkishStemmer

from sinkaf import Sinkaf

from google.colab import drive
drive.mount('/content/drive')

# jpype
from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java, isJVMStarted

# install and download stanza's turkish model
stanza.download("tr", verbose=False)
print("Stanza Turkish model downloaded!")

nlp = stanza.Pipeline("tr", use_gpu=True)

print("Libraries and NLTK datasets loaded!")

zemberek_jar_path = "/content/drive/My Drive/zemberek.jar"

if not isJVMStarted():
    startJVM(getDefaultJVMPath(), f"-Djava.class.path={zemberek_jar_path}", "-ea")

TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
TurkishSpellChecker = JClass('zemberek.normalization.TurkishSpellChecker')

morphology = TurkishMorphology.createWithDefaults()
spell_checker = TurkishSpellChecker(morphology)

dataset = load_dataset("Overfit-GM/turkish-toxic-language")
train = dataset['train'].to_pandas()

def correct_spelling(text):
    words = text.split()
    corrected_words = []
    for word in words:
        if spell_checker.check(word):
            corrected_words.append(str(word))
        else:
            suggestions = spell_checker.suggestForWord(word)
            if suggestions:
                corrected_words.append(str(suggestions[0]))
            else:
                corrected_words.append(str(word))
    return " ".join(corrected_words)

tqdm.pandas()
train['corrected_text'] = train['text'].progress_apply(correct_spelling)

train.to_csv("/content/drive/My Drive/corrected_dataset.csv", index=False)

shutdownJVM()

train

turkish_characters = "ğüşıöçĞÜŞİÖÇ"
train = train[train['corrected_text'].str.contains(f"[{turkish_characters}]", na=False)]
train

# sinkaf profanity filter
profanity_filter = Sinkaf(model = "bert_pre")

nltk.download("stopwords")
stop_words = set(stopwords.words("turkish"))

# function: remove special char
def remove_special_characters(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[\U00010000-\U0010FFFF]', '', text)
    return text

stemmer = TurkishStemmer()

# function: manual stemming (cutting to 5 characters if longer)
def manual_stem(text):
    return text[:5] if len(text) > 5 else text

# preprocessing function
def preprocess_text_stanza_with_sinkaf(text):
    text = remove_special_characters(text.lower())

    doc = nlp(text)
    lemmatized_tokens = []

    for sentence in doc.sentences:
        for word in sentence.words:
            if word.text.isalpha() and word.text not in stop_words:
                if word.lemma is not None:
                    lemmatized_tokens.append(word.lemma)
                else:
                    profanity_score = profanity_filter.tahminlik([word.text])[0]
                    if profanity_score > 0.9:
                        lemmatized_tokens.append(word.text)
                    else:
                        stemmed_word = stemmer.stem(word.text)
                        if len(stemmed_word) == len(word.text):
                            stemmed_word = manual_stem(word.text)
                        lemmatized_tokens.append(stemmed_word)

    return " ".join(lemmatized_tokens)

tqdm.pandas()

train["cleaned_text"] = train["corrected_text"].progress_apply(preprocess_text_stanza_with_sinkaf)

train

output_file = "/content/drive/My Drive/cleaned_dataset.csv"

train.to_csv(output_file, index=False, encoding="utf-8")

print(f"Cleaned dataset saved to Google Drive at {output_file}")

input_file = "/content/drive/My Drive/cleaned_dataset.csv"

train = pd.read_csv(input_file, encoding="utf-8")

train

train = train[~train['target'].isin(['RACIST', 'SEXIST'])].reset_index(drop=True)

train['target'] = train['target'].replace({'INSULT': 'OFFENSIVE', 'PROFANITY': 'OFFENSIVE'})

print(train['target'].value_counts())
train

def clean_corrected_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r"[^\w\s.,!?-]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

train['cleaned_corrected_text'] = train['corrected_text'].apply(clean_corrected_text)


print(train['target'].value_counts())
train

from imblearn.under_sampling import RandomUnderSampler

X = train[['cleaned_text','cleaned_corrected_text']]
y = train['is_toxic']

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

print("Örnek sayıları:")
print(y_resampled.value_counts())

undersampled_data = pd.concat([X_resampled, y_resampled], axis=1)

undersampled_data.to_csv("undersampled_dataset.csv", index=False)

undersampled_data

kelimeler = ["kanka", "lan", "knk", "bro", "la", "birader", "kanki"]

def kelime_cikartici(metin):
    for kelime in kelimeler:
        metin = metin.replace(kelime, "")
    return metin.strip()

undersampled_data["cleaned_corrected_text"] = undersampled_data["cleaned_corrected_text"].apply(kelime_cikartici)

undersampled_data

data = undersampled_data

X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned_corrected_text'], data['is_toxic'], test_size=0.2, random_state=42
)

model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

train_dataset = ToxicityDataset(X_train, y_train, tokenizer)
test_dataset = ToxicityDataset(X_test, y_test, tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./turkish_toxic_language_detection_model")
tokenizer.save_pretrained("./turkish_toxic_language_detection_model")

print("train finished")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_model(test_sentences, model_path="./turkish_toxic_language_detection_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.eval()

    predictions = []

    with torch.no_grad():
        for sentence in test_sentences:
            encoding = tokenizer(
                sentence,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predicted_label = torch.argmax(logits, dim=1).item()
            predictions.append(predicted_label)

    return predictions

test_sentences = [
    "zeki insanlar kitap okumuştur",
    "suratlarına tükürsek zerre utanmazlar.. kk  pkkcıydı demi bu ... gargameline göre.. şu çağrıyı kk yapsaydı hapse atardı bu o....ları ... ",
    "onemli oranda aliyorda, siaysetciler kendi atadıklari gudumlu hakimin yerine adil bir yapay zeka istemez",
    "Bu arada sozlesmeli personelin kadroya geçmesini tüm sendikalar destekler ve başarı diye anlatirlar"
]

predictions = test_model(test_sentences)

for sentence, prediction in zip(test_sentences, predictions):
    print(f"Cümle: {sentence} -> Tahmin: {'Toksik' if prediction == 1 else 'Toksik Değil'}")


# classification report


# generate predictions
predictions = trainer.predict(test_dataset)
predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=1)

# performance metrics
print("Classification Report:")
print(classification_report(y_test, predicted_labels, target_names=["Non-Toxic", "Toxic"]))

# confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_labels)

# visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Toxic", "Toxic"], yticklabels=["Non-Toxic", "Toxic"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
