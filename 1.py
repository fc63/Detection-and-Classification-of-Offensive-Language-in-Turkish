#!pip install datasets stanza
#!pip install transformers

from datasets import load_dataset

dataset = load_dataset("Overfit-GM/turkish-toxic-language")

dataset

# ignore unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

X = train["cleaned_text"]
y = train["is_toxic"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(train.isnull().sum())

# we use a pre-trained bert model
model_name = "dbmdz/bert-base-turkish-cased"

# tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

class ToxicDataset(Dataset):
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

# datasets preparation
train_dataset = ToxicDataset(X_train, y_train, tokenizer)
test_dataset = ToxicDataset(X_test, y_test, tokenizer)

# training settings
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
    report_to="none",
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=1,
)

# trainer (Hugging Face Trainer API)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# which epoch is used
best_checkpoint = trainer.state.best_model_checkpoint
checkpoint_step = int(re.search(r"checkpoint-(\d+)", best_checkpoint).group(1))
total_training_steps = len(train_dataset) // training_args.per_device_train_batch_size
steps_per_epoch = total_training_steps // training_args.num_train_epochs
epoch_used = checkpoint_step / steps_per_epoch
print(f"Best model is from epoch: {epoch_used:.2f}")

# classification report


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

# target column for categorization model
X_category = train["cleaned_text"]
y_category = train["target"]

# convert categories in `target` column to numeric values
label_encoder = LabelEncoder()
y_category = label_encoder.fit_transform(y_category)

# compute class weights for imbalanced dataset
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_category),
    y=y_category
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# separate training and test sets
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    X_category, y_category, test_size=0.2, random_state=42
)

# tokenizer & model
category_tokenizer = AutoTokenizer.from_pretrained(model_name)
category_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

class CategoryDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels[idx]
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

# datasets preparation
train_dataset_cat = CategoryDataset(X_train_cat, y_train_cat, category_tokenizer)
test_dataset_cat = CategoryDataset(X_test_cat, y_test_cat, category_tokenizer)

# custom trainer class for loss function with class weights
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# training settings
training_args_cat = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,  
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=32,
    num_train_epochs=5,  
    weight_decay=0.1,  
    gradient_accumulation_steps=2,  
    logging_dir="./logs",
    report_to="none",
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=1,
)

# trainer (Hugging Face Trainer API)
category_trainer = CustomTrainer(
    model=category_model,
    args=training_args_cat,
    train_dataset=train_dataset_cat,
    eval_dataset=test_dataset_cat,
    tokenizer=category_tokenizer,
)

category_trainer.train()

# which epoch is used for categorization model
best_checkpoint = category_trainer.state.best_model_checkpoint
checkpoint_step = int(re.search(r"checkpoint-(\d+)", best_checkpoint).group(1))
total_training_steps = len(train_dataset_cat) // training_args_cat.per_device_train_batch_size
steps_per_epoch = total_training_steps // training_args_cat.num_train_epochs
epoch_used = checkpoint_step / steps_per_epoch
print(f"Best model for categorization is from epoch: {epoch_used:.2f}")

# classification report for categorization model

predictions = category_trainer.predict(test_dataset_cat)
predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=1)

# performance metrics
print("Classification Report for Categorization Model:")
print(classification_report(
    y_test_cat, 
    predicted_labels.numpy(), 
    target_names=label_encoder.classes_
))

# confusion matrix
conf_matrix = confusion_matrix(y_test_cat, predicted_labels.numpy())

# visualization of confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=label_encoder.classes_, 
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Categorization Model")
plt.show()
