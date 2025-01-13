from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

repo_id = "fc63/toxic-classification-model"
model = AutoModelForSequenceClassification.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

class_labels = {
    0: "Non-Toxic",
    1: "Toxic"
}

"""
Uyarı: Aşağıdaki cümleler, modelimizi test etmek içindir.
Herhangi bir şekilde, dini, ideolojik, siyasi, felsefi, hayat görüşü vb.
fikrimizi yansıtma amacı taşımamaktadır.
Modelimizin racismi, sexismi vs. tespit edip edemeyeceğini test etmek için kullanılmıştır.
"""

test_sentences = [
    "Bu çok aptalca bir hareketti!",  # Toxic
    "Bugün hava oldukça güzel.",      # Non-Toxic
    "Sen nasıl bir piç kurususun lan",  # Toxic
    "Bu armutlar tam salaklar.",      # Toxic
    "Bu arabanın çiziği yok", # Non-Toxic
    "Bitti demediniz mi", # Non-Toxic
]

for sentence in test_sentences:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    print(f"Test cümlesi: {sentence}")
    print(f"Predicted class: {predicted_class} ({class_labels[predicted_class]})\n")
