from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

repo_id = "fc63/turkish-toxic-language-detection"

model = AutoModelForSequenceClassification.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

class_labels = {
    0: "Non-Toxic",
    1: "Toxic"
}

test_sentences = [
    "Bu çok aptalca bir hareketti!",  # Toxic
    "Bugün hava oldukça güzel.",      # Non-Toxic
    "Sen nasıl bir piç kurususun lan",  # Toxic
    "Kürtler gerizekalı.",      # Toxic
    "elinde divit kalem, katlime ferman yazar. ne anlıyorsun?", # Non-Toxic
    "Bitti demediniz mi", # Non-Toxic
]

for sentence in test_sentences:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    print(f"Test sentence: {sentence}")
    print(f"Predicted class: {predicted_class} ({class_labels[predicted_class]})\n")
