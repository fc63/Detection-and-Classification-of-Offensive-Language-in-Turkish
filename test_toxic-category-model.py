from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

repo_id = "fc63/toxic-category-model"
model = AutoModelForSequenceClassification.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

class_labels = {
    0: "Insult",
    1: "Other",
    2: "Profanity",
    3: "Racist",
    4: "Sexist"
}
test_sentences = [
    "Bu çok aptalca bir hareketti!",  # Insult
    "Bugün hava oldukça güzel.",      # Other
    "Sen nasıl bir piç kurususun lan",  # Profanity
    "Kürtler gerizekalı.",      # Racist
    "Kadınlar araba kullanmayı beceremez.", # Sexist
]

for sentence in test_sentences:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    print(f"Test cümlesi: {sentence}")
    print(f"Predicted class: {predicted_class} ({class_labels[predicted_class]})\n")
