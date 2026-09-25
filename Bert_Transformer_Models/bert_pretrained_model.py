import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = REPO_ROOT / "data" / "questions.csv"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
data = pd.read_csv(DATA_CSV)

# Encode labels
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])

# Split data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(data['Text'], data['Label'], test_size=0.2, random_state=42)

# Tokenize data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

train_encodings = tokenize(train_texts.tolist())
test_encodings = tokenize(test_texts.tolist())

# Create PyTorch Dataset
class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextClassificationDataset(train_encodings, train_labels.tolist())
test_dataset = TextClassificationDataset(test_encodings, test_labels.tolist())

# Define model and training parameters
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data['Label'].unique()))

training_args = TrainingArguments(
    output_dir=str(MODELS_DIR / "bert-results"),
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=str(MODELS_DIR / "bert-logs"),
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# Classify input questions
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([predicted_class_id])[0]

# Interactive input
while True:
    question = input("Enter your question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    label = classify_text(question)
    print(f"The question is classified as: {label}")
