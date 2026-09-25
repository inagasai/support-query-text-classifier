import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = REPO_ROOT / "data" / "questions.csv"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load the data
df = pd.read_csv(DATA_CSV)

# Preprocess the data
def preprocess_text(text):
    text = text.lower()  # convert text to lowercase
    text = re.sub(r'\d+', '', text)  # remove digits
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    text = re.sub(r'\W', ' ', text)  # remove special characters
    return text.strip()

df['Text'] = df['Text'].apply(preprocess_text)

# Split the data into training and testing sets
X = df['Text']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Assets', 'Tickets', 'Invoices'])

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Save the model
model_path = MODELS_DIR / "RandomForestClassifier.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(pipeline, f)

print(f'Model saved to {model_path}')

# Function to classify a new question
def classify_question(model, question):
    question = preprocess_text(question)
    prediction = model.predict([question])
    return prediction[0]

# Load the model and classify a new question
if __name__ == "__main__":
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    while True:
        question = input("Enter a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        label = classify_question(loaded_model, question)
        print(f'The question is classified as: {label}')
