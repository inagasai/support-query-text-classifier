import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load  # For saving and loading model
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = REPO_ROOT / "data" / "questions.csv"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load the CSV file
data = pd.read_csv(DATA_CSV)

# Tokenizer parameters
max_words = 10000
max_len = 100

# Initialize and fit tokenizer
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data['Text'])
sequences = tokenizer.texts_to_sequences(data['Text'])
X = pad_sequences(sequences, maxlen=max_len)

# Convert labels to categorical format
y = pd.get_dummies(data['Label']).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
model.add(LSTM(64, return_sequences=True))
model.add(GlobalAveragePooling1D())
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Save the model
model_path = MODELS_DIR / "lstm_text_classification.h5"
model.save(model_path)

# Load the model (optional, for demonstration)
from tensorflow.keras.models import load_model
model = load_model(model_path)

# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print("Accuracy:", accuracy_score(y_test_labels, y_pred))
print("Classification Report:\n", classification_report(y_test_labels, y_pred))

# Function to classify a new question
def classify_question(question):
    sequence = tokenizer.texts_to_sequences([question])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    return np.argmax(prediction)

# Console input loop
print("Enter a question to classify (type 'exit' to quit):")
while True:
    user_input = input("Question: ")
    if user_input.lower() == 'exit':
        break
    classification_index = classify_question(user_input)
    classification = data['Label'].unique()[classification_index]
    print(f"Classified as: {classification}\n")
