import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load  # Import joblib for saving and loading the model
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = REPO_ROOT / "data" / "questions.csv"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load spaCy model
# nlp = spacy.load('en_core_web_sm') --93%
# nlp = spacy.load('en_core_web_md') --966667%
# nlp = spacy.load('en_core_web_lg') --93%
# nlp = spacy.load('en_core_web_trf')--96667%

nlp = spacy.load('en_core_web_md')

# Load the CSV file
data = pd.read_csv(DATA_CSV)

# Preprocess text
def preprocess(text):
    return " ".join([token.lemma_ for token in nlp(text) if not token.is_stop])

# Preprocess the questions
data['Preprocessed_Text'] = data['Text'].apply(preprocess)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Preprocessed_Text'], data['Label'], test_size=0.1, random_state=42)

# Create a text classification pipeline using TF-IDF and LinearSVC
model = make_pipeline(TfidfVectorizer(), LinearSVC())
model.fit(X_train, y_train)

# Save the model
model_path = MODELS_DIR / "spacy_linear_svm.joblib"
dump(model, model_path)

# Load the model (optional, for demonstration)
model = load(model_path)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Function to classify a new question
def classify_question(question):
    preprocessed_question = preprocess(question)
    return model.predict([preprocessed_question])[0]

# Console input loop
print("Enter a question to classify (type 'exit' to quit):")
while True:
    user_input = input("Question: ")
    if user_input.lower() == 'exit':
        break
    classification = classify_question(user_input)
    print(f"Classified as: {classification}\n")
