import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump, load  # Import joblib for saving and loading the model
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = REPO_ROOT / "data" / "questions.csv"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load spaCy model
nlp = spacy.load('en_core_web_md')

# Load the CSV file
data = pd.read_csv(DATA_CSV)

# Preprocess text
def preprocess(text):
    return " ".join([token.lemma_ for token in nlp(text) if not token.is_stop])

# Preprocess the questions
data['Preprocessed_Text'] = data['Text'].apply(preprocess)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Preprocessed_Text'], data['Label'], test_size=0.2, random_state=42)

# Create a text classification pipeline using TF-IDF and LinearSVC
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svc', LinearSVC())
])

# Define the hyperparameters to tune
param_grid = {
    'tfidf__max_df': [0.8, 0.9, 1.0],
    'tfidf__min_df': [1, 2, 5],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'svc__C': [0.1, 1, 10, 100],
    'svc__max_iter': [1000, 2000, 3000]
}

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Save the best model
model_path = MODELS_DIR / "spacy_linear_svm_tuned.joblib"
dump(best_model, model_path)

# Load the model (optional, for demonstration)
best_model = load(model_path)

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to classify a new question
def classify_question(question):
    preprocessed_question = preprocess(question)
    return best_model.predict([preprocessed_question])[0]

# Console input loop
print("Enter a question to classify (type 'exit' to quit):")
while True:
    user_input = input("Question: ")
    if user_input.lower() == 'exit':
        break
    classification = classify_question(user_input)
    print(f"Classified as: {classification}\n")
