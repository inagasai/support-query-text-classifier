import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import joblib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = REPO_ROOT / "data" / "questions.csv"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

if __name__ == "__main__":
    df = pd.read_csv(DATA_CSV)
    df['cleaned_question'] = df['Text'].apply(preprocess_text)

    vectorizer = CountVectorizer()

    X = df['cleaned_question']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=512)

    # Uncomment this section if you want to use GridSearchCV with a RandomForestClassifier
    # pipeline = Pipeline([
    #     ('vectorizer', vectorizer),
    #     ('classifier', RandomForestClassifier())
    # ])
    #
    # param_grid = {
    #     'classifier__n_estimators': [100, 200, 300],
    #     'classifier__max_depth': [None, 10, 20, 30],
    #     'classifier__min_samples_split': [2, 5, 10],
    # }
    #
    # grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    # grid_search.fit(X_train, y_train)
    #
    # best_model = grid_search.best_estimator_

    model = make_pipeline(vectorizer, MultinomialNB())
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    for i, val1 in enumerate(list(y_test)):
        val2 = list(y_pred)[i]
        if val2 != val1:
            print(f"{i}. Predicted: {val2}, Actual: {val1}")
            print(f"{list(X_test)[i]}")

    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    joblib.dump(model, MODELS_DIR / "nltk_multinomial_nb.joblib")
