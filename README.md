# support-query-text-classifier

Train and compare text classifiers for **support queries** labeled as **Assets**, **Tickets**, or **Invoices**.

This repo holds standalone training scripts (not a product template): classical ML, spaCy + linear SVM, LSTM, and BERT fine-tuning on the same dataset.

## Dataset

- **Path:** `data/questions.csv`
- **Columns:** `Text`, `Label`
- **Labels:** `Assets`, `Tickets`, `Invoices`
- **Note:** Example questions use synthetic names, identifiers, and ticket numbers for public sharing.

## Project layout

| Directory | Approach |
|-----------|----------|
| `Machine_learning_Models/` | scikit-learn (logistic regression, random forest, NLTK + sklearn) |
| `Spacy_Models/` | spaCy preprocessing + TF-IDF + LinearSVC |
| `Deep_Learning_Models/` | Keras LSTM |
| `Bert_Transformer_Models/` | Hugging Face `bert-base-uncased` |

Trained artifacts are written under `models/` (ignored by git).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

NLTK scripts download `stopwords`, `wordnet`, and `punkt` on first run.

## Run a script

From the repository root:

```bash
python Machine_learning_Models/logistic_regression_model.py
python Spacy_Models/spacy_linearSVM_model.py
python Deep_Learning_Models/LSTM_model.py
python Bert_Transformer_Models/bert_pretrained_model.py
```

Each script loads `data/questions.csv` via paths relative to the repo root.

## Requirements

See `requirements.txt`. Versions are pinned from the environment used during development; adjust for your Python/CUDA setup if needed.
