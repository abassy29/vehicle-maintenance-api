# train_driver_report_full_preprocessing.py

import os
import re
import pandas as pd
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from text_utils import preprocess

# Uncomment and run once to download NLTK data:
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# 1. Prepare preprocessing components
stop_words = set(stopwords.words('english'))
stop_words.update(['car', 'vehicle'])  # domain‑specific
lemmatizer = WordNetLemmatizer()
slang_map = {
    "shakin": "shaking",
    "ride": "car",
    "engin": "engine",
    "knoking": "knocking"
}

# def preprocess(text: str) -> str:
#     """Normalize, clean, tokenize, remove stop‑words, and lemmatize."""
#     text = text.lower()
#     tokens = text.split()
#     tokens = [slang_map.get(tok, tok) for tok in tokens]
#     text = ' '.join(tokens)

#     # Remove punctuation and digits
#     text = re.sub(r'[^a-z\s]', ' ', text)
#     text = re.sub(r'\s+', ' ', text).strip()

#     # Tokenize and lemmatize + remove stop‑words
#     tokens = text.split()
#     tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
#     return ' '.join(tokens)

def main():
    # 1. Load datasets
    df_train = pd.read_csv('data/driver_reports_updated.csv')
    df_test  = pd.read_csv('data/vehicle_issues_test.csv')

    X_train, y_train_cat, y_train_urg = (
        df_train['report_text'],
        df_train['category'],
        df_train['urgency'],
    )
    X_test, y_test_cat, y_test_urg = (
        df_test['report_text'],
        df_test['category'],
        df_test['urgency'],
    )

    # 2. TF‑IDF vectorizer (fit on train only)
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess,
        max_df=0.8,
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # 3. Train category classifier
    cat_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    cat_model.fit(X_train_tfidf, y_train_cat)

    # 4. Train urgency classifier
    urg_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    urg_model.fit(X_train_tfidf, y_train_urg)

    # 5. Evaluate on the TEST set only
    print("\n=== Category Classifier Evaluation (TEST SET) ===")
    y_pred_cat = cat_model.predict(X_test_tfidf)
    print(classification_report(y_test_cat, y_pred_cat))
    print("Confusion Matrix:\n", confusion_matrix(y_test_cat, y_pred_cat))
    print("Accuracy:", accuracy_score(y_test_cat, y_pred_cat))

    print("\n=== Urgency Classifier Evaluation (TEST SET) ===")
    y_pred_urg = urg_model.predict(X_test_tfidf)
    print(classification_report(y_test_urg, y_pred_urg))
    print("Confusion Matrix:\n", confusion_matrix(y_test_urg, y_pred_urg))
    print("Accuracy:", accuracy_score(y_test_urg, y_pred_urg))

    # Show the predictions
    print("Urgency predictions for test set:", list(y_pred_urg))

    # 6. Save artifacts for deployment
    os.makedirs('model', exist_ok=True)
    joblib.dump(vectorizer, 'test/tfidf_vectorizer.pkl')
    joblib.dump(cat_model,   'test/incident_classifier.pkl')
    joblib.dump(urg_model,   'test/incident_urgency_classifier.pkl')
    print("\nSaved artifacts to ./model/")

if __name__ == '__main__':
    main()
