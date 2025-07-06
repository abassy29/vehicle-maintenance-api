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
from sklearn.metrics import classification_report, confusion_matrix
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
    # 2. Load dataset
    df = pd.read_csv('data/driver_reports_updated.csv')
    X = df['report_text']
    y = df['category']
    y_urg = df['urgency']

    # 3. Split into train/test (stratified on both y and y_urg)
    X_train, X_test, y_train, y_test, y_train_urg, y_test_urg = train_test_split(
        X, y, y_urg,
        test_size=0.2,
        random_state=42,
        stratify=y  # primary stratification on categories
    )

    # 4. TF‑IDF vectorizer with custom preprocessing
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess,
        stop_words=None,
        max_df=0.8,
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # 5. Train category classifier
    cat_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    cat_model.fit(X_train_tfidf, y_train)

    # 6. Train urgency classifier
    urg_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    urg_model.fit(X_train_tfidf, y_train_urg)

    # 7. Evaluate both
    print("\n=== Category Classification Report ===")
    y_pred = cat_model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))

    print("\n=== Category Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred, labels=cat_model.classes_)
    print(pd.DataFrame(cm, index=cat_model.classes_, columns=cat_model.classes_))

    print("\n=== Urgency Classification Report ===")
    y_pred_urg = urg_model.predict(X_test_tfidf)
    print(classification_report(y_test_urg, y_pred_urg))

    # 8. Ensure output directory exists
    os.makedirs('model', exist_ok=True)

    # 9. Save all artifacts
    joblib.dump(vectorizer, 'model/tfidf_vectorizer_fullprep.pkl')
    joblib.dump(cat_model, 'model/incident_classifier_lr_fullprep.pkl')
    joblib.dump(urg_model, 'model/incident_urgency_lr_fullprep.pkl')

    print("\nSaved artifacts to ./model/")

if __name__ == '__main__':
    main()

#pip install pandas scikit-learn joblib nltk
