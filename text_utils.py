import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
stop_words.update(["car", "vehicle"])

lemmatizer = WordNetLemmatizer()

slang_map = {
    "shakin": "shaking",
    "ride": "car",
    "engin": "engine",
    "knoking": "knocking"
}

def preprocess(text: str) -> str:
    text = text.lower()
    tokens = text.split()
    tokens = [slang_map.get(tok, tok) for tok in tokens]
    text = ' '.join(tokens)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return ' '.join(tokens)
