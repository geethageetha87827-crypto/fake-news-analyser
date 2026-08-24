import re
import nltk

for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text)

    # Remove Reuters-style datelines e.g. "WASHINGTON (Reuters) - "
    text = re.sub(r'^[A-Z\s]+\(Reuters\)\s*-\s*', '', text)
    text = re.sub(r'\(Reuters\)', '', text)

    # Remove other common source/agency tags that could leak the label
    text = re.sub(r'\b(Reuters|Associated Press|AP)\b', '', text, flags=re.IGNORECASE)

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)