import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    # Case folding
    text = text.lower()

    # Hapus karakter selain huruf
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenizing
    tokens = word_tokenize(text)

    # Stopword removal
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)
