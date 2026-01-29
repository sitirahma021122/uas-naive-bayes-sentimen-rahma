import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing_rahma import clean_text

def proses_training():
    # Load dataset
    data = pd.read_csv("dataset_sentimen.csv")

    # Preprocessing
    data['clean_review'] = data['review_text'].apply(clean_text)

    X = data['clean_review']
    y = data['sentiment']

    # Split data 80:20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # TF-IDF
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Model Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    # Prediksi
    y_pred = nb_model.predict(X_test_tfidf)

    # Evaluasi
    print("=== HASIL EVALUASI MODEL ===")
    print(classification_report(y_test, y_pred))
    print("=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))

    return nb_model, tfidf
