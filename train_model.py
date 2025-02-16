import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler

DATA_DIR = "data"

LABEL_MAP = {"negative": -1, "neutral": 0, "positive": 1}

MODEL_PATH = "models/logistic_regression_stock_sentiment.pk1"
VECTORIZER_PATH = "models/tfidf_vectorizer.pk1"

def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load and preprocess sentiment-labeled text files from the given directory.

    Args:
        data_dir (str): Path to the directory containing labeled text files.

    Returns:
        pd.DataFrame: Processed dataset with "sentence" and "label" columns.
    """
    sentences, labels = [], []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(data_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "@" in line:
                        sentence, sentiment = line.rsplit("@", 1)
                        sentence = re.sub(r"[^\w\s]", "", sentence.strip().lower())  # Normalize text
                        sentiment = sentiment.strip().lower()

                        if sentiment in LABEL_MAP:
                            sentences.append(sentence)
                            labels.append(LABEL_MAP[sentiment])

    if not sentences:
        raise ValueError("No valid data found in the provided directory.")

    return pd.DataFrame({"sentence": sentences, "label": labels})

def train_model(df: pd.DataFrame) -> tuple[LogisticRegression, TfidfVectorizer]:
    """
    Train a logistic regression model for sentiment analysis using TF-IDF features.

    Args:
        df (pd.DataFrame): DataFrame containing "sentence" and "label" columns.

    Returns:
        tuple[LogisticRegression, TfidfVectorizer]: Trained model and vectorizer.
    """
    # Convert text into TF-IDF features
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 3), min_df=2)
    X = vectorizer.fit_transform(df["sentence"])
    y = df["label"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance with under-sampling
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, C=5.0)
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluate model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    return model, vectorizer

def save_model(model: LogisticRegression, vectorizer: TfidfVectorizer, model_path: str, vectorizer_path: str) -> None:
    """
    Save the trained model and vectorizer to disk.

    Args:
        model (LogisticRegression): Trained sentiment analysis model.
        vectorizer (TfidfVectorizer): TF-IDF vectorizer used for feature extraction.
        model_path (str): File path to save the model.
        vectorizer_path (str): File path to save the vectorizer.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Model and vectorizer saved to '{model_path}' and '{vectorizer_path}'.")

if __name__ == "__main__":
    dataset = load_data(DATA_DIR)
    trained_model, trained_vectorizer = train_model(dataset)
    save_model(trained_model, trained_vectorizer, MODEL_PATH, VECTORIZER_PATH)
