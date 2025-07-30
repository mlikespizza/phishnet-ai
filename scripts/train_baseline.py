import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Load data
df = pd.read_csv("data/processed/emails.csv")

# Basic preprocessing (optional: you can add more here)
df["text"] = df["text"].fillna("").str.lower()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/logreg_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("💾 Model and vectorizer saved to models/")
