# scripts/batch_predict.py

import os
import joblib
import argparse

# Load model + vectorizer once
model = joblib.load("models/logreg_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def load_email(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def predict_email(text):
    """
    Predict a single email's label and confidence.
    """
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0][prediction]
    return prediction, confidence

def batch_predict(folder_path):
    """
    Scans a folder of email text files and returns a list of predictions.
    Returns: List of dicts with filename, prediction, confidence
    """
    results = []
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for fname in files:
        path = os.path.join(folder_path, fname)
        content = load_email(path)
        label, conf = predict_email(content)

        results.append({
            "filename": fname,
            "prediction": label,
            "confidence": conf
        })

    return results

# CLI usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to folder containing email files")
    args = parser.parse_args()

    results = batch_predict(args.folder)

    print(f"\n🔎 Scanning {len(results)} emails...\n")
    for r in results:
        print(f"📧 {r['filename']}")
        print(f"→ Prediction: {'Phishing 🛑' if r['prediction'] == 1 else 'Ham ✅'}")
        print(f"→ Confidence: {r['confidence'] * 100:.2f}%\n")
