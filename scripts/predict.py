# scripts/predict.py

import joblib
import sys

# Load model and vectorizer only once
model = joblib.load("models/logreg_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def predict_email(email_text: str):
    """
    Predicts whether an email is phishing or ham.
    Returns: (label, confidence)
    """
    features = vectorizer.transform([email_text])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction]
    return prediction, probability

# CLI usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py \"Your email content here...\"")
        sys.exit(1)

    email_input = sys.argv[1]
    label, confidence = predict_email(email_input)

    print("\n Email Prediction:")
    print(f"→ Prediction: {'Phishing 🛑' if label == 1 else 'Ham ✅'}")
    print(f"→ Confidence: {confidence * 100:.2f}%")
