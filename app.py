import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# Load model and vectorizer
model = joblib.load("models/logreg_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Set dark mode aesthetic
st.set_page_config(
    page_title="PhishNet AI",
    layout="wide",
    page_icon="🛡️"
)
st.markdown("""
    <style>
    body {
        background-color: #0f1117;
        color: #f0f2f6;
    }
    .stApp {
        background-color: #0f1117;
    }
    </style>
""", unsafe_allow_html=True)

# Predict a single email
def predict_email(email_text):
    X = vectorizer.transform([email_text])
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0][prediction]
    return prediction, confidence

# Predict multiple emails and return dataframe
def batch_predict(files):
    results = []

    for file in files:
        content = file.read().decode("utf-8", errors="ignore")
        pred, conf = predict_email(content)
        results.append({
            "File Name": file.name,
            "Prediction": "Phishing 🛑" if pred == 1 else "Ham ✅",
            "Confidence (%)": round(conf * 100, 2)
        })

    return pd.DataFrame(results)

# Bar chart function
def show_confidence_chart(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["File Name"], df["Confidence (%)"], color="#0ea5e9")
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Confidence Levels per Email")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# Streamlit UI
st.title("🛡️ PhishNet AI - Email Phishing Detection")

tabs = st.tabs(["📧 Single Email", "📂 Batch Upload"])

# Single Prediction Tab
with tabs[0]:
    st.subheader("Predict a Single Email")
    email_input = st.text_area("Enter email content below", height=200)

    if st.button("Predict"):
        if email_input.strip():
            label, conf = predict_email(email_input)
            st.markdown(f"**Prediction:** {'🛑 Phishing' if label == 1 else '✅ Ham'}")
            st.markdown(f"**Confidence:** `{conf * 100:.2f}%`")
        else:
            st.warning("Please enter email content.")

# Batch Upload Tab
with tabs[1]:
    st.subheader("Upload .txt Files for Batch Prediction")
    uploaded_files = st.file_uploader(
        "Drop or select multiple .txt files",
        type=["txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        df = batch_predict(uploaded_files)
        st.success(f"✅ Scanned {len(df)} file(s)")
        st.dataframe(df, use_container_width=True)

        # Confidence bar chart
        st.subheader("📊 Confidence Chart")
        show_confidence_chart(df)

        # Download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name="phishnet_batch_results.csv",
            mime="text/csv"
        )
