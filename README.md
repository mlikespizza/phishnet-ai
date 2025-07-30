

## PhishNet AI

**PhishNet AI** is an email phishing detection tool that uses machine learning to classify emails as either *phishing* or *legitimate (ham)*. The app includes both single and batch email prediction features, and a simple, tech-inspired dark UI built with Streamlit.

### Features

*  Detect phishing emails using logistic regression and TF-IDF
*  Upload single emails or batch `.txt` files for analysis
*  Visualize prediction confidence with bar charts
*  Download scan results as a `.csv` file
*  Dark mode minimalist interface
*  Built with Python, scikit-learn, Streamlit


---

### How It Works

1. Preprocesses and vectorizes email text using TF-IDF
2. Applies a trained logistic regression model
3. Outputs:

   * Prediction label: Ham or Phishing
   * Confidence score
   * Visuals and downloadable results

---

### Run Locally

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/phishnet-ai.git
cd phishnet-ai
```

#### 2. Install Requirements

```bash
pip install -r requirements.txt
```

If you get a `ModuleNotFoundError`, install missing packages individually (e.g., `pip install matplotlib`).

#### 3. Run the App

```bash
streamlit run app.py
```

---

### Project Structure

```
phishnet-ai/
│
├── models/               # Trained model & TF-IDF vectorizer
├── data/to_scan/         # Folder for batch prediction email .txt files
├── scripts/
│   ├── predict.py        # Single email prediction via CLI
│   └── batch_predict.py  # Folder-based batch predictions via CLI
├── app.py                # Streamlit frontend
├── requirements.txt
└── README.md
```

---

### Future Plans

* Switch to a Flask-based UI for production deployment
* Improve model accuracy with more training data
* Integrate real-time email scanning (IMAP/Gmail API)
* Deploy online for public use

---

### License

MIT License. Feel free to use, modify, or contribute.

