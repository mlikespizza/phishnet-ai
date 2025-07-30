import os
import email
import pandas as pd
from email import policy
from email.parser import BytesParser
from tqdm import tqdm
import mailbox
import random

# Extract body from individual .eml or text file
def extract_email_body(file_path):
    try:
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        return msg.get_body(preferencelist=('plain')).get_content()
    except Exception:
        return ""

# Extract body from an mbox message object
def extract_mbox_body(msg):
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode(errors="ignore")
        else:
            return msg.get_payload(decode=True).decode(errors="ignore")
    except Exception:
        return ""

# Load .eml/.txt files or plain files from Enron
def load_emails_from_dir(root_dir, label, max_emails=None):
    email_texts = []
    count = 0
    for root, _, files in os.walk(root_dir):
        for file in tqdm(files, desc="📥 Loading ham"):
            if max_emails and count >= max_emails:
                return email_texts
            path = os.path.join(root, file)
            if os.path.isfile(path):
                body = extract_email_body(path)
                if body and body.strip():
                    email_texts.append((body, label))
                    count += 1
    return email_texts

# Load phishing emails from mbox
def load_emails_from_mbox(mbox_path, label):
    email_texts = []
    mbox = mailbox.mbox(mbox_path)
    for msg in mbox:
        body = extract_mbox_body(msg)
        if body and body.strip():
            email_texts.append((body, label))
    return email_texts

if __name__ == "__main__":
    print("🔍 Preprocessing emails...")

    phishing_path = "data/phishing_raw/phishing-2024.mbox"
    enron_path = "data/enron_raw"

    emails = []

    # Load phishing emails
    if os.path.exists(phishing_path):
        print(f"📁 Loading phishing emails from {phishing_path}")
        phishing_emails = load_emails_from_mbox(phishing_path, label=1)
        emails += phishing_emails
        print(f"🐟 Total phishing emails: {len(phishing_emails)}")
    else:
        phishing_emails = []
        print(f"⚠️ MBOX not found at {phishing_path}")

    # Load matching number of ham emails
    if os.path.exists(enron_path):
        print(f"📁 Loading Enron emails from {enron_path}")
        max_ham = len(phishing_emails)
        ham_emails = load_emails_from_dir(enron_path, label=0, max_emails=max_ham)

        if len(ham_emails) < max_ham:
            print(f"📦 Found {len(ham_emails)} ham emails")
            print("⚠️ Not enough ham to match phishing count — using all ham")
        else:
            print(f"📦 Sampled {len(ham_emails)} ham emails")

        emails += ham_emails
    else:
        print(f"❌ Enron directory not found at {enron_path}")

    df = pd.DataFrame(emails, columns=["text", "label"])
    print(f"📊 Final dataset: {len(df)} emails (Balanced 1:1)")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/emails.csv", index=False)
    print("💾 Saved to data/processed/emails.csv")
