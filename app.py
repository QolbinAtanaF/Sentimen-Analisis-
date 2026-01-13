import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import os
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# =========================
# SETUP
# =========================
nltk.download("stopwords")
stop_words = stopwords.words("indonesian")

st.set_page_config(page_title="Sentimen Analisis YouTube", layout="wide")
st.title("📊 Dashboard Sentimen Analisis Komentar YouTube (ML Auto)")
st.markdown("""
**Metode:** Lexicon-Based + TF-IDF + SVM  
**Preprocessing:** Case Folding, Cleaning, Stopword Removal
""")
st.divider()

# =========================
# LOAD KAMUS
# =========================
def load_lexicon(file):
    with open(file, "r", encoding="utf-8") as f:
        return set(f.read().splitlines())

positive_words = load_lexicon("kamus_positif.txt")
negative_words = load_lexicon("kamus_negatif.txt")

# =========================
# PREPROCESSING
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# =========================
# AUTO LABEL (LEXICON)
# =========================
def auto_label(text):
    score = 0
    for word in text.split():
        if word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1
    if score > 0:
        return "positif"
    elif score < 0:
        return "negatif"
    else:
        return "netral"

# =========================
# MODEL FILE
# =========================
MODEL_FILE = "svm_model.pkl"
VECT_FILE = "tfidf.pkl"

# =========================
# TRAIN MODEL
# =========================
def train_model(df):
    df["clean_text"] = df["textDisplay"].apply(clean_text)
    df["sentiment"] = df["clean_text"].apply(auto_label)

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df["clean_text"])
    y = df["sentiment"]

    svm_model = SVC(kernel="linear")
    svm_model.fit(X, y)

    with open(VECT_FILE, "wb") as f:
        pickle.dump(tfidf, f)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(svm_model, f)

    return tfidf, svm_model

# =========================
# UPLOAD CSV
# =========================
st.subheader("📂 Upload File CSV Komentar YouTube")
uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "textDisplay" not in df.columns:
        st.error("❌ Kolom 'textDisplay' tidak ditemukan")
        st.stop()

    st.success(f"✅ File berhasil dibaca ({len(df)} komentar)")

    if st.button("🔍 Analisis Sentimen"):
        with st.spinner("Menganalisis komentar..."):
            if not os.path.exists(MODEL_FILE):
                tfidf, svm_model = train_model(df)
            else:
                with open(VECT_FILE, "rb") as f:
                    tfidf = pickle.load(f)
                with open(MODEL_FILE, "rb") as f:
                    svm_model = pickle.load(f)

            df["clean_text"] = df["textDisplay"].apply(clean_text)
            X_new = tfidf.transform(df["clean_text"])
            df["sentiment"] = svm_model.predict(X_new)

        st.success("🎉 Analisis selesai")

        st.subheader("📋 Hasil Analisis")
        st.dataframe(df[["authorDisplayName", "textDisplay", "sentiment"]])

        st.subheader("🥧 Distribusi Sentimen (%)")
        sent_ratio = df["sentiment"].value_counts(normalize=True) * 100

        fig, ax = plt.subplots()
        ax.pie(sent_ratio, labels=sent_ratio.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

        st.download_button(
            "⬇ Download Hasil CSV",
            df.to_csv(index=False),
            "hasil_sentimen.csv",
            "text/csv"
        )
