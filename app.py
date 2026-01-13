import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# =========================
# DASHBOARD
# =========================
st.set_page_config(page_title="Sentimen Analisis YouTube", layout="wide")
st.title("📊 Dashboard Sentimen Analisis Komentar YouTube (ML Auto)")
st.markdown("""
**Metode:** Lexicon-Based + TF-IDF + SVM (otomatis training jika model belum ada)  
**Preprocessing:** Case Folding, Cleaning, Tokenizing & Stopword Removal, Stemming
""")
st.divider()

# =========================
# DOWNLOAD STOPWORDS
# =========================
nltk.download("stopwords")
stop_words = stopwords.words("indonesian")

# =========================
# STEMMER (SASTRAWI)
# =========================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# =========================
# LOAD KAMUS POSITIF & NEGATIF
# =========================
def load_lexicon(file):
    with open(file, "r", encoding="utf-8") as f:
        return set(f.read().splitlines())

positive_words = load_lexicon("kamus_positif.txt")
negative_words = load_lexicon("kamus_negatif.txt")

# =========================
# PREPROCESSING LENGKAP
# =========================
def clean_text(text):
    # 1. Case Folding
    text = str(text).lower()
    # 2. Cleaning: hapus link, angka, tanda baca, emoji
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    # 3. Tokenizing & Stopword Removal
    tokens = [w for w in text.split() if w not in stop_words]
    # 4. Stemming
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# =========================
# AUTO LABELING (LEXICON)
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
# FILE MODEL
# =========================
MODEL_FILE = "svm_model.pkl"
VECT_FILE = "tfidf.pkl"

# =========================
# TRAIN MODEL (TF-IDF + SVM)
# =========================
def train_model(df):
    # Preprocessing teks
    df["clean_text"] = df["textDisplay"].apply(clean_text)
    # Auto-label lexicon untuk training
    df["sentiment"] = df["clean_text"].apply(auto_label)

    # TF-IDF
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df["clean_text"])
    y = df["sentiment"]

    # SVM training
    svm_model = SVC(kernel="linear", probability=True)
    svm_model.fit(X, y)

    # Simpan model
    with open(VECT_FILE, "wb") as f:
        pickle.dump(tfidf, f)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(svm_model, f)

    return tfidf, svm_model

# =========================
# UPLOAD CSV
# =========================
st.subheader("📂 Upload File CSV Komentar YouTube")
uploaded_file = st.file_uploader("Pilih file CSV dari komputer", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "textDisplay" not in df.columns:
        st.error("❌ Kolom 'textDisplay' tidak ditemukan")
        st.stop()
    else:
        st.success(f"✅ File berhasil dibaca ({len(df)} komentar)")

        if st.button("🔍 Analisis Sentimen"):
            with st.spinner("Menganalisis komentar..."):
                # TRAIN MODEL JIKA BELUM ADA
                if not os.path.exists(MODEL_FILE) or not os.path.exists(VECT_FILE):
                    st.info("Model belum ada, melakukan training otomatis dari lexicon...")
                    tfidf, svm_model = train_model(df)
                    st.success("✅ Model berhasil dibuat dan disimpan")
                else:
                    with open(VECT_FILE, "rb") as f:
                        tfidf = pickle.load(f)
                    with open(MODEL_FILE, "rb") as f:
                        svm_model = pickle.load(f)

                # PREDIKSI
                df["clean_text"] = df["textDisplay"].apply(clean_text)
                X_new = tfidf.transform(df["clean_text"])
                df["sentiment"] = svm_model.predict(X_new)

            st.success("🎉 Analisis selesai")

            # =========================
            # TABEL HASIL
            # =========================
            st.subheader("📋 Hasil Analisis Sentimen")
            st.dataframe(
                df[["authorDisplayName", "textDisplay", "sentiment"]],
                use_container_width=True
            )

            # =========================
            # PIE CHART
            # =========================
            st.subheader("🥧 Distribusi Sentimen (%)")
            sent_ratio = df["sentiment"].value_counts(normalize=True) * 100
            sent_ratio = sent_ratio.round(2)

            fig, ax = plt.subplots()
            ax.pie(sent_ratio, labels=sent_ratio.index, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

            # =========================
            # METRIC BOX
            # =========================
            c1, c2, c3 = st.columns(3)
            c1.metric("Positif (%)", sent_ratio.get("positif", 0))
            c2.metric("Negatif (%)", sent_ratio.get("negatif", 0))
            c3.metric("Netral (%)", sent_ratio.get("netral", 0))

            # =========================
            # DOWNLOAD HASIL CSV
            # =========================
            st.download_button(
                label="⬇ Download Hasil CSV",
                data=df.to_csv(index=False),
                file_name="hasil_sentimen.csv",
                mime="text/csv"
            )
