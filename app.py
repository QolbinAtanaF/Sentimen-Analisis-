import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords

# =========================
# KONFIGURASI DASHBOARD
# =========================
st.set_page_config(
    page_title="Sentimen Analisis YouTube",
    layout="wide"
)

st.title("📊 Dashboard Sentimen Analisis Komentar YouTube")
st.markdown("""
**Metode:**  
Auto Sentiment Labeling berbasis kamus kata (Lexicon-Based)
""")

st.divider()

# =========================
# NLTK STOPWORDS
# =========================
nltk.download("stopwords")
stop_words = stopwords.words("indonesian")

# =========================
# LOAD KAMUS POSITIF & NEGATIF
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
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

# =========================
# AUTO SENTIMENT LABELING
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
# DASHBOARD UPLOAD FILE
# =========================
st.subheader("📂 Upload File CSV Komentar YouTube")
uploaded_file = st.file_uploader(
    "Pilih file CSV dari komputer",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "textDisplay" not in df.columns:
        st.error("❌ Kolom 'textDisplay' tidak ditemukan")
        st.stop()
    else:
        st.success(f"✅ File berhasil dibaca ({len(df)} komentar)")

        if st.button("🔍 Analisis Sentimen"):
            with st.spinner("Menganalisis komentar..."):
                df["clean_text"] = df["textDisplay"].apply(clean_text)
                df["sentiment"] = df["clean_text"].apply(auto_label)

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