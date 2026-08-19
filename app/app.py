import streamlit as st
import joblib
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from clean_text import clean_text

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'fake_news_model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'tfidf_vectorizer.pkl'))
st.set_page_config(page_title="Fake News Analyser", page_icon="📰", layout="wide")

# --- Custom styling ---
st.markdown("""
    <div style="text-align: center; margin-bottom: 10px;">
        <span style="font-size: 80px;">📰</span>
        <span style="
            font-size: 80px;
            font-weight: 900;
            background: linear-gradient(90deg, #FF4B4B, #7C3AED);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">Fake News Analyser</span>
    </div>
""", unsafe_allow_html=True)

st.markdown(
    '<p style="text-align: center; color: #9CA3AF; font-size: 18px; margin-bottom: 35px;">'
    'Paste a news article or headline to check if it looks Real or Fake</p>',
    unsafe_allow_html=True
)

user_input = st.text_area("News text", height=220, placeholder="Paste the article or headline here...", label_visibility="collapsed")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_clicked = st.button("🔍 Analyze", use_container_width=True)

if analyze_clicked:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]

            raw_score = model.decision_function(vec)[0]
            confidence = 1 / (1 + np.exp(-abs(raw_score)))
            confidence_pct = round(confidence * 100, 1)

        st.divider()

        if prediction == 1:
            st.success(f"### ✅ This looks like REAL news")
        else:
            st.error(f"### 🚫 This looks like FAKE news")

        st.metric(label="Model Confidence", value=f"{confidence_pct}%")
        st.progress(confidence)

        with st.expander("See cleaned text used by the model"):
            st.write(cleaned)