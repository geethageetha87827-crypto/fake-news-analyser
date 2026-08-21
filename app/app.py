import streamlit as st
import joblib
import sys
import os
import numpy as np

def get_top_contributing_words(text_vector, vectorizer, model, top_n=5):
    """Returns top words pushing toward Fake and toward Real."""
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]  # PassiveAggressiveClassifier has one coef row for binary classification

    # Get indices of non-zero features in this specific input
    nonzero_indices = text_vector.nonzero()[1]

    # Compute each word's contribution = tfidf_value * model_weight
    contributions = []
    for idx in nonzero_indices:
        word = feature_names[idx]
        tfidf_val = text_vector[0, idx]
        weight = coefs[idx]
        contribution = tfidf_val * weight
        contributions.append((word, contribution))

    contributions.sort(key=lambda x: x[1], reverse=True)

    top_real = [w for w, c in contributions if c > 0][:top_n]
    top_fake = [w for w, c in contributions if c < 0][:top_n]

    return top_fake, top_real

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

                # Reliability tier based on confidence
        if confidence_pct >= 80:
            reliability = "High"
        elif confidence_pct >= 60:
            reliability = "Moderate"
        else:
            reliability = "Low"

        if prediction == 1:
            st.success(f"### ✅ Likely Reliable")
        else:
            st.error(f"### 🚫 Likely Misleading")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(label="Confidence", value=f"{confidence_pct}%")
        with col_b:
            st.metric(label="Reliability", value=reliability)

        st.progress(confidence)

        # Explainability section
        top_fake_words, top_real_words = get_top_contributing_words(vec, vectorizer, model)

        st.markdown("#### Why this prediction?")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Words suggesting Fake:**")
            if top_fake_words:
                st.write(", ".join(top_fake_words))
            else:
                st.write("None detected")
        with col2:
            st.markdown("**Words suggesting Real:**")
            if top_real_words:
                st.write(", ".join(top_real_words))
            else:
                st.write("None detected")

        st.caption("⚠️ This is an AI-assisted assessment based on text patterns, not verified proof of truth or falsehood.")

        with st.expander("See cleaned text used by the model"):
            st.write(cleaned)