import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin


# ── CUSTOM TRANSFORMER (Modelin yüklenmesi için şart) ────────────────────────
class AgeGroupTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bins = [17, 25, 35, 50, np.inf]
        self.labels = ['Young', 'Early_Career', 'Middle_Age', 'Senior']

    def fit(self, X, y=None): return self

    def transform(self, X):
        X_copy = X.copy()
        # ColumnTransformer tek bir sütun gönderdiği için iloc[:, 0] güvenlidir
        X_binned = pd.cut(X_copy.iloc[:, 0], bins=self.bins, labels=self.labels)
        return pd.DataFrame(X_binned)


# ── MODEL YÜKLEME ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = 'best_credit_model.pkl'
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Model yüklenirken bir hata oluştu: {e}")
            return None
    else:
        st.error("Hata: 'best_credit_model.pkl' dosyası bulunamadı. Lütfen notebook üzerinden modeli kaydedin.")
        return None


model = load_model()

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSight · Credit Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── GLOBAL CSS (Senin harika tasarımın korunmuştur) ──────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e4dc;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(255,200,80,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(80,180,255,0.05) 0%, transparent 55%),
        #0a0a0f;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 3rem 4rem 5rem; max-width: 1200px; }

h1, h2, h3 { font-family: 'DM Serif Display', serif; font-weight: 400; }

.hero {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    padding: 3.5rem 0 3rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 3rem;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.25em;
    color: #f5c842;
    text-transform: uppercase;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.2rem, 5vw, 4.2rem);
    line-height: 1.05;
    color: #f0ece4;
    letter-spacing: -0.02em;
}
.hero-title em { font-style: italic; color: #f5c842; }

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    margin-bottom: 1.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.stButton > button {
    background: #f5c842 !important;
    color: #0a0a0f !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    padding: 0.85rem 2.4rem !important;
    box-shadow: 0 4px 20px rgba(245,200,66,0.25) !important;
}

.result-card {
    border-radius: 16px;
    padding: 2.8rem 3rem;
    margin-top: 2.5rem;
    position: relative;
    border: 1px solid rgba(255,255,255,0.08);
}
.result-approved { background: rgba(34, 197, 94, 0.08); border-color: #22c55e44; }
.result-rejected { background: rgba(239, 68, 68, 0.08); border-color: #ef444444; }

.prob-bar-track {
    height: 8px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    margin: 1rem 0;
}
.prob-bar-fill-approved { height: 100%; border-radius: 99px; background: #22c55e; }
.prob-bar-fill-rejected { height: 100%; border-radius: 99px; background: #ef4444; }
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">◈ LoanSight · Credit Intelligence System</div>
    <h1 class="hero-title">Should this loan<br>be <em>approved?</em></h1>
</div>
""", unsafe_allow_html=True)

# ── INPUTS ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">01 · Applicant Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income ($)", min_value=1, max_value=1_000_000, value=50000, step=1000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=1, max_value=1_000_000, value=15000, step=500)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    experience = st.number_input("Work Experience (Years)", 0, 50, 5)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])
    city = st.selectbox("City", ["New York", "Los Angeles", "Chicago", "Houston", "San Francisco"])
    employment = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-Employed", "Unemployed"])

st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)

# ── RUN ANALYSIS ──────────────────────────────────────────────────────────────
if st.button("Run Analysis →"):
    if model is not None:
        # SÜTUN SIRALAMASI: Notebook'taki X_train ile birebir aynı olmalı
        input_df = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Income': [income],
            'Education': [education],
            'EmploymentType': [employment],
            'LoanAmount': [loan_amount],
            'CreditScore': [credit_score],
            'City': [city],
            'YearsExperience': [experience]
        })

        # Tahmin
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        prob_pct = round(prob * 100, 1)

        # Karar Görselleştirme
        is_approved = prediction == 1
        card_style = "result-approved" if is_approved else "result-rejected"
        verdict = "Approved" if is_approved else "Rejected"
        color = "#4ade80" if is_approved else "#f87171"
        bar_style = "prob-bar-fill-approved" if is_approved else "prob-bar-fill-rejected"

        # LTI (Loan to Income) Hesaplama
        lti_ratio = round(loan_amount / income, 2)

        st.markdown(f"""
        <div class="result-card {card_style}">
            <div style="font-family: 'DM Serif Display'; font-size: 2.5rem; color: {color};">
                {"✦" if is_approved else "✕"} {verdict}
            </div>
            <div style="font-family: 'DM Mono'; font-size: 0.8rem; opacity: 0.6; margin-bottom: 2rem;">
                {"Müşteri kredi kriterlerini karşılıyor." if is_approved else "Başvuru yüksek risk taşımaktadır."}
            </div>

            <div style="font-family: 'DM Mono'; font-size: 0.7rem; text-transform: uppercase; opacity: 0.5;">Approval Probability</div>
            <div style="font-family: 'DM Serif Display'; font-size: 3rem;">{prob_pct}%</div>

            <div class="prob-bar-track">
                <div class="{bar_style}" style="width: {prob_pct}%;"></div>
            </div>

            <div style="display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.05); padding: 5px 12px; border-radius: 20px; font-family: 'DM Mono'; font-size: 0.7rem;">LTI Ratio: {lti_ratio}</span>
                <span style="background: rgba(255,255,255,0.05); padding: 5px 12px; border-radius: 20px; font-family: 'DM Mono'; font-size: 0.7rem;">Credit Score: {credit_score}</span>
                <span style="background: rgba(255,255,255,0.05); padding: 5px 12px; border-radius: 20px; font-family: 'DM Mono'; font-size: 0.7rem;">Exp: {experience}y</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Model yüklenemediği için analiz yapılamıyor.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top: 5rem; padding: 2rem 0; border-top: 1px solid rgba(255,255,255,0.05); font-family: 'DM Mono'; font-size: 0.6rem; opacity: 0.3; display: flex; justify-content: space-between;">
    <span>◈ LoanSight Intelligence Engine v1.0</span>
    <span>Powered by LightGBM & Streamlit</span>
</div>
""", unsafe_allow_html=True)