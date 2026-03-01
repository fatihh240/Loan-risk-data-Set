import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin


# ── CUSTOM TRANSFORMER ────────────────────────────────────────────────────────
class AgeGroupTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bins = [17, 25, 35, 50, np.inf]
        self.labels = ['Young', 'Early_Career', 'Middle_Age', 'Senior']

    def fit(self, X, y=None): return self

    def transform(self, X):
        X_copy = X.copy()
        X_binned = pd.cut(X_copy.iloc[:, 0], bins=self.bins, labels=self.labels)
        return pd.DataFrame(X_binned)


# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = 'best_credit_model.pkl'
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Model yuklenirken hata: {e}")
            return None
    else:
        st.error("'best_credit_model.pkl' bulunamadi.")
        return None


model = load_model()

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSight · Credit Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
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

.hero { display: flex; flex-direction: column; gap: 0.4rem; padding: 3.5rem 0 3rem; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 3rem; }
.hero-eyebrow { font-family: 'DM Mono', monospace; font-size: 0.68rem; letter-spacing: 0.25em; color: #f5c842; text-transform: uppercase; }
.hero-title { font-family: 'DM Serif Display', serif; font-size: clamp(2.2rem, 5vw, 4.2rem); line-height: 1.05; color: #f0ece4; letter-spacing: -0.02em; }
.hero-title em { font-style: italic; color: #f5c842; }

.section-label { font-family: 'DM Mono', monospace; font-size: 0.62rem; letter-spacing: 0.2em; text-transform: uppercase; color: rgba(255,255,255,0.3); margin-bottom: 1.2rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.06); }

.stButton > button { background: #f5c842 !important; color: #0a0a0f !important; border: none !important; border-radius: 8px !important; font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; padding: 0.85rem 2.4rem !important; box-shadow: 0 4px 20px rgba(245,200,66,0.25) !important; }
.stButton > button:hover { background: #ffd84a !important; transform: translateY(-1px) !important; }

.result-card { border-radius: 16px; padding: 2.8rem 3rem; margin-top: 2.5rem; border: 1px solid rgba(255,255,255,0.08); }
.result-approved { background: rgba(34,197,94,0.08); border-color: rgba(34,197,94,0.27); }
.result-rejected { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.27); }

.verdict { font-family: 'DM Serif Display', serif; font-size: 2.5rem; line-height: 1.1; margin-bottom: 0.4rem; }
.verdict-approved { color: #4ade80; }
.verdict-rejected { color: #f87171; }

.verdict-sub { font-family: 'DM Mono', monospace; font-size: 0.78rem; opacity: 0.5; margin-bottom: 2rem; letter-spacing: 0.05em; }

.prob-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.15em; opacity: 0.45; margin-bottom: 0.4rem; }
.prob-value-approved { font-family: 'DM Serif Display', serif; font-size: 3rem; color: #4ade80; line-height: 1; }
.prob-value-rejected { font-family: 'DM Serif Display', serif; font-size: 3rem; color: #f87171; line-height: 1; }

.prob-bar-track { height: 8px; background: rgba(255,255,255,0.07); border-radius: 99px; margin: 1rem 0 1.5rem; overflow: hidden; }
.prob-bar-approved { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #22c55e, #4ade80); }
.prob-bar-rejected { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #ef4444, #f87171); }

.chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1.5rem; }
.chip { font-family: 'DM Mono', monospace; font-size: 0.68rem; padding: 0.3rem 0.75rem; border-radius: 99px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); color: rgba(232,228,220,0.6); }

.app-footer { margin-top: 5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.05); font-family: 'DM Mono', monospace; font-size: 0.6rem; opacity: 0.25; display: flex; justify-content: space-between; }
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">&#9672; LoanSight &middot; Credit Intelligence System</div>
    <h1 class="hero-title">Should this loan<br>be <em>approved?</em></h1>
</div>
""", unsafe_allow_html=True)

# ── INPUTS ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">01 &middot; Applicant Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    age          = st.number_input("Age", min_value=18, max_value=100, value=30)
    income       = st.number_input("Annual Income ($)", min_value=1, max_value=1_000_000, value=50000, step=1000)
    loan_amount  = st.number_input("Loan Amount ($)", min_value=1, max_value=1_000_000, value=15000, step=500)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    experience   = st.number_input("Work Experience (Years)", 0, 50, 5)

with col2:
    gender     = st.selectbox("Gender", ["Male", "Female"])
    education  = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])
    city       = st.selectbox("City", ["New York", "Los Angeles", "Chicago", "Houston", "San Francisco"])
    employment = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-Employed", "Unemployed"])

st.markdown("<div style='margin:2rem 0'></div>", unsafe_allow_html=True)

# ── RUN ───────────────────────────────────────────────────────────────────────
if st.button("Run Analysis →"):
    if model is not None:
        input_df = pd.DataFrame({
            'Age':             [age],
            'Gender':          [gender],
            'Income':          [income],
            'Education':       [education],
            'EmploymentType':  [employment],
            'LoanAmount':      [loan_amount],
            'CreditScore':     [credit_score],
            'City':            [city],
            'YearsExperience': [experience]
        })

        prediction = model.predict(input_df)[0]
        prob       = model.predict_proba(input_df)[0][1]
        prob_pct   = round(prob * 100, 1)
        lti        = round(loan_amount / income, 2)

        approved     = prediction == 1
        card_cls     = "result-approved" if approved else "result-rejected"
        verdict_cls  = "verdict-approved" if approved else "verdict-rejected"
        prob_val_cls = "prob-value-approved" if approved else "prob-value-rejected"
        bar_cls      = "prob-bar-approved" if approved else "prob-bar-rejected"
        icon         = "&#10022;" if approved else "&#10005;"
        verdict_txt  = "Approved" if approved else "Rejected"
        sub_txt      = "Applicant meets the credit criteria." if approved else "Application carries elevated risk."

        # Tum HTML string concat ile olusturuluyor — tırnak catismasi yok
        html = (
            '<div class="result-card ' + card_cls + '">'
            '<div class="verdict ' + verdict_cls + '">' + icon + ' ' + verdict_txt + '</div>'
            '<div class="verdict-sub">' + sub_txt + '</div>'
            '<div class="prob-label">Approval Probability</div>'
            '<div class="' + prob_val_cls + '">' + str(prob_pct) + '%</div>'
            '<div class="prob-bar-track">'
            '<div class="' + bar_cls + '" style="width:' + str(prob_pct) + '%"></div>'
            '</div>'
            '<div class="chips">'
            '<div class="chip">Credit Score: ' + str(credit_score) + '</div>'
            '<div class="chip">Income: $' + f'{income:,}' + '</div>'
            '<div class="chip">Loan: $' + f'{loan_amount:,}' + '</div>'
            '<div class="chip">LTI: ' + str(lti) + '</div>'
            '<div class="chip">Exp: ' + str(experience) + 'y</div>'
            '<div class="chip">' + employment + '</div>'
            '</div>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.warning("Model yuklenemedi, analiz yapilamiyor.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-footer">'
    '<span>&#9672; LoanSight Intelligence Engine v1.0</span>'
    '<span>Powered by LightGBM &amp; Streamlit</span>'
    '</div>',
    unsafe_allow_html=True
)
