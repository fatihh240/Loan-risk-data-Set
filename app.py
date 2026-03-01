import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ── CUSTOM TRANSFORMER ────────────────────────────────────────────────────────
class AgeGroupTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bins   = [17, 25, 35, 50, np.inf]
        self.labels = ['Young', 'Early_Career', 'Middle_Age', 'Senior']

    def fit(self, X, y=None): return self

    def transform(self, X):
        X_copy  = X.copy()
        X_binned = pd.cut(X_copy.iloc[:, 0], bins=self.bins, labels=self.labels)
        return pd.DataFrame(X_binned)


# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('best_credit_model.pkl')

model = load_model()

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSight · Credit Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── reset & base ── */
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

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 3rem 4rem 5rem; max-width: 1200px; }

/* ── typography ── */
h1, h2, h3 { font-family: 'DM Serif Display', serif; font-weight: 400; }

/* ── hero header ── */
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
    font-size: clamp(2.8rem, 5vw, 4.2rem);
    line-height: 1.05;
    color: #f0ece4;
    letter-spacing: -0.02em;
}
.hero-title em { font-style: italic; color: #f5c842; }
.hero-sub {
    font-size: 0.95rem;
    color: rgba(232,228,220,0.45);
    max-width: 520px;
    line-height: 1.65;
    margin-top: 0.6rem;
}

/* ── section label ── */
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

/* ── input cards ── */
.input-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 2.5rem;
}
@media (max-width: 768px) { .input-grid { grid-template-columns: 1fr; } }

/* ── streamlit widget overrides ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"] > div > div:focus {
    border-color: rgba(245,200,66,0.5) !important;
    box-shadow: 0 0 0 2px rgba(245,200,66,0.08) !important;
    outline: none !important;
}
label, [data-testid="stWidgetLabel"] p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: rgba(232,228,220,0.5) !important;
    margin-bottom: 0.35rem !important;
}

/* slider accent */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #f5c842 !important;
    border-color: #f5c842 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stSlider"] > div > div > div:nth-child(2) {
    background: #f5c842 !important;
}

/* ── divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 2.5rem 0;
}

/* ── run button ── */
.stButton > button {
    background: #f5c842 !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 2.4rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(245,200,66,0.25) !important;
    width: auto !important;
}
.stButton > button:hover {
    background: #ffd84a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(245,200,66,0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── result card ── */
.result-card {
    border-radius: 16px;
    padding: 2.8rem 3rem;
    margin-top: 2.5rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 16px;
    padding: 1px;
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
}
.result-approved {
    background: rgba(34, 197, 94, 0.06);
}
.result-approved::before {
    background: linear-gradient(135deg, rgba(34,197,94,0.4), rgba(34,197,94,0.1));
}
.result-rejected {
    background: rgba(239, 68, 68, 0.06);
}
.result-rejected::before {
    background: linear-gradient(135deg, rgba(239,68,68,0.4), rgba(239,68,68,0.1));
}

.result-verdict {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(1.8rem, 3vw, 2.6rem);
    line-height: 1.1;
    margin-bottom: 0.6rem;
}
.verdict-approved { color: #4ade80; }
.verdict-rejected { color: #f87171; }

.result-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    color: rgba(232,228,220,0.4);
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ── probability meter ── */
.prob-section { margin-top: 1.5rem; }
.prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(232,228,220,0.35);
    margin-bottom: 0.6rem;
}
.prob-bar-track {
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    overflow: hidden;
    margin-bottom: 0.45rem;
}
.prob-bar-fill-approved {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #22c55e, #4ade80);
    transition: width 0.8s cubic-bezier(0.23, 1, 0.32, 1);
}
.prob-bar-fill-rejected {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #ef4444, #f87171);
    transition: width 0.8s cubic-bezier(0.23, 1, 0.32, 1);
}
.prob-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    line-height: 1;
    letter-spacing: -0.03em;
}
.prob-value-approved { color: #4ade80; }
.prob-value-rejected { color: #f87171; }

/* ── feature chips ── */
.chips-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 2rem;
}
.chip {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    padding: 0.3rem 0.75rem;
    border-radius: 99px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: rgba(232,228,220,0.55);
}
.chip span { color: rgba(232,228,220,0.9); margin-left: 0.3em; }

/* ── footer ── */
.footer {
    margin-top: 5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: rgba(232,228,220,0.2);
    display: flex;
    justify-content: space-between;
}
</style>
""", unsafe_allow_html=True)


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">◈ LoanSight · Credit Intelligence System</div>
    <h1 class="hero-title">Should this loan<br>be <em>approved?</em></h1>
    <p class="hero-sub">
        Enter the applicant's profile below. The model evaluates credit risk
        using a LightGBM classifier trained on historical approval data.
    </p>
</div>
""", unsafe_allow_html=True)


# ── INPUTS ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">01 · Applicant Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    age           = st.number_input("Age", min_value=18, max_value=100, value=32)
    income        = st.number_input("Annual Income ($)", min_value=0, max_value=1_000_000, value=62_000, step=1000)
    loan_amount   = st.number_input("Loan Amount ($)", min_value=0, max_value=1_000_000, value=18_000, step=500)
    credit_score  = st.slider("Credit Score", min_value=300, max_value=850, value=680)
    experience    = st.number_input("Work Experience (years)", min_value=0, max_value=50, value=6)

with col2:
    gender     = st.selectbox("Gender", ["Male", "Female"])
    education  = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])
    city       = st.selectbox("City", ["New York", "Los Angeles", "Chicago", "Houston", "San Francisco"])
    employment = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-Employed", "Unemployed"])

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ── RUN BUTTON ────────────────────────────────────────────────────────────────
col_btn, _ = st.columns([1, 4])
with col_btn:
    run = st.button("Run Analysis →")


# ── RESULT ────────────────────────────────────────────────────────────────────
if run:
    input_df = pd.DataFrame({
        'Age':             [age],
        'Gender':          [gender],
        'Income':          [income],
        'Education':       [education],
        'EmploymentType':  [employment],
        'LoanAmount':      [loan_amount],
        'CreditScore':     [credit_score],
        'City':            [city],
        'YearsExperience': [experience],
    })

    prediction  = model.predict(input_df)[0]
    prob        = model.predict_proba(input_df)[0][1]
    prob_pct    = round(prob * 100, 1)

    # Determine style
    approved     = prediction == 1
    card_class   = "result-approved" if approved else "result-rejected"
    verdict_cls  = "verdict-approved" if approved else "verdict-rejected"
    bar_cls      = "prob-bar-fill-approved" if approved else "prob-bar-fill-rejected"
    pval_cls     = "prob-value-approved" if approved else "prob-value-rejected"
    verdict_text = "Approved" if approved else "Rejected"
    verdict_line = "The applicant meets the credit criteria." if approved else "The application carries elevated risk."
    icon         = "✦" if approved else "✕"

    # Loan-to-income ratio
    lti = round(loan_amount / income, 2) if income > 0 else "—"

    chips_html = f"""
    <div class="chips-row">
        <div class="chip">Credit Score <span>{credit_score}</span></div>
        <div class="chip">Income <span>${income:,}</span></div>
        <div class="chip">Loan <span>${loan_amount:,}</span></div>
        <div class="chip">LTI Ratio <span>{lti}</span></div>
        <div class="chip">Employment <span>{employment}</span></div>
        <div class="chip">Education <span>{education}</span></div>
    </div>
    """

    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-verdict {verdict_cls}">{icon} {verdict_text}</div>
        <div class="result-sub">{verdict_line}</div>

        <div class="prob-section">
            <div class="prob-label">Approval Probability</div>
            <div class="prob-value {pval_cls}">{prob_pct}<span style="font-size:1.2rem;opacity:0.5">%</span></div>
            <div style="margin-top:1rem;">
                <div class="prob-bar-track">
                    <div class="{bar_cls}" style="width:{prob_pct}%;"></div>
                </div>
            </div>
        </div>
        {chips_html}
    </div>
    """, unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>◈ LoanSight · Credit Intelligence</span>
    <span>LightGBM · sklearn Pipeline · SHAP</span>
</div>
""", unsafe_allow_html=True)