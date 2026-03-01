# 💳 Credit Risk Classification

A machine learning project that predicts loan approval decisions using LightGBM, built with a focus on preventing data leakage, handling class imbalance, and model interpretability.

🌐 Live Demo

> ⚠️ **Disclaimer:** This application is a simulation and educational project only. Results do not constitute a real credit assessment and should not be used for financial decision-making.
> 
🚀 You can test the deployed model here:

🔗 Streamlit App:
https://fatih-loan-risk-data-set.streamlit.app/

---

## 📁 Project Structure

```
├── model.ipynb                      # Main notebook
├── loan_risk_prediction_dataset.csv # Dataset
└── README.md
```

---

## 🔍 Problem Definition

Binary classification task: predict whether a loan application will be **approved (1)** or **rejected (0)** based on applicant features available at the time of application.

---

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| `Age` | Numerical | Applicant age |
| `Income` | Numerical | Annual income |
| `CreditScore` | Numerical | Credit score |
| `LoanAmount` | Numerical | Requested loan amount |
| `YearsExperience` | Numerical | Work experience |
| `Education` | Ordinal | High School → Bachelors → Masters → PhD |
| `Gender` | Categorical | Applicant gender |
| `City` | Categorical | City of residence |
| `EmploymentType` | Categorical | Employment status |
| `LoanApproved` | Target (0/1) | Loan approval decision |

---

## ⚙️ Pipeline Architecture

All preprocessing steps are wrapped inside a `sklearn Pipeline` to prevent data leakage.

```
ColumnTransformer
├── Numerical   → SimpleImputer(median) → StandardScaler
├── Education   → SimpleImputer(most_frequent) → OrdinalEncoder
├── Categorical → SimpleImputer(most_frequent) → OneHotEncoder
└── Age         → AgeGroupTransformer (binning) → OneHotEncoder
        └── Groups: Young / Early_Career / Middle_Age / Senior
```

---

## 🧪 Modeling & Evaluation

### Baseline — Logistic Regression
- Evaluated with **Stratified 5-Fold Cross-Validation** on training data only
- `class_weight='balanced'` used to handle class imbalance

### Final Model — LightGBM
- Hyperparameter tuning with **RandomizedSearchCV** (40 iterations, 5-fold CV)
- Tuned on `X_train` only — test set never touched during training
- Best CV ROC-AUC: **~0.95**

### Evaluation Metrics
- Classification Report (Precision / Recall / F1)
- Confusion Matrix
- ROC-AUC Curve

---

## 🔎 Interpretability

**SHAP (SHapley Additive exPlanations)** was used to explain model decisions:

- `CreditScore` — dominant feature driving approvals/rejections
- `Income` & `LoanAmount` — secondary importance
- `EmploymentType` (Unemployed) — significantly lowers approval probability
- Age groups, City, Gender — minor influence

LightGBM feature importances are consistent with SHAP results, confirming the model behaves in a financially intuitive way.

---

## 🛡️ Data Leakage Prevention

- No fit/transform applied to the full dataset before splitting
- All imputers, scalers, and encoders live inside the pipeline
- `cross_val_predict` and `RandomizedSearchCV` both operate on `X_train` only
- No post-approval features (e.g., last payment date, overdue interest) included

---

## 🏗️ Tech Stack

| Library | Usage |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Pipeline, preprocessing, CV, metrics |
| `lightgbm` | Gradient boosting classifier |
| `shap` | Model explainability |

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm shap

# 2. Place dataset in project root
# loan_risk_prediction_dataset.csv

# 3. Run the notebook
jupyter notebook model.ipynb
```

---

## 📈 Results Summary

| Metric | Score |
|---|---|
| CV ROC-AUC (best) | ~0.95 |
| Test Accuracy | Strong |
| Overfitting Gap | Reduced after tuning |
| Class Imbalance | Handled via `class_weight='balanced'` |
