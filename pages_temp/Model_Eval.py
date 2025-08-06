# Pages/Model_Eval.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
import os

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Evaluation", layout="wide", initial_sidebar_state="expanded"
)

st.title("📈 Model Evaluation")

# ─── Load Model & Data ───────────────────────────────────────────────────────────
model_path = "models/diabetes_model_combined.pkl"
data_path = "data/processed/diabetes_combined.csv"

model = joblib.load(model_path)
df = pd.read_csv(data_path)

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── Predictions ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ─── Create output folder ───────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

# ─── A. Confusion Matrix ────────────────────────────────────────────────────────
st.subheader("🔲 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
ax1.set_title("Confusion Matrix")
st.pyplot(fig1)
fig1.savefig("outputs/confusion_matrix.png")

# ─── B. Classification Report ───────────────────────────────────────────────────
st.subheader("🧾 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).T)

# ─── C. ROC Curve ───────────────────────────────────────────────────────────────
st.subheader("📉 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", color="blue")
ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend()
st.pyplot(fig2)
fig2.savefig("outputs/roc_curve.png")

# ─── D. Precision-Recall Curve ──────────────────────────────────────────────────
st.subheader("📈 Precision–Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
fig3, ax3 = plt.subplots()
ax3.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}", color="green")
ax3.set_xlabel("Recall")
ax3.set_ylabel("Precision")
ax3.set_title("Precision–Recall Curve")
ax3.legend()
st.pyplot(fig3)
fig3.savefig("outputs/precision_recall_curve.png")

# ─── E. Calibration Curve ───────────────────────────────────────────────────────
st.subheader("⚖️ Calibration Curve")
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
fig4, ax4 = plt.subplots()
ax4.plot(prob_pred, prob_true, marker="o", label="Model")
ax4.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
ax4.set_xlabel("Predicted Probability")
ax4.set_ylabel("True Proportion")
ax4.set_title("Calibration Curve")
ax4.legend()
st.pyplot(fig4)
fig4.savefig("outputs/calibration_curve.png")

# ─── Interpretation ─────────────────────────────────────────────────────────────
st.markdown("### 📌 Interpretation")
st.markdown(f"""
- **ROC AUC Score**: **{auc_score:.3f}** – reflects overall classification ability  
- **PR AUC Score**: **{pr_auc:.3f}** – useful for imbalanced datasets  
- **Calibration Curve**: Compares predicted probabilities to actual outcomes  
""")
