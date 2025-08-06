import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import os
import base64
from fpdf import FPDF

# ─── Page Title ────────────────────────────────────────────────────────────────
st.title("📊 Global Health Insights")


# ─── Load Data & Model ─────────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_data
def load_model(path):
    return joblib.load(path)


data_path = "data/processed/diabetes_combined.csv"
model_path = "models/diabetes_model_combined.pkl"

df = load_data(data_path)
model = load_model(model_path)
X = df.drop(columns=["Outcome"])

st.info("This page shows SHAP explanations and correlations from the full dataset.")

# ─── SHAP Global Explanation ───────────────────────────────────────────────────
st.subheader("🔍 SHAP Explanation (Random Sample)")
explainer = shap.TreeExplainer(model)
X_sample = X.sample(1, random_state=42)
shap_values = explainer.shap_values(X_sample)

# Get scalar base value
expected_val = explainer.expected_value
base_val = (
    expected_val[0] if isinstance(expected_val, (list, np.ndarray)) else expected_val
)

# Get SHAP values for the sample
shap_val = shap_values[0][0] if isinstance(shap_values, list) else shap_values[0]

# Plot SHAP waterfall
with st.spinner("Generating SHAP plot..."):
    shap.initjs()
    fig1 = plt.figure()
    shap.plots._waterfall.waterfall_legacy(base_val, shap_val, X_sample.iloc[0])
    st.pyplot(fig1)

# Save SHAP plot
os.makedirs("outputs", exist_ok=True)
shap_img_path = "outputs/shap_global_sample.png"
fig1.savefig(shap_img_path)

# Download SHAP plot
with open(shap_img_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<a href="data:file/png;base64,{b64}" download="shap_global_sample.png">📥 Download SHAP Plot (PNG)</a>',
        unsafe_allow_html=True,
    )

# ─── Correlation Heatmap ───────────────────────────────────────────────────────
st.subheader("🔗 Feature Correlation Heatmap")

corr = X.corr()
fig2, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Correlation Between Features")
st.pyplot(fig2)

# Save heatmap
heatmap_path = "outputs/correlation_heatmap.png"
fig2.savefig(heatmap_path)

# Download heatmap
with open(heatmap_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<a href="data:file/png;base64,{b64}" download="correlation_heatmap.png">📥 Download Heatmap (PNG)</a>',
        unsafe_allow_html=True,
    )

# ─── Export to PDF ─────────────────────────────────────────────────────────────
st.markdown("### 📄 Export Summary as PDF")
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, "Diabetes SHAP Insight Summary", ln=True, align="C")
pdf.ln(10)

for col, val in X_sample.iloc[0].items():
    pdf.cell(200, 10, f"{col}: {val}", ln=True)

pdf_path = "outputs/shap_summary.pdf"
pdf.output(pdf_path)

with open(pdf_path, "rb") as f:
    st.download_button(
        "📄 Download PDF Summary",
        data=f,
        file_name="shap_summary.pdf",
        mime="application/pdf",
    )

# ─── Feature Glossary ──────────────────────────────────────────────────────────
st.markdown("### ℹ️ Feature Glossary")
st.markdown("""
- **Glucose**: Blood sugar level (mg/dL)  
- **BMI**: Body Mass Index = weight / height²  
- **Insulin**: Insulin level in blood (μU/mL)  
- **Pregnancies**: Number of times pregnant  
- **DPF**: Genetic likelihood of diabetes  
- **Age**: Age in years  
- **BloodPressure**: Blood pressure (mm Hg)  
- **SkinThickness**: Skin fold thickness (mm)  
""")
