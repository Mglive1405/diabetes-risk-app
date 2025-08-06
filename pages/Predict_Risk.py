import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import contextlib
from fpdf import FPDF
import os

# ─── Load Model ─────────────────────────────────────────────────────────────────
MODEL_PATH = "models/diabetes_model_combined.pkl"
model = joblib.load(MODEL_PATH)

# ─── “I DON’T KNOW” Defaults ─────────────────────────────────────────────────────
averages = {
    "Glucose": 120,
    "BloodPressure": 80,
    "SkinThickness": 25,
    "Insulin": 100,
    "DiabetesPedigreeFunction": 0.5,
}

# ─── Page Title ─────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Prediction Tool")

with st.sidebar:
    st.header("⚙️ Settings")
    gender = st.radio("Select Gender", ["Female", "Male"])
    threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.01)
    show_tips = st.checkbox("Show Health Tips", value=True)

st.subheader("📋 Enter Medical Information")

with st.form(key="input_form"):
    col1, col2, col3, col4 = st.columns(4)

    # Pregnancies (only if Female)
    pregnancies = (
        col1.number_input("Pregnancies", 0, 50, 1, step=1) if gender == "Female" else 0
    )

    glucose = col2.number_input("Glucose (mg/dL)", 30, 500, averages["Glucose"])
    blood_pressure = col3.number_input(
        "Blood Pressure (mm Hg)", 30, 300, averages["BloodPressure"]
    )
    skin_thickness = col4.number_input(
        "Skin Thickness (mm)", 0, 200, averages["SkinThickness"]
    )
    insulin = col1.number_input("Insulin Level (μU/mL)", 0, 2000, averages["Insulin"])
    dpf = col2.number_input(
        "Diabetes Pedigree Function",
        0.0,
        5.0,
        averages["DiabetesPedigreeFunction"],
        step=0.01,
    )

    height = col3.number_input("Height (cm)", 50, 250, 165)
    weight = col4.number_input("Weight (kg)", 10, 300, 70)
    bmi = round(weight / ((height / 100) ** 2), 2)
    col4.markdown(f"**Calculated BMI:** {bmi}")

    age = col1.number_input("Age", 0, 120, 30)
    submitted = st.form_submit_button("🔮 Predict Risk")

# ─── Risk Prediction ─────────────────────────────────────────────────────────────
if submitted:
    user_input = pd.DataFrame(
        [
            [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                dpf,
                age,
            ]
        ],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
    )

    prob = model.predict_proba(user_input)[0][1]
    risk_label = "High" if prob >= threshold else "Borderline" if prob >= 0.4 else "Low"
    color = {"High": "#ff4b4b", "Borderline": "#f2c94c", "Low": "#6fcf97"}

    st.markdown("### 📊 Prediction Result")
    st.progress(prob)
    st.markdown(
        f"### Risk: <span style='color:{color[risk_label]}'>{risk_label} ({prob * 100:.2f}%)</span>",
        unsafe_allow_html=True,
    )

    # ─── SHAP Explanation ────────────────────────────────────────────────────────
    st.subheader("🔍 SHAP Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_input)

    # Safely get scalar base value
    expected_val = explainer.expected_value
    base_val = (
        expected_val[0]
        if isinstance(expected_val, (list, np.ndarray))
        else expected_val
    )

    # Safely get SHAP values
    shap_val = shap_values[0][0] if isinstance(shap_values, list) else shap_values[0]

    with contextlib.redirect_stdout(None):
        shap.initjs()

    fig = plt.figure()
    shap.plots._waterfall.waterfall_legacy(base_val, shap_val, user_input.iloc[0])
    st.pyplot(fig)

    # ─── Advice ──────────────────────────────────────────────────────────────────
    if show_tips:
        st.subheader("💡 Health Tips")
        important_feats = (
            pd.Series(np.abs(shap_val), index=user_input.columns).nlargest(3).index
        )
        tips = {
            "Glucose": "🍭 Cut sugar and refined carbs.",
            "BMI": "🏃‍♂️ Exercise and manage weight.",
            "Age": "⌛ Screen regularly after 30.",
            "Insulin": "💉 Regular insulin checkups.",
            "Pregnancies": "👶 Monitor after childbirth.",
            "BloodPressure": "❤️ Lower salt and stress.",
            "SkinThickness": "💪 Stay active.",
            "DiabetesPedigreeFunction": "👨‍👩‍👧 Family history = be vigilant.",
        }
        for feat in important_feats:
            st.markdown(f"- **{feat}**: {tips.get(feat, 'Stay healthy!')}")

    # ─── Export (PDF + CSV) ───────────────────────────────────────────────────────
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Diabetes Risk Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, f"Risk Level: {risk_label} ({prob * 100:.2f}%)", ln=True)
    pdf.cell(200, 10, f"Top Factors: {', '.join(important_feats)}", ln=True)
    pdf.cell(200, 10, f"Gender: {gender}, BMI: {bmi}, Age: {age}", ln=True)

    os.makedirs("outputs", exist_ok=True)
    pdf.output("outputs/prediction_report.pdf")

    st.download_button(
        "📄 Download Report (PDF)",
        data=open("outputs/prediction_report.pdf", "rb"),
        file_name="prediction_report.pdf",
        mime="application/pdf",
    )

    csv_path = "outputs/user_input.csv"
    user_input["RiskProbability"] = prob
    user_input["RiskLabel"] = risk_label
    user_input.to_csv(csv_path, index=False)

    st.download_button(
        "📥 Download CSV",
        data=open(csv_path, "rb"),
        file_name="user_input.csv",
        mime="text/csv",
    )
