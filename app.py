# app.py

import streamlit as st

# Page config
st.set_page_config(page_title="Diabetes Risk Tool", page_icon="🩺", layout="wide")

st.title("🏠 Welcome to the Diabetes Risk Assessment Tool")
st.markdown(
    """
    Use this app to:
    - 🔮 Predict your diabetes risk
    - 📊 Evaluate model performance
    - 📈 See global SHAP insights
    - ℹ️ View model & app details
    """
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ("Predict Risk", "Model Evaluation", "Global Insights", "Model & App Info")
)

# Page routing
if page == "Predict Risk":
    from Pages import Predict_Risk
elif page == "Model Evaluation":
    from Pages import Model_Eval
elif page == "Global Insights":
    from Pages import Insights
elif page == "Model & App Info":
    from Pages import Model_Info
