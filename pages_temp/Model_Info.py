import streamlit as st

# ─── Page Config ────────────────────────────────────────────────────────────────
st.title("📦 Model and App Info")

st.markdown("""
### 🧠 Model Details
- **Algorithm Used**: Gradient Boosting Classifier  
- **Hyperparameters**:  
  - `n_estimators`: 150  
  - `learning_rate`: 0.1  
  - `max_depth`: 5  
- **Training Dataset**: Combined PIMA + Germany Diabetes Dataset  
- **Train/Test Split**: 2,214 / 554  
- **Balancing Method**: SMOTE  
- **Total Samples After Balancing**: 3,632  
- **Features Used**: Glucose, BMI, Insulin, Age, etc.  

### 📊 Performance on Test Set
- **Accuracy**: 99.09%  
- **F1 Score**: 0.9844  
- **Precision**: 0.9791  
- **Recall**: 0.9791  
- **ROC AUC Score**: 0.9998  
- **PR AUC Score**: 0.9996  
- **Test Samples**: 554  

### 🛠️ App Features
- Streamlit web app with modular pages  
- Risk prediction with threshold logic  
- SHAP explanations (global + local)  
- Auto-filled values for unknown inputs  
- PDF/CSV export support  

### ⚠️ Disclaimer
This app is for educational and prototyping purposes only.  
It is **not** a substitute for professional medical advice.
""")
