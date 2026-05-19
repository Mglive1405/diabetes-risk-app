# DiabetesIQ — AI-Powered Diabetes Risk Assessment

A modern, privacy-first diabetes risk assessment tool powered by machine learning. The AI model runs **entirely in your browser** — no data is ever sent to any server.

🌐 **[Live Demo](https://mglive1405.github.io/diabetes-risk-app/)**

## Features

- 🧠 **AI-Powered** — Gradient Boosting model trained on 778 clinical records
- 🔒 **100% Private** — ONNX Runtime Web runs inference client-side
- 📊 **Explainable** — See which health factors contribute to your risk
- ⚡ **Instant** — Results in milliseconds, no server required
- 📱 **Responsive** — Works on desktop, tablet, and mobile
- 🎨 **Premium UI** — Dark mode with glassmorphism design

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 73.1% |
| Precision | 61.8% |
| Recall | 61.8% |
| F1 Score | 61.8% |
| ROC AUC | 82.1% |
| CV AUC (5-fold) | 82.0% ± 2.9% |

## Tech Stack

- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **ML Inference**: ONNX Runtime Web
- **Model**: scikit-learn GradientBoostingClassifier → ONNX
- **Hosting**: GitHub Pages (static)

## Project Structure

```
├── docs/                  # Frontend (GitHub Pages source)
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── diabetes_model.onnx
├── models/                # Model artifacts
│   └── model_metadata.json
├── data/                  # Training data
│   ├── raw/
│   └── processed/
├── notebooks/             # Training notebooks
├── train_model.py         # Model training script
└── requirements.txt       # Python deps (for training only)
```

## Deploying to GitHub Pages

1. Go to your repo **Settings → Pages**
2. Set Source to **Deploy from a branch**
3. Select **main** branch and **/docs** folder
4. Save — your site will be live at `https://<username>.github.io/diabetes-risk-app/`

## Disclaimer

This tool is for **educational purposes only** and is not a substitute for professional medical advice, diagnosis, or treatment.
