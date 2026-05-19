"""
Phase 2 & 3: Model Training + ONNX Export
- Deduplicates PIMA + Germany datasets
- Fixes invalid zero values via median imputation
- Trains GradientBoosting with proper stratified CV
- Exports to both .pkl and .onnx
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─── 1. Load & Merge Datasets ──────────────────────────────────────────────────
print("=" * 60)
print("PHASE 2: MODEL TRAINING WITH PROPER VALIDATION")
print("=" * 60)

pima = pd.read_csv("data/raw/pima.csv")
germany = pd.read_csv("data/raw/germany.csv")

print(f"\nPIMA shape: {pima.shape}")
print(f"Germany shape: {germany.shape}")

# Merge
combined = pd.concat([pima, germany], ignore_index=True)
print(f"Combined (before dedup): {combined.shape}")

# ─── 2. Deduplicate ────────────────────────────────────────────────────────────
combined = combined.drop_duplicates()
print(f"Combined (after dedup): {combined.shape}")

# ─── 3. Fix Invalid Zero Values ────────────────────────────────────────────────
# These features cannot biologically be zero
zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in zero_invalid_cols:
    n_zeros = (combined[col] == 0).sum()
    if n_zeros > 0:
        median_val = combined.loc[combined[col] != 0, col].median()
        combined.loc[combined[col] == 0, col] = median_val
        print(f"  Fixed {n_zeros} zero values in {col} -> median {median_val}")

print(f"\nFinal dataset shape: {combined.shape}")
print(f"Class distribution:\n{combined['Outcome'].value_counts()}")
print(f"Class ratio: {combined['Outcome'].mean():.2%} positive")

# ─── 4. Train/Test Split ───────────────────────────────────────────────────────
X = combined.drop(columns=["Outcome"])
y = combined["Outcome"]

FEATURE_NAMES = list(X.columns)
print(f"\nFeatures: {FEATURE_NAMES}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")

# ─── 5. Train Model ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING GRADIENT BOOSTING CLASSIFIER")
print("=" * 60)

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# ─── 6. Evaluate ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION ON TEST SET")
print("=" * 60)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  ROC AUC:   {auc:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─── 7. Cross-Validation ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"\n  CV AUC scores: {cv_scores}")
print(f"  Mean AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── 8. Save Model (pkl) ───────────────────────────────────────────────────────
import os
os.makedirs("models", exist_ok=True)

pkl_path = "models/diabetes_model.pkl"
joblib.dump(model, pkl_path)
print(f"\n✅ Model saved: {pkl_path}")

# ─── 9. Export to ONNX ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3: ONNX EXPORT")
print("=" * 60)

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [("float_input", FloatTensorType([None, len(FEATURE_NAMES)]))]

onnx_model = convert_sklearn(
    model, initial_types=initial_type,
    options={type(model): {"zipmap": False}}
)

onnx_path = "models/diabetes_model.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ ONNX model saved: {onnx_path}")

# ─── 10. Verify ONNX Predictions Match ─────────────────────────────────────────
import onnxruntime as ort

sess = ort.InferenceSession(onnx_path)
input_name = sess.get_inputs()[0].name

# Test with first 5 samples
test_samples = X_test.head(5).values.astype(np.float32)

# Original predictions
orig_probs = model.predict_proba(test_samples)[:, 1]

# ONNX predictions
onnx_results = sess.run(None, {input_name: test_samples})
onnx_probs = onnx_results[1][:, 1]

print(f"\nPrediction Comparison (first 5 test samples):")
print(f"  Original:  {np.round(orig_probs, 4)}")
print(f"  ONNX:      {np.round(onnx_probs, 4)}")

max_diff = np.max(np.abs(orig_probs - onnx_probs))
print(f"  Max difference: {max_diff:.8f}")

if max_diff < 0.001:
    print("  ✅ ONNX predictions match original model!")
else:
    print("  ⚠️ Warning: ONNX predictions differ slightly")

# ─── 11. Save metadata for frontend ────────────────────────────────────────────
metadata = {
    "features": FEATURE_NAMES,
    "metrics": {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "cv_auc_mean": round(cv_scores.mean(), 4),
        "cv_auc_std": round(cv_scores.std(), 4),
    },
    "training": {
        "algorithm": "GradientBoostingClassifier",
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 4,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "total_samples": int(combined.shape[0]),
        "positive_ratio": round(combined["Outcome"].mean(), 4),
    }
}

meta_path = "models/model_metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Metadata saved: {meta_path}")
print("\n" + "=" * 60)
print("ALL PHASES COMPLETE!")
print("=" * 60)
