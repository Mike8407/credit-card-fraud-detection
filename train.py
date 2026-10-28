"""
Credit Card Fraud Detection - working model

Train + evaluate + save:
    python train.py --train creditcard_fraud.csv

Predict on new transactions (CSV with the same feature columns, no 'Class'):
    python train.py --predict new_transactions.csv
"""
import argparse
import json
import os
import sklearn
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)

MODEL_FILE = "model/fraud_model.joblib"
TARGET = "Class"
DATASET_PATH = "data/creditcard_fraud.csv"

def build_pipeline():
    # Scaler + model in one object, so new data is always scaled the same way.
    # class_weight='balanced' handles the heavy fraud/legit imbalance.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def metrics_at(y_true, probs, t):
    pred = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": t,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "FP": fp, "FN": fn, "TP": tp, "TN": tn,
    }


def train(csv_path):
    os.makedirs("model", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    df = pd.read_csv(csv_path)
    X, y = df.drop(TARGET, axis=1), df[TARGET]

    print("Class distribution:")
    print(y.value_counts(normalize=True).round(5).to_string(), "\n")

    # 60% train / 20% validation (pick threshold) / 20% test (final report)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp)

    model = build_pipeline()
    model.fit(X_train, y_train)

    # Choose the threshold on the validation set (best F1), not on the test set
    val_probs = model.predict_proba(X_val)[:, 1]
    grid = np.concatenate([[0.01, 0.02, 0.05], np.arange(0.1, 1.0, 0.1),
                           [0.95, 0.97, 0.98, 0.99, 0.995, 0.999]])
    val_table = pd.DataFrame([metrics_at(y_val, val_probs, t) for t in grid])
    best_t = float(val_table.loc[val_table["f1"].idxmax(), "threshold"])
    print("Threshold analysis (validation set):")
    print(val_table.round(4).to_string(index=False))
    print(f"\nChosen threshold (best F1 on validation): {best_t:.3f}\n")

    # Final evaluation on untouched test set
    probs = model.predict_proba(X_test)[:, 1]
    pred = (probs >= best_t).astype(int)

    print("=== TEST SET RESULTS ===")
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, pred, zero_division=0):.4f}")
    print(f"F1-Score : {f1_score(y_test, pred, zero_division=0):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, probs):.4f}")
    print(f"Avg Prec : {average_precision_score(y_test, probs):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # Plots (saved as PNG and shown)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Legit", "Fraud"],
        cmap="Blues", ax=ax, values_format=",")
    ax.set_title(f"Confusion Matrix (threshold={best_t:.3f})")
    plt.tight_layout(); plt.savefig("static/confusion_matrix.png", dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="blue", lw=2,
             label=f"Model (AUC={roc_auc_score(y_test, probs):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Random")
    plt.xlabel("FPR (False Alarm Rate)"); plt.ylabel("TPR (Recall)")
    plt.title("ROC Curve"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("static/roc_curve.png", dpi=150); plt.close()

    p, r, _ = precision_recall_curve(y_test, probs)
    plt.figure(figsize=(6, 6))
    plt.plot(r, p, color="green", lw=2,
             label=f"AP={average_precision_score(y_test, probs):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("static/pr_curve.png", dpi=150); plt.close()

    # Save model + threshold + expected columns
    joblib.dump({"model": model, "threshold": best_t,
                 "columns": list(X.columns)}, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")

    # Demo dataset bundled with the website: ONLY rows from the held-out test set,
    # so the live accuracy shown in the app is an honest out-of-sample result.
    os.makedirs("data", exist_ok=True)
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    frauds = test_df[test_df[TARGET] == 1]
    legit = test_df[test_df[TARGET] == 0].sample(n=5000 - len(frauds), random_state=42)
    demo = pd.concat([frauds, legit]).sample(frac=1, random_state=42)
    demo.insert(0, "tx_id", demo.index)
    demo.to_csv("data/demo_transactions.csv.gz", index=False, compression="gzip")
    print(f"Saved data/demo_transactions.csv.gz ({len(demo)} rows, {len(frauds)} fraud)")
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
        "avg_precision": average_precision_score(y_test, probs),
        "threshold": best_t, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "fraud_rate": float(y.mean()), "sklearn_version": sklearn.__version__,
    }
    json.dump(metrics, open("model/metrics.json", "w"), indent=2)
    print("Saved model/metrics.json")


def predict(csv_path, out_path="predictions.csv"):
    bundle = joblib.load(MODEL_FILE)
    df = pd.read_csv(csv_path)
    if TARGET in df.columns:
        df = df.drop(TARGET, axis=1)
    df = df[bundle["columns"]]  # same columns, same order as training

    probs = bundle["model"].predict_proba(df)[:, 1]
    out = df.copy()
    out["fraud_probability"] = probs
    out["prediction"] = np.where(probs >= bundle["threshold"], "FRAUD", "Legit")
    out.to_csv(out_path, index=False)
    print(out["prediction"].value_counts().to_string())
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    train(DATASET_PATH)