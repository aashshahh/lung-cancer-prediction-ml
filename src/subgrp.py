"""
subgrp.py — subgroup fairness analysis.
Evaluates model performance broken down by gender, age group, smoking status.
Run with: python src/subgrp.py
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import FEATURES, TARGET, MODELS_PATH, RESULTS_PATH
from dataloader import load_data
from preprocessor import preprocess

os.makedirs(RESULTS_PATH, exist_ok=True)


def run_subgroup_analysis():
    print("[subgrp] Loading model and data...")

    model_path = os.path.join(MODELS_PATH, "best_model.pkl")
    scaler_path = os.path.join(MODELS_PATH, "scaler.pkl")

    if not os.path.exists(model_path):
        print("[subgrp] ERROR: No best_model.pkl found. Run src/train.py first.")
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    df = load_data()

    # Encode same way as preprocessor
    df_encoded = df.copy()
    df_encoded[TARGET] = df_encoded[TARGET].map({"YES": 1, "NO": 0})
    df_encoded["GENDER"] = df_encoded["GENDER"].map({"M": 1, "F": 0})

    X = scaler.transform(df_encoded[FEATURES].values)
    y = df_encoded[TARGET].values

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Work with original (readable) values for grouping
    df_work = df_encoded.copy()
    df_work["y_true"] = y
    df_work["y_pred"] = y_pred
    df_work["y_prob"] = y_prob

    results = []

    def evaluate_group(subset, label):
        if len(subset) < 5:
            return
        yt, yp, ypr = subset["y_true"], subset["y_pred"], subset["y_prob"]
        try:
            auc = roc_auc_score(yt, ypr) if yt.nunique() > 1 else None
        except Exception:
            auc = None
        results.append({
            "Subgroup": label,
            "N": len(subset),
            "Accuracy": f"{accuracy_score(yt, yp)*100:.1f}%",
            "Recall": f"{recall_score(yt, yp, zero_division=0)*100:.1f}%",
            "AUC-ROC": f"{auc*100:.1f}%" if auc else "N/A",
        })

    # Gender
    evaluate_group(df_work[df_work["GENDER"] == 1], "Gender: Male")
    evaluate_group(df_work[df_work["GENDER"] == 0], "Gender: Female")

    # Age groups
    df_work["age_group"] = pd.cut(
        df_work["AGE"], bins=[0, 50, 65, 120],
        labels=["Age: <50", "Age: 50-65", "Age: 65+"]
    )
    for grp in ["Age: <50", "Age: 50-65", "Age: 65+"]:
        evaluate_group(df_work[df_work["age_group"] == grp], grp)

    # Smoking status (original encoding: 2=yes, 1=no in the dataset)
    evaluate_group(df_work[df_work["SMOKING"] == 2], "Smoking: Yes")
    evaluate_group(df_work[df_work["SMOKING"] == 1], "Smoking: No")

    results_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_PATH, "subgroup_results.csv")
    results_df.to_csv(out_path, index=False)

    print("\n[subgrp] SUBGROUP FAIRNESS RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nSaved to {out_path}")
    print("\n>>> Copy these numbers into the Subgroup table in README.md")


if __name__ == "__main__":
    run_subgroup_analysis()