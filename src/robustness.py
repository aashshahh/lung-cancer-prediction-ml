"""
robustness.py — perturbation robustness testing.
Flips binary features and measures how much predictions change.
Run with: python src/robustness.py
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import FEATURES, TARGET, BINARY_FEATURES, MODELS_PATH, RESULTS_PATH
from dataloader import load_data
from preprocessor import preprocess

os.makedirs(RESULTS_PATH, exist_ok=True)

N_TRIALS = 50
FLIP_RATE = 0.2


def run_robustness_test():
    print("[robustness] Loading model and test data...")

    model_path = os.path.join(MODELS_PATH, "best_model.pkl")
    scaler_path = os.path.join(MODELS_PATH, "scaler.pkl")

    if not os.path.exists(model_path):
        print("[robustness] ERROR: No best_model.pkl found. Run src/train.py first.")
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    df = load_data()
    _, X_test, _, y_test, _, _ = preprocess(df)

    baseline_preds = model.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_preds)
    print(f"[robustness] Baseline accuracy on test set: {baseline_acc*100:.2f}%")

    rng = np.random.default_rng(42)
    results = []

    for feat_name in BINARY_FEATURES:
        if feat_name not in FEATURES:
            continue
        feat_idx = FEATURES.index(feat_name)

        trial_accs = []
        trial_change_rates = []

        for _ in range(N_TRIALS):
            X_perturbed = X_test.copy()
            flip_mask = rng.random(len(X_test)) < FLIP_RATE

            # Flip: the features are standardized so we can't simply do 1-x
            # Instead we negate the standardized value (equivalent to flipping direction)
            X_perturbed[flip_mask, feat_idx] = -X_perturbed[flip_mask, feat_idx]

            perturbed_preds = model.predict(X_perturbed)
            trial_accs.append(accuracy_score(y_test, perturbed_preds))
            trial_change_rates.append(
                np.mean(perturbed_preds != baseline_preds)
            )

        results.append({
            "Feature": feat_name,
            "Baseline Acc": f"{baseline_acc*100:.1f}%",
            "Acc Under Perturbation": f"{np.mean(trial_accs)*100:.1f}%",
            "Acc Variance": f"{np.var(trial_accs)*100:.4f}%",
            "Prediction Change Rate": f"{np.mean(trial_change_rates)*100:.1f}%",
        })

    results_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_PATH, "robustness_results.csv")
    results_df.to_csv(out_path, index=False)

    print("\n[robustness] PERTURBATION ROBUSTNESS RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nSaved to {out_path}")
    print("\n>>> Copy these numbers into the Robustness table in README.md")


if __name__ == "__main__":
    run_robustness_test()