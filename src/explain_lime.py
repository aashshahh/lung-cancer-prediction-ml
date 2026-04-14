"""
explain_lime.py — LIME per-patient explanations for 3 patient profiles.
Run with: python src/explain_lime.py
"""
import os
import sys
import pickle
import warnings
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import FEATURES, MODELS_PATH, PLOTS_PATH
from dataloader import load_data
from preprocessor import preprocess

os.makedirs(PLOTS_PATH, exist_ok=True)

PROFILES = {
    "profile_1_smoker": 0,
    "profile_2_young": 15,
    "profile_3_elderly": 50,
}


def run_lime():
    print("[lime] Loading model and data...")

    model_path = os.path.join(MODELS_PATH, "best_model.pkl")
    if not os.path.exists(model_path):
        print("[lime] ERROR: No best_model.pkl found. Run src/train.py first.")
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = load_data()
    X_train, X_test, _, _, _, _ = preprocess(df)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=FEATURES,
        class_names=["No Cancer", "Cancer"],
        mode="classification",
        random_state=42,
    )

    for label, idx in PROFILES.items():
        if idx >= len(X_test):
            idx = 0

        print(f"[lime] Explaining {label} (test index {idx})...")
        exp = explainer.explain_instance(
            X_test[idx],
            model.predict_proba,
            num_features=10,
        )

        fig = exp.as_pyplot_figure()
        fig.set_size_inches(10, 6)
        plt.title(f"LIME Explanation — {label.replace('_', ' ').title()}")
        plt.tight_layout()
        out_path = os.path.join(PLOTS_PATH, f"lime_{label}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

        # Print feature weights to terminal
        print(f"  Top features:")
        for feat, weight in exp.as_list():
            direction = "↑ Cancer" if weight > 0 else "↓ No Cancer"
            print(f"    {feat}: {weight:+.4f} ({direction})")
        print()

    print("[lime] Done. All plots saved to artifacts/plots/")


if __name__ == "__main__":
    run_lime()