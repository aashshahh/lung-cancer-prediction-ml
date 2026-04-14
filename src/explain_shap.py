"""
explain_shap.py — SHAP global + local explanations for best model.
Run with: python src/explain_shap.py
"""
import os
import sys
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import FEATURES, MODELS_PATH, PLOTS_PATH
from dataloader import load_data
from preprocessor import preprocess
os.makedirs(PLOTS_PATH, exist_ok=True)

# Patient profiles for individual explanations
PROFILES = {
    "middle_aged_smoker": 0,
    "young_nonsmoker": 15,
    "elderly_comorbid": 50,
}


def run_shap():
    print("[shap] Loading model and data...")

    model_path = os.path.join(MODELS_PATH, "best_model.pkl")
    if not os.path.exists(model_path):
        print("[shap] ERROR: No best_model.pkl found. Run src/train.py first.")
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = load_data()
    X_train, X_test, y_train, y_test, scaler, _ = preprocess(df)

    print(f"[shap] Model type: {type(model).__name__}")

    # Choose explainer based on model type
    model_name = type(model).__name__
    try:
        if model_name in ["RandomForestClassifier", "GradientBoostingClassifier",
                          "XGBClassifier", "LGBMClassifier"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            # For tree models, shap_values may be a list [class0, class1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            expected_value = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )
        else:
            # For MLP, SVM, LR — use KernelExplainer (slower but universal)
            print("[shap] Using KernelExplainer (this may take 1-2 min for MLP/SVM)...")
            background = shap.sample(X_train, 50)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values_raw = explainer.shap_values(X_test[:30])
            shap_values = shap_values_raw[1]
            expected_value = explainer.expected_value[1]
            X_test = X_test[:30]  # trim for speed

    except Exception as e:
        print(f"[shap] Explainer failed: {e}")
        print("[shap] Falling back to KernelExplainer...")
        background = shap.sample(X_train, 30)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values_raw = explainer.shap_values(X_test[:20])
        shap_values = shap_values_raw[1]
        expected_value = explainer.expected_value[1]
        X_test = X_test[:20]

    # 1. Global summary plot
    print("[shap] Generating summary plot...")
    plt.figure()
    shap.summary_plot(
        shap_values, X_test,
        feature_names=FEATURES,
        show=False, plot_size=(10, 6)
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, "shap_summary.png"), dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_PATH}shap_summary.png")

    # 2. Bar plot (mean |SHAP|)
    plt.figure()
    shap.summary_plot(
        shap_values, X_test,
        feature_names=FEATURES,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, "shap_bar.png"), dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_PATH}shap_bar.png")

    # 3. Individual force plots
    for label, idx in PROFILES.items():
        if idx >= len(X_test):
            idx = 0
        try:
            shap.force_plot(
                expected_value,
                shap_values[idx],
                X_test[idx],
                feature_names=FEATURES,
                matplotlib=True,
                show=False
            )
            plt.savefig(
                os.path.join(PLOTS_PATH, f"shap_force_{label}.png"),
                dpi=150, bbox_inches="tight"
            )
            plt.close()
            print(f"  Saved: shap_force_{label}.png")
        except Exception as e:
            print(f"  [WARN] Force plot failed for {label}: {e}")

    print("\n[shap] Done. All plots saved to artifacts/plots/")


if __name__ == "__main__":
    run_shap()