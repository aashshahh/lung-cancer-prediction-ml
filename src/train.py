"""
train.py — runs all augmentation × classifier combinations.
Logs every run to MLflow. Saves best model to artifacts/models/.
Run with: python src/train.py
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import mlflow

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RANDOM_SEED, CV_FOLDS, MODELS_PATH, RESULTS_PATH
from dataloader import load_data
from preprocessor import preprocess
from aug import get_augmenter, AUGMENTATION_METHODS

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


def get_classifier(name: str):
    classifiers = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED
        ),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "xgboost": XGBClassifier(
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            verbosity=0
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=RANDOM_SEED
        ),
        "svm": SVC(probability=True, random_state=RANDOM_SEED),
        "lightgbm": LGBMClassifier(
            random_state=RANDOM_SEED, verbose=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=RANDOM_SEED
        ),
    }
    return classifiers[name]


CLASSIFIERS = list(get_classifier.__code__.co_consts)  # not ideal, just list them:
CLASSIFIER_NAMES = [
    "logistic_regression", "knn", "xgboost",
    "random_forest", "mlp", "svm", "lightgbm"
]


def run_one_experiment(X, y, augment_name: str, clf_name: str) -> dict:
    """
    Runs one augmentation + classifier combo with 5-fold stratified CV.
    Augmentation is applied ONLY inside training folds (no data leakage).
    Returns a dict of averaged metrics with 95% CI.
    """
    skf = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )
    augmenter = get_augmenter(augment_name)
    clf = get_classifier(clf_name)

    acc, auc, f1, rec, prec = [], [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val = X[train_idx], X[val_idx]
        y_train_fold, y_val = y[train_idx], y[val_idx]

        # Augment only training fold — this is the key line
        try:
            X_res, y_res = augmenter.fit_resample(X_train_fold, y_train_fold)
        except Exception as e:
            print(f"  [WARN] Augmentation failed for {augment_name}: {e}. Using original.")
            X_res, y_res = X_train_fold, y_train_fold

        clf.fit(X_res, y_res)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]

        acc.append(accuracy_score(y_val, y_pred))
        auc.append(roc_auc_score(y_val, y_prob))
        f1.append(f1_score(y_val, y_pred, zero_division=0))
        rec.append(recall_score(y_val, y_pred, zero_division=0))
        prec.append(precision_score(y_val, y_pred, zero_division=0))

    n = CV_FOLDS
    result = {
        "augmentation": augment_name,
        "classifier": clf_name,
        "accuracy": round(np.mean(acc) * 100, 2),
        "accuracy_ci": round(1.96 * np.std(acc) / np.sqrt(n) * 100, 2),
        "auc_roc": round(np.mean(auc) * 100, 2),
        "auc_roc_ci": round(1.96 * np.std(auc) / np.sqrt(n) * 100, 2),
        "f1": round(np.mean(f1) * 100, 2),
        "recall": round(np.mean(rec) * 100, 2),
        "precision": round(np.mean(prec) * 100, 2),
    }
    return result


def main():
    print("=" * 60)
    print("LUNG CANCER RISK PREDICTION — FULL EXPERIMENT")
    print("=" * 60)

    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    # Combine train+test for CV (CV does its own splitting)
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    all_results = []
    best_auc = 0
    best_model = None
    best_combo = ("", "")

    mlflow.set_experiment("lung_cancer_xai")

    total = len(AUGMENTATION_METHODS) * len(CLASSIFIER_NAMES)
    done = 0

    for aug_name in AUGMENTATION_METHODS:
        for clf_name in CLASSIFIER_NAMES:
            done += 1
            print(f"[{done}/{total}] Running: {aug_name} + {clf_name} ...")

            with mlflow.start_run(run_name=f"{aug_name}__{clf_name}"):
                result = run_one_experiment(X_all, y_all, aug_name, clf_name)
                all_results.append(result)

                mlflow.log_params({
                    "augmentation": aug_name,
                    "classifier": clf_name,
                    "cv_folds": CV_FOLDS,
                    "random_seed": RANDOM_SEED,
                })
                mlflow.log_metrics({
                    "accuracy": result["accuracy"],
                    "auc_roc": result["auc_roc"],
                    "f1": result["f1"],
                    "recall": result["recall"],
                    "precision": result["precision"],
                })

                print(
                    f"    Accuracy: {result['accuracy']}% | "
                    f"AUC-ROC: {result['auc_roc']}% | "
                    f"F1: {result['f1']}%"
                )

                # Track best model
                if result["auc_roc"] > best_auc:
                    best_auc = result["auc_roc"]
                    best_combo = (aug_name, clf_name)

    # Save results table
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("auc_roc", ascending=False)
    results_path = os.path.join(RESULTS_PATH, "all_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Train best model on full training data and save
    print(f"\nBest combo: {best_combo[0]} + {best_combo[1]} (AUC-ROC: {best_auc}%)")
    best_augmenter = get_augmenter(best_combo[0])
    best_clf = get_classifier(best_combo[1])
    X_res_final, y_res_final = best_augmenter.fit_resample(X_train, y_train)
    best_clf.fit(X_res_final, y_res_final)

    with open(os.path.join(MODELS_PATH, "best_model.pkl"), "wb") as f:
        pickle.dump(best_clf, f)
    with open(os.path.join(MODELS_PATH, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_PATH, "X_train.pkl"), "wb") as f:
        pickle.dump(X_train, f)
    with open(os.path.join(MODELS_PATH, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)

    print("Best model saved to artifacts/models/")
    print("\nTop 5 combinations:")
    print(results_df[["augmentation", "classifier", "accuracy", "auc_roc", "f1"]].head())


if __name__ == "__main__":
    main()