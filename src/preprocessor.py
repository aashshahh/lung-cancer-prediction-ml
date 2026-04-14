"""
preprocessor.py — encode, scale, split.
Always call preprocess(df) and get back train/test arrays + scaler.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import FEATURES, TARGET, TEST_SIZE, RANDOM_SEED


def preprocess(df: pd.DataFrame):
    """
    Encode categorical columns, scale all features, do stratified split.

    Returns
    -------
    X_train, X_test : np.ndarray  (scaled)
    y_train, y_test : np.ndarray
    scaler          : fitted StandardScaler
    feature_names   : list[str]
    """
    df = df.copy()

    # --- Encode target ---
    df[TARGET] = df[TARGET].map({"YES": 1, "NO": 0})
    if df[TARGET].isnull().any():
        raise ValueError(
            f"[preprocessor] Unknown values in {TARGET}. "
            f"Expected YES/NO, got: {df[TARGET].unique()}"
        )

    # --- Encode GENDER ---
    df["GENDER"] = df["GENDER"].map({"M": 1, "F": 0})
    if df["GENDER"].isnull().any():
        # Try alternate encoding
        df["GENDER"] = df["GENDER"].map({1: 1, 2: 0, "MALE": 1, "FEMALE": 0})

    # --- Select features and target ---
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"[preprocessor] Missing columns: {missing}")

    X = df[FEATURES].values.astype(float)
    y = df[TARGET].values.astype(int)

    # --- Train / test split (stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # --- Scale (fit ONLY on train) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[preprocessor] Train : {X_train.shape[0]} samples  "
          f"| class balance {np.bincount(y_train)}")
    print(f"[preprocessor] Test  : {X_test.shape[0]}  samples  "
          f"| class balance {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test, scaler, FEATURES


if __name__ == "__main__":
    from src.data_loader import load_data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)
    print("Preprocessing done.")