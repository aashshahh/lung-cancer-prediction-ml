"""
data_loader.py — loads and validates the raw CSV.
Tries multiple path locations automatically so it works
whether you run from project root or inside src/.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import FEATURES, TARGET


def load_data(path: str = None) -> pd.DataFrame:
    """
    Load raw CSV. If path is None, searches common locations automatically.
    Returns a cleaned DataFrame with normalized column names.
    """
    if path is None:
        candidates = [
            "data/raw/survey_lung_cancer.csv",
            "data/survey_lung_cancer.csv",
            "../data/raw/survey_lung_cancer.csv",
            "../data/survey_lung_cancer.csv",
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
        if path is None:
            raise FileNotFoundError(
                "\n[data_loader] ERROR: Could not find survey_lung_cancer.csv.\n"
                "Run this from your project root and make sure the file is in data/ or data/raw/"
            )

    df = pd.read_csv(path)

    # Normalize column names: strip spaces, uppercase, underscores
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )

    print(f"[data_loader] Loaded : {path}")
    print(f"[data_loader] Shape  : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"[data_loader] Target distribution:\n{df[TARGET].value_counts()}\n")

    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())