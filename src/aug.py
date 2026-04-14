"""
aug.py — all 7 augmentation strategies in one place.
Import get_augmenter(name) anywhere in the project.
"""
from imblearn.over_sampling import (
    SMOTE, ADASYN, BorderlineSMOTE,
    RandomOverSampler, KMeansSMOTE
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import RANDOM_SEED

AUGMENTATION_METHODS = [
    "smote",
    "kmeans_smote",
    "adasyn",
    "borderline_smote",
    "smote_enn",
    "random_over",
    "random_under",
]


def get_augmenter(method: str):
    """
    Returns an imblearn resampler object for the given method name.
    All augmenters use RANDOM_SEED from config for reproducibility.
    """
    augmenters = {
        "smote": SMOTE(random_state=RANDOM_SEED),
        "kmeans_smote": KMeansSMOTE(
            random_state=RANDOM_SEED,
            cluster_balance_threshold=0.1
        ),
        "adasyn": ADASYN(random_state=RANDOM_SEED),
        "borderline_smote": BorderlineSMOTE(random_state=RANDOM_SEED),
        "smote_enn": SMOTEENN(random_state=RANDOM_SEED),
        "random_over": RandomOverSampler(random_state=RANDOM_SEED),
        "random_under": RandomUnderSampler(random_state=RANDOM_SEED),
    }

    if method not in augmenters:
        raise ValueError(
            f"Unknown augmentation method: '{method}'. "
            f"Choose from: {list(augmenters.keys())}"
        )

    return augmenters[method]


if __name__ == "__main__":
    for name in AUGMENTATION_METHODS:
        aug = get_augmenter(name)
        print(f"OK: {name} -> {type(aug).__name__}")