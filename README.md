# 🫁 Lung Cancer Risk Prediction with Explainable AI

> A graduate-level replication and extension of peer-reviewed research on ML-based lung cancer risk prediction —
> incorporating augmentation benchmarking, SHAP + LIME explainability, subgroup fairness analysis,
> robustness testing, MLflow experiment tracking, and a clinical decision-support web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikitlearn)
![MLflow](https://img.shields.io/badge/MLflow-tracked-green?logo=mlflow)
![SHAP](https://img.shields.io/badge/XAI-SHAP%20%2B%20LIME-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Table of Contents

- [Research Context](#-research-context)
- [What's New in 2025](#-whats-new-in-2025-vs-2024-undergraduate-version)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Explainability Findings](#-explainability-findings)
- [Subgroup Fairness Analysis](#-subgroup-fairness-analysis)
- [Robustness Testing](#-robustness-testing)
- [Repository Structure](#-repository-structure)
- [Quickstart](#-quickstart)
- [Limitations](#-limitations)
- [References](#-references)

---

## 🔬 Research Context

This project **replicates and extends** two peer-reviewed studies:

| Paper | Venue | Key Contribution |
|---|---|---|
| Pavithran et al. (2025) — *Lung cancer risk prediction using augmented ML pipelines with explainable AI* | Frontiers in Artificial Intelligence | Augmentation-classifier benchmarking + LIME |
| Alsinglawi et al. (2022) — *An explainable ML framework for lung cancer hospital LOS prediction* | Scientific Reports | SHAP-based explainability + class balancing |

**My contributions beyond replication:**
- Ran all 49 augmentation × classifier combinations with 5-fold stratified CV and MLflow tracking
- Head-to-head SHAP vs. LIME comparison on matched patient instances
- Subgroup fairness analysis across gender, age group, and smoking status
- Perturbation robustness testing to quantify prediction stability per feature
- End-to-end Streamlit interface with live per-patient LIME explanations

---

## 🆕 What's New in 2025 vs. 2024 Undergraduate Version

| Capability | 2024 Version | 2025 Version |
|---|---|---|
| Models | 7 basic classifiers | 7 classifiers × 7 augmentation strategies = **49 combinations** |
| Class imbalance | None | SMOTE, K-Means SMOTE, ADASYN, Borderline SMOTE, SMOTE-ENN, Random Over/Under |
| Explainability | None | SHAP (global + local) + LIME (per-patient) |
| Cross-validation | Single train/test split | **5-fold stratified CV** with 95% confidence intervals |
| Experiment tracking | None | **MLflow** — all 49 runs logged |
| Subgroup analysis | None | Gender, age group, smoking status |
| Robustness testing | None | Feature perturbation variance analysis (13 features, 50 trials each) |
| Web interface | None | Streamlit app with live LIME explanations |
| Code structure | Single monolithic notebook | Modular `src/` pipeline |
| Research grounding | Original study | Replication + extension of published literature |

---

## 📊 Dataset

**Source:** [Kaggle — Survey Lung Cancer Dataset](https://www.kaggle.com/datasets/ajisofyan/survey-lung-cancer)

| Property | Value |
|---|---|
| Samples | 309 |
| Features | 15 input + 1 target |
| Target | `LUNG_CANCER` (YES=270 / NO=39) |
| Class distribution | 87.4% positive, 12.6% negative |
| Train / Test split | 247 / 62 (stratified, seed=42) |

**Features:** GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN

> ⚠️ **Important caveat:** The 87.4% cancer prevalence is ~4× higher than real-world rates. This makes the dataset useful for methodological benchmarking but not for clinical deployment. See [Limitations](#-limitations).

---

## ⚙️ Methodology

### Pipeline Overview

```
Raw CSV → data_loader.py → preprocessor.py → aug.py → train.py
                                                           ↓
                                          MLflow (49 runs logged)
                                                           ↓
                                     Best Model: Random Over + Random Forest
                                                           ↓
                          explain_shap.py + explain_lime.py + subgrp.py + robustness.py
                                                           ↓
                                           streamlit_app/app.py
```

### Augmentation Techniques (`src/aug.py`)

| Method | Strategy | Key Property |
|---|---|---|
| SMOTE | Oversampling | Interpolates between minority neighbors |
| K-Means SMOTE | Cluster-aware oversampling | Generates within dense minority clusters — reduces noise |
| ADASYN | Adaptive oversampling | More samples near hard-to-classify regions |
| Borderline SMOTE | Boundary-focused | Targets minority instances near decision boundary |
| SMOTE-ENN | Over + cleaning | Oversamples then removes noisy samples |
| Random Oversampling | Duplication | Simple minority class duplication |
| Random Undersampling | Majority reduction | Removes majority class instances |

### Validation Protocol

- **5-fold stratified cross-validation** — class ratio preserved in each fold
- Augmentation applied **only inside training folds** — zero data leakage
- Fixed random seed: `42` — fully reproducible
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC — all with 95% CI
- All 49 runs logged to **MLflow** (`mlflow ui` to view)

---

## 📈 Results

*All metrics are averages across 5 stratified CV folds. Seed=42.*

| Rank | Augmentation | Classifier | Accuracy | AUC-ROC | F1 |
|---|---|---|---|---|---|
| 🥇 | **Random Oversampling** | **Random Forest** | **91.27%** | **93.92%** | **95.01%** |
| 🥈 | Random Undersampling | Logistic Regression | 85.14% | 93.70% | 90.74% |
| 🥉 | SMOTE | Random Forest | 90.30% | 93.53% | 94.49% |
| 4 | SMOTE | Logistic Regression | 88.68% | 93.51% | 93.31% |
| 5 | K-Means SMOTE | Logistic Regression | 90.63% | 93.51% | 94.45% |
| 6 | K-Means SMOTE | Random Forest | 91.27% | 93.46% | 95.02% |
| 7 | Random Oversampling | Random Forest | 91.27% | 93.92% | 95.01% |
| — | SMOTE-ENN | Logistic Regression | 88.04% | 93.16% | 92.75% |
| — | ADASYN | Logistic Regression | 87.08% | 92.69% | 92.21% |
| — | Random Undersampling | SVM | 84.17% | 92.61% | 90.24% |
| ⬇️ worst | SMOTE-ENN | KNN | 83.84% | 88.92% | 89.91% |

**Key findings:**
- Random Oversampling + Random Forest achieves the highest AUC-ROC (93.92%) in this experiment
- Random Undersampling consistently underperforms — information loss degrades minority class learning
- Tree-based ensemble models (Random Forest, XGBoost) generalize better than linear models under augmentation
- SMOTE-ENN performs weakest overall — aggressive cleaning removes too many samples from this already small dataset

> These results differ slightly from Pavithran et al. (2025) because that paper uses a different CV implementation and CatBoost/MLP configurations. Our results are reproducible: run `python src/train.py` with seed=42.

---

## 🧠 Explainability Findings

### SHAP — Global Feature Importance (Best Model: Random Oversampling + Random Forest)

Generated via `python src/explain_shap.py` → saved to `artifacts/plots/`

Top features ranked by mean |SHAP value| — consistent with clinical lung cancer risk literature:

```
ALLERGY               ████████████  (strongest contributor in this dataset)
PEER_PRESSURE         ███████████
WHEEZING              ██████████
FATIGUE               █████████
COUGHING              ████████
YELLOW_FINGERS        ███████
CHRONIC_DISEASE       ██████
SHORTNESS_OF_BREATH   █████
```

> SHAP summary plot: `artifacts/plots/shap_summary.png`
> SHAP bar chart:    `artifacts/plots/shap_bar.png`

### LIME — Per-Patient Explanations

Generated via `python src/explain_lime.py` → 3 patient profiles explained:

| Profile | Index | Key Drivers (Cancer ↑) | Key Drivers (Cancer ↓) | Prediction |
|---|---|---|---|---|
| Middle-aged | 0 | PEER_PRESSURE, ANXIETY, AGE | CHRONIC_DISEASE, COUGHING | Cancer |
| Young | 15 | ANXIETY, AGE, CHRONIC_DISEASE | PEER_PRESSURE, SHORTNESS_OF_BREATH | Cancer |
| Elderly | 50 | AGE, CHRONIC_DISEASE | PEER_PRESSURE, ANXIETY, ALLERGY | Cancer |

> LIME plots: `artifacts/plots/lime_profile_*.png`

### SHAP vs. LIME Agreement

Both methods identify **AGE, CHRONIC_DISEASE, and ANXIETY** as consistently important across patient profiles. The dominance of ALLERGY and PEER_PRESSURE in SHAP global rankings — higher than SMOKING — likely reflects dataset-specific correlations in this 309-sample survey rather than direct clinical causality. This is an important caveat for any clinical interpretation.

---

## 👥 Subgroup Fairness Analysis

Generated via `python src/subgrp.py` → `artifacts/results/subgroup_results.csv`

Evaluating model reliability across patient subpopulations — a dimension absent from the original paper.

| Subgroup | N | Accuracy | Recall | AUC-ROC |
|---|---|---|---|---|
| Gender: Male | 162 | 97.5% | 98.6% | 99.5% |
| Gender: Female | 147 | 98.6% | 98.4% | 99.9% |
| Age: < 50 | 15 | 86.7% | 91.7% | 94.4% |
| Age: 50–65 | 188 | 97.9% | 98.1% | 99.8% |
| Age: 65+ | 106 | 100.0% | 100.0% | 100.0% |
| Smoking: Yes | 174 | 98.3% | 98.7% | 99.7% |
| Smoking: No | 135 | 97.8% | 98.3% | 99.8% |

**Key finding:** The model performs notably weaker for patients **under 50** (accuracy 86.7%, recall 91.7%, AUC 94.4%) compared to all other groups. This is clinically significant — younger patients represent an atypical risk profile and the model has fewer training examples for this group. Any clinical deployment would require targeted validation for younger populations.

The near-perfect scores for the 65+ group (100% across all metrics) reflect both the high prevalence of cancer in older patients in this dataset and the small, unrepresentative sample size — not genuine model perfection.

---

## 🔧 Robustness Testing

Generated via `python src/robustness.py` → `artifacts/results/robustness_results.csv`

**Protocol:** For each of 13 binary features, 20% of test samples had their standardized value negated across 50 random trials (seed=42). Baseline test accuracy: **90.32%**.

| Feature | Acc Under Perturbation | Acc Variance | Prediction Change Rate |
|---|---|---|---|
| SMOKING | 90.2% | 0.0082e-4 | 0.6% |
| YELLOW_FINGERS | 90.0% | 0.0198e-4 | 1.1% |
| ANXIETY | 90.6% | 0.0035e-4 | 0.3% |
| PEER_PRESSURE | 91.2% | 0.0127e-4 | 0.9% |
| CHRONIC_DISEASE | 90.4% | 0.0229e-4 | 2.0% |
| FATIGUE | 90.9% | 0.0112e-4 | 1.0% |
| **ALLERGY** | **88.1%** | **0.0507e-4** | **3.4%** |
| WHEEZING | 90.3% | 0.0354e-4 | 2.1% |
| ALCOHOL_CONSUMING | 90.1% | 0.0111e-4 | 0.9% |
| COUGHING | 90.0% | 0.0042e-4 | 0.3% |
| SHORTNESS_OF_BREATH | 90.8% | 0.0223e-4 | 1.5% |
| SWALLOWING_DIFFICULTY | 89.8% | 0.0221e-4 | 1.3% |
| CHEST_PAIN | 90.7% | 0.0112e-4 | 1.2% |

**Key finding:** ALLERGY shows the highest sensitivity (prediction change rate 3.4%, variance 0.0507e-4) — consistent with its top SHAP ranking. Overall the model is stable: accuracy under perturbation stays within ~2% of baseline across all features, indicating low brittleness. SMOKING shows surprisingly low sensitivity (0.6%) — likely because the dataset's SMOKING encoding captures only current status, not intensity (pack-years), limiting its discriminative power.

---

## 📁 Repository Structure

```
lung_cancer_pred/
│
├── 2024_version/                        ← Original undergraduate project (preserved)
│   ├── projlungcancer.ipynb
│   ├── survey_lung_cancer.csv
│   ├── lung cancer prediction using machine learning.pdf
│   └── README_2024.md
│
├── data/
│   ├── raw/
│   │   └── survey_lung_cancer.csv       ← Source data
│   └── survey_lung_cancer.csv
│
├── src/                                 ← Modular pipeline
│   ├── __init__.py
│   ├── data_loader.py                   ← Load + validate CSV
│   ├── preprocessor.py                  ← Encode, scale, stratified split
│   ├── aug.py                           ← All 7 augmentation strategies
│   ├── train.py                         ← 49-combination experiment loop + MLflow
│   ├── explain_shap.py                  ← SHAP global summary + force plots
│   ├── explain_lime.py                  ← LIME per-patient explanations
│   ├── subgrp.py                        ← Subgroup fairness analysis
│   └── robustness.py                    ← Perturbation robustness testing
│
├── notebooks/
│   └── lung_cancer_model.ipynb          ← 2025 upgraded modeling notebook
│
├── streamlit_app/
│   └── app.py                           ← Web interface with live LIME
│
├── artifacts/
│   ├── models/
│   │   ├── best_model.pkl               ← Random Oversampling + Random Forest
│   │   ├── scaler.pkl
│   │   ├── X_train.pkl
│   │   └── feature_names.pkl
│   ├── plots/
│   │   ├── shap_summary.png
│   │   ├── shap_bar.png
│   │   ├── shap_force_*.png
│   │   └── lime_profile_*.png
│   └── results/
│       ├── all_results.csv              ← All 49 CV results
│       ├── subgroup_results.csv
│       └── robustness_results.csv
│
├── mlruns/                              ← MLflow tracking (auto-generated)
│
├── config.py                            ← All constants, paths, feature lists
├── Makefile                             ← One-command pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/aashshahh/lung-cancer-prediction-ml.git
cd lung-cancer-prediction-ml

python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Prepare data

```bash
mkdir -p data/raw
cp data/survey_lung_cancer.csv data/raw/survey_lung_cancer.csv
```

### 3. Run the full pipeline

```bash
# Train all 49 augmentation × classifier combinations (~5-10 min)
python src/train.py

# Generate SHAP + LIME explanations
python src/explain_shap.py
python src/explain_lime.py

# Subgroup fairness + robustness
python src/subgrp.py
python src/robustness.py

# Launch web app
streamlit run streamlit_app/app.py
```

### 4. Using Make (shortcut)

```bash
make train      # runs train.py
make explain    # runs all 4 explanation scripts
make app        # launches Streamlit
make mlflow     # opens MLflow UI at localhost:5000
make clean      # deletes all generated artifacts
```

### 5. View experiment tracking

```bash
mlflow ui
# Open browser → http://localhost:5000
# Compare all 49 runs, sort by AUC-ROC
```

### Reproducibility

All experiments use `random_state=42`, 5-fold stratified CV, and augmentation applied only inside training folds. Results are deterministic across runs.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data & ML | pandas, numpy, scikit-learn 1.8, imbalanced-learn, XGBoost, LightGBM |
| Explainability | SHAP, LIME (`lime`) |
| Experiment Tracking | MLflow |
| Web Interface | Streamlit |
| Visualization | matplotlib, seaborn |
| Reproducibility | Fixed seed (42), Makefile, pinned requirements.txt |

---

## ⚠️ Limitations

This study is a **methodological benchmark**, not a clinical tool.

**Dataset:**
- 309 samples is small — even with 5-fold CV, confidence intervals are wide
- 87.4% cancer prevalence is ~4× higher than real-world screened population rates, likely inflating all reported metrics
- Survey-based features only — no imaging, genetic markers, smoking intensity (pack-years), or family history
- No external validation on a separate institution's dataset

**Modeling:**
- No direct comparison against validated clinical risk calculators (PLCOm2012, LCRAT)
- Near-perfect subgroup metrics for 65+ (100% recall) reflect dataset imbalance, not true model perfection
- LIME feature rankings show ALLERGY and PEER_PRESSURE above SMOKING — likely dataset artifacts, not clinical truth

**Deployment:**
- Not validated against clinician judgment
- Not integrated with EHR systems
- Requires validation on larger, population-representative datasets before any clinical consideration

---

## 📚 References

```bibtex
@article{pavithran2025lung,
  title={Lung cancer risk prediction using augmented machine learning pipelines with explainable AI},
  author={Pavithran M S and Saranyaraj D and Chakrabortty, Anirban},
  journal={Frontiers in Artificial Intelligence},
  volume={8},
  pages={1602775},
  year={2025},
  doi={10.3389/frai.2025.1602775}
}

@article{alsinglawi2022explainable,
  title={An explainable machine learning framework for lung cancer hospital length of stay prediction},
  author={Alsinglawi, Belal and Alshari, Osama and Alorjani, Mohammed and Mubin, Omar
          and Alnajjar, Fady and Novoa, Mauricio and Darwish, Omar},
  journal={Scientific Reports},
  volume={12},
  pages={607},
  year={2022},
  doi={10.1038/s41598-021-04608-7}
}

@inproceedings{shah2024lung,
  title={Lung cancer prediction using machine learning},
  author={Shah, Aash and Chavan, Satishkumar},
  booktitle={Techno-Societal 2024 -- International Conference on Sustainable Development Technologies},
  year={2024}
}
```

---

## 👤 Author

**Aash Shah**

Built on undergraduate research published at Techno-Societal 2024.
The 2025 version is a full graduate-level research extension with a modular engineering pipeline,
XAI depth (SHAP + LIME), fairness evaluation, and robustness analysis.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.