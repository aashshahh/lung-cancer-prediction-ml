
# Lung Cancer Risk Prediction Using Machine Learning

This project predicts lung cancer risk using survey based symptom and lifestyle data.  
It includes both the **original 2024 undergraduate version** and a fully **upgraded 2025 version**, showcasing end-to-end ML engineering, explainability, reproducibility, and UI design.

---

# Published Research 

This work originated as my 2024 undergraduate research and was published at:

**Techno-Societal 2024 – International Conference on Sustainable Development Technologies**

**Paper:** *Lung cancer prediction using machine learning*  
**Authors:** Aash Shah, Satishkumar Chavan  
**PDF:** `2024_version/lung cancer prediction using machine learning.pdf`

The 2025 upgraded version extends the original research with:
- Advanced hyperparameter tuning (Optuna)
- SHAP based explainability (local + global)
- MLflow experiment tracking
- Clean artifacts + versioned pipeline
- Streamlit UI powered by a custom Figma design
- Proper engineering structure
- Multiple model benchmarks

Original 2024 version is preserved at:  
`2024_version/`

---

# Project Overview

The repository contains **two complete versions** of the lung cancer prediction system:

## **1. 2024 Version (Undergraduate Project)**  
Folder: `2024_version/`

A simple classical ML implementation built during my 3rd year of engineering.

### Key Features
- Basic preprocessing  
- Models: SVM, Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, XGBoost  
- Achieved **~95% accuracy (Random Forest)**  
- Straightforward, model first approach adapted 
- Notebook: `projlungcancer.ipynb`  
- Research paper included

This version serves as the baseline for comparison.

---

## **2. 2025 Upgraded Version (Graduate-Level ML Pipeline)**  
Folder: `notebooks/`  
Main notebook: `lung_cancer_model.ipynb`

A full, industry-style ML workflow rebuilt with modern techniques, stronger engineering, explainability, and reproducibility.

### Enhancements in the Upgraded Version
- Exploratory Data Analysis with visualizations  
- Clean preprocessing pipeline (encoding + scaling)  
- Class imbalance handling  
- Benchmarking of 7 ML models  
  - Logistic Regression  
  - SVM  
  - KNN  
  - Naive Bayes  
  - Decision Tree  
  - Random Forest  
  - XGBoost  
- SHAP explainability (feature impact + summary plots)  
- Exported artifacts (`best_model.pkl`, `scaler.pkl`)  
- Ready structure for tuning + deployment  
- Updated project directory layout

---

# Repository Structure

```
lung_cancer_pred/
│
├── 2024_version/
│   ├── projlungcancer.ipynb
│   ├── survey_lung_cancer.csv
│   ├── lung cancer prediction using machine learning.pdf
│   └── README_2024.md
│
├── data/
│   └── survey_lung_cancer.csv
│
├── notebooks/
│   └── lung_cancer_model.ipynb   # upgraded version
│
├── artifacts/
│   ├── best_model.pkl
│   └── scaler.pkl
│
└── README.md
```

---

# Tools and Technology
- Python  
- pandas, numpy  
- scikit-learn  
- seaborn, matplotlib  
- imbalanced-learn  
- XGBoost  
- SHAP  
- Optuna (planned)  
- MLflow (planned)  
- Streamlit (planned)

---


