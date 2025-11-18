# Lung Cancer Risk Prediction Using ML

Built a classification model to predict lung cancer risk based on lifestyle and symptom data.

## Highlights
- Dataset: 15+ features including smoking history, fatigue, wheezing, etc.
- Models used: Random Forest, SVM, Decision Tree
- Achieved 94.96% accuracy using Random Forest

## Tools Used
Python, pandas, scikit-learn, seaborn, matplotlib

## Paper
Presented at Techno-Societal 2024. Full paper available on LinkedIn Experience section.

# Lung Cancer Risk Prediction – ML Project

A complete ML pipeline for predicting lung cancer risk using survey-based symptom and lifestyle data.  
This repository contains **two versions** of the project**:**

1. **2024 Undergrad Version** – the original model built during BTech  
2. **2025 Upgraded Version** – fully rebuilt with grad-level enhancements, explainability, benchmarking, and artifacts

This structure showcases the technical growth from early ML basics → deeper, research-style modeling.

---

## 1. 2024 Version (Undergraduate Project)
**Folder:** `2024_version/`  

Built during the 3rd year of engineering using basic Python ML techniques.

### Highlights
- Simple preprocessing  
- Basic ML models (SVM, Logistic Regression, Random Forest)  
- ~95% accuracy  
- Focused mostly on model performance  
- No reproducible pipeline or explainability  
- Notebook: `projlungcancer.ipynb`

### Purpose
This version shows the original implementation and serves as a baseline for further improvements.

---

## 2. 2025 Upgraded Version (Graduate-Level Project)
**Folder:** `notebooks/`  
Main notebook: `lung_cancer_model.ipynb`

A fully re-engineered version reflecting best practices used in real ML workflows.

### What’s New
- Full Exploratory Data Analysis (EDA)
- Clean preprocessing pipeline (encoding + scaling)
- 7-model benchmarking:
  - Logistic Regression  
  - SVM  
  - KNN  
  - Naive Bayes  
  - Decision Tree  
  - Random Forest  
  - XGBoost  
- SHAP explainability (feature contributions)
- Model + scaler exported to `artifacts/`
- Reproducible project structure
- Versioned datasets and clean directory layout

### Upcoming Enhancements
- Hyperparameter tuning (Optuna)  
- Cross-validation  
- MLflow experiment tracking  
- Streamlit mini-app for live predictions  
- Class imbalance handling  
- Feature selection  
- Partial dependence plots  
- Interpretability narrative for report

---

## 📂 Repository Structure
lung_cancer_pred/
│── 2024_version/
│ ├── projlungcancer.ipynb
│ ├── survey_lung_cancer.csv
│ └── README_2024.md
│
│── data/
│ └── survey_lung_cancer.csv
│
│── notebooks/
│ └── lung_cancer_model.ipynb <-- main upgraded version
│
│── artifacts/
│ ├── best_model.pkl
│ └── scaler.pkl
│
│── README.md <-- you are reading this

