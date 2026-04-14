"""
app.py — Lung Cancer Risk Prediction — Streamlit Web Interface
Run from project root: streamlit run streamlit_app/app.py
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(
    page_title="Lung Cancer Risk Predictor",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
    "PEER_PRESSURE", "CHRONIC_DISEASE", "FATIGUE", "ALLERGY",
    "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING",
    "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN",
]

MODELS_PATH  = "artifacts/models/"
RESULTS_PATH = "artifacts/results/"
PLOTS_PATH   = "artifacts/plots/"


@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        with open(os.path.join(MODELS_PATH, "best_model.pkl"), "rb") as f:
            artifacts["model"] = pickle.load(f)
        with open(os.path.join(MODELS_PATH, "scaler.pkl"), "rb") as f:
            artifacts["scaler"] = pickle.load(f)
        with open(os.path.join(MODELS_PATH, "X_train.pkl"), "rb") as f:
            artifacts["X_train"] = pickle.load(f)
        artifacts["loaded"] = True
    except FileNotFoundError:
        artifacts["loaded"] = False
    return artifacts


artifacts = load_artifacts()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 About")
    st.markdown("""
    **Lung Cancer Risk Predictor**

    Graduate ML project replicating and extending:
    - Pavithran et al. (2025) — *Frontiers in AI*
    - Alsinglawi et al. (2022) — *Scientific Reports*

    **Best Model:** Random Oversampling + Random Forest
    **AUC-ROC:** 93.92% (5-fold CV, seed=42)

    ---
    ⚠️ *Research only. Not a medical device.*
    """)
    st.markdown("---")
    st.markdown("[📄 Paper 1](https://doi.org/10.3389/frai.2025.1602775)")
    st.markdown("[📄 Paper 2](https://doi.org/10.1038/s41598-021-04608-7)")
    st.markdown("[💻 GitHub](https://github.com/aashshahh/lung-cancer-prediction-ml)")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🫁 Lung Cancer Risk Prediction with Explainable AI")
st.markdown(
    "Enter patient symptoms and demographics below to receive a risk prediction "
    "with a live LIME explanation showing which features drove the result."
)

if not artifacts["loaded"]:
    st.warning("⚠️ Model not found. Run `python src/train.py` first, then restart.")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Risk Prediction",
    "📊 Model Performance",
    "🧠 Explainability",
    "👥 Subgroup & Robustness",
])

# ── TAB 1: Prediction ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Patient Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender          = st.selectbox("Gender",             ["Male", "Female"])
        age             = st.slider("Age", 20, 90, 55)
        smoking         = st.selectbox("Smoking",            ["Yes", "No"])
        alcohol         = st.selectbox("Alcohol Consuming",  ["Yes", "No"])

    with col2:
        st.markdown("**Symptoms**")
        coughing        = st.selectbox("Coughing",                ["Yes", "No"])
        shortness       = st.selectbox("Shortness of Breath",     ["Yes", "No"])
        wheezing        = st.selectbox("Wheezing",                ["Yes", "No"])
        chest_pain      = st.selectbox("Chest Pain",              ["Yes", "No"])
        fatigue         = st.selectbox("Fatigue",                 ["Yes", "No"])
        swallowing      = st.selectbox("Swallowing Difficulty",   ["Yes", "No"])

    with col3:
        st.markdown("**Risk Factors**")
        yellow_fingers  = st.selectbox("Yellow Fingers",     ["Yes", "No"])
        anxiety         = st.selectbox("Anxiety",            ["Yes", "No"])
        peer_pressure   = st.selectbox("Peer Pressure",      ["Yes", "No"])
        chronic_disease = st.selectbox("Chronic Disease",    ["Yes", "No"])
        allergy         = st.selectbox("Allergy",            ["Yes", "No"])

    st.divider()
    predict_btn = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

    if predict_btn:
        if not artifacts["loaded"]:
            st.error("Model not loaded. Run `python src/train.py` first.")
        else:
            def yn(v): return 2 if v == "Yes" else 1
            def gn(v): return 1 if v == "Male" else 0

            input_array = np.array([[
                gn(gender), age, yn(smoking), yn(yellow_fingers),
                yn(anxiety), yn(peer_pressure), yn(chronic_disease),
                yn(fatigue), yn(allergy), yn(wheezing),
                yn(alcohol), yn(coughing), yn(shortness),
                yn(swallowing), yn(chest_pain),
            ]])

            model        = artifacts["model"]
            scaler       = artifacts["scaler"]
            input_scaled = scaler.transform(input_array)
            prediction   = model.predict(input_scaled)[0]
            probability  = model.predict_proba(input_scaled)[0][1]

            res1, res2 = st.columns([1, 1])

            with res1:
                if prediction == 1:
                    st.error("### ⚠️ HIGH RISK")
                else:
                    st.success("### ✅ LOW RISK")
                st.metric("Predicted Cancer Probability", f"{probability:.1%}")
                st.progress(float(probability))
                st.caption("⚠️ Research only. Consult a qualified physician.")

            with res2:
                st.markdown("**LIME Explanation**")
                try:
                    import lime
                    import lime.lime_tabular

                    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                        artifacts["X_train"],
                        feature_names=FEATURES,
                        class_names=["No Cancer", "Cancer"],
                        mode="classification",
                        random_state=42,
                    )
                    exp = lime_explainer.explain_instance(
                        input_scaled[0],
                        model.predict_proba,
                        num_features=8,
                    )
                    fig = exp.as_pyplot_figure()
                    fig.set_size_inches(7, 4)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.info(f"LIME unavailable: {e}")

# ── TAB 2: Results Table ──────────────────────────────────────────────────────
with tab2:
    st.subheader("All 49 Augmentation × Classifier Results")
    st.caption("5-fold stratified CV, seed=42, sorted by AUC-ROC.")

    results_path = os.path.join(RESULTS_PATH, "all_results.csv")
    if os.path.exists(results_path):
        df_res = pd.read_csv(results_path).sort_values("auc_roc", ascending=False).reset_index(drop=True)

        def highlight_best(row):
            return (["background-color: #1a472a; color: white"] * len(row)
                    if row.name == 0 else [""] * len(row))

        cols = ["augmentation", "classifier", "accuracy", "auc_roc", "f1", "recall", "precision"]
        st.dataframe(df_res[cols].style.apply(highlight_best, axis=1),
                     use_container_width=True, height=600)

        m1, m2, m3, m4 = st.columns(4)
        b = df_res.iloc[0]
        m1.metric("Best Accuracy",  f"{b['accuracy']}%")
        m2.metric("Best AUC-ROC",   f"{b['auc_roc']}%")
        m3.metric("Best F1",        f"{b['f1']}%")
        m4.metric("Best Combo",     f"{b['augmentation']} + {b['classifier']}")
    else:
        st.info("Run `python src/train.py` to generate this table.")
        st.dataframe(pd.DataFrame([
            {"Augmentation": "Random Oversampling", "Classifier": "Random Forest",      "Accuracy": "91.27%", "AUC-ROC": "93.92%", "F1": "95.01%"},
            {"Augmentation": "Random Undersampling","Classifier": "Logistic Regression", "Accuracy": "85.14%", "AUC-ROC": "93.70%", "F1": "90.74%"},
            {"Augmentation": "SMOTE",               "Classifier": "Random Forest",       "Accuracy": "90.30%", "AUC-ROC": "93.53%", "F1": "94.49%"},
        ]), use_container_width=True)

# ── TAB 3: Explainability ─────────────────────────────────────────────────────
with tab3:
    st.subheader("SHAP Global + LIME Per-Patient Explanations")

    e1, e2 = st.columns(2)

    with e1:
        st.markdown("**SHAP Summary Plot (Beeswarm)**")
        p = os.path.join(PLOTS_PATH, "shap_summary.png")
        st.image(p, use_container_width=True) if os.path.exists(p) else st.info("Run `python src/explain_shap.py`")

        st.markdown("**SHAP Feature Importance (Bar)**")
        p = os.path.join(PLOTS_PATH, "shap_bar.png")
        st.image(p, use_container_width=True) if os.path.exists(p) else st.info("Run `python src/explain_shap.py`")

    with e2:
        st.markdown("**LIME — Three Patient Profiles**")
        for label, fname in {
            "Profile 1 — Middle-aged Smoker": "lime_profile_1_smoker.png",
            "Profile 2 — Young Non-smoker":   "lime_profile_2_young.png",
            "Profile 3 — Elderly Patient":    "lime_profile_3_elderly.png",
        }.items():
            p = os.path.join(PLOTS_PATH, fname)
            if os.path.exists(p):
                st.markdown(f"*{label}*")
                st.image(p, use_container_width=True)

    st.divider()
    st.info(
        "Both SHAP and LIME consistently identify **AGE, CHRONIC_DISEASE, and ANXIETY** "
        "as important. High ALLERGY ranking likely reflects dataset-specific correlations "
        "rather than direct clinical causality — an important limitation to acknowledge."
    )

# ── TAB 4: Subgroup + Robustness ──────────────────────────────────────────────
with tab4:
    st.subheader("Subgroup Fairness Analysis")
    st.caption("Model performance by gender, age group, and smoking status.")

    subgrp_path = os.path.join(RESULTS_PATH, "subgroup_results.csv")
    if os.path.exists(subgrp_path):
        st.dataframe(pd.read_csv(subgrp_path), use_container_width=True)
    else:
        st.dataframe(pd.DataFrame([
            {"Subgroup": "Gender: Male",   "N": 162, "Accuracy": "97.5%",  "Recall": "98.6%",  "AUC-ROC": "99.5%"},
            {"Subgroup": "Gender: Female", "N": 147, "Accuracy": "98.6%",  "Recall": "98.4%",  "AUC-ROC": "99.9%"},
            {"Subgroup": "Age: <50",       "N":  15, "Accuracy": "86.7%",  "Recall": "91.7%",  "AUC-ROC": "94.4%"},
            {"Subgroup": "Age: 50-65",     "N": 188, "Accuracy": "97.9%",  "Recall": "98.1%",  "AUC-ROC": "99.8%"},
            {"Subgroup": "Age: 65+",       "N": 106, "Accuracy": "100.0%", "Recall": "100.0%", "AUC-ROC": "100.0%"},
            {"Subgroup": "Smoking: Yes",   "N": 174, "Accuracy": "98.3%",  "Recall": "98.7%",  "AUC-ROC": "99.7%"},
            {"Subgroup": "Smoking: No",    "N": 135, "Accuracy": "97.8%",  "Recall": "98.3%",  "AUC-ROC": "99.8%"},
        ]), use_container_width=True)

    st.warning(
        "**Key finding:** Under-50 patients show notably lower accuracy (86.7%) — "
        "a critical gap that would need targeted validation before any clinical use."
    )

    st.divider()
    st.subheader("Robustness Testing")
    st.caption("20% feature perturbation, 50 trials. Baseline accuracy: 90.32%")

    rob_path = os.path.join(RESULTS_PATH, "robustness_results.csv")
    if os.path.exists(rob_path):
        st.dataframe(pd.read_csv(rob_path), use_container_width=True)
    else:
        st.dataframe(pd.DataFrame([
            {"Feature": "ALLERGY",    "Acc Under Perturbation": "88.1%", "Prediction Change Rate": "3.4%"},
            {"Feature": "WHEEZING",   "Acc Under Perturbation": "90.3%", "Prediction Change Rate": "2.1%"},
            {"Feature": "SMOKING",    "Acc Under Perturbation": "90.2%", "Prediction Change Rate": "0.6%"},
            {"Feature": "COUGHING",   "Acc Under Perturbation": "90.0%", "Prediction Change Rate": "0.3%"},
        ]), use_container_width=True)
        st.info("Run `python src/robustness.py` for full results.")
