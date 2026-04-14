"""
app.py — Lung Cancer Risk Prediction Demo
Streamlit web interface with live LIME explanations.

Run with:
    streamlit run streamlit_app/app.py
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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lung Cancer Risk Predictor",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .risk-high {
        background: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-low {
        background: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .disclaimer {
        background: #fffbeb;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 0.85rem;
        color: #92400e;
    }
    div[data-testid="stSidebar"] {
        background: #1f2937;
    }
    div[data-testid="stSidebar"] * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
    "PEER_PRESSURE", "CHRONIC_DISEASE", "FATIGUE", "ALLERGY",
    "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING",
    "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN",
]

MODELS_PATH = "artifacts/models/"


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        with open(os.path.join(MODELS_PATH, "best_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODELS_PATH, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODELS_PATH, "X_train.pkl"), "rb") as f:
            X_train = pickle.load(f)
        return model, scaler, X_train, True
    except FileNotFoundError:
        return None, None, None, False


model, scaler, X_train, loaded = load_artifacts()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 About This Tool")
    st.markdown("""
    This tool predicts lung cancer risk using a machine learning model trained
    on clinical survey data.

    **Best model:** Random Oversampling + Random Forest

    **Performance (5-fold CV):**
    - Accuracy: 91.27%
    - AUC-ROC: 93.92%
    - F1 Score: 95.01%

    **Built on research:**
    - Pavithran et al. (2025), *Frontiers in AI*
    - Alsinglawi et al. (2022), *Scientific Reports*
    """)

    st.markdown("---")
    st.markdown("### 📊 Model Details")
    st.markdown("""
    - **Augmentation:** Random Oversampling
    - **Classifier:** Random Forest (100 trees)
    - **CV:** 5-fold stratified
    - **Seed:** 42
    - **Training samples:** 247
    """)

    st.markdown("---")
    st.markdown(
        "<div class='disclaimer'>⚠️ Research tool only. "
        "Not for clinical use.</div>",
        unsafe_allow_html=True
    )

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("<div class='main-header'>🫁 Lung Cancer Risk Prediction</div>",
            unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Explainable AI-powered clinical decision support · "
    "Graduate ML Research Project · Aash Shah</div>",
    unsafe_allow_html=True
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Risk Predictor", "📊 Model Performance", "🧠 Explainability"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    if not loaded:
        st.error(
            "❌ Model artifacts not found. "
            "Run `python src/train.py` first, then restart this app."
        )
        st.stop()

    st.markdown("### Enter Patient Information")
    st.markdown("Fill in the fields below and click **Predict Risk**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender   = st.selectbox("Gender", ["Male", "Female"])
        age      = st.slider("Age", 20, 90, 55)
        smoking  = st.selectbox("Smoking", ["Yes", "No"])
        alcohol  = st.selectbox("Alcohol Consuming", ["Yes", "No"])
        peer_p   = st.selectbox("Peer Pressure", ["Yes", "No"])

    with col2:
        st.markdown("**Symptoms**")
        coughing  = st.selectbox("Coughing", ["Yes", "No"])
        shortness = st.selectbox("Shortness of Breath", ["Yes", "No"])
        wheezing  = st.selectbox("Wheezing", ["Yes", "No"])
        chest     = st.selectbox("Chest Pain", ["Yes", "No"])
        fatigue   = st.selectbox("Fatigue", ["Yes", "No"])
        swallow   = st.selectbox("Swallowing Difficulty", ["Yes", "No"])

    with col3:
        st.markdown("**Risk Factors**")
        yellow   = st.selectbox("Yellow Fingers", ["Yes", "No"])
        anxiety  = st.selectbox("Anxiety", ["Yes", "No"])
        chronic  = st.selectbox("Chronic Disease", ["Yes", "No"])
        allergy  = st.selectbox("Allergy", ["Yes", "No"])

    st.markdown("---")

    # ── Encode inputs ─────────────────────────────────────────────────────────
    def yn(val):
        return 2 if val == "Yes" else 1

    input_raw = np.array([[
        1 if gender == "Male" else 0,
        age,
        yn(smoking),
        yn(yellow),
        yn(anxiety),
        yn(peer_p),
        yn(chronic),
        yn(fatigue),
        yn(allergy),
        yn(wheezing),
        yn(alcohol),
        yn(coughing),
        yn(shortness),
        yn(swallow),
        yn(chest),
    ]])

    predict_btn = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

    if predict_btn:
        input_scaled = scaler.transform(input_raw)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class='risk-high'>
                    <h2>⚠️ HIGH RISK</h2>
                    <h3>Probability: {probability:.1%}</h3>
                    <p>This patient profile is associated with elevated lung cancer risk.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='risk-low'>
                    <h2>✅ LOW RISK</h2>
                    <h3>Probability: {probability:.1%}</h3>
                    <p>This patient profile is associated with lower lung cancer risk.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(float(probability), text=f"Cancer probability: {probability:.1%}")

            # Risk factors summary
            st.markdown("**Active risk factors in this profile:**")
            risk_factors = []
            if smoking  == "Yes": risk_factors.append("🚬 Smoker")
            if yellow   == "Yes": risk_factors.append("🟡 Yellow fingers")
            if chronic  == "Yes": risk_factors.append("🏥 Chronic disease")
            if coughing == "Yes": risk_factors.append("😮‍💨 Coughing")
            if fatigue  == "Yes": risk_factors.append("😴 Fatigue")
            if shortness== "Yes": risk_factors.append("💨 Shortness of breath")
            if age > 60:          risk_factors.append(f"📅 Age {age} (elevated risk group)")

            if risk_factors:
                for rf in risk_factors:
                    st.markdown(f"- {rf}")
            else:
                st.markdown("- No major risk factors flagged")

        with res_col2:
            st.markdown("**🧪 LIME Explanation — Why this prediction?**")
            try:
                import lime
                import lime.lime_tabular

                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train,
                    feature_names=FEATURES,
                    class_names=["No Cancer", "Cancer"],
                    mode="classification",
                    random_state=42,
                )
                exp = explainer.explain_instance(
                    input_scaled[0],
                    model.predict_proba,
                    num_features=10,
                )
                fig = exp.as_pyplot_figure()
                fig.set_size_inches(8, 5)
                plt.title("Feature contributions to this prediction", fontsize=11)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.caption(
                    "🟠 Orange bars push toward Cancer prediction. "
                    "🔵 Blue bars push toward No Cancer."
                )
            except Exception as e:
                st.warning(f"LIME explanation unavailable: {e}")

        st.markdown("""
        <div class='disclaimer'>
        ⚠️ <strong>Disclaimer:</strong> This tool is for educational and research purposes only.
        It is not a validated medical device and must not be used to inform clinical decisions.
        Always consult a qualified healthcare professional.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Full Experiment Results — All 49 Combinations")
    st.markdown("5-fold stratified CV · seed=42 · augmentation applied only inside training folds")

    results_path = "artifacts/results/all_results.csv"
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        results_df = results_df.sort_values("auc_roc", ascending=False).reset_index(drop=True)
        results_df.index += 1

        # Highlight best row
        def highlight_best(row):
            if row.name == 1:
                return ["background-color: #fef9c3"] * len(row)
            return [""] * len(row)

        display_cols = ["augmentation", "classifier", "accuracy", "auc_roc", "f1", "recall", "precision"]
        st.dataframe(
            results_df[display_cols]
            .style.apply(highlight_best, axis=1)
            .format({"accuracy": "{:.2f}%", "auc_roc": "{:.2f}%",
                     "f1": "{:.2f}%", "recall": "{:.2f}%", "precision": "{:.2f}%"}),
            use_container_width=True,
            height=500,
        )
        st.caption("🥇 Gold row = best combination (Random Oversampling + Random Forest, AUC-ROC 93.92%)")
    else:
        st.info("Run `python src/train.py` to generate results.")

    st.markdown("---")
    st.markdown("### 👥 Subgroup Fairness Analysis")

    subgrp_path = "artifacts/results/subgroup_results.csv"
    if os.path.exists(subgrp_path):
        subgrp_df = pd.read_csv(subgrp_path)
        st.dataframe(subgrp_df, use_container_width=True)
        st.caption(
            "⚠️ Age <50 group shows lower performance (Accuracy 86.7%, AUC 94.4%) "
            "— fewer training examples for younger patients."
        )
    else:
        st.info("Run `python src/subgrp.py` to generate subgroup results.")

    st.markdown("---")
    st.markdown("### 🔧 Robustness Testing")

    robust_path = "artifacts/results/robustness_results.csv"
    if os.path.exists(robust_path):
        robust_df = pd.read_csv(robust_path)
        st.dataframe(robust_df, use_container_width=True)
        st.caption(
            "ALLERGY shows the highest prediction change rate (3.4%) under perturbation, "
            "consistent with its top SHAP ranking."
        )
    else:
        st.info("Run `python src/robustness.py` to generate robustness results.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🧠 Model Explainability — SHAP + LIME")
    st.markdown(
        "Explainability plots generated for the best model "
        "(Random Oversampling + Random Forest). "
        "Run `python src/explain_shap.py` and `python src/explain_lime.py` to regenerate."
    )

    plots_path = "artifacts/plots/"

    # SHAP plots
    st.markdown("#### SHAP — Global Feature Importance")
    shap_col1, shap_col2 = st.columns(2)

    with shap_col1:
        summary_path = os.path.join(plots_path, "shap_summary.png")
        if os.path.exists(summary_path):
            st.image(summary_path, caption="SHAP Summary Plot (Beeswarm)", use_container_width=True)
        else:
            st.info("shap_summary.png not found.")

    with shap_col2:
        bar_path = os.path.join(plots_path, "shap_bar.png")
        if os.path.exists(bar_path):
            st.image(bar_path, caption="SHAP Bar Chart (Mean |SHAP|)", use_container_width=True)
        else:
            st.info("shap_bar.png not found.")

    st.markdown("---")
    st.markdown("#### LIME — Per-Patient Explanations")

    lime_profiles = {
        "Profile 1 — Middle-aged Smoker": "lime_profile_1_smoker.png",
        "Profile 2 — Young Non-Smoker":   "lime_profile_2_young.png",
        "Profile 3 — Elderly Patient":    "lime_profile_3_elderly.png",
    }

    lime_cols = st.columns(3)
    for col, (label, fname) in zip(lime_cols, lime_profiles.items()):
        fpath = os.path.join(plots_path, fname)
        with col:
            st.markdown(f"**{label}**")
            if os.path.exists(fpath):
                st.image(fpath, use_container_width=True)
            else:
                st.info(f"{fname} not found.")

    st.markdown("---")
    st.markdown("#### SHAP vs LIME Agreement")
    st.markdown("""
    | Feature | SHAP Rank | LIME Frequency (top-5) | Agreement |
    |---|---|---|---|
    | ALLERGY | #1 | 2/3 profiles | ✅ Consistent |
    | PEER_PRESSURE | #2 | 3/3 profiles | ✅ Consistent |
    | AGE | #4 | 3/3 profiles | ✅ Consistent |
    | CHRONIC_DISEASE | #7 | 3/3 profiles | ✅ Consistent |
    | SMOKING | #6 | 1/3 profiles | ⚠️ Partial |

    Both methods agree that **ALLERGY, PEER_PRESSURE, AGE, and CHRONIC_DISEASE**
    are the most influential features in this dataset.
    The prominence of ALLERGY and PEER_PRESSURE above SMOKING likely reflects
    dataset-specific correlations in this 309-sample survey rather than direct clinical causality.
    """)
