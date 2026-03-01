import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from feature_extractor import extract_features, TEXT_FEATURES

# All features = text features + post-admin features (must match training order)
POST_ADMIN_FEATURES = TEXT_FEATURES + [
    "avg_response_time_sec",
    "std_response_time_sec",
    "discrimination_index",
    "point_biserial_corr",
    "irt_a_param",
]

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'files')

# --- Page Config ---
st.set_page_config(
    page_title="Difficulty Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional Aesthetic ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e2e8f0;
        background-color: #0f172a; 
    }
    
    /* Input Fields */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div > div {
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 6px;
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #64748b;
        box-shadow: none;
    }

    /* Buttons */
    .stButton > button {
        background-color: #334155;
        color: #f8fafc;
        border: 1px solid #475569;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #475569;
        border-color: #64748b;
        color: #ffffff;
    }
    .stButton > button:active {
        background-color: #1e293b;
    }

    /* Cards & Metrics */
    .metric-container {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #f8fafc;
    }
    .metric-sub {
        font-size: 0.875rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Section Headers */
    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    h1 { margin-bottom: 0.5rem; }
    p { color: #94a3b8; }

    /* Tables/Dataframes */
    .stDataFrame { border: none; }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1e293b;
        color: #e2e8f0;
        border-radius: 6px;
    }
    
    /* Removing default Streamlit decoration */
    .stDeployButton {display:none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}
    
    /* Feature Grid */
    .feature-row {
        display: flex;
        justify-content: space-between;
        padding: 0.75rem 0;
        border-bottom: 1px solid #334155;
    }
    .feature-row:last-child { border-bottom: none; }
    .feature-name { color: #94a3b8; font-size: 0.9rem; }
    .feature-val { color: #f8fafc; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; }

</style>
""", unsafe_allow_html=True)


# --- Base Dirs ---
MODEL_DIR = os.path.join(BASE_DIR, 'model')
FILES_DIR = os.path.join(BASE_DIR, 'files')

# --- Load Models (All-Features XGBoost B) ---
@st.cache_resource
def load_models_all():
    try:
        reg_booster = xgb.Booster()
        reg_booster.load_model(os.path.join(FILES_DIR, 'xgb_all_reg_model_B.json'))
        
        clf_booster = xgb.Booster()
        clf_booster.load_model(os.path.join(FILES_DIR, 'xgb_all_clf_model_B.json'))
        
        le = joblib.load(os.path.join(FILES_DIR, 'xgb_all_model.pkl'))
        return reg_booster, clf_booster, le
    except Exception as e:
        st.error(f"System Error: All-features model loading failed ({e})")
        return None, None, None

# --- Load Models (Text-Only XGBoost A) ---
@st.cache_resource
def load_models_text():
    try:
        reg_booster = xgb.Booster()
        reg_booster.load_model(os.path.join(FILES_DIR, 'xgb_reg_model_A.json'))
        
        clf_booster = xgb.Booster()
        clf_booster.load_model(os.path.join(FILES_DIR, 'xgb_clf_model_A.json'))
        
        le = joblib.load(os.path.join(FILES_DIR, 'xgb_text_model.pkl'))
        return reg_booster, clf_booster, le
    except Exception as e:
        st.error(f"System Error: Text-only model loading failed ({e})")
        return None, None, None

reg_booster, clf_booster, le = load_models_all()
text_reg_booster, text_clf_booster, text_le = load_models_text()
CLASS_NAMES = le.classes_.tolist() if le is not None else ['Easy', 'Hard', 'Medium']

# --- Navigation ---
page = st.sidebar.radio("Navigation", ["Post-Exam Analysis", "Pre-Exam Analysis", "About the Model"])

if page == "Post-Exam Analysis":
    # --- Post-Exam Analysis (All Features) ---
    st.title("Post-Exam Difficulty Analysis")
    st.markdown("Uses **all features** including post-administration stats for highest accuracy predictions.")
    st.markdown("---")

    # Input Section
    col_main, col_side = st.columns([2, 1], gap="large")

    with col_main:
        st.markdown("### Question Content")
        question_text = st.text_area("Question Text", height=150, placeholder="Enter the full question text here...")
        
        c1, c2 = st.columns(2)
        with c1:
            answer_a = st.text_input("Option A", placeholder="Answer choice A")
            answer_c = st.text_input("Option C", placeholder="Answer choice C")
        with c2:
            answer_b = st.text_input("Option B", placeholder="Answer choice B")
            answer_d = st.text_input("Option D", placeholder="Answer choice D")

    with col_side:
        st.markdown("### Metadata")
        subject_tier = st.selectbox("Subject Tier", options=[1, 2, 3, 4, 5], index=0, help="Academic difficulty level (1-5)")
        num_misconceptions = st.number_input("Misconception Count", min_value=0, max_value=10, value=0)
        construct_frequency = st.number_input("Construct Frequency", min_value=1, value=10)

        st.markdown("### Post-Administration Stats")
        avg_response_time = st.number_input("Avg Response Time (sec)", min_value=0.0, value=60.0, step=1.0, help="Average time students spent on this question")
        std_response_time = st.number_input("Std Response Time (sec)", min_value=0.0, value=20.0, step=1.0, help="Standard deviation of response times")
        discrimination_index = st.number_input("Discrimination Index", min_value=-1.0, max_value=1.0, value=0.3, step=0.05, help="How well the question discriminates between strong and weak students")
        point_biserial_corr = st.number_input("Point-Biserial Correlation", min_value=-1.0, max_value=1.0, value=0.3, step=0.05, help="Correlation between item score and total score")
        irt_a_param = st.number_input("IRT a-Parameter", min_value=0.0, value=1.0, step=0.1, help="Item discrimination (IRT)")

        st.markdown("### Analysis")
        predict_btn = st.button("Run Prediction", use_container_width=True)

    # Prediction Logic
    if predict_btn:
        if not question_text.strip():
            st.warning("Input required: Question text cannot be empty.")
            st.stop()
            
        # extract features
        features = extract_features(
            question_text, answer_a, answer_b, answer_c, answer_d,
            num_misconceptions=num_misconceptions,
            subject_difficulty_tier=subject_tier,
            construct_frequency=construct_frequency,
        )

        # add post-admin features to the feature dict
        features["avg_response_time_sec"] = avg_response_time
        features["std_response_time_sec"] = std_response_time
        features["discrimination_index"] = discrimination_index
        features["point_biserial_corr"] = point_biserial_corr
        features["irt_a_param"] = irt_a_param

        # predict using all 30 features
        input_df = pd.DataFrame([features])[POST_ADMIN_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
        dinput = xgb.DMatrix(input_df, feature_names=POST_ADMIN_FEATURES)

        p_value = float(reg_booster.predict(dinput)[0])
        clf_proba_raw = clf_booster.predict(dinput)[0]

        if clf_proba_raw.ndim == 0:
            # Binary case handling (unlikely given 3 classes, but good safety)
            clf_proba = np.array([1 - clf_proba_raw, clf_proba_raw])
        else:
            clf_proba = clf_proba_raw

        # --- Heuristic Adjustment based on Subject Tier ---
        # User Rule: 1,2 -> Easy; 3 -> Medium; 4 -> Medium/Hard; 5 -> Hard
        # We apply a gentle weight multiplier to nudge the model without ignoring text features completely.
        
        classes = list(le.classes_)
        if 'Easy' in classes and 'Medium' in classes and 'Hard' in classes:
            easy_idx = classes.index('Easy')
            med_idx = classes.index('Medium')
            hard_idx = classes.index('Hard')
            
            # Adjustment factors (multipliers) - REDUCED IMPACT
            # We want "Question Content" to matter more.
            weights = np.ones(len(classes))
            
            # Only apply tier bias if the question isn't undeniably complex
            # (assuming text_complexity_score > 6.0 is very hard)
            complexity_score = features.get('text_complexity_score', 0)
            word_count = features.get('word_count', 0)
            
            if word_count < 4 and complexity_score < 2.0:
                 # Super short/simple input -> Force Easy
                 weights[easy_idx] = 5.0 
            elif complexity_score < 6.0:  # If not super complex, let Tier nudge it
                if subject_tier <= 2:
                    weights[easy_idx] = 1.5  # Was 2.0
                elif subject_tier == 3:
                    weights[med_idx] = 1.3   # Was 2.0
                elif subject_tier == 4:
                    weights[med_idx] = 1.2
                    weights[hard_idx] = 1.5  # Increased to give more weight to hard
                elif subject_tier == 5:
                    weights[hard_idx] = 4.0  # Increased to give more weight to hard
            
            # Apply weights and re-normalize
            clf_proba = clf_proba * weights
            clf_proba = clf_proba / clf_proba.sum()

        clf_pred_idx = int(np.argmax(clf_proba))
        probs = clf_proba # Alias for downstream plotting code
        
        predicted_class = le.inverse_transform([clf_pred_idx])[0]
        confidence = float(np.max(clf_proba))

        st.markdown("---")
        st.markdown("### Analysis Results")
        
        # 1. Top Level Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Predicted Difficulty</div>
                <div class="metric-value">{predicted_class}</div>
                <div class="metric-sub">Classification</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Difficulty Index (p-value)</div>
                <div class="metric-value">{p_value:.4f}</div>
                <div class="metric-sub">Regression Output (0.0 - 1.0)</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-value">{confidence*100:.1f}%</div>
                <div class="metric-sub">Probability Score</div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Detailed Data (Two Column Layout)
        d1, d2 = st.columns([1, 1], gap="large")
        
        with d1:
            st.markdown("#### Class Probabilities")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            
            # Monochrome bars
            y_pos = np.arange(len(CLASS_NAMES))
            ax.barh(y_pos, probs, align='center', color='#94a3b8', height=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(CLASS_NAMES, color='#cbd5e1', fontsize=9)
            ax.invert_yaxis()
            ax.tick_params(axis='x', colors='#64748b', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_visible(False)
            
            for i, v in enumerate(probs):
                ax.text(v + 0.02, i, f"{v:.2f}", color='#f8fafc', va='center', fontsize=8)
                
            st.pyplot(fig)
            
            st.markdown("#### Text Analysis")
            st.markdown(f"""
            <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem;">
                <div class="feature-row"><span class="feature-name">Word Count</span><span class="feature-val">{features['word_count']}</span></div>
                <div class="feature-row"><span class="feature-name">Sentence Count</span><span class="feature-val">{features['sentence_count']}</span></div>
                <div class="feature-row"><span class="feature-name">LaTeX Commands</span><span class="feature-val">{features['latex_command_count']}</span></div>
                <div class="feature-row"><span class="feature-name">Math Operators</span><span class="feature-val">{features['math_operator_count']}</span></div>
                <div class="feature-row"><span class="feature-name">Complexity Score</span><span class="feature-val">{features['text_complexity_score']:.2f}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with d2:
            st.markdown("#### Feature Detail")
            with st.container():
                st.markdown(
                    """
                    <style>
                    .feature-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
                    .feature-table td { padding: 8px 0; border-bottom: 1px solid #334155; color: #cbd5e1; }
                    .feature-table td:last-child { text-align: right; font-family: 'JetBrains Mono', monospace; color: #f8fafc; }
                    .feature-table tr:last-child td { border-bottom: none; }
                    </style>
                    <table class="feature-table">
                    """, unsafe_allow_html=True
                )
                
                domain_tags = []
                if features['has_algebra_terms']: domain_tags.append("Algebra")
                if features['has_geometry_terms']: domain_tags.append("Geometry")
                if features['has_stats_terms']: domain_tags.append("Statistics")
                if features['has_advanced_terms']: domain_tags.append("Advanced")
                if not domain_tags: domain_tags.append("General")
                
                table_html = "<table class='feature-table'>"
                table_html += f"<tr><td>Detected Domain</td><td>{', '.join(domain_tags)}</td></tr>"
                table_html += f"<tr><td>Avg Word Length</td><td>{features['avg_word_length']:.2f}</td></tr>"
                table_html += f"<tr><td>Vocabulary Richness</td><td>{features['vocab_richness']:.2f}</td></tr>"
                table_html += f"<tr><td>Avg Answer Length</td><td>{features['avg_answer_length']:.1f}</td></tr>"
                table_html += f"<tr><td>LaTeX Density</td><td>{features['latex_density']:.3f}</td></tr>"
                table_html += "</table>"
                
                st.markdown(table_html, unsafe_allow_html=True)

elif page == "Pre-Exam Analysis":
    # --- Pre-Exam Analysis (Text-Only Features) ---
    st.title("Pre-Exam Difficulty Analysis")
    st.markdown("Predict difficulty **before** the exam is administered — uses text features only (no post-admin stats required).")
    st.markdown("---")

    col_main, col_side = st.columns([2, 1], gap="large")

    with col_main:
        st.markdown("### Question Content")
        pre_question_text = st.text_area("Question Text", height=150, placeholder="Enter the full question text here...", key="pre_q")
        
        c1, c2 = st.columns(2)
        with c1:
            pre_answer_a = st.text_input("Option A", placeholder="Answer choice A", key="pre_a")
            pre_answer_c = st.text_input("Option C", placeholder="Answer choice C", key="pre_c")
        with c2:
            pre_answer_b = st.text_input("Option B", placeholder="Answer choice B", key="pre_b")
            pre_answer_d = st.text_input("Option D", placeholder="Answer choice D", key="pre_d")

    with col_side:
        st.markdown("### Metadata")
        pre_subject_tier = st.selectbox("Subject Tier", options=[1, 2, 3, 4, 5], index=0, help="Academic difficulty level (1-5)", key="pre_tier")
        pre_num_misconceptions = st.number_input("Misconception Count", min_value=0, max_value=10, value=0, key="pre_misc")
        pre_construct_frequency = st.number_input("Construct Frequency", min_value=1, value=10, key="pre_freq")
        st.markdown("### Analysis")
        pre_predict_btn = st.button("Run Pre-Exam Prediction", use_container_width=True, key="pre_btn")

    if pre_predict_btn:
        if not pre_question_text.strip():
            st.warning("Input required: Question text cannot be empty.")
            st.stop()

        features = extract_features(
            pre_question_text, pre_answer_a, pre_answer_b, pre_answer_c, pre_answer_d,
            num_misconceptions=pre_num_misconceptions,
            subject_difficulty_tier=pre_subject_tier,
            construct_frequency=pre_construct_frequency,
        )

        input_df = pd.DataFrame([features])[TEXT_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
        dinput = xgb.DMatrix(input_df, feature_names=TEXT_FEATURES)

        p_value = float(text_reg_booster.predict(dinput)[0])
        clf_proba_raw = text_clf_booster.predict(dinput)[0]

        if clf_proba_raw.ndim == 0:
            clf_proba = np.array([1 - clf_proba_raw, clf_proba_raw])
        else:
            clf_proba = clf_proba_raw

        # Heuristic adjustment for text-only model
        classes = list(text_le.classes_)
        if 'Easy' in classes and 'Medium' in classes and 'Hard' in classes:
            easy_idx = classes.index('Easy')
            med_idx = classes.index('Medium')
            hard_idx = classes.index('Hard')
            weights = np.ones(len(classes))
            complexity_score = features.get('text_complexity_score', 0)
            word_count = features.get('word_count', 0)
            if word_count < 4 and complexity_score < 2.0:
                weights[easy_idx] = 5.0
            elif complexity_score < 6.0:
                if pre_subject_tier <= 2:
                    weights[easy_idx] = 1.5
                elif pre_subject_tier == 3:
                    weights[med_idx] = 1.3
                elif pre_subject_tier == 4:
                    weights[med_idx] = 1.2
                    weights[hard_idx] = 1.5
                elif pre_subject_tier == 5:
                    weights[hard_idx] = 4.0
            clf_proba = clf_proba * weights
            clf_proba = clf_proba / clf_proba.sum()

        clf_pred_idx = int(np.argmax(clf_proba))
        probs = clf_proba
        predicted_class = text_le.inverse_transform([clf_pred_idx])[0]
        confidence = float(np.max(clf_proba))

        st.markdown("---")
        st.markdown("### Pre-Exam Analysis Results")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Predicted Difficulty</div>
                <div class="metric-value">{predicted_class}</div>
                <div class="metric-sub">Text-Only Classification</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Difficulty Index (p-value)</div>
                <div class="metric-value">{p_value:.4f}</div>
                <div class="metric-sub">Regression Output (0.0 - 1.0)</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-value">{confidence*100:.1f}%</div>
                <div class="metric-sub">Text-Only Probability</div>
            </div>
            """, unsafe_allow_html=True)

        d1, d2 = st.columns([1, 1], gap="large")
        with d1:
            st.markdown("#### Class Probabilities")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            y_pos = np.arange(len(CLASS_NAMES))
            ax.barh(y_pos, probs, align='center', color='#64748b', height=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(CLASS_NAMES, color='#cbd5e1', fontsize=9)
            ax.invert_yaxis()
            ax.tick_params(axis='x', colors='#64748b', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_visible(False)
            for i, v in enumerate(probs):
                ax.text(v + 0.02, i, f"{v:.2f}", color='#f8fafc', va='center', fontsize=8)
            st.pyplot(fig)

            st.markdown("#### Text Analysis")
            st.markdown(f"""
            <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem;">
                <div class="feature-row"><span class="feature-name">Word Count</span><span class="feature-val">{features['word_count']}</span></div>
                <div class="feature-row"><span class="feature-name">Sentence Count</span><span class="feature-val">{features['sentence_count']}</span></div>
                <div class="feature-row"><span class="feature-name">LaTeX Commands</span><span class="feature-val">{features['latex_command_count']}</span></div>
                <div class="feature-row"><span class="feature-name">Math Operators</span><span class="feature-val">{features['math_operator_count']}</span></div>
                <div class="feature-row"><span class="feature-name">Complexity Score</span><span class="feature-val">{features['text_complexity_score']:.2f}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with d2:
            st.markdown("#### Feature Detail")
            domain_tags = []
            if features['has_algebra_terms']: domain_tags.append("Algebra")
            if features['has_geometry_terms']: domain_tags.append("Geometry")
            if features['has_stats_terms']: domain_tags.append("Statistics")
            if features['has_advanced_terms']: domain_tags.append("Advanced")
            if not domain_tags: domain_tags.append("General")

            table_html = "<table class='feature-table'>"
            table_html += f"<tr><td>Detected Domain</td><td>{', '.join(domain_tags)}</td></tr>"
            table_html += f"<tr><td>Avg Word Length</td><td>{features['avg_word_length']:.2f}</td></tr>"
            table_html += f"<tr><td>Vocabulary Richness</td><td>{features['vocab_richness']:.2f}</td></tr>"
            table_html += f"<tr><td>Avg Answer Length</td><td>{features['avg_answer_length']:.1f}</td></tr>"
            table_html += f"<tr><td>LaTeX Density</td><td>{features['latex_density']:.3f}</td></tr>"
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)

            st.markdown("#### Model Info")
            st.markdown(f"""
            <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem;">
                <div class="feature-row"><span class="feature-name">Model</span><span class="feature-val">XGBoost (Text-Only)</span></div>
                <div class="feature-row"><span class="feature-name">Features Used</span><span class="feature-val">25</span></div>
                <div class="feature-row"><span class="feature-name">R² (Train)</span><span class="feature-val">0.5693</span></div>
                <div class="feature-row"><span class="feature-name">Accuracy (Train)</span><span class="feature-val">83.34%</span></div>
            </div>
            """, unsafe_allow_html=True)

elif page == "About the Model":
    # --- About Page Logic ---
    st.title("About the Model")
    st.markdown("A comprehensive overview of the two deployed models, the feature pipeline, and known constraints.")
    st.markdown("---")

    # --- Overview ---
    st.markdown("### How It Works")
    st.markdown("""
    This application predicts the **difficulty of exam questions** using machine-learning models trained on a dataset of **50,000 mathematics questions**.
    
    Given question text and metadata, the system produces:
    - A **continuous difficulty index** (p-value, 0.0 = hardest → 1.0 = easiest)
    - A **categorical difficulty label** (Easy / Medium / Hard)
    """)

    # --- Two Model Cards ---
    st.markdown("### Deployed Models")
    st.markdown("Two separate XGBoost model variants are available, tailored for different stages of the exam lifecycle:")

    col_pre, col_post = st.columns(2, gap="large")

    with col_pre:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Pre-Exam Model (Text-Only)</div>
            <div class="metric-sub" style="margin-top:0.5rem;">
                Use <b>before</b> the exam is administered.<br>
                Relies solely on 25 text-derived features — no student data required.
            </div>
            <hr style="border-color:#334155; margin:0.75rem 0;">
            <div class="feature-row"><span class="feature-name">Features</span><span class="feature-val">25</span></div>
            <div class="feature-row"><span class="feature-name">R² Score</span><span class="feature-val">0.5693</span></div>
            <div class="feature-row"><span class="feature-name">MAE</span><span class="feature-val">0.0772</span></div>
            <div class="feature-row"><span class="feature-name">RMSE</span><span class="feature-val">0.0983</span></div>
            <div class="feature-row"><span class="feature-name">Accuracy</span><span class="feature-val">83.34%</span></div>
            <div class="feature-row"><span class="feature-name">F1 (weighted)</span><span class="feature-val">0.80</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_post:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Post-Exam Model (All Features)</div>
            <div class="metric-sub" style="margin-top:0.5rem;">
                Use <b>after</b> a pilot administration.<br>
                Combines text features with response time, discrimination, and IRT statistics.
            </div>
            <hr style="border-color:#334155; margin:0.75rem 0;">
            <div class="feature-row"><span class="feature-name">Features</span><span class="feature-val">30</span></div>
            <div class="feature-row"><span class="feature-name">R² Score</span><span class="feature-val">0.9761</span></div>
            <div class="feature-row"><span class="feature-name">MAE</span><span class="feature-val">0.0162</span></div>
            <div class="feature-row"><span class="feature-name">RMSE</span><span class="feature-val">0.0227</span></div>
            <div class="feature-row"><span class="feature-name">Accuracy</span><span class="feature-val">95.34%</span></div>
            <div class="feature-row"><span class="feature-name">F1 (weighted)</span><span class="feature-val">0.95</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Feature Pipeline ---
    st.markdown("### Feature Pipeline")
    st.markdown("All features are extracted by `feature_extractor.py`. The 25 text features are grouped as follows:")

    f1, f2 = st.columns(2, gap="large")
    with f1:
        st.markdown("""
        **Lexical (4)**  
        `text_length` · `word_count` · `sentence_count` · `avg_word_length`

        **LaTeX & Math (5)**  
        `latex_command_count` · `has_latex` · `latex_density` · `math_operator_count` · `number_count`

        **Complexity (2)**  
        `vocab_richness` · `text_complexity_score`
        """)
    with f2:
        st.markdown("""
        **Answer Options (5)**  
        `answer_{a,b,c,d}_length` · `avg_answer_length` · `answer_length_variance`

        **Domain Detection (4)**  
        `has_advanced_terms` · `has_algebra_terms` · `has_geometry_terms` · `has_stats_terms`

        **Metadata (5)**  
        `num_misconceptions` · `has_misconception` · `subject_difficulty_tier` · `construct_frequency`
        """)

    st.markdown("""
    The **Post-Exam Model** adds 5 additional features available only after students sit the exam:
    `avg_response_time_sec` · `std_response_time_sec` · `discrimination_index` · `point_biserial_corr` · `irt_a_param`
    """)

    st.markdown("---")

    # --- Training Details ---
    st.markdown("### Training Details")
    st.markdown("""
    | Parameter | Value |
    |---|---|
    | **Dataset** | 50,000 exam questions (cleaned) |
    | **Train / Test Split** | 80% / 20% (stratified) |
    | **Cross-Validation** | 5-fold stratified |
    | **Algorithm** | XGBoost (n_estimators=500, max_depth=6, lr=0.05) |
    | **Classifier Objective** | `multi:softprob` (3-class) |
    | **Regressor Objective** | `reg:squarederror` |
    """)

    st.markdown("---")

    # --- Limitations ---
    st.markdown("### Limitations & Considerations")
    st.warning("""
    **Class Imbalance** — The dataset is heavily skewed toward Easy questions (~80%). The Pre-Exam model achieves only 11% recall on Hard questions. Consider this when interpreting results for minority classes.
    """)
    st.info("""
    **No Visual Understanding** — Neither model can interpret images, diagrams, or graphs. Geometry questions that depend on visual context may be inaccurately classified.
    """)
    st.info("""
    **English Only** — Feature extraction (keyword matching, sentence splitting) is designed for English-language mathematics questions. Non-English input will produce unreliable results.
    """)
    st.info("""
    **Synthetic Training Data** — The 50k dataset includes synthetically augmented question variants. Real-world questions with unusual formatting may produce lower-confidence predictions.
    """)