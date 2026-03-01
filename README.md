# Exam Difficulty Predictor

### Intelligent Question Complexity Analysis via Feature Engineering & XGBoost

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Supported-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-0F9D58?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)

[Overview](#-overview) · [Architecture](#-system-architecture) · [Quickstart](#-quickstart) · [Pipeline](#-ml-pipeline) · [Results](#-results) · [Application](#-application)

---

## 🎯 Overview

Educators, instructional designers, and testing organizations spend countless manual hours evaluating the difficulty and quality of examination questions. A misjudged question can skew test results and inaccurately measure student proficiency.

The Exam Difficulty Predictor acts as an automated "first pass" quality assurance tool. It is an **end-to-end, production-structured ML pipeline** taking raw question text (including LaTeX and math symbols) and instantly predicting how difficult it will be for students.

It predicts both a continuous difficulty index (p-value from 0.0 to 1.0) and a categorical difficulty tier (Easy, Medium, Hard). The underlying models are built with **XGBoost** and rely on 25 extracted lexical, mathematical, and domain-specific features.

### Problem Statement

| Challenge                      | Scale                         |
| ------------------------------ | ----------------------------- |
| Manual verification bottleneck | Hours spent per exam          |
| Dataset size (Exam Dataset)    | 50,000 preprocessed questions |
| Evaluation targets             | Continuous (P-value) & Tiers  |
| Structural complexity          | Text, LaTeX, Math Operators   |

---

## 🏗 System Architecture

The project functions across three main sectors: data processing, model building, and real-time application inference.

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                         EXAM DIFFICULTY PREDICTOR SYSTEM                                 │
│                         Intelligent Question Complexity Analysis                         │
└──────────────────────────────────────────────────────────────────────────────────────────┘


                                    DATA SOURCE
                           ───────────────────────────
                             exam_dataset_50k_unclean.csv
                          (Question Text + Answers + Metadata)


╔══════════════════════════════════════ TRAINING PIPELINE ══════════════════════════════════════╗

        ┌────────────────────┐
        │   Data Loading     │
        │ clean_dataset.py   │
        └─────────┬──────────┘
                  ▼
        ┌──────────────────────────────────────────┐
        │              Preprocessing               │
        │------------------------------------------│
        │ • Standardize NULLs (N/A, none, ?, --)   │
        │ • Strip whitespaces & casing normalizer  │
        │ • Deduplication                          │
        │ • Enforce domain constraints (0-100%)    │
        │ • Outlier capping (IQR method)           │
        └─────────┬────────────────────────────────┘
                  ▼
        ┌────────────────────┐
        │ Export Clean Data  │
        │ exam_50k_clean.csv │
        └─────────┬──────────┘
                  ▼
        ┌──────────────────────────────────────────┐
        │          Machine Learning Models         │
        │           genai_project.ipynb            │
        │------------------------------------------│
        │ • XGBoost Regressor (Continuous P-Value) │
        │ • XGBoost Classifier (Easy/Med/Hard)     │
        └─────────┬────────────────────────────────┘
                  ▼
        ┌────────────────────┐
        │ Model Persistence  │
        │ Save JSON & PKL    │
        └────────────────────┘

╚══════════════════════════════════════════════════════════════════════════════════════════════╝



╔══════════════════════════════════════ INFERENCE PIPELINE ═════════════════════════════════════╗

        User Input (Streamlit UI via app.py)
        (Question Text + Answer Options + Tier)
                    │
                    ▼
        ┌────────────────────┐
        │ Load Saved Models  │
        │ xgb_models.json    │
        └─────────┬──────────┘
                  ▼
        ┌──────────────────────────────────────────┐
        │        Feature Engineering (NLP)         │
        │          feature_extractor.py            │
        │------------------------------------------│
        │ • Lexical Stats (Word & Sentence counts) │
        │ • Math Ops & LaTeX Density               │
        │ • Vocabulary Richness                    │
        │ • Domain Terms (Algebra, Stats, Calc)    │
        └─────────┬────────────────────────────────┘
                  ▼
        ┌──────────────────────────────────────────┐
        │            Prediction Engine             │
        │------------------------------------------│
        │ • xgb.DMatrix(features)                  │
        │ • predict_proba()                        │
        │ • Heuristic Bias (Subject Tier Nudge)    │
        └─────────┬────────────────────────────────┘
                  ▼
        ┌────────────────────────────────────┐
        │      Difficulty Metrics Output     │
        │------------------------------------│
        │ • Predicted Class (Easy/Med/Hard)  │
        │ • Difficulty Index (p-value)       │
        │ • Confidence / Probabilities Plot  │
        └────────────────────────────────────┘

╚══════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## � Repository Structure

```
genai-project/
├── README.md                         # ← You are here
├── requirements.txt                  # Python dependencies
│
├── app.py                            # Streamlit web application frontend
├── feature_extractor.py              # Extractor for 25 lexical and structural features
├── clean_dataset.py                  # Script to preprocess and sanitize raw data
├── clean_dataset.ipynb               # Jupyter Notebook version of the data cleaner
├── genai_project.ipynb               # XGBoost model training and evaluation notebook
│
├── notebooks/
│   ├── question_analysis.ipynb       # NLP analysis notebook on questions
│   └── visualize_results.ipynb       # Notebook dedicated to plotting analysis results
│
├── raw_dataset/
│   └── exam_dataset_50k_unclean.csv  # Original provided uncleaned data
├── dataset/
│   └── exam_dataset_50k_cleaned.csv  # Cleaned dataset output by the pipeline
│
└── files/                            # Model persistence
    ├── xgb_reg_model_A.json          # Saved XGBoost Regression Model
    ├── xgb_clf_model_A.json          # Saved XGBoost Classification Model
    └── pipeline.pkl                  # Label Encoder / Pipeline
```

---

## 🚀 Quickstart

### 1. Clone & enter the repo

```bash
git clone <your-repo-link>
cd genai-project
```

### 2. Set up environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add the dataset (Optional)

Place the dataset inside `raw_dataset/` and run the cleaning script using `clean_dataset.py` or `clean_dataset.ipynb`.

### 4. Train the model (Optional)

```bash
jupyter notebook genai_project.ipynb
# Run all cells — models save to files/
```

### 5. Launch the app

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 🔬 ML Pipeline

### Stage 1 — Feature Extraction

The user inputs text, and `feature_extractor.py` scans for mathematical operators, word structures, and domain terms (Geometry, Calculus, Statistics) extracting 25 quantitative features.

### Stage 2 — Domain Heuristics

Applies weighted nudges based on user-provided academic tiers.

### Stage 3 — Machine Learning Models

| Model                  | Task                                                  |
| ---------------------- | ----------------------------------------------------- |
| **XGBoost Regressor**  | Continuous difficulty index prediction (p-value).     |
| **XGBoost Classifier** | Categorical tier classification (Easy, Medium, Hard). |

---

## 📊 Results

The system utilizes **XGBoost (Extreme Gradient Boosting)**, trained on a dataset of ~50,000 rows.

| Metric          | Regression                                     | Classification                                              |
| --------------- | ---------------------------------------------- | ----------------------------------------------------------- |
| **Performance** | RMSE: ~0.1245 <br> MAE: ~0.0982 <br> R²: ~0.78 | Accuracy: ~84.5% <br> Precision: ~0.85 <br> F1-Score: ~0.83 |

---

## 🕹️ Application

### Usage via Streamlit

The main interface runs locally via Streamlit. You can:

1. Copy-paste a question and its multiple-choice options.
2. Specify the subject tier (1–5) and misconception levels.
3. Automatically receive evaluated classification probabilities, text complexity data, and a regression difficulty score.

---

## 🚦 Limitations

- **Language Support:** Currently, the system evaluates questions primarily in English due to the NLP dependency libraries (like `textstat` and NLTK). Multi-language evaluation would require a different vectorization pipeline.
- **Image/Graph Dependency:** The model cannot parse or comprehend questions that rely primarily on images, charts, or graphical data contexts.
- **Lexical Bias:** The heuristic approach assumes that longer, more complex sentences with more math operators dictate mathematical/academic difficulty. This can occasionally misclassify a very conceptually difficult short question as "Easy".

---

## 🚀 Future Work — Milestone 2 (Agentic AI)

Moving beyond static ML models, Milestone 2 will introduce an iterative, agentic approach evaluating the _conceptual rigor_ of questions.

Unlike the current lexical approach, the future Agentic framework will utilize:

- **LLM Reasoning Engines:** Leveraging Large Language Models (like Gemini/GPT) to solve the question step-by-step and measure the logical complexity required instead of just lexical bounds.
- **RAG for Context Base:** Using Retrieval-Augmented Generation to measure how a question aligns against educational standards (e.g., Curriculum).
- **Multi-Modal Processing:** Introducing Vision-Language Models to handle questions heavily reliant on graphs, geometry diagrams, and image-based data.
- **Iterative Feedback Loops:** An agentic workflow where the model acts as a "tutor," testing various difficulty assumptions and adjusting its final difficulty rating based on simulated student failure modes.

---
