# Exam Question Difficulty Predictor

### Intelligent Question Complexity Analysis via Feature Engineering & XGBoost

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Supported-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-0F9D58?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)

[Overview](#-overview) · [Architecture](#-system-architecture) · [Quickstart](#-quickstart) · [Models](#-models) · [Results](#-results) · [Application](#-application)

---

## 🎯 Overview

Educators, instructional designers, and testing organizations spend countless manual hours evaluating the difficulty and quality of examination questions. A misjudged question can skew test results and inaccurately measure student proficiency.

The **Exam Question Difficulty Predictor** acts as an automated "first pass" quality assurance tool. It is an end-to-end, production-structured ML pipeline that takes raw question text (including LaTeX and math symbols) and instantly predicts how difficult it will be for students.

It predicts both a **continuous difficulty index** (p-value from 0.0 to 1.0) and a **categorical difficulty tier** (Easy, Medium, Hard). Two XGBoost model variants are deployed — one for **pre-exam** analysis (text-only) and one for **post-exam** analysis (all features).

### Problem Statement

| Challenge                      | Scale                         |
| ------------------------------ | ----------------------------- |
| Manual verification bottleneck | Hours spent per exam          |
| Dataset size                   | 50,000 preprocessed questions |
| Evaluation targets             | Continuous (P-value) & Tiers  |
| Structural complexity          | Text, LaTeX, Math Operators   |

---

## 🏗 System Architecture

The project functions across three main sectors: data processing, model building, and real-time application inference.

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                     EXAM QUESTION DIFFICULTY PREDICTOR SYSTEM                             │
│                     Intelligent Question Complexity Analysis                              │
└──────────────────────────────────────────────────────────────────────────────────────────┘


                                    DATA SOURCE
                           ───────────────────────────
                             exam_dataset_50k_unclean.csv
                          (Question Text + Answers + Metadata)


╔══════════════════════════════════════ TRAINING PIPELINE ══════════════════════════════════════╗

        ┌────────────────────┐
        │   Data Loading     │
        │ clean_dataset.ipynb│
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
        ┌────────────────────────────────────────────────────┐
        │              Machine Learning Models               │
        │----------------------------------------------------│
        │  Pre-Exam (Text-Only)  │  Post-Exam (All Features) │
        │  xgboost_pre_exam/     │  xgboost_post_exam/       │
        │  25 features           │  30 features              │
        └─────────┬──────────────┴───────────┬───────────────┘
                  ▼                          ▼
        ┌────────────────────────────────────────┐
        │         Model Persistence              │
        │  xgb_reg_model_A.json  (pre-exam)      │
        │  xgb_clf_model_A.json  (pre-exam)      │
        │  xgb_all_reg_model_B.json (post-exam)  │
        │  xgb_all_clf_model_B.json (post-exam)  │
        │  xgb_text_model.pkl / xgb_all_model.pkl│
        └────────────────────────────────────────┘

╚══════════════════════════════════════════════════════════════════════════════════════════════╝



╔══════════════════════════════════════ INFERENCE PIPELINE ═════════════════════════════════════╗

        User Input (Streamlit UI via streamlit/app.py)
        (Question Text + Answer Options + Tier + Post-Admin Stats)
                    │
                    ▼
        ┌────────────────────┐
        │ Load Saved Models  │
        │ files/*.json       │
        └─────────┬──────────┘
                  ▼
        ┌──────────────────────────────────────────┐
        │        Feature Engineering (NLP)         │
        │     streamlit/feature_extractor.py       │
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

## 📂 Repository Structure

```
genai-project/
├── README.md                         # ← You are here
├── requirements.txt                  # Python dependencies
│
├── streamlit/                        # Streamlit web application
│   ├── app.py                        # Main application (3 pages)
│   └── feature_extractor.py          # 25-feature extraction module
│
├── files/                            # Serialised model artefacts
│   ├── xgb_reg_model_A.json          # XGBoost Regressor (text-only, pre-exam)
│   ├── xgb_clf_model_A.json          # XGBoost Classifier (text-only, pre-exam)
│   ├── xgb_text_model.pkl            # LabelEncoder for pre-exam models
│   ├── xgb_all_reg_model_B.json      # XGBoost Regressor (all features, post-exam)
│   ├── xgb_all_clf_model_B.json      # XGBoost Classifier (all features, post-exam)
│   └── xgb_all_model.pkl             # LabelEncoder for post-exam models
│
├── notebooks/
│   ├── clean_dataset.ipynb           # Data preprocessing notebook
│   ├── visualize_results.ipynb       # Visualisation notebook
│   ├── compare/
│   │   └── compare.ipynb             # Side-by-side model comparison
│   ├── xgboost_pre_exam/
│   │   └── xgboost_pre_exam.ipynb    # Pre-exam (text-only) model training
│   └── xgboost_post_exam/
│       └── xgboost_post_exam.ipynb   # Post-exam (all features) model training
│
├── report/
│   ├── report.tex                    # Full LaTeX technical report
│   └── report.pdf                    # Compiled PDF report
│
├── raw_dataset/
│   └── exam_dataset_50k_unclean.csv  # Original uncleaned data
└── cleaned_dataset/
    └── exam_dataset_50k_cleaned.csv  # Cleaned dataset
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

### 3. Launch the app

```bash
streamlit run streamlit/app.py
# Opens at http://localhost:8501
```

### 4. Train models (optional)

Open the notebooks in `notebooks/xgboost_pre_exam/` or `notebooks/xgboost_post_exam/` and run all cells. Trained models will be saved to `files/`.

---

## 🧠 Models

Two XGBoost model variants are deployed, each targeting a different stage of the exam lifecycle:

### Pre-Exam Model (Text-Only)

- **Use case:** Predict difficulty **before** the exam is administered
- **Features:** 25 text-derived features (lexical, LaTeX, domain terms, answer complexity)
- **Model files:** `xgb_reg_model_A.json`, `xgb_clf_model_A.json`, `xgb_text_model.pkl`

### Post-Exam Model (All Features)

- **Use case:** Predict difficulty **after** a pilot administration
- **Features:** 30 features (25 text + 5 post-admin: response time, discrimination index, IRT params)
- **Model files:** `xgb_all_reg_model_B.json`, `xgb_all_clf_model_B.json`, `xgb_all_model.pkl`

---

## 📊 Results

| Metric            | Pre-Exam (Text-Only) | Post-Exam (All Features) |
| ----------------- | -------------------- | ------------------------ |
| **MAE**           | 0.0772               | **0.0162**               |
| **RMSE**          | 0.0983               | **0.0227**               |
| **R² Score**      | 0.5693               | **0.9761**               |
| **Accuracy**      | 83.34%               | **95.34%**               |
| **F1 (weighted)** | 0.80                 | **0.95**                 |

> **Note:** The post-exam model dramatically outperforms the pre-exam model because post-administration features (response time, discrimination index) are strong proxies for difficulty. However, these features are only available after students have taken the exam.

---

## 🕹️ Application

The Streamlit app (`streamlit/app.py`) provides **three pages**:

### 1. Post-Exam Analysis

Enter question text, answer options, metadata, **and** post-administration statistics (response time, discrimination index, IRT parameters) to get the highest-accuracy predictions.

### 2. Pre-Exam Analysis

Enter only question text, answer options, and metadata. No post-admin data required — ideal for evaluating questions **before** they go live.

### 3. About the Model

Side-by-side comparison cards for both models, feature pipeline documentation, training details, and known limitations.

---

## 🚦 Limitations

- **Language:** Optimised for English-language mathematics questions only.
- **No Visual Understanding:** Cannot interpret images, diagrams, or graphs.
- **Class Imbalance:** The dataset is ~80% Easy questions, leading to low recall for Hard questions in the pre-exam model.
- **Synthetic Data:** Training data includes synthetic augmentations; unusual real-world formatting may reduce confidence.

---

## 📄 Report

A comprehensive LaTeX technical report is available in `report/`:

- **Source:** `report/report.tex`
- **Compiled:** `report/report.pdf` (16 pages)

Covers: introduction, dataset, feature engineering, model architecture, training, results comparison, system architecture, application interface, heuristic adjustment, limitations, and future work.

---

## 🚀 Future Work — Milestone 2 (Agentic AI)

- **LLM Reasoning:** Leverage Gemini/GPT to solve questions step-by-step and measure conceptual complexity.
- **RAG for Curriculum Alignment:** Retrieval-Augmented Generation against educational standards.
- **Multi-Modal Processing:** Vision-Language Models for diagram-dependent questions.
- **Iterative Feedback Loops:** Agentic workflows simulating student failure modes.

---
