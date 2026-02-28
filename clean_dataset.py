import pandas as pd
import numpy as np

# ── 0. Load ──────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("exam_dataset_50k_unclean.csv", low_memory=False)
print(f"  Raw shape: {df.shape}")


# ── 1. Standardise NULL representations ──────────────────────────────────────
print("Step 1: Replacing junk null representations with NaN...")

NULL_VALUES = {"N/A", "n/a", "NA", "na", "NULL", "null", "None", "none",
               "NaN", "nan", "-", "--", "?", "", "  "}

def replace_nulls(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in NULL_VALUES or s == "":
        return np.nan
    return val

df = df.map(replace_nulls)


# ── 2. Strip leading/trailing whitespace from all string cells ────────────────
print("Step 2: Stripping whitespace from string columns...")

def strip_whitespace(val):
    if isinstance(val, str):
        return val.strip()
    return val

df = df.map(strip_whitespace)

# Re-apply null check after stripping (catches "   " cases)
df = df.map(lambda x: np.nan if (isinstance(x, str) and x.strip() == "") else x)


# ── 3. Remove duplicate rows ──────────────────────────────────────────────────
print("Step 3: Removing duplicates...")
before = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"  Removed {before - len(df)} duplicate rows → {len(df)} rows remain")


# ── 4. Normalise categorical / text columns to consistent casing ──────────────
print("Step 4: Normalising casing in categorical columns...")

CATEGORICAL_COLS = [
    "difficulty_label",       # Easy / Medium / Hard / Very Easy / Very Hard
    "difficulty_label_5",     # same set
    "discrimination_quality", # Good / Acceptable / Poor / Excellent
    "question_quality",       # Good / Acceptable / Poor / Excellent
    "source",                 # original / synthetic
    "CorrectAnswer",          # A / B / C / D
]

for col in CATEGORICAL_COLS:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: str(x).strip().title() if pd.notna(x) else x)

# SubjectName and ConstructName → Title Case (fix ALL-CAPS / all-lower variants)
for col in ["SubjectName", "ConstructName"]:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: str(x).strip().title() if pd.notna(x) else x)

# ── 5. Cast numeric columns ───────────────────────────────────────────────────
print("Step 5: Coercing columns to correct numeric types...")

# Float columns
FLOAT_COLS = [
    "avg_word_length", "latex_density", "vocab_richness", "text_complexity_score",
    "avg_answer_length", "answer_length_variance",
    "pct_correct", "pct_chose_A", "pct_chose_B", "pct_chose_C", "pct_chose_D",
    "avg_response_time_sec", "std_response_time_sec",
    "difficulty_p_value", "irt_b_param", "irt_a_param",
    "point_biserial_corr", "discrimination_index",
    "p_value_upper27", "p_value_lower27",
    "SubjectId",
]

# Integer columns (stored as float after coercion since NaNs exist)
INT_COLS = [
    "ConstructId",
    "num_misconceptions", "has_misconception",
    "MisconceptionAId", "MisconceptionBId", "MisconceptionCId", "MisconceptionDId",
    "text_length", "word_count", "sentence_count",
    "latex_command_count", "has_latex",
    "math_operator_count", "number_count",
    "answer_a_length", "answer_b_length", "answer_c_length", "answer_d_length",
    "has_advanced_terms", "has_algebra_terms", "has_geometry_terms", "has_stats_terms",
    "subject_difficulty_tier", "construct_frequency",
    "n_students_attempted", "n_correct_responses", "n_incorrect_responses",
    "difficulty_numeric", "OriginalQuestionId",
]

for col in FLOAT_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in INT_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ── 6. Remove / cap outliers using IQR method ─────────────────────────────────
print("Step 6: Capping outliers using IQR (1.5× rule)...")

OUTLIER_COLS = [
    "pct_correct", "avg_response_time_sec", "std_response_time_sec",
    "difficulty_p_value", "irt_b_param", "irt_a_param",
    "n_students_attempted", "text_complexity_score",
    "point_biserial_corr", "discrimination_index",
]

outlier_report = {}
for col in OUTLIER_COLS:
    if col not in df.columns:
        continue
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    flagged = df[col].notna() & ((df[col] < lower) | (df[col] > upper))
    outlier_report[col] = int(flagged.sum())
    df[col] = df[col].clip(lower=lower, upper=upper)

print("  Outliers capped per column:")
for k, v in outlier_report.items():
    print(f"    {k}: {v} values capped")


# ── 7. Enforce domain-specific constraints ────────────────────────────────────
print("Step 7: Enforcing domain constraints...")

# Percentages must be 0–100
for col in ["pct_correct", "pct_chose_A", "pct_chose_B", "pct_chose_C", "pct_chose_D",
            "p_value_upper27", "p_value_lower27", "difficulty_p_value"]:
    if col in df.columns:
        df[col] = df[col].clip(lower=0, upper=100)

# Binary flags must be 0 or 1
for col in ["has_misconception", "has_latex", "has_advanced_terms",
            "has_algebra_terms", "has_geometry_terms", "has_stats_terms"]:
    if col in df.columns:
        df[col] = df[col].where(df[col].isin([0.0, 1.0, np.nan]), other=np.nan)

# Counts must be non-negative
for col in ["num_misconceptions", "text_length", "word_count", "sentence_count",
            "latex_command_count", "math_operator_count", "number_count",
            "n_students_attempted", "n_correct_responses", "n_incorrect_responses"]:
    if col in df.columns:
        df[col] = df[col].where(df[col].isna() | (df[col] >= 0), other=np.nan)


# ── 8. Final dtype cleanup ────────────────────────────────────────────────────
print("Step 8: Final dtype assignment...")

# Downcast int cols to Int64 (nullable integer) for clean display
for col in INT_COLS:
    if col in df.columns:
        df[col] = pd.array(df[col], dtype=pd.Int64Dtype())


# ── 9. Save ───────────────────────────────────────────────────────────────────
output_path = "exam_dataset_50k_cleaned.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Cleaning complete!")
print(f"   Final shape  : {df.shape}")
print(f"   Output file  : {output_path}")
print(f"\nMissing value summary (top 10 cols with most NaNs):")
print(df.isnull().sum().sort_values(ascending=False).head(10).to_string())