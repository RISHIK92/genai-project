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

