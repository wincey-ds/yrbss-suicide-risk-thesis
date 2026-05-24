"""
YRBSS 2023 MS Combined State Dataset — Suicide-related risk dataset preparation

Main thesis dataset:
    - Uses 2023 state data only.
    - Target: suicide_related_risk
        1 = student reported at least one suicide-related risk indicator:
            qn15 = seriously considered suicide
            qn16 = made a suicide plan
            qn17 = attempted suicide
        0 = student answered at least one of qn15/qn16/qn17 and none were positive
        NaN = all three target items missing
    - Predictors:
        * demographic variables: age, sex, grade, race4
        * remaining qn variables
        * excludes qn15, qn16, qn17 to prevent target leakage
        * excludes qn45 in the main model because it directly measures poor mental health

Robustness datasets:
    1. Same as main model, but includes qn45
    2. Same as main model, but includes sitecode

Sample weights:
    - Kept separately
    - Not used as model features

Important:
    - Missing values are not imputed in this script.
    - Imputation should be performed inside the modelling pipeline to prevent data leakage.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------
# 1) File paths
# ---------------------------------------------------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")
MODELS_DIR = Path("outputs/models")

for folder in [RAW_DIR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

SAS_PATH = RAW_DIR / "2023-SADC-MS-SAS-Input-Program.sas"
DATA_AM = RAW_DIR / "sadc_ms_2023_state_a_m.dat"
DATA_NZ = RAW_DIR / "sadc_ms_2023_state_n_z.dat"


# ---------------------------------------------------------------------
# 2) Parse SAS input program
# ---------------------------------------------------------------------
def extract_width(sas_format: str) -> int | None:
    """Extract fixed width from SAS format string."""
    m = re.search(r"(\d+)", sas_format)
    return int(m.group(1)) if m else None


def parse_sas_input(sas_text: str) -> tuple[list[tuple[int, int]], list[str]]:
    """
    Parse SAS INPUT block and return fixed-width column specs and variable names.
    Supports @position formats and range formats.
    """
    upper = sas_text.upper()
    idx = upper.find("INPUT")

    if idx == -1:
        raise ValueError("Could not find an INPUT block in the SAS file.")

    window = sas_text[idx: idx + 20000]

    pat_at = re.compile(
        r"@(\d+)\s+([A-Za-z_]\w*)\s*(?::\s*)?(\$?[A-Za-z]*\d+(?:\.\d+)?)\.",
        re.IGNORECASE,
    )

    at_matches = pat_at.findall(window)

    if at_matches:
        starts = [int(s) for s, _, _ in at_matches]
        names = [v.lower() for _, v, _ in at_matches]
        widths = [extract_width(fmt) for _, _, fmt in at_matches]

        if any(w is None for w in widths):
            raise ValueError("Could not extract width from one or more SAS formats.")

        colspecs = [(s - 1, (s - 1) + w) for s, w in zip(starts, widths)]
        return colspecs, names

    pat_rng = re.compile(
        r"\b([A-Za-z_]\w*)\s+\$?\s*(\d+)\s*-\s*(\d+)",
        re.IGNORECASE,
    )

    rng_matches = pat_rng.findall(window)

    if rng_matches:
        names = [v.lower() for v, _, _ in rng_matches]
        colspecs = [(int(a) - 1, int(b)) for _, a, b in rng_matches]
        return colspecs, names

    raise ValueError("No recognizable column definitions found in SAS INPUT block.")


# ---------------------------------------------------------------------
# 3) Read and merge state data
# ---------------------------------------------------------------------
sas_text = SAS_PATH.read_text(errors="ignore")
colspecs, names = parse_sas_input(sas_text)

df_am = pd.read_fwf(DATA_AM, colspecs=colspecs, names=names)
df_nz = pd.read_fwf(DATA_NZ, colspecs=colspecs, names=names)

df = pd.concat([df_am, df_nz], ignore_index=True)

df.columns = df.columns.str.lower()

df = df.replace(".", np.nan)
df = df.replace(r"^\s*$", np.nan, regex=True)

print("A–M shape:", df_am.shape)
print("N–Z shape:", df_nz.shape)
print("Combined state shape:", df.shape)


# ---------------------------------------------------------------------
# 4) Variable labels / data dictionary
# ---------------------------------------------------------------------
label_pattern = re.compile(r'(\w+)\s*=\s*"([^"]+)"')
labels_dict = {k.lower(): v for k, v in label_pattern.findall(sas_text)}

labels_dict["qn17"] = "Ever tried to kill themselves"

labels_df = pd.DataFrame({"variable": df.columns})
labels_df["label"] = labels_df["variable"].map(labels_dict)
labels_df.to_csv(PROCESSED_DIR / "variable_labels_yrbss_ms_2023_corrected.csv", index=False)

print("\n--- Suicide-related variable labels ---")
print(labels_df[labels_df["variable"].isin(["q15", "q16", "q17", "qn15", "qn16", "qn17"])])


# ---------------------------------------------------------------------
# 5) Convert year and target variables for full-dataset EDA
# ---------------------------------------------------------------------
df["year"] = pd.to_numeric(df["year"], errors="coerce")

target_vars = [
    "qn15",  # ever seriously thought about killing themselves
    "qn16",  # ever made a plan about how they would kill themselves
    "qn17",  # ever tried to kill themselves
]

missing_target_vars = [c for c in target_vars if c not in df.columns]
if missing_target_vars:
    raise ValueError(f"Missing target variables: {missing_target_vars}")

for col in target_vars:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ---------------------------------------------------------------------
# 6) EDA: justify using 2023 only
# ---------------------------------------------------------------------
print("\n================ EDA: YEAR COVERAGE AND MISSINGNESS ================")

records_by_year = (
    df.groupby("year")
    .size()
    .reset_index(name="n_records")
    .sort_values("year")
)

print("\nRecords by year:")
print(records_by_year)

df["n_suicide_target_observed_temp"] = df[target_vars].notna().sum(axis=1)

df["suicide_related_risk_temp"] = np.where(
    df["n_suicide_target_observed_temp"] == 0,
    np.nan,
    np.where((df[target_vars] == 1).any(axis=1), 1, 0)
)

target_by_year = (
    df.groupby("year")
    .agg(
        n_records=("year", "size"),
        n_target_observed=("suicide_related_risk_temp", lambda x: x.notna().sum()),
        target_missing_rate=("suicide_related_risk_temp", lambda x: x.isna().mean()),
        risk_prevalence=("suicide_related_risk_temp", lambda x: x.dropna().mean()),
    )
    .reset_index()
)

print("\nTarget availability and prevalence by year:")
print(target_by_year)

important_vars = [
    "qn11",  # neighborhood violence exposure
    "qn12",  # unfair treatment because of race/ethnicity
    "qn13",  # bullied at school
    "qn14",  # electronically bullied
    "qn44",  # social media use
    "qn45",  # poor mental health
    "qn46",  # sleep
    "qn47",  # unstable housing
    "qn48",  # grades
]

important_vars = [c for c in important_vars if c in df.columns]

important_missing_by_year = (
    df.groupby("year")[important_vars]
    .apply(lambda g: g.isna().mean())
    .reset_index()
)

print("\nMissingness of key theoretically relevant variables by year:")
print(important_missing_by_year)

all_qn_cols = [c for c in df.columns if c.startswith("qn")]

exclude_qn = {"qn15", "qn16", "qn17"}
candidate_qn_cols = [c for c in all_qn_cols if c not in exclude_qn]

availability_rows = []

for year, g in df.groupby("year"):
    missing_rates = g[candidate_qn_cols].isna().mean()

    availability_rows.append({
        "year": year,
        "n_records": len(g),
        "n_qn_predictors_total": len(candidate_qn_cols),
        "n_qn_predictors_missing_lt_25pct": (missing_rates < 0.25).sum(),
        "n_qn_predictors_missing_lt_50pct": (missing_rates < 0.50).sum(),
        "n_qn_predictors_missing_lt_85pct": (missing_rates < 0.85).sum(),
        "mean_qn_missingness": missing_rates.mean(),
    })

predictor_availability_by_year = pd.DataFrame(availability_rows).sort_values("year")

print("\nPredictor availability by year:")
print(predictor_availability_by_year)

records_by_year.to_csv(TABLES_DIR / "eda_records_by_year.csv", index=False)
target_by_year.to_csv(TABLES_DIR / "eda_target_by_year.csv", index=False)
important_missing_by_year.to_csv(TABLES_DIR / "eda_key_variable_missingness_by_year.csv", index=False)
predictor_availability_by_year.to_csv(TABLES_DIR / "eda_predictor_availability_by_year.csv", index=False)

print("\nSaved EDA justification tables:")
print("eda_records_by_year.csv")
print("eda_target_by_year.csv")
print("eda_key_variable_missingness_by_year.csv")
print("eda_predictor_availability_by_year.csv")


# ---------------------------------------------------------------------
# 7) Filter to 2023 only
# ---------------------------------------------------------------------
df_2023 = df[df["year"] == 2023].copy()

print("\n--- 2023-only state dataset ---")
print("2023 shape:", df_2023.shape)

if df_2023.empty:
    raise ValueError("No rows found for year == 2023. Check how the year variable is coded.")


# ---------------------------------------------------------------------
# 8) Define final suicide-related risk target for 2023
# ---------------------------------------------------------------------
df_2023["n_suicide_target_observed"] = df_2023[target_vars].notna().sum(axis=1)

df_2023["suicide_related_risk"] = np.where(
    df_2023["n_suicide_target_observed"] == 0,
    np.nan,
    np.where((df_2023[target_vars] == 1).any(axis=1), 1, 0)
)

print("\n--- 2023 suicide-related risk target distribution ---")
print(df_2023["suicide_related_risk"].value_counts(dropna=False))

print("\nTarget proportions:")
print(df_2023["suicide_related_risk"].value_counts(normalize=True, dropna=False))

print("\n--- Individual suicide-related items, 2023 ---")
for col in target_vars:
    print(f"\n{col}: {labels_dict.get(col, '')}")
    print(df_2023[col].value_counts(dropna=False).sort_index())
    print(df_2023[col].value_counts(normalize=True, dropna=False).sort_index())


# ---------------------------------------------------------------------
# 9) Verification: q17 and qn17 should match
# ---------------------------------------------------------------------
if "q17" in df_2023.columns and "qn17" in df_2023.columns:
    df_2023["q17"] = pd.to_numeric(df_2023["q17"], errors="coerce")

    print("\n--- Verification: q17 vs qn17, 2023 ---")
    print(pd.crosstab(df_2023["q17"], df_2023["qn17"], dropna=False))


# ---------------------------------------------------------------------
# 10) Keep rows with known target
# ---------------------------------------------------------------------
df_model = df_2023[df_2023["suicide_related_risk"].notna()].copy()
df_model["suicide_related_risk"] = df_model["suicide_related_risk"].astype(int)

df_model["record_id"] = df_model.index.astype(int)

print("\nModelling shape after dropping missing target:", df_model.shape)

print("\nFinal target distribution:")
print(df_model["suicide_related_risk"].value_counts())

print("\nFinal target proportions:")
print(df_model["suicide_related_risk"].value_counts(normalize=True))


# ---------------------------------------------------------------------
# 11) Keep sample weights and survey design variables separately
# ---------------------------------------------------------------------
weights_cols = ["record_id", "weight", "stratum", "psu", "sitecode", "year"]
available_weight_cols = [c for c in weights_cols if c in df_model.columns]

if "weight" in df_model.columns:
    weights_df = df_model[available_weight_cols].copy()
    weights_df["target"] = df_model["suicide_related_risk"].values

    weights_df.to_csv(PROCESSED_DIR / "yrbss_2023_suicide_related_risk_sample_weights.csv", index=False)
    print("\nSaved: yrbss_2023_suicide_related_risk_sample_weights.csv")
else:
    print("\nNo sample weight variable found.")


# ---------------------------------------------------------------------
# 12) Build feature sets
# ---------------------------------------------------------------------
base_features_main = ["age", "sex", "grade", "race4"]

base_features_with_site = ["sitecode", "age", "sex", "grade", "race4"]

target_vars_to_remove = {
    "q15", "q16", "q17",
    "qn15", "qn16", "qn17",
}

direct_mental_health_var = "qn45"

all_qn_cols = [c for c in df_model.columns if c.startswith("qn")]

main_qn_cols = [
    c for c in all_qn_cols
    if c not in target_vars_to_remove
    and c != direct_mental_health_var
]

robust_qn_cols = [
    c for c in all_qn_cols
    if c not in target_vars_to_remove
]

feature_cols_main = [c for c in base_features_main if c in df_model.columns] + main_qn_cols
feature_cols_robust_qn45 = [c for c in base_features_main if c in df_model.columns] + robust_qn_cols
feature_cols_robust_site = [c for c in base_features_with_site if c in df_model.columns] + main_qn_cols

X_main = df_model[feature_cols_main].copy()
X_robust_qn45 = df_model[feature_cols_robust_qn45].copy()
X_robust_site = df_model[feature_cols_robust_site].copy()

y = df_model["suicide_related_risk"].copy()


# ---------------------------------------------------------------------
# 13) Recode feature types
# ---------------------------------------------------------------------
def recode_features(
    X: pd.DataFrame,
    categorical_base_features: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    
    X = X.copy()

    categorical_features = [c for c in categorical_base_features if c in X.columns]

    for col in categorical_features:
        X[col] = X[col].astype("object")

    qn_feature_cols = [c for c in X.columns if c.startswith("qn")]

    for col in qn_feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].map({1: 1, 2: 0})

    numeric_qn_features = qn_feature_cols

    return X, categorical_features, numeric_qn_features


X_main, categorical_main, numeric_main = recode_features(
    X_main,
    base_features_main,
)

X_robust_qn45, categorical_robust_qn45, numeric_robust_qn45 = recode_features(
    X_robust_qn45,
    base_features_main,
)

X_robust_site, categorical_robust_site, numeric_robust_site = recode_features(
    X_robust_site,
    base_features_with_site,
)


# ---------------------------------------------------------------------
# 14) Drop qn columns with extreme missingness
# ---------------------------------------------------------------------
def drop_high_missing_qn(
    X: pd.DataFrame,
    numeric_qn_features: list[str],
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Drop qn columns with missingness above threshold.
    Only applies to qn variables, not demographics.
    """
    missing_ratio = X.isna().mean()

    high_missing_cols = [
        c for c in numeric_qn_features
        if missing_ratio[c] > threshold
    ]

    X = X.drop(columns=high_missing_cols)
    numeric_qn_features = [c for c in numeric_qn_features if c not in high_missing_cols]

    return X, numeric_qn_features, high_missing_cols


missing_threshold = 0.85

X_main, numeric_main, high_missing_main = drop_high_missing_qn(
    X_main,
    numeric_main,
    threshold=missing_threshold,
)

X_robust_qn45, numeric_robust_qn45, high_missing_robust_qn45 = drop_high_missing_qn(
    X_robust_qn45,
    numeric_robust_qn45,
    threshold=missing_threshold,
)

X_robust_site, numeric_robust_site, high_missing_robust_site = drop_high_missing_qn(
    X_robust_site,
    numeric_robust_site,
    threshold=missing_threshold,
)

print(f"\nDropped qn columns with > {missing_threshold:.0%} missingness in main dataset:")
print(high_missing_main if high_missing_main else "None")

print(f"\nDropped qn columns with > {missing_threshold:.0%} missingness in qn45 robustness dataset:")
print(high_missing_robust_qn45 if high_missing_robust_qn45 else "None")

print(f"\nDropped qn columns with > {missing_threshold:.0%} missingness in sitecode robustness dataset:")
print(high_missing_robust_site if high_missing_robust_site else "None")


# ---------------------------------------------------------------------
# 15) Dataset checks
# ---------------------------------------------------------------------
def print_dataset_checks(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: list[str],
    numeric_qn_features: list[str],
):
    print(f"\n================ {name} ================")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nTarget proportions:")
    print(y.value_counts(normalize=True))

    print("\nTop 20 missing values:")
    print(X.isna().sum().sort_values(ascending=False).head(20))

    print("\nCategorical features:")
    print(categorical_features)

    print("\nNumeric qn features:")
    print(numeric_qn_features)

    print("\nX head:")
    print(X.head())


print_dataset_checks(
    "MAIN DATASET: 2023, WITHOUT qn45, WITHOUT sitecode",
    X_main,
    y,
    categorical_main,
    numeric_main,
)

print_dataset_checks(
    "ROBUSTNESS DATASET: 2023, WITH qn45",
    X_robust_qn45,
    y,
    categorical_robust_qn45,
    numeric_robust_qn45,
)

print_dataset_checks(
    "ROBUSTNESS DATASET: 2023, WITH sitecode, WITHOUT qn45",
    X_robust_site,
    y,
    categorical_robust_site,
    numeric_robust_site,
)


# ---------------------------------------------------------------------
# 16) Subgroup descriptives 
# ---------------------------------------------------------------------
def print_subgroup_target_distribution(X: pd.DataFrame, y: pd.Series, group_col: str):
    if group_col in X.columns:
        print(f"\n--- Target by {group_col} ---")
        print(pd.crosstab(X[group_col], y, normalize="index"))
        print("\nCounts:")
        print(pd.crosstab(X[group_col], y))


for group_col in ["sex", "grade", "race4", "age"]:
    print_subgroup_target_distribution(X_main, y, group_col)


# ---------------------------------------------------------------------
# 17) Save model datasets and feature info
# ---------------------------------------------------------------------
def save_model_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    record_ids: pd.Series,
    filename_dataset: str,
    filename_feature_info: str,
    categorical_features: list[str],
    numeric_qn_features: list[str],
    labels_dict: dict,
):
    model_df = X.copy()
    model_df.insert(0, "record_id", record_ids.values)
    model_df["target"] = y.values

    dataset_path = Path(filename_dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROCESSED_DIR / dataset_path
    model_df.to_csv(dataset_path, index=False)

    feature_info = pd.DataFrame({
        "feature": X.columns,
        "dtype": [str(X[c].dtype) for c in X.columns],
        "missing_ratio": [X[c].isna().mean() for c in X.columns],
        "is_categorical": [c in categorical_features for c in X.columns],
        "is_numeric_qn": [c in numeric_qn_features for c in X.columns],
        "label": [labels_dict.get(c, np.nan) for c in X.columns],
    })

    feature_info_path = Path(filename_feature_info)
    if not feature_info_path.is_absolute():
        feature_info_path = PROCESSED_DIR / feature_info_path
    feature_info.to_csv(feature_info_path, index=False)


record_ids = df_model["record_id"]

save_model_dataset(
    X_main,
    y,
    record_ids,
    "yrbss_2023_model_dataset_suicide_risk_main_no_qn45_no_sitecode.csv",
    "yrbss_2023_feature_info_suicide_risk_main_no_qn45_no_sitecode.csv",
    categorical_main,
    numeric_main,
    labels_dict,
)

save_model_dataset(
    X_robust_qn45,
    y,
    record_ids,
    "yrbss_2023_model_dataset_suicide_risk_robust_with_qn45.csv",
    "yrbss_2023_feature_info_suicide_risk_robust_with_qn45.csv",
    categorical_robust_qn45,
    numeric_robust_qn45,
    labels_dict,
)

save_model_dataset(
    X_robust_site,
    y,
    record_ids,
    "yrbss_2023_model_dataset_suicide_risk_robust_with_sitecode.csv",
    "yrbss_2023_feature_info_suicide_risk_robust_with_sitecode.csv",
    categorical_robust_site,
    numeric_robust_site,
    labels_dict,
)

print("\nSaved files:")
print(PROCESSED_DIR / "variable_labels_yrbss_ms_2023_corrected.csv")
print(TABLES_DIR / "eda_records_by_year.csv")
print(TABLES_DIR / "eda_target_by_year.csv")
print(TABLES_DIR / "eda_key_variable_missingness_by_year.csv")
print(TABLES_DIR / "eda_predictor_availability_by_year.csv")
print("yrbss_2023_model_dataset_suicide_risk_main_no_qn45_no_sitecode.csv")
print("yrbss_2023_feature_info_suicide_risk_main_no_qn45_no_sitecode.csv")
print("yrbss_2023_model_dataset_suicide_risk_robust_with_qn45.csv")
print("yrbss_2023_feature_info_suicide_risk_robust_with_qn45.csv")
print("yrbss_2023_model_dataset_suicide_risk_robust_with_sitecode.csv")
print("yrbss_2023_feature_info_suicide_risk_robust_with_sitecode.csv")

if "weight" in df_model.columns:
    print(PROCESSED_DIR / "yrbss_2023_suicide_related_risk_sample_weights.csv")