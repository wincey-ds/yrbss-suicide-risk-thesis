from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")

DATA_PATH = PROCESSED_DIR / "yrbss_2023_model_dataset_suicide_risk_main_no_qn45_no_sitecode.csv"
FEATURE_INFO_PATH = PROCESSED_DIR / "yrbss_2023_feature_info_suicide_risk_main_no_qn45_no_sitecode.csv"
feature_info = pd.read_csv(FEATURE_INFO_PATH)

df = pd.read_csv(DATA_PATH)

print(df.shape)
print(df["target"].value_counts())
print(df["target"].value_counts(normalize=True))

target_distribution = (
    df["target"]
    .value_counts()
    .rename_axis("target")
    .reset_index(name="count")
)

target_distribution["proportion"] = target_distribution["count"] / len(df)

target_distribution.to_csv(
    TABLES_DIR / "descriptive_target_distribution.csv",
    index=False
)

print("\nSaved: descriptive_target_distribution.csv")
print(target_distribution)

target_counts = df["target"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(["No suicide-related risk", "Suicide-related risk"], target_counts.values)
plt.ylabel("Number of students")
plt.title("Distribution of suicide-related risk target")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=300)
plt.close()

print("\nSaved: target_distribution.png")

target_by_sex = (
    pd.crosstab(df["sex"], df["target"], normalize="index")
    .reset_index()
)

target_by_sex_counts = (
    pd.crosstab(df["sex"], df["target"])
    .reset_index()
)

target_by_sex.to_csv(TABLES_DIR / "descriptive_target_by_sex_proportions.csv", index=False)
target_by_sex_counts.to_csv(TABLES_DIR / "descriptive_target_by_sex_counts.csv", index=False)

print("\nSaved: descriptive_target_by_sex_proportions.csv")
print("Saved: descriptive_target_by_sex_counts.csv")
print(target_by_sex)

target_by_grade = (
    pd.crosstab(df["grade"], df["target"], normalize="index")
    .reset_index()
)

target_by_grade_counts = (
    pd.crosstab(df["grade"], df["target"])
    .reset_index()
)

target_by_grade.to_csv(TABLES_DIR / "descriptive_target_by_grade_proportions.csv", index=False)
target_by_grade_counts.to_csv(TABLES_DIR / "descriptive_target_by_grade_counts.csv", index=False)

print("\nSaved: descriptive_target_by_grade_proportions.csv")
print("Saved: descriptive_target_by_grade_counts.csv")
print(target_by_grade)

target_by_race4 = (
    pd.crosstab(df["race4"], df["target"], normalize="index")
    .reset_index()
)

target_by_race4_counts = (
    pd.crosstab(df["race4"], df["target"])
    .reset_index()
)

target_by_race4.to_csv(TABLES_DIR / "descriptive_target_by_race4_proportions.csv", index=False)
target_by_race4_counts.to_csv(TABLES_DIR / "descriptive_target_by_race4_counts.csv", index=False)

print("\nSaved: descriptive_target_by_race4_proportions.csv")
print("Saved: descriptive_target_by_race4_counts.csv")
print(target_by_race4)

target_by_age = (
    pd.crosstab(df["age"], df["target"], normalize="index")
    .reset_index()
)

target_by_age_counts = (
    pd.crosstab(df["age"], df["target"])
    .reset_index()
)

target_by_age.to_csv(TABLES_DIR / "descriptive_target_by_age_proportions.csv", index=False)
target_by_age_counts.to_csv(TABLES_DIR / "descriptive_target_by_age_counts.csv", index=False)

print("\nSaved: descriptive_target_by_age_proportions.csv")
print("Saved: descriptive_target_by_age_counts.csv")
print(target_by_age)

feature_missingness = (
    df.drop(columns=["record_id", "target"])
    .isna()
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

feature_missingness.columns = ["feature", "missing_ratio"]

feature_missingness = feature_missingness.merge(
    feature_info[["feature", "label"]],
    on="feature",
    how="left"
)

feature_missingness.to_csv(
    TABLES_DIR / "descriptive_feature_missingness.csv",
    index=False
)

print("\nSaved: descriptive_feature_missingness.csv")
print(feature_missingness.head(20))

top_missing = feature_missingness.head(15).copy()

plt.figure(figsize=(8, 5))
plt.barh(top_missing["feature"], top_missing["missing_ratio"])
plt.xlabel("Missing ratio")
plt.ylabel("Feature")
plt.title("Top 15 predictors with highest missingness")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(FIGURES_DIR / "top15_feature_missingness.png", dpi=300)
plt.close()

print("\nSaved: top15_feature_missingness.png")

subgroup_tables = []

for group_col in ["sex", "grade", "race4", "age"]:
    counts = pd.crosstab(df[group_col], df["target"])
    props = pd.crosstab(df[group_col], df["target"], normalize="index")

    temp = counts.reset_index().melt(
        id_vars=group_col,
        var_name="target",
        value_name="count"
    )

    temp_props = props.reset_index().melt(
        id_vars=group_col,
        var_name="target",
        value_name="proportion"
    )

    temp = temp.merge(temp_props, on=[group_col, "target"])
    temp = temp.rename(columns={group_col: "group"})
    temp.insert(0, "subgroup_variable", group_col)

    subgroup_tables.append(temp)

subgroup_distribution = pd.concat(subgroup_tables, ignore_index=True)

subgroup_distribution.to_csv(
    TABLES_DIR / "descriptive_subgroup_target_distribution.csv",
    index=False
)

print("\nSaved: descriptive_subgroup_target_distribution.csv")
print(subgroup_distribution.head(20))

selected_vars = [
    "target",
    "qn13",      # bullied at school
    "qn14",      # electronically bullied
    "qn12",      # unfair treatment at school
    "qnbk7day",  # ate breakfast all 7 days
    "qn46",      # 8+ hours of sleep
    "qn40",      # physically active 5+ days
    "qn10",      # physical fight
    "qn42",      # sports team participation
    "qn9",       # rode with drinking driver
    "qn11",      # saw neighborhood violence
    "qn26",      # ever drank alcohol
    "qn22",      # ever used electronic vapor product
]

selected_vars = [var for var in selected_vars if var in df.columns]

corr_df = df[selected_vars].copy()

for col in corr_df.columns:
    corr_df[col] = pd.to_numeric(corr_df[col], errors="coerce")

corr_matrix = corr_df.corr(method="pearson")

label_map = {
    "target": "Suicide-related risk",
    "qn13": "Bullied at school",
    "qn14": "Electronically bullied",
    "qn12": "Unfair treatment at school",
    "qnbk7day": "Ate breakfast all 7 days",
    "qn46": "8+ hours of sleep",
    "qn40": "Physically active 5+ days",
    "qn10": "Physical fight",
    "qn42": "Sports team participation",
    "qn9": "Rode with drinking driver",
    "qn11": "Saw neighborhood violence",
    "qn26": "Ever drank alcohol",
    "qn22": "Ever used e-vapor product",
}

corr_matrix = corr_matrix.rename(index=label_map, columns=label_map)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(11, 9))

im = ax.imshow(
    np.ma.masked_where(mask, corr_matrix),
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.index)))

ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(corr_matrix.index, fontsize=9)

for i in range(len(corr_matrix.index)):
    for j in range(len(corr_matrix.columns)):
        if not mask[i, j]:
            value = corr_matrix.iloc[i, j]
            if not np.isnan(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black"
                )


cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson correlation", fontsize=10)

ax.set_title(
    "Correlation heatmap for selected YRBS predictors",
    fontsize=13,
    fontweight="bold",
    pad=14
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "correlation_heatmap_selected_predictors.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

corr_matrix.to_csv(
    TABLES_DIR / "correlation_matrix_selected_predictors.csv"
)

print("\nSaved: correlation_heatmap_selected_predictors.png")
print("Saved: correlation_matrix_selected_predictors.csv")

appendix_feature_overview = feature_info[
    [
        "feature",
        "label",
        "dtype",
        "missing_ratio",
        "is_categorical",
        "is_numeric_qn"
    ]
].copy()

appendix_feature_overview = appendix_feature_overview.rename(columns={
    "feature": "Variable",
    "label": "Description",
    "dtype": "Data type",
    "missing_ratio": "Missing ratio",
    "is_categorical": "Categorical",
    "is_numeric_qn": "YRBS dichotomous variable"
})

appendix_feature_overview["Missing ratio"] = appendix_feature_overview["Missing ratio"].round(3)

appendix_feature_overview.to_csv(
    TABLES_DIR / "appendix_a_feature_overview.csv",
    index=False
)

print("\nSaved: appendix_a_feature_overview.csv")
print(appendix_feature_overview.head(20))