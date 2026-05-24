import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")
MODELS_DIR = Path("outputs/models")

df = pd.read_csv(PROCESSED_DIR / "yrbss_2023_model_dataset_suicide_risk_main_no_qn45_no_sitecode.csv")

print(df.shape)
print(df["target"].value_counts())
print(df["target"].value_counts(normalize=True))

BEST_MODEL_NAME = "XGBoost"
BEST_IMBALANCE_STRATEGY = "random_undersampling"

print("Chosen model:")
print(BEST_MODEL_NAME, "|", BEST_IMBALANCE_STRATEGY)

record_id = df["record_id"]
y = df["target"]
X = df.drop(columns=["record_id", "target"])

X_train, X_test, y_train, y_test, record_id_train, record_id_test = train_test_split(
    X,
    y,
    record_id,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nTest target distribution:")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True))

MODEL_PATH = MODELS_DIR / "XGBoost_random_undersampling_cv.pkl"

def convert_to_string(X):
    return X.astype(str)

best_model_object = joblib.load(MODEL_PATH)

fitted_preprocessor = best_model_object["preprocessor"]
fitted_model = best_model_object["model"]

X_test_processed = fitted_preprocessor.transform(X_test)

y_pred = fitted_model.predict(X_test_processed)

print("\nBest model loaded for subgroup error analysis.")
print("Model:", BEST_MODEL_NAME, "|", BEST_IMBALANCE_STRATEGY)
print("Best parameters:", best_model_object["best_params"])
print("Predictions created:", len(y_pred))

test_results = X_test.copy()
test_results["record_id"] = record_id_test.values
test_results["y_true"] = y_test.values
test_results["y_pred"] = y_pred

print("\nTest results created:")
print(test_results[["record_id", "y_true", "y_pred", "sex", "grade", "race4", "age"]].head())

subgroup_metrics = []

for group_col in ["sex", "grade", "race4", "age"]:
    for group_value, group_df in test_results.groupby(group_col):
        y_true_group = group_df["y_true"]
        y_pred_group = group_df["y_pred"]

        if y_true_group.nunique() < 2:
            continue

        cm = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        subgroup_metrics.append({
            "subgroup_variable": group_col,
            "group": group_value,
            "n": len(group_df),
            "recall": recall_score(y_true_group, y_pred_group, zero_division=0),
            "precision": precision_score(y_true_group, y_pred_group, zero_division=0),
            "f1_score": f1_score(y_true_group, y_pred_group, zero_division=0),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })

subgroup_metrics_df = pd.DataFrame(subgroup_metrics)

subgroup_metrics_df.to_csv(
    TABLES_DIR / "subgroup_error_metrics.csv",
    index=False
)

print("\nSaved: subgroup_error_metrics.csv")
print(subgroup_metrics_df)

subgroup_metrics_filtered = subgroup_metrics_df[subgroup_metrics_df["n"] >= 100].copy()

subgroup_metrics_filtered.to_csv(
    TABLES_DIR / "subgroup_error_metrics_n100.csv",
    index=False
)

print("\nSaved: subgroup_error_metrics_n100.csv")
print(subgroup_metrics_filtered)

plot_df = subgroup_metrics_filtered.copy()
plot_df["label"] = (
    plot_df["subgroup_variable"].astype(str)
    + " = "
    + plot_df["group"].astype(str)
)

plot_df = plot_df.sort_values("recall", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(plot_df["label"], plot_df["recall"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Recall")
plt.xlabel("Subgroup")
plt.title("Recall by demographic subgroup")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "subgroup_recall.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved: subgroup_recall.png")

plot_df = subgroup_metrics_filtered.copy()
plot_df["label"] = (
    plot_df["subgroup_variable"].astype(str)
    + " = "
    + plot_df["group"].astype(str)
)

plot_df = plot_df.sort_values("f1_score", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(plot_df["label"], plot_df["f1_score"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("F1-score")
plt.xlabel("Subgroup")
plt.title("F1-score by demographic subgroup")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "subgroup_f1_score.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved: subgroup_f1_score.png")