import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier


PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")
MODELS_DIR = Path("outputs/models")

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MAIN_DATA_PATH = PROCESSED_DIR / "yrbss_2023_model_dataset_suicide_risk_main_no_qn45_no_sitecode.csv"

ROBUSTNESS_DATASETS = {
    "main_no_qn45_no_sitecode": MAIN_DATA_PATH,
    "with_qn45": PROCESSED_DIR / "yrbss_2023_model_dataset_suicide_risk_robust_with_qn45.csv",
    "with_sitecode": PROCESSED_DIR / "yrbss_2023_model_dataset_suicide_risk_robust_with_sitecode.csv",
}

MAIN_MODEL_PATH = MODELS_DIR / "XGBoost_random_undersampling_cv.pkl"


def convert_to_string(X):
    return X.astype(str)


best_model_object = joblib.load(MAIN_MODEL_PATH)

best_params = best_model_object["best_params"].copy()

best_params.pop("cv_f1", None)

print("\nBest model parameters loaded from 03:")
print(best_params)


def build_preprocessor(X):
    categorical_features = ["age", "sex", "grade", "race4"]

    if "sitecode" in X.columns:
        categorical_features.append("sitecode")

    categorical_features = [col for col in categorical_features if col in X.columns]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    categorical_preprocessor = Pipeline(steps=[
        ("to_string", FunctionTransformer(convert_to_string, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    numeric_preprocessor = FeatureUnion(transformer_list=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("missing_indicator", MissingIndicator(features="missing-only")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_preprocessor, categorical_features),
            ("numeric", numeric_preprocessor, numeric_features),
        ]
    )

    return preprocessor, categorical_features, numeric_features


def evaluate_predictions(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def train_and_evaluate_dataset(dataset_name, dataset_path):
    if not dataset_path.exists():
        print(f"\nSkipping {dataset_name}: file not found at {dataset_path}")
        return None

    print(f"\nRunning robustness check: {dataset_name}")
    print(f"Data path: {dataset_path}")

    df = pd.read_csv(dataset_path)

    record_id = df["record_id"]
    y = df["target"]
    X = df.drop(columns=["record_id", "target"])

    X_train, X_test, y_train, y_test, record_id_train, record_id_test = train_test_split(
        X,
        y,
        record_id,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor, categorical_features, numeric_features = build_preprocessor(X)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    sampler = RandomUnderSampler(random_state=42)
    X_train_under, y_train_under = sampler.fit_resample(X_train_processed, y_train)

    model = XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    model.fit(X_train_under, y_train_under)

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    metrics = evaluate_predictions(y_test, y_pred, y_proba)

    result = {
        "robustness_model": dataset_name,
        "n_records": len(df),
        "n_features": X.shape[1],
        "positive_class_count": int(y.sum()),
        "positive_class_proportion": y.mean(),
        "categorical_features": ", ".join(categorical_features),
        "numeric_feature_count": len(numeric_features),
        **metrics,
    }

    model_object = {
        "dataset_name": dataset_name,
        "preprocessor": preprocessor,
        "model": model,
        "best_params_from_main_model": best_params,
        "imbalance_strategy": "random_undersampling",
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
    }

    joblib.dump(
        model_object,
        MODELS_DIR / f"robustness_{dataset_name}_xgboost_random_undersampling.pkl"
    )

    print("Results:")
    print({
        "recall": round(metrics["recall"], 3),
        "precision": round(metrics["precision"], 3),
        "f1_score": round(metrics["f1_score"], 3),
        "pr_auc": round(metrics["pr_auc"], 3),
        "roc_auc": round(metrics["roc_auc"], 3),
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
    })

    return result


robustness_results = []

for dataset_name, dataset_path in ROBUSTNESS_DATASETS.items():
    result = train_and_evaluate_dataset(dataset_name, dataset_path)

    if result is not None:
        robustness_results.append(result)


robustness_df = pd.DataFrame(robustness_results)

robustness_df.to_csv(
    TABLES_DIR / "robustness_results_summary.csv",
    index=False
)

print("\nSaved: robustness_results_summary.csv")
print(robustness_df)


robustness_thesis = robustness_df.copy()

metric_cols = ["recall", "precision", "f1_score", "pr_auc", "roc_auc", "positive_class_proportion"]

for col in metric_cols:
    robustness_thesis[col] = robustness_thesis[col].round(3)

robustness_thesis.to_csv(
    TABLES_DIR / "robustness_results_summary_for_thesis.csv",
    index=False
)

print("\nSaved: robustness_results_summary_for_thesis.csv")
print(robustness_thesis)


plot_df = robustness_thesis.copy()

label_map = {
    "main_no_qn45_no_sitecode": "Main model",
    "with_qn45": "With qn45",
    "with_sitecode": "With sitecode"
}

plot_df["plot_label"] = plot_df["robustness_model"].replace(label_map)

if not plot_df.empty:
    metrics_to_plot = ["recall", "precision", "f1_score", "pr_auc"]
    metric_labels = {
        "recall": "Recall",
        "precision": "Precision",
        "f1_score": "F1-score",
        "pr_auc": "PR-AUC"
    }

    x = np.arange(len(plot_df))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 5.5))

    colors = ["#1F4E79", "#3D7EA6", "#73A9C2", "#BFD7EA"]

    for i, metric in enumerate(metrics_to_plot):
        bars = ax.bar(
            x + (i - 1.5) * width,
            plot_df[metric],
            width,
            label=metric_labels[metric],
            color=colors[i]
        )

        ax.bar_label(
            bars,
            labels=[f"{value:.3f}" for value in plot_df[metric]],
            padding=2,
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["plot_label"], fontsize=10)

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Robustness check comparison", fontsize=13, fontweight="bold")

    ax.set_ylim(0.45, 0.80)
    ax.legend(frameon=True, fontsize=9, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    ax.tick_params(axis="y", labelsize=10)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "robustness_comparison_metrics.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print("\nSaved: robustness_comparison_metrics.png")
