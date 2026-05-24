from pathlib import Path
import itertools
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
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
    make_scorer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

# ---------------------------------------------------------------------
# 1) Paths
# ---------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")
MODELS_DIR = Path("outputs/models")

for folder in [TABLES_DIR, FIGURES_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROCESSED_DIR / "yrbss_2023_model_dataset_suicide_risk_main_no_qn45_no_sitecode.csv"

# ---------------------------------------------------------------------
# 2) Load data
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

record_id = df["record_id"]
y = df["target"]
X = df.drop(columns=["record_id", "target"])

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nTarget distribution:")
print(y.value_counts())
print(y.value_counts(normalize=True))

# ---------------------------------------------------------------------
# 3) Train-test split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test, record_id_train, record_id_test = train_test_split(
    X,
    y,
    record_id,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ---------------------------------------------------------------------
# 4) Feature types
# ---------------------------------------------------------------------
categorical_features = ["age", "sex", "grade", "race4"]
numeric_features = [col for col in X.columns if col not in categorical_features]

print("\nCategorical features:")
print(categorical_features)
print("\nNumber of numeric/qn features:", len(numeric_features))

# ---------------------------------------------------------------------
# 5) Preprocessing
# ---------------------------------------------------------------------
def convert_to_string(X):
    return X.astype(str)

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

print("\nPreprocessing pipeline created.")

# ---------------------------------------------------------------------
# 6) Helpers
# ---------------------------------------------------------------------
f1_scorer = make_scorer(f1_score, pos_label=1)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
best_params = []
cv_fold_scores = []


def evaluate_and_store(model_name, imbalance_strategy, y_test, y_pred, y_proba, best_param_dict=None, model_object=None):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    row = {
        "model": model_name,
        "imbalance_strategy": imbalance_strategy,
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    results.append(row)

    best_params.append({
        "model": model_name,
        "imbalance_strategy": imbalance_strategy,
        "best_params": best_param_dict if best_param_dict is not None else {},
    })

    print(f"\n{model_name} | {imbalance_strategy}")
    print("Best params:", best_param_dict)
    print("Recall:", row["recall"])
    print("Precision:", row["precision"])
    print("F1-score:", row["f1_score"])
    print("PR-AUC:", row["pr_auc"])
    print("ROC-AUC:", row["roc_auc"])
    print("Confusion matrix:")
    print(cm)

    if model_object is not None:
        safe_name = f"{model_name}_{imbalance_strategy}".replace(" ", "_").replace("/", "_")
        joblib.dump(model_object, MODELS_DIR / f"{safe_name}_cv.pkl")


def collect_cv_fold_scores_sklearn(model_name, imbalance_strategy, best_model):
    scoring = {
        "f1_score": "f1",
        "recall": "recall",
        "pr_auc": "average_precision",
    }

    scores = cross_validate(
        best_model,
        X_train,
        y_train,
        scoring=scoring,
        cv=cv5,
        n_jobs=1,
    )

    for fold_idx in range(5):
        cv_fold_scores.append({
            "model": model_name,
            "imbalance_strategy": imbalance_strategy,
            "fold": fold_idx + 1,
            "f1_score": scores["test_f1_score"][fold_idx],
            "recall": scores["test_recall"][fold_idx],
            "pr_auc": scores["test_pr_auc"][fold_idx],
        })


def run_gridsearch_sklearn(model_name, imbalance_strategy, estimator, param_grid):
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=cv5,
        n_jobs=1,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    evaluate_and_store(
        model_name=model_name,
        imbalance_strategy=imbalance_strategy,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        best_param_dict=grid.best_params_,
        model_object=best_model,
    )

    collect_cv_fold_scores_sklearn(
        model_name=model_name,
        imbalance_strategy=imbalance_strategy,
        best_model=best_model,
    )


def iter_param_grid(param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def collect_cv_fold_scores_xgboost(model_name, imbalance_strategy, best_params_local, sampler=None, scale_pos_weight=None):
    fold_idx = 1

    for train_idx, val_idx in cv5.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        fold_preprocessor = clone(preprocessor)
        X_tr_processed = fold_preprocessor.fit_transform(X_tr)
        X_val_processed = fold_preprocessor.transform(X_val)

        y_tr_model = y_tr
        if sampler is not None:
            X_tr_processed, y_tr_model = sampler.fit_resample(X_tr_processed, y_tr)

        xgb_kwargs = dict(
            **best_params_local,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )

        if scale_pos_weight is not None:
            xgb_kwargs["scale_pos_weight"] = scale_pos_weight

        model = XGBClassifier(**xgb_kwargs)
        model.fit(X_tr_processed, y_tr_model)

        y_val_pred = model.predict(X_val_processed)
        y_val_proba = model.predict_proba(X_val_processed)[:, 1]

        cv_fold_scores.append({
            "model": model_name,
            "imbalance_strategy": imbalance_strategy,
            "fold": fold_idx,
            "f1_score": f1_score(y_val, y_val_pred, zero_division=0),
            "recall": recall_score(y_val, y_val_pred, zero_division=0),
            "pr_auc": average_precision_score(y_val, y_val_proba),
        })

        fold_idx += 1


def run_xgboost_manual_cv(model_name, imbalance_strategy, sampler=None, scale_pos_weight=None):
    xgb_param_grid = {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    best_score = -np.inf
    best_params_local = None

    for params in iter_param_grid(xgb_param_grid):
        fold_scores = []

        for train_idx, val_idx in cv5.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            fold_preprocessor = clone(preprocessor)
            X_tr_processed = fold_preprocessor.fit_transform(X_tr)
            X_val_processed = fold_preprocessor.transform(X_val)

            y_tr_model = y_tr
            if sampler is not None:
                X_tr_processed, y_tr_model = sampler.fit_resample(X_tr_processed, y_tr)

            xgb_kwargs = dict(
                **params,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            )
            if scale_pos_weight is not None:
                xgb_kwargs["scale_pos_weight"] = scale_pos_weight

            model = XGBClassifier(**xgb_kwargs)
            model.fit(X_tr_processed, y_tr_model)

            y_val_pred = model.predict(X_val_processed)
            fold_scores.append(f1_score(y_val, y_val_pred, zero_division=0))

        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params_local = params

    final_preprocessor = clone(preprocessor)
    X_train_processed = final_preprocessor.fit_transform(X_train)
    X_test_processed = final_preprocessor.transform(X_test)

    y_train_model = y_train
    if sampler is not None:
        X_train_processed, y_train_model = sampler.fit_resample(X_train_processed, y_train)

    final_kwargs = dict(
        **best_params_local,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    if scale_pos_weight is not None:
        final_kwargs["scale_pos_weight"] = scale_pos_weight

    final_model = XGBClassifier(**final_kwargs)
    final_model.fit(X_train_processed, y_train_model)

    y_pred = final_model.predict(X_test_processed)
    y_proba = final_model.predict_proba(X_test_processed)[:, 1]

    combined_model_object = {
        "preprocessor": final_preprocessor,
        "model": final_model,
        "best_params": best_params_local,
        "imbalance_strategy": imbalance_strategy,
    }

    evaluate_and_store(
        model_name=model_name,
        imbalance_strategy=imbalance_strategy,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        best_param_dict={**best_params_local, "cv_f1": best_score},
        model_object=combined_model_object,
    )

    collect_cv_fold_scores_xgboost(
        model_name=model_name,
        imbalance_strategy=imbalance_strategy,
        best_params_local=best_params_local,
        sampler=sampler,
        scale_pos_weight=scale_pos_weight,
    )

# ---------------------------------------------------------------------
# 7) Hyperparameter grids
# ---------------------------------------------------------------------
logreg_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
}

rf_grid = {
    "classifier__n_estimators": [100, 300, 500],
    "classifier__max_depth": [None, 5, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}

# ---------------------------------------------------------------------
# 8) Logistic Regression
# ---------------------------------------------------------------------
run_gridsearch_sklearn(
    model_name="Logistic Regression",
    imbalance_strategy="none",
    estimator=Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    param_grid=logreg_grid,
)

run_gridsearch_sklearn(
    model_name="Logistic Regression",
    imbalance_strategy="class_weight_balanced",
    estimator=Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ]),
    param_grid=logreg_grid,
)

run_gridsearch_sklearn(
    model_name="Logistic Regression",
    imbalance_strategy="random_undersampling",
    estimator=ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("sampler", RandomUnderSampler(random_state=42)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    param_grid=logreg_grid,
)

run_gridsearch_sklearn(
    model_name="Logistic Regression",
    imbalance_strategy="SMOTE",
    estimator=ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("sampler", SMOTE(random_state=42)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    param_grid=logreg_grid,
)

# ---------------------------------------------------------------------
# 9) Random Forest
# ---------------------------------------------------------------------
run_gridsearch_sklearn(
    model_name="Random Forest",
    imbalance_strategy="none",
    estimator=Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ]),
    param_grid=rf_grid,
)

run_gridsearch_sklearn(
    model_name="Random Forest",
    imbalance_strategy="class_weight_balanced",
    estimator=Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")),
    ]),
    param_grid=rf_grid,
)

run_gridsearch_sklearn(
    model_name="Random Forest",
    imbalance_strategy="random_undersampling",
    estimator=ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("sampler", RandomUnderSampler(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ]),
    param_grid=rf_grid,
)

run_gridsearch_sklearn(
    model_name="Random Forest",
    imbalance_strategy="SMOTE",
    estimator=ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("sampler", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ]),
    param_grid=rf_grid,
)

# ---------------------------------------------------------------------
# 10) XGBoost
# ---------------------------------------------------------------------
negative_count = y_train.value_counts()[0]
positive_count = y_train.value_counts()[1]
scale_pos_weight = negative_count / positive_count

run_xgboost_manual_cv(
    model_name="XGBoost",
    imbalance_strategy="none",
    sampler=None,
    scale_pos_weight=None,
)

run_xgboost_manual_cv(
    model_name="XGBoost",
    imbalance_strategy="class_weight_balanced",
    sampler=None,
    scale_pos_weight=scale_pos_weight,
)

run_xgboost_manual_cv(
    model_name="XGBoost",
    imbalance_strategy="random_undersampling",
    sampler=RandomUnderSampler(random_state=42),
    scale_pos_weight=None,
)

run_xgboost_manual_cv(
    model_name="XGBoost",
    imbalance_strategy="SMOTE",
    sampler=SMOTE(random_state=42),
    scale_pos_weight=None,
)

# ---------------------------------------------------------------------
# 11) Save results
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(TABLES_DIR / "model_results_test_all_models_cv.csv", index=False)

best_params_df = pd.DataFrame(best_params)
best_params_df.to_csv(TABLES_DIR / "model_best_params_cv.csv", index=False)

cv_fold_scores_df = pd.DataFrame(cv_fold_scores)
cv_fold_scores_df.to_csv(TABLES_DIR / "model_cv_fold_scores.csv", index=False)

print("\nSaved: model_results_test_all_models_cv.csv")
print("Saved: model_best_params_cv.csv")
print("Saved: model_cv_fold_scores.csv")

print("\nFinal test-set results:")
print(results_df)

print("\nSorted by F1-score:")
print(results_df.sort_values("f1_score", ascending=False))


