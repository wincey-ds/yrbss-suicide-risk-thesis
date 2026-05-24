import pandas as pd
import numpy as np
import matplotlib
import sklearn
import imblearn
import xgboost
import shap
import joblib
import sys
import ast
import re

df = pd.read_csv("outputs/tables/model_cv_fold_scores.csv")

df["imbalance_strategy"] = df["imbalance_strategy"].replace({
    "none": "None",
    "class_weight_balanced": "Class weighting",
    "random_undersampling": "Random undersampling",
    "SMOTE": "SMOTE"
})

df["f1_score"] = df["f1_score"].round(3)
df["recall"] = df["recall"].round(3)
df["pr_auc"] = df["pr_auc"].round(3)

df.to_csv("outputs/tables/model_cv_fold_scores_clean.csv", index=False)

df = pd.read_csv("outputs/tables/model_best_params_cv.csv")

df["imbalance_strategy"] = df["imbalance_strategy"].replace({
    "none": "None",
    "class_weight_balanced": "Class weighting",
    "random_undersampling": "Random undersampling",
    "SMOTE": "SMOTE"
})

def format_params(param_string):
    s = str(param_string)

    s = s.replace("classifier__", "")
    s = re.sub(r"np\.float64\(([^)]*)\)", r"\1", s)
    s = re.sub(r"np\.int64\(([^)]*)\)", r"\1", s)

    try:
        params = ast.literal_eval(s)
        if isinstance(params, dict):
            clean_params = []
            for key, value in params.items():
                key = key.replace("classifier__", "")
                if key == "cv_f1":
                    continue
                clean_params.append(f"{key} = {value}")
            return "; ".join(clean_params)
    except Exception:
        pass

    s = s.replace("{", "").replace("}", "")
    s = s.replace("'", "")
    s = s.replace(": ", " = ")
    s = s.replace(",", ";")

    parts = [part.strip() for part in s.split(";")]
    parts = [part for part in parts if not part.startswith("cv_f1")]
    return "; ".join(parts)

df["best_params"] = df["best_params"].apply(format_params)

df.to_csv("outputs/tables/model_best_params_cv_clean.csv", index=False)


df = pd.read_csv("outputs/tables/subgroup_error_metrics_n100.csv")

df["subgroup_variable"] = df["subgroup_variable"].replace({
    "sex": "Sex",
    "grade": "Grade",
    "race4": "Race/ethnicity",
    "age": "Age"
})

def map_group(row):
    variable = row["subgroup_variable"]
    group = row["group"]

    if variable == "Sex":
        return {
            1.0: "Female",
            2.0: "Male"
        }.get(group, group)

    if variable == "Grade":
        return {
            1.0: "6th grade",
            2.0: "7th grade",
            3.0: "8th grade",
            4.0: "Ungraded/other"
        }.get(group, group)

    if variable == "Race/ethnicity":
        return {
            1.0: "White",
            2.0: "Black or African American",
            3.0: "Hispanic/Latino",
            4.0: "All Other Races"
        }.get(group, group)

    if variable == "Age":
        return {
            1.0: r"$\leq$10 years",
            2.0: "11 years",
            3.0: "12 years",
            4.0: "13 years",
            5.0: "14 years",
            6.0: "15 years",
            7.0: r"$\geq$16 years"
        }.get(group, group)

    return group

df["group"] = df.apply(map_group, axis=1)

df["recall"] = df["recall"].round(3)
df["precision"] = df["precision"].round(3)
df["f1_score"] = df["f1_score"].round(3)

df = df[["subgroup_variable", "group", "n", "recall", "precision", "f1_score"]]

df.to_csv("outputs/tables/subgroup_error_metrics_n100_clean.csv", index=False)

library_versions = pd.DataFrame({
    "Library": [
        "Python",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "imbalanced-learn",
        "xgboost",
        "shap",
        "joblib"
    ],
    "Version": [
        sys.version.split()[0],
        pd.__version__,
        np.__version__,
        matplotlib.__version__,
        sklearn.__version__,
        imblearn.__version__,
        xgboost.__version__,
        shap.__version__,
        joblib.__version__
    ]
})

library_versions.to_csv(
    "outputs/tables/python_library_versions.csv",
    index=False
)

print(library_versions)