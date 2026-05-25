**Predicting Suicide-Related Risk Among Adolescents Using Machine Learning**

This repository contains the code used for my Master Thesis in Data Science & Society at Tilburg University.

Project description:

This project uses the 2023 Youth Risk Behavior Survey (YRBSS) middle school state dataset to predict suicide-related risk among adolescents. The target variable indicates whether a student reported at least one suicide-related risk indicator: seriously considering suicide, making a suicide plan, or attempting suicide.

The analysis compares Logistic Regression, Random Forest, and XGBoost models using different class imbalance strategies. The final model is interpreted using SHAP and evaluated through subgroup error analysis and robustness checks.

The raw YRBSS data are not included in this repository. The data can be obtained from the official CDC YRBSS data portal. After downloading the raw data, the scripts can be run sequentially to reproduce the processed datasets, model results, figures, and tables.

Repository structure:

```text
scripts/
    01_data_preparation.py
    02_descriptive_analysis.py
    03_model_training.py
    04_model_results_visualisation.py
    05_subgroup_error_analysis.py
    06_shap_interpretation.py
    07_robustness_checks.py
    08_format_tables.py

data/
    raw/          # not included in repository
    processed/    # not included in repository

outputs/
    tables/
    figures/
    models/       # not included in repository
