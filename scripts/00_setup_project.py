from pathlib import Path

# Folders to create
folders = [
    "data/raw",
    "data/processed",
    "outputs/tables",
    "outputs/figures",
    "outputs/models",
]

# Python scripts to create
files = [
    "01_prepare_data.py",
    "02_descriptive_eda.py",
    "03_model_training.py",
    "04_evaluate_models.py",
    "05_subgroup_error_analysis.py",
    "06_shap_explainability.py",
    "07_robustness_checks.py",
    "08_format_tables"
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

for file in files:
    Path(file).touch(exist_ok=True)

print("Project structure created.")