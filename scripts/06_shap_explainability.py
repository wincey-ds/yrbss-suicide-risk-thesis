import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ---------------------------------------------------------------------
# 1) Paths
# ---------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")
MODELS_DIR = Path("outputs/models")

DATA_PATH = PROCESSED_DIR / "yrbss_2023_model_dataset_suicide_risk_main_no_qn45_no_sitecode.csv"
FEATURE_INFO_PATH = PROCESSED_DIR / "yrbss_2023_feature_info_suicide_risk_main_no_qn45_no_sitecode.csv"
MODEL_PATH = MODELS_DIR / "XGBoost_random_undersampling_cv.pkl"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 2) Load data
# ---------------------------------------------------------------------

feature_info = pd.read_csv(FEATURE_INFO_PATH)
df = pd.read_csv(DATA_PATH)

record_id = df["record_id"]
y = df["target"]
X = df.drop(columns=["record_id", "target"])

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())
print(y.value_counts(normalize=True))

print("\nChosen model for SHAP:")
print("XGBoost | random_undersampling")

# ---------------------------------------------------------------------
# 3) Train-test split 
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------
# 4) Load best model
# ---------------------------------------------------------------------

def convert_to_string(X):
    return X.astype(str)

best_model_object = joblib.load(MODEL_PATH)

fitted_preprocessor = best_model_object["preprocessor"]
fitted_classifier = best_model_object["model"]

X_test_processed = fitted_preprocessor.transform(X_test)

print("\nBest model loaded for SHAP:")
print("Imbalance strategy:", best_model_object["imbalance_strategy"])
print("Best parameters:", best_model_object["best_params"])

print("\nProcessed test shape:", X_test_processed.shape)
print("Classifier:", fitted_classifier)

feature_names = fitted_preprocessor.get_feature_names_out()

print("\nNumber of transformed feature names:", len(feature_names))
print(feature_names[:20])

# ---------------------------------------------------------------------
# 5) Compute SHAP values
# ---------------------------------------------------------------------

explainer = shap.TreeExplainer(fitted_classifier)

shap_values = explainer.shap_values(X_test_processed)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

print("\nSHAP values computed.")
print("SHAP values shape:", shap_values.shape)

# ---------------------------------------------------------------------
# 6) Map transformed features back to original features
# ---------------------------------------------------------------------

def map_to_original_feature(processed_feature_name):
    name = processed_feature_name

    if name.startswith("categorical__"):
        clean = name.replace("categorical__", "")

        if clean.startswith("sex_"):
            return "sex"
        if clean.startswith("age_"):
            return "age"
        if clean.startswith("grade_"):
            return "grade"
        if clean.startswith("race4_"):
            return "race4"

        return clean

    if name.startswith("numeric__imputer__"):
        return name.replace("numeric__imputer__", "")

    if name.startswith("numeric__missing_indicator__missingindicator_"):
        original = name.replace("numeric__missing_indicator__missingindicator_", "")
        return f"{original}_missing"

    return name


original_feature_names = [map_to_original_feature(f) for f in feature_names]

shap_grouped = pd.DataFrame(
    shap_values,
    columns=original_feature_names
)

shap_grouped = shap_grouped.T.groupby(level=0).sum().T

grouped_mean_abs_shap = np.abs(shap_grouped).mean(axis=0)

shap_importance = pd.DataFrame({
    "feature": grouped_mean_abs_shap.index,
    "mean_abs_shap": grouped_mean_abs_shap.values
}).sort_values("mean_abs_shap", ascending=False)

shap_importance = shap_importance.merge(
    feature_info[["feature", "label"]],
    on="feature",
    how="left"
)

manual_labels = {
    "sex": "Sex",
    "age": "Age",
    "grade": "Grade",
    "race4": "Race/ethnicity"
}

shap_importance["label"] = shap_importance["label"].fillna(
    shap_importance["feature"].map(manual_labels)
)

shap_importance.to_csv(
    TABLES_DIR / "shap_feature_importance_grouped_with_labels.csv",
    index=False
)

print("\nSaved: shap_feature_importance_grouped_with_labels.csv")
print(shap_importance.head(20))

# ---------------------------------------------------------------------
# 7) Readable labels
# ---------------------------------------------------------------------

label_shortcuts = {
    "Were ever electronically bullied": "Electronically bullied",
    "Were ever bullied on school property": "Bullied at school",
    "Felt that they were ever treated badly or unfairly in school because of their race or ethnicity": "Unfair treatment at school",
    "Ate breakfast on all 7 days": "Ate breakfast all 7 days",
    "Ever rode with a driver who had been drinking alcohol": "Rode with drinking driver",
    "Were physically active at least 60 minutes per day on 5 or more days": "Physically active 5+ days",
    "Get 8 or more hours of sleep": "8+ hours of sleep",
    "Were ever in a physical fight": "Physical fight",
    "Ever used an electronic vapor product": "Ever used e-vapor product",
    "Ever saw someone get physically attacked, beaten, stabbed, or shot in their neighborhood": "Saw neighborhood violence",
    "Ever took prescription pain medicine without a doctor's prescription or differently than how a doctor told them to use it": "Prescription pain medicine misuse",
    "Played on at least one sports team": "Sports team participation",
    "Drank alcohol for the first time before age 11 years": "Drank alcohol before age 11",
    "Did not eat breakfast": "Did not eat breakfast",
    "Currently smoked cigarettes or cigars": "Current cigarette/cigar use",
    "Currently smoked cigarettes or used electronic vapor products": "Current cigarette/e-vapor use",
    "Rarely or never wore a bicycle helmet": "Rarely/never bicycle helmet"
}


def readable_feature_label(feature):
    label_lookup = dict(zip(feature_info["feature"], feature_info["label"]))

    label = label_lookup.get(feature, feature)

    if pd.isna(label):
        label = feature

    label = label_shortcuts.get(label, label)

    if isinstance(label, str) and "treated badly or unfairly" in label.lower():
        label = "Unfair treatment at school"

    if isinstance(label, str) and "prescription pain medicine" in label.lower():
        label = "Prescription pain medicine misuse"

    if isinstance(label, str) and "physically attacked" in label.lower():
        label = "Saw neighborhood violence"

    if feature == "sex":
        label = "Sex"
    if feature == "age":
        label = "Age"
    if feature == "grade":
        label = "Grade"
    if feature == "race4":
        label = "Race/ethnicity"

    return label

# ---------------------------------------------------------------------
# 8) Global SHAP top-15 feature importance barplot
# ---------------------------------------------------------------------

top15 = shap_importance.head(15).copy()

top15["plot_label"] = top15["feature"].apply(readable_feature_label)

top15 = top15.sort_values("mean_abs_shap", ascending=True)

fig, ax = plt.subplots(figsize=(8.5, 6.5))

ax.barh(
    top15["plot_label"],
    top15["mean_abs_shap"],
    color="#2C7FB8",
    edgecolor="white"
)

ax.set_title("Top 15 SHAP feature importances", fontsize=12, fontweight="bold")
ax.set_xlabel("Mean absolute SHAP value", fontsize=10)
ax.set_ylabel("")

ax.tick_params(axis="y", labelsize=9)
ax.tick_params(axis="x", labelsize=9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(left=0.34)

plt.savefig(
    FIGURES_DIR / "shap_top15_feature_importance.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print("\nSaved: shap_top15_feature_importance.png")

# ---------------------------------------------------------------------
# 9) Global grouped SHAP summary plot
# ---------------------------------------------------------------------

top_features = (
    shap_importance
    .head(15)["feature"]
    .tolist()
)

top_features_for_summary = [
    feature for feature in top_features
    if feature in X_test.columns
]

shap_grouped_top = shap_grouped[top_features_for_summary].copy()

X_original_for_summary = X_test[top_features_for_summary].copy()

for col in X_original_for_summary.columns:
    X_original_for_summary[col] = pd.to_numeric(
        X_original_for_summary[col],
        errors="coerce"
    )

display_names = [
    readable_feature_label(feature)
    for feature in top_features_for_summary
]

shap_grouped_top.columns = display_names
X_original_for_summary.columns = display_names

plt.figure(figsize=(10, 7), facecolor="white")

shap.summary_plot(
    shap_grouped_top.values,
    features=X_original_for_summary,
    feature_names=display_names,
    max_display=15,
    plot_type="dot",
    show=False,
    color_bar=True
)

fig = plt.gcf()
ax = plt.gca()

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.set_title(
    "Grouped SHAP summary plot",
    fontsize=14,
    fontweight="bold",
    pad=14
)

ax.set_xlabel(
    "SHAP value (impact on model output)",
    fontsize=11
)

ax.tick_params(axis="both", labelsize=9)

ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.25)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "shap_summary_top15_grouped_dot.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print("\nSaved: shap_summary_top15_grouped_dot.png")

# ---------------------------------------------------------------------
# 10) SHAP beeswarm plots by subgroup variable
# ---------------------------------------------------------------------

MIN_SUBGROUP_N = 100
MAX_DISPLAY_BEESWARM = 10

DEMOGRAPHIC_FEATURES_TO_EXCLUDE = ["sex", "age", "grade", "race4"]

subgroup_definitions = {
    "sex": {
        1.0: "Female students",
        2.0: "Male students"
    },
    "race4": {
        1.0: "White students",
        2.0: "Black students",
        3.0: "Hispanic/Latino students",
        4.0: "All other race/ethnicity students"
    },
    "age": {
        1.0: "10 years old or younger",
        2.0: "11 years old",
        3.0: "12 years old",
        4.0: "13 years old",
        5.0: "14 years old",
        6.0: "15 years old",
        7.0: "16 years old or older"
    },
    "grade": {
        1.0: "6th grade",
        2.0: "7th grade",
        3.0: "8th grade",
        4.0: "Ungraded/other"
    }
}

subgroup_title_map = {
    "sex": "sex",
    "race4": "race/ethnicity",
    "age": "age",
    "grade": "grade"
}


def safe_subgroup_filename(text):
    return (
        str(text)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )


def filter_substantive_features(feature_list):
    return [
        feature for feature in feature_list
        if feature not in DEMOGRAPHIC_FEATURES_TO_EXCLUDE
        and not feature.endswith("_missing")
    ]


def get_grouped_shap_for_mask(mask):
    shap_sub = shap_values[mask.values]

    shap_sub_grouped = pd.DataFrame(
        shap_sub,
        columns=original_feature_names
    )

    shap_sub_grouped = shap_sub_grouped.T.groupby(level=0).sum().T

    return shap_sub_grouped


def make_subgroup_feature_table(subgroup_variable, subgroup_value, subgroup_label):
    subgroup_mask = X_test[subgroup_variable] == subgroup_value
    subgroup_n = int(subgroup_mask.sum())

    if subgroup_n < MIN_SUBGROUP_N:
        print(
            f"Skipping {subgroup_variable} = {subgroup_label}: "
            f"n = {subgroup_n}, below minimum n = {MIN_SUBGROUP_N}"
        )
        return None

    shap_sub_grouped = get_grouped_shap_for_mask(subgroup_mask)

    usable_features = filter_substantive_features(shap_sub_grouped.columns)
    shap_sub_grouped = shap_sub_grouped[usable_features]

    X_sub_original = X_test.loc[subgroup_mask, usable_features].copy()

    for col in X_sub_original.columns:
        X_sub_original[col] = pd.to_numeric(
            X_sub_original[col],
            errors="coerce"
        )

    mean_abs_shap = np.abs(shap_sub_grouped).mean(axis=0)

    importance_df = pd.DataFrame({
        "subgroup_variable": subgroup_variable,
        "subgroup_value": subgroup_value,
        "subgroup_label": subgroup_label,
        "n": subgroup_n,
        "feature": mean_abs_shap.index,
        "mean_abs_shap": mean_abs_shap.values
    }).sort_values("mean_abs_shap", ascending=False)

    importance_df["label"] = importance_df["feature"].apply(readable_feature_label)

    return {
        "subgroup_label": subgroup_label,
        "n": subgroup_n,
        "importance_df": importance_df,
        "shap_grouped": shap_sub_grouped,
        "X_original": X_sub_original
    }


# ---------------------------------------------------------------------
# Main beeswarm loop
# ---------------------------------------------------------------------

for subgroup_variable, subgroup_map in subgroup_definitions.items():

    print(f"\nCreating appendix SHAP beeswarm figure for: {subgroup_variable}")

    subgroup_outputs = []

    for subgroup_value, subgroup_label in subgroup_map.items():

        subgroup_result = make_subgroup_feature_table(
            subgroup_variable=subgroup_variable,
            subgroup_value=subgroup_value,
            subgroup_label=subgroup_label
        )

        if subgroup_result is not None:
            subgroup_outputs.append(subgroup_result)

            output_table_name = (
                f"shap_subgroup_importance_"
                f"{subgroup_variable}_"
                f"{safe_subgroup_filename(subgroup_label)}_"
                f"appendix.csv"
            )

            subgroup_result["importance_df"].to_csv(
                TABLES_DIR / output_table_name,
                index=False
            )

            print(f"Saved: {output_table_name}")

    if len(subgroup_outputs) == 0:
        print(f"No valid categories for {subgroup_variable}; skipping figure.")
        continue

    # --------------------------------------------------
    # Shared top-10 features per subgroup variable
    # --------------------------------------------------

    combined_importance = pd.concat(
        [item["importance_df"] for item in subgroup_outputs],
        ignore_index=True
    )

    shared_top_features = (
        combined_importance
        .groupby("feature")["mean_abs_shap"]
        .mean()
        .sort_values(ascending=False)
        .head(MAX_DISPLAY_BEESWARM)
        .index
        .tolist()
    )

    plot_features = shared_top_features[::-1]
    plot_labels = [
        readable_feature_label(feature)
        for feature in plot_features
    ]

    n_panels = len(subgroup_outputs)

    if n_panels == 2:
        nrows, ncols = 1, 2
        fig_width, fig_height = 12.0, 6.5

    elif n_panels == 3:
        nrows, ncols = 2, 2
        fig_width, fig_height = 13.0, 13.0

    elif n_panels == 4:
        nrows, ncols = 2, 2
        fig_width, fig_height =13.0, 13.0

    else:
        nrows, ncols = n_panels, 1
        fig_width, fig_height = 8.0, max(5.5, 4.5 * n_panels)

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")


    gs = GridSpec(
        nrows=nrows,
        ncols=ncols,
        figure=fig,
        width_ratios=[1.0] * ncols,
        wspace=0.70,
        hspace=0.35
    )

    # fig.subplots_adjust(
    #     top=0.86,
    #     bottom=0.07,
    #     left=0.08,
    #     right=0.88
    # )

    if n_panels == 2:
        fig.suptitle(
            f"Grouped SHAP summary plots by {subgroup_title_map[subgroup_variable]}",
            fontsize=14,
            fontweight="bold",
            y=0.97
        )

        fig.subplots_adjust(
            top=0.78,
            bottom=0.12,
            left=0.08,
            right=0.88
        )

    elif n_panels == 3:
        fig.suptitle(
            f"Grouped SHAP summary plots by {subgroup_title_map[subgroup_variable]}",
            fontsize=14,
            fontweight="bold",
            y=0.97
        )

        fig.subplots_adjust(
            top=0.88,
            bottom=0.08,
            left=0.08,
            right=0.88
        )

    elif n_panels == 4:
        fig.suptitle(
            f"Grouped SHAP summary plots by {subgroup_title_map[subgroup_variable]}",
            fontsize=14,
            fontweight="bold",
            y=0.97
        )

        fig.subplots_adjust(
            top=0.90,
            bottom=0.07,
            left=0.08,
            right=0.88
        )

    axes = []

    for i in range(n_panels):
        row = i // ncols
        col = i % ncols
        axes.append(fig.add_subplot(gs[row, col]))

    cax = fig.add_axes([0.91, 0.16, 0.018, 0.68])

    all_x_values = []

    for item in subgroup_outputs:
        shap_values_for_range = item["shap_grouped"][shared_top_features].values.flatten()
        all_x_values.extend(shap_values_for_range)

    all_x_values = np.array(all_x_values)
    all_x_values = all_x_values[~np.isnan(all_x_values)]

    if len(all_x_values) > 0:
        x_abs_max = np.nanpercentile(np.abs(all_x_values), 98)
        x_abs_max = max(x_abs_max, 0.30)

        x_abs_max = np.ceil(x_abs_max / 0.10) * 0.10

        x_abs_max = min(x_abs_max, 0.45)

        x_lim = (-x_abs_max, x_abs_max)

        if x_abs_max <= 0.40:
            x_tick_step = 0.20
        else:
            x_tick_step = 0.30
    else:
        x_lim = None
        x_tick_step = 0.20

    norm = Normalize(vmin=0, vmax=1)
    cmap = "coolwarm"

    for ax_idx, item in enumerate(subgroup_outputs):

        ax = axes[ax_idx]

        subgroup_label = item["subgroup_label"]
        subgroup_n = item["n"]

        shap_sub = item["shap_grouped"][shared_top_features].copy()
        X_sub = item["X_original"][shared_top_features].copy()

        for feature_idx, feature in enumerate(plot_features):

            x_values = shap_sub[feature].values
            color_values = X_sub[feature].values

            rng = np.random.default_rng(42 + feature_idx)
            jitter = rng.normal(loc=0, scale=0.055, size=len(x_values))
            y_values = np.full(len(x_values), feature_idx) + jitter

            non_missing_mask = ~pd.isna(color_values)

            ax.scatter(
                x_values[non_missing_mask],
                y_values[non_missing_mask],
                c=color_values[non_missing_mask],
                cmap=cmap,
                norm=norm,
                s=7,
                alpha=0.70,
                linewidths=0
            )

            if (~non_missing_mask).sum() > 0:
                ax.scatter(
                    x_values[~non_missing_mask],
                    y_values[~non_missing_mask],
                    color="lightgrey",
                    s=7,
                    alpha=0.70,
                    linewidths=0
                )

        ax.axvline(0, color="grey", linewidth=1)

        ax.set_yticks(np.arange(len(plot_features)))
        ax.set_yticklabels(plot_labels, fontsize=8)

        ax.set_title(
            f"{subgroup_label}\n(n = {subgroup_n})",
            fontsize=11,
            fontweight="bold"
        )

        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.35)
        ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.25)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(axis="x", labelsize=8)

        if x_lim is not None:
            ax.set_xlim(x_lim)
            ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Feature value", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    fig.supxlabel(
        "SHAP value (impact on model output)",
        fontsize=10,
        y=0.02
)

    fig.suptitle(
        f"Grouped SHAP summary plots by {subgroup_title_map[subgroup_variable]}",
        fontsize=14,
        fontweight="bold",
        y=0.995
    )

    output_figure_name = (
        f"shap_appendix_beeswarm_"
        f"{subgroup_variable}_"
        f"top10_no_demographics_no_missing.png"
    )

    plt.savefig(
        FIGURES_DIR / output_figure_name,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"Saved: {output_figure_name}")