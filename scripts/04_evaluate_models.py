from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

TABLES_DIR = Path("outputs/tables")

RESULTS_PATH = TABLES_DIR / "model_results_test_all_models_cv.csv"
CV_SCORES_PATH = TABLES_DIR / "model_cv_fold_scores.csv"

results = pd.read_csv(RESULTS_PATH)

print(results)
print("\nSorted by F1-score:")
print(results.sort_values("f1_score", ascending=False))

print("\nSorted by recall:")
print(results.sort_values("recall", ascending=False))

print("\nSorted by PR-AUC:")
print(results.sort_values("pr_auc", ascending=False))

results.sort_values("f1_score", ascending=False).to_csv(
    TABLES_DIR / "model_results_sorted_by_f1.csv",
    index=False
)

results.sort_values("recall", ascending=False).to_csv(
    TABLES_DIR / "model_results_sorted_by_recall.csv",
    index=False
)

results.sort_values("pr_auc", ascending=False).to_csv(
    TABLES_DIR / "model_results_sorted_by_pr_auc.csv",
    index=False
)

print("\nSaved sorted model result tables.")

summary = results.copy()

metric_cols = ["recall", "precision", "f1_score", "pr_auc", "roc_auc"]
summary[metric_cols] = summary[metric_cols].round(3)

summary = summary.sort_values("f1_score", ascending=False)

summary.to_csv(
    TABLES_DIR / "model_results_summary_for_thesis.csv",
    index=False
)

print("\nSaved: model_results_summary_for_thesis.csv")
print(summary)

TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

results = pd.read_csv(TABLES_DIR / "model_results_summary_for_thesis.csv")

results["model_strategy"] = results["model"] + " | " + results["imbalance_strategy"]

results = results.sort_values("f1_score", ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(results["model_strategy"], results["f1_score"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("F1-score")
plt.xlabel("Model and imbalance strategy")
plt.title("Comparison of F1-scores across models and imbalance strategies")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "f1_comparison_models.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: f1_comparison_models.png")

recall_results = results.copy()
recall_results["model_strategy"] = (
    recall_results["model"] + " | " + recall_results["imbalance_strategy"]
)
recall_results = recall_results.sort_values("recall", ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(recall_results["model_strategy"], recall_results["recall"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Recall")
plt.xlabel("Model and imbalance strategy")
plt.title("Comparison of recall across models and imbalance strategies")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "recall_comparison_models.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: recall_comparison_models.png")

pr_results = results.copy()
pr_results["model_strategy"] = (
    pr_results["model"] + " | " + pr_results["imbalance_strategy"]
)
pr_results = pr_results.sort_values("pr_auc", ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(pr_results["model_strategy"], pr_results["pr_auc"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("PR-AUC")
plt.xlabel("Model and imbalance strategy")
plt.title("Comparison of PR-AUC across models and imbalance strategies")
plt.tight_layout()

plt.savefig(FIGURES_DIR / "pr_auc_comparison_models.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: pr_auc_comparison_models.png")


CV_SCORES_PATH = TABLES_DIR / "model_cv_fold_scores.csv"

cv_scores = pd.read_csv(CV_SCORES_PATH)

cv_scores["model_strategy"] = (
    cv_scores["model"] + " | " + cv_scores["imbalance_strategy"]
)

def make_cv_boxplot(metric, title, filename, xlabel):
    model_order = (
        cv_scores
        .groupby("model_strategy")[metric]
        .mean()
        .sort_values(ascending=True)
        .index
    )

    data = [
        cv_scores.loc[
            cv_scores["model_strategy"] == model_strategy,
            metric
        ].values
        for model_strategy in model_order
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    bp = ax.boxplot(
        data,
        vert=False,
        tick_labels=model_order,
        patch_artist=True,
        widths=0.6,
        showmeans=False,
        showfliers=False
    )

    for box in bp["boxes"]:
        box.set(facecolor="#D9D9D9", edgecolor="black", linewidth=1.0)

    for whisker in bp["whiskers"]:
        whisker.set(color="black", linewidth=1.0)

    for cap in bp["caps"]:
        cap.set(color="black", linewidth=1.0)

    for median in bp["medians"]:
        median.set(color="black", linewidth=1.2)

    for i, scores in enumerate(data, start=1):
        if len(scores) > 1:
            y_jitter = np.linspace(-0.10, 0.10, len(scores))
        else:
            y_jitter = [0]

        y_positions = i + y_jitter

        ax.scatter(
            scores,
            y_positions,
            s=18,
            marker="o",
            color="black",
            zorder=3
        )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Model and imbalance strategy", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.yaxis.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")


make_cv_boxplot(
    metric="f1_score",
    title="F1-score across models and imbalance strategies",
    filename="cv_boxplot_f1_score.png",
    xlabel="F1-score"
)

make_cv_boxplot(
    metric="recall",
    title="Recall across models and imbalance strategies",
    filename="cv_boxplot_recall.png",
    xlabel="Recall"
)

make_cv_boxplot(
    metric="pr_auc",
    title="PR-AUC across models and imbalance strategies",
    filename="cv_boxplot_pr_auc.png",
    xlabel="PR-AUC"
)

tradeoff_df = pd.read_csv(TABLES_DIR / "model_results_summary_for_thesis.csv")

model_abbrev = {
    "Logistic Regression": "LR",
    "Random Forest": "RF",
    "XGBoost": "XGB"
}

strategy_abbrev = {
    "none": "None",
    "class_weight_balanced": "CW",
    "random_undersampling": "Under",
    "SMOTE": "SMOTE"
}

model_markers = {
    "Logistic Regression": "o",
    "Random Forest": "s",
    "XGBoost": "^"
}

strategy_colors = {
    "none": "#4d4d4d",
    "class_weight_balanced": "#377eb8",
    "random_undersampling": "#4daf4a",
    "SMOTE": "#e41a1c"
}
fig, ax = plt.subplots(figsize=(8.5, 6))

fig.patch.set_facecolor("white")
ax.set_facecolor("#E5E5E5")
ax.grid(True, color="white", linewidth=1.2)
ax.set_axisbelow(True)

for _, row in tradeoff_df.iterrows():
    ax.scatter(
        row["precision"],
        row["recall"],
        marker=model_markers[row["model"]],
        s=55,
        c=strategy_colors[row["imbalance_strategy"]],
        edgecolor="#333333",
        linewidth=0.8,
        alpha=0.95
    )

ax.set_title("Precision–recall trade-off", fontsize=14, fontweight="bold")
ax.set_xlabel("Precision", fontsize=11)
ax.set_ylabel("Recall", fontsize=11)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#666666")
ax.spines["bottom"].set_color("#666666")

legend_handles = [
    Line2D([], [], linestyle="none", label="Model"),
    Line2D([], [], marker="o", color="black", markerfacecolor="white", linestyle="none", label="  LR"),
    Line2D([], [], marker="s", color="black", markerfacecolor="white", linestyle="none", label="  RF"),
    Line2D([], [], marker="^", color="black", markerfacecolor="white", linestyle="none", label="  XGB"),

    Line2D([], [], linestyle="none", label=""),

    Line2D([], [], linestyle="none", label="Strategy"),
    Line2D([], [], marker="o", color="gray", linestyle="none", label="  None"),
    Line2D([], [], marker="o", color="#1f77b4", linestyle="none", label="  CW"),
    Line2D([], [], marker="o", color="#2ca02c", linestyle="none", label="  Under"),
    Line2D([], [], marker="o", color="#d62728", linestyle="none", label="  SMOTE"),
]

leg = ax.legend(
    handles=legend_handles,
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="0.7",
    fontsize=9,
    handlelength=1.2,
    handletextpad=0.8,
    borderpad=0.8,
    labelspacing=0.45
)

for txt in leg.get_texts():
    if txt.get_text() in ["Model", "Strategy"]:
        txt.set_fontweight("bold")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "precision_recall_tradeoff.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: precision_recall_tradeoff.png")


best_model_row = tradeoff_df.sort_values("f1_score", ascending=False).iloc[0]

tn = int(best_model_row["tn"])
fp = int(best_model_row["fp"])
fn = int(best_model_row["fn"])
tp = int(best_model_row["tp"])

cm = np.array([
    [tn, fp],
    [fn, tp]
])

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm, cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Predicted no risk", "Predicted risk"], fontsize=11)
ax.set_yticklabels(["Actual no risk", "Actual risk"], fontsize=11)

ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)

ax.set_title(
    "Confusion matrix — best-performing model",
    fontsize=13,
    fontweight="bold",
    pad=14
)

threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        value = cm[i, j]
        text_color = "white" if value > threshold else "black"

        ax.text(
            j,
            i,
            f"{value:,}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=12
        )

cbar = plt.colorbar(
    im,
    ax=ax,
    fraction=0.046,   
    pad=0.04,         
    shrink=0.80       
)

cbar.set_label("Number of students", fontsize=10)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "best_model_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: best_model_confusion_matrix.png")
