"""Generate IEEE double-column friendly, publication-style figures (PNG+PDF).

Uses existing artifacts:
- reports/tuning_test/*_threshold_tuned_test.json
- reports/initial_test/*_test_predictions.npz (ROC/PR)
- reports/friday_ml_tuned_subset.json, reports/snort_friday_evaluation_final.json (Fig 7)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
TUNED_DIR = ROOT / "reports" / "tuning_test"
BASE_DIR = ROOT / "reports" / "initial_test"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["rf", "xgb", "nn"]
NAMES = {"rf": "Random Forest", "xgb": "XGBoost", "nn": "Neural Network (MLP)"}
# Colorblind-friendly (Okabe-Ito inspired)
COLORS = {"rf": "#0072B2", "xgb": "#009E73", "nn": "#D55E00", "snort": "#CC79A7"}
THRESHOLDS = {"rf": 0.35, "xgb": 0.70, "nn": 0.70}

# IEEE double-column: single column ~3.5in, double ~7.16in
SINGLE_COL = (3.5, 2.6)
DOUBLE_COL = (7.16, 3.0)
DOUBLE_TALL = (7.16, 3.6)
SINGLE_TALL = (3.5, 3.8)
DPI = 600


def set_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.dpi": DPI,
        }
    )
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)


def save(fig, stem: str):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")
    plt.close(fig)
    print(f"saved {stem}.png/.pdf")


def load_tuned():
    tuned = {}
    for m in MODELS:
        tuned[m] = json.loads((TUNED_DIR / f"{m}_threshold_tuned_test.json").read_text())
    return tuned


def fig1_overall(tuned):
    metrics = [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]
    x = np.arange(len(metrics))
    w = 0.23
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    for i, m in enumerate(MODELS):
        vals = [tuned[m][k] for k, _ in metrics]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=f"{NAMES[m]}", color=COLORS[m], edgecolor="black", linewidth=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=5.8, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylim(0.86, 1.04)
    ax.set_ylabel("Score")
    ax.legend(frameon=True, loc="lower left", ncol=1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "01_overall_comparison")


def fig1b_fpr(tuned):
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    x = np.arange(len(MODELS))
    vals = [tuned[m]["false_positive_rate"] for m in MODELS]
    bars = ax.bar(x, vals, color=[COLORS[m] for m in MODELS], edgecolor="black", linewidth=0.3, width=0.55)
    for b, v, m in zip(bars, vals, MODELS):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.03, f"{v:.4f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[m].replace(" (MLP)", "") for m in MODELS], rotation=10)
    ax.set_ylabel("False Positive Rate")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "01b_fpr_comparison")


def fig2_confusion(tuned):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
    labels = ["Benign", "Attack"]
    for ax, m in zip(axes, MODELS):
        cm = tuned[m]["confusion_matrix"]
        mat = np.array([[cm["true_negative"], cm["false_positive"]], [cm["false_negative"], cm["true_positive"]]], dtype=float)
        row_sums = mat.sum(axis=1, keepdims=True)
        norm = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)
        annot = np.empty_like(mat, dtype=object)
        for i in range(2):
            for j in range(2):
                annot[i, j] = f"{int(mat[i,j]):,}\n({norm[i,j]*100:.2f}%)"
        sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, ax=ax,
                    xticklabels=labels, yticklabels=labels if m == MODELS[0] else ["", ""],
                    cbar=(m == MODELS[-1]), cbar_kws={"shrink": 0.8, "label": "Row %"} if m == MODELS[-1] else {},
                    linewidths=0.4, linecolor="white", annot_kws={"fontsize": 6})
        ax.set_title(f"{NAMES[m].split()[0] if m!='nn' else 'MLP'} (τ={THRESHOLDS[m]:.2f})", fontsize=7.5)
        ax.set_xlabel("Predicted" if m == MODELS[1] else "")
        ax.set_ylabel("Actual" if m == MODELS[0] else "")
    fig.tight_layout(w_pad=0.4)
    save(fig, "02_confusion_matrices")


def load_predictions():
    probs, y = {}, None
    for m in MODELS:
        d = np.load(BASE_DIR / f"{m}_test_predictions.npz")
        probs[m] = d["attack_probabilities"].astype(np.float64)
        yy = d["true_labels"].astype(np.uint8)
        if y is None:
            y = yy
    return y, probs


def fig3_roc(y, probs):
    fig, ax = plt.subplots(figsize=SINGLE_TALL)
    for m in MODELS:
        fpr, tpr, _ = roc_curve(y, probs[m])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{NAMES[m]} (AUC={roc_auc:.4f})", color=COLORS[m])
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "03_roc_curves")


def fig4_pr(y, probs):
    fig, ax = plt.subplots(figsize=SINGLE_TALL)
    for m in MODELS:
        prec, rec, _ = precision_recall_curve(y, probs[m])
        ap = auc(rec, prec)
        ax.plot(rec, prec, label=f"{NAMES[m]} (AP={ap:.4f})", color=COLORS[m])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.85, 1.005)
    ax.legend(frameon=True, loc="lower left")
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "04_pr_curves")


def fig5_xgb_per_attack(tuned):
    attacks_d = tuned["xgb"]["per_attack_test_recall"]
    items = sorted(attacks_d.items(), key=lambda kv: kv[1]["recall"])
    labels = [k.replace("Web Attack � ", "Web ").replace("Web Attack – ", "Web ").replace("Web Attack \u2013 ", "Web ") for k, _ in items]
    vals = [v["recall"] for _, v in items]
    fig, ax = plt.subplots(figsize=(3.5, 4.2))
    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=COLORS["xgb"], edgecolor="black", linewidth=0.25, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Recall")
    ax.set_xlim(0, 1.12)
    for b, v, (_, d) in zip(bars, vals, items):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f} (n={d.get('support', d.get('count', d.get('total','?')))})", va="center", fontsize=5.5)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "05_xgb_per_attack_recall")


def fig6_per_attack_compare(tuned):
    attacks_all = list(tuned["xgb"]["per_attack_test_recall"].keys())
    # sort by xgb recall for readability
    attacks_all.sort(key=lambda a: tuned["xgb"]["per_attack_test_recall"][a]["recall"])
    labels = [a.replace("Web Attack � ", "Web ").replace("Web Attack – ", "Web ").replace("Web Attack \u2013 ", "Web ") for a in attacks_all]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.16, 3.8))
    for i, m in enumerate(MODELS):
        vals = [tuned[m]["per_attack_test_recall"][a]["recall"] for a in attacks_all]
        ax.bar(x + (i - 1) * w, vals, w, label=NAMES[m].replace(" (MLP)", ""), color=COLORS[m], edgecolor="black", linewidth=0.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.2)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=True, ncol=3, loc="upper left", fontsize=6.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "06_per_attack_comparison")


def fig7_snort_ml():
    ml = json.loads((ROOT / "reports" / "friday_ml_tuned_subset.json").read_text())
    sn = json.loads((ROOT / "reports" / "snort_friday_evaluation_final.json").read_text())
    rows = [("RF\n(τ=0.35)", ml["rf"]), ("XGBoost\n(τ=0.70)", ml["xgb"]), ("MLP\n(τ=0.70)", ml["nn"]), ("Snort 3\n(Community)", sn)]
    labels = ["Precision", "Recall", "F1-score", "FPR"]
    data = []
    for name, m in rows:
        if "metrics" in m:
            met = m["metrics"]
            f1, fpr = met.get("f1_score", met.get("f1")), met.get("false_positive_rate", met.get("fpr"))
        else:
            f1, fpr = m.get("f1"), m.get("fpr")
        data.append([m.get("precision", met.get("precision") if 'met' in locals() else 0), m.get("recall", met.get("recall") if 'met' in locals() else 0), f1, fpr])
    # fix snort/others robustly
    data = []
    for name, m in rows:
        mm = m.get("metrics", m)
        data.append([mm["precision"], mm["recall"], mm.get("f1_score", mm.get("f1")), mm.get("false_positive_rate", mm.get("fpr"))])
    data = np.array(data, float)
    x = np.arange(len(labels))
    w = 0.18
    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    cset = [COLORS["rf"], COLORS["xgb"], COLORS["nn"], COLORS["snort"]]
    names = [r[0].replace("\n", " ") for r in rows]
    for i, (nm, row, c) in enumerate(zip(names, data, cset)):
        ax.bar(x + (i - 1.5) * w, row, w, label=nm, color=c, edgecolor="black", linewidth=0.25)
        for j, v in enumerate(row):
            ax.text(x[j] + (i - 1.5) * w, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=5.2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Score")
    ax.legend(frameon=True, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=6.2)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "07_snort_ml_comparison")


def main():
    set_style()
    tuned = load_tuned()
    fig1_overall(tuned)
    fig1b_fpr(tuned)
    fig2_confusion(tuned)
    y, probs = load_predictions()
    fig3_roc(y, probs)
    fig4_pr(y, probs)
    fig5_xgb_per_attack(tuned)
    fig6_per_attack_compare(tuned)
    fig7_snort_ml()
    print(f"All IEEE-style figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
