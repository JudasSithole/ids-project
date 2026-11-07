import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "reports" / "initial_test"
FIG_DIR = ROOT / "reports" / "figures_baseline"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load results
base = {}
for name in ['rf', 'xgb', 'nn']:
    base[name] = json.loads((BASE_DIR / f"{name}_test.json").read_text())

models = ['rf', 'xgb', 'nn']
names_map = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'nn': 'Neural Network'}
colors = {'rf': '#1f77b4', 'xgb': '#2ca02c', 'nn': '#d62728'}

# 1) Overall comparison bar chart
metrics = ['f1', 'precision', 'recall']
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.25
for i, m in enumerate(models):
    vals = [base[m][metric] for metric in metrics]
    ax.bar(x + i*width, vals, width, label=f"{names_map[m]} (t=0.5)", color=colors[m])
ax.set_ylabel('Score')
ax.set_title('Overall Performance Comparison (Baseline - t=0.5)')
ax.set_xticks(x + width)
ax.set_xticklabels([m.upper() for m in metrics])
ax.legend()
ax.set_ylim(0.8, 1.0)
plt.tight_layout()
plt.savefig(FIG_DIR / '01_overall_comparison_baseline.png', dpi=300)
plt.savefig(FIG_DIR / '01_overall_comparison_baseline.pdf')
plt.close()

# FPR separate
fig, ax = plt.subplots(figsize=(8, 6))
for i, m in enumerate(models):
    ax.bar(i, base[m]['false_positive_rate'], color=colors[m], label=names_map[m])
ax.set_ylabel('False Positive Rate')
ax.set_title('False Positive Rate Comparison (Baseline - t=0.5)')
ax.set_xticks(range(len(models)))
ax.set_xticklabels([names_map[m] for m in models])
plt.tight_layout()
plt.savefig(FIG_DIR / '01b_fpr_comparison_baseline.png', dpi=300)
plt.savefig(FIG_DIR / '01b_fpr_comparison_baseline.pdf')
plt.close()

# 2) Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, m in enumerate(models):
    cm = base[m]['confusion_matrix']
    matrix = np.array([[cm['true_negative'], cm['false_positive']],
                       [cm['false_negative'], cm['true_positive']]])
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Predicted Benign', 'Predicted Attack'],
                yticklabels=['Actual Benign', 'Actual Attack'])
    axes[i].set_title(f"{names_map[m]} (t=0.5)")
plt.suptitle('Confusion Matrices - Baseline Models on Test Set', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / '02_confusion_matrices_baseline.png', dpi=300)
plt.savefig(FIG_DIR / '02_confusion_matrices_baseline.pdf')
plt.close()

# 3) ROC and PR curves
print("Loading data for ROC/PR curves...", flush=True)
import gc
import joblib
import xgboost as xgb
from threadpoolctl import threadpool_limits
import pandas as pd

ROOT2 = ROOT
DATA_DIR = ROOT2 / "data" / "processed"
REPORT_DIR = ROOT2 / "reports"
MODEL_DIR = ROOT2 / "models"

dataset_path = DATA_DIR / "cleaned_flows.pkl"
split_report = json.loads((REPORT_DIR / "split_report.json").read_text())
features = json.loads((REPORT_DIR / "cleaning_report.json").read_text())["features"]
data = pd.read_pickle(dataset_path)
with np.load(DATA_DIR / "split_indices.npz") as splits:
    test_idx = splits["test"]
X_test = data[features].to_numpy(dtype=np.float32)
y_test = data["target"].to_numpy(dtype=np.uint8)
del data
gc.collect()

BATCH_SIZE = 4096
def predict_in_batches(estimator, values, scaler=None):
    probs = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(values))
        batch = values[start:end]
        if scaler is not None:
            batch = scaler.transform(batch, copy=True)
        probs[start:end] = estimator.predict_proba(batch)[:, 1]
    return probs

with threadpool_limits(limits=2):
    rf_bundle = joblib.load(MODEL_DIR / "random_forest_baseline.joblib")
    rf_model = rf_bundle["model"]; rf_model.set_params(n_jobs=2)
    xgb_model = xgb.XGBClassifier(); xgb_model.load_model(MODEL_DIR / "xgboost_baseline.ubj"); xgb_model.set_params(n_jobs=2, device="cpu")
    nn_bundle = joblib.load(MODEL_DIR / "neural_network_baseline.joblib")
    nn_model = nn_bundle["model"]; nn_scaler = nn_bundle["scaler"]
    probs = {}
    probs['rf'] = predict_in_batches(rf_model, X_test, None)
    probs['xgb'] = predict_in_batches(xgb_model, X_test, None)
    probs['nn'] = predict_in_batches(nn_model, X_test, nn_scaler)

# ROC
fig, ax = plt.subplots(figsize=(8, 6))
for m in models:
    fpr, tpr, _ = roc_curve(y_test, probs[m])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{names_map[m]} (AUC={roc_auc:.6f})", color=colors[m])
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - Baseline Models')
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / '03_roc_curves_baseline.png', dpi=300)
plt.savefig(FIG_DIR / '03_roc_curves_baseline.pdf')
plt.close()

# PR Curves
fig, ax = plt.subplots(figsize=(8, 6))
for m in models:
    prec, rec, _ = precision_recall_curve(y_test, probs[m])
    pr_auc = auc(rec, prec)
    ax.plot(rec, prec, label=f"{names_map[m]} (AP={pr_auc:.6f})", color=colors[m])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves - Baseline Models')
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / '04_pr_curves_baseline.png', dpi=300)
plt.savefig(FIG_DIR / '04_pr_curves_baseline.pdf')
plt.close()

# 5) Per-attack recall for XGBoost baseline
fig, ax = plt.subplots(figsize=(10, 8))
xgb_attacks = base['xgb']['per_attack_test_recall']
attacks = sorted(xgb_attacks.keys(), key=lambda x: xgb_attacks[x]['recall'])
recalls = [xgb_attacks[a]['recall'] for a in attacks]
bars = ax.barh(attacks, recalls, color=colors['xgb'])
ax.set_xlabel('Recall')
ax.set_title('XGBoost (Baseline, t=0.5) Per-Attack Recall')
ax.set_xlim(0, 1.05)
for bar, rec in zip(bars, recalls):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{rec:.3f}', va='center')
plt.tight_layout()
plt.savefig(FIG_DIR / '05_xgb_per_attack_recall_baseline.png', dpi=300)
plt.savefig(FIG_DIR / '05_xgb_per_attack_recall_baseline.pdf')
plt.close()

# 6) Per-attack comparison
fig, ax = plt.subplots(figsize=(12, 10))
attacks_all = sorted(base['xgb']['per_attack_test_recall'].keys())
x = np.arange(len(attacks_all))
width = 0.25
for i, m in enumerate(models):
    vals = [base[m]['per_attack_test_recall'][a]['recall'] for a in attacks_all]
    ax.bar(x + i*width, vals, width, label=names_map[m], color=colors[m])
ax.set_ylabel('Recall')
ax.set_title('Per-Attack Recall Comparison (Baseline Models)')
ax.set_xticks(x + width)
ax.set_xticklabels(attacks_all, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(FIG_DIR / '06_per_attack_comparison_baseline.png', dpi=300)
plt.savefig(FIG_DIR / '06_per_attack_comparison_baseline.pdf')
plt.close()

print(f"Done. Baseline figures saved to {FIG_DIR}")
