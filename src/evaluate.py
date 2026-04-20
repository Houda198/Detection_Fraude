"""Model Evaluation & Visualization.

Evalue les modeles sur le TEST SET avec les thresholds
optimises sur le validation set.

Toutes les metriques sont calculees sur le TEST SET
(distribution reelle, jamais vu pendant l'entrainement).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score
)
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_PLOTS, log

plt.style.use("seaborn-v0_8-darkgrid")
FIG_DPI = 150

def evaluate_model(model, X_test, y_test, name, threshold):
    """Evalue un modele sur le test set."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba)
    pr_auc_val = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    log.info(f"\n[EVAL] === {name} (threshold={threshold:.2f}) ===")
    log.info(f"  Accuracy  : {acc:.6f}")
    log.info(f"  Precision : {prec:.4f}")
    log.info(f"  Recall    : {rec:.4f}")
    log.info(f"  F1-Score  : {f1:.4f}")
    log.info(f"  ROC-AUC   : {roc:.4f}")
    log.info(f"  PR-AUC    : {pr_auc_val:.4f}")
    log.info(f"  Confusion Matrix:")
    log.info(f"    TN={cm[0][0]:,}  FP={cm[0][1]}")
    log.info(f"    FN={cm[1][0]:,}  TP={cm[1][1]}")
    log.info(classification_report(y_test, y_pred,
             target_names=['Legitime', 'Fraude']))

    return {
        "name": name, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "roc_auc": roc, "pr_auc": pr_auc_val,
        "confusion_matrix": cm, "y_proba": y_proba,
        "y_pred": y_pred, "threshold": threshold,
    }

def plot_confusion_matrices(all_results):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, all_results):
        sns.heatmap(r["confusion_matrix"], annot=True, fmt=",d",
                    cmap="Blues", ax=ax,
                    xticklabels=["Legitime", "Fraude"],
                    yticklabels=["Legitime", "Fraude"],
                    linewidths=1, linecolor="white")
        ax.set_title(f"{r['name']}\nF1={r['f1']:.4f}",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS["confusion_matrices"], dpi=FIG_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("[EVAL] Confusion matrices saved")

def plot_roc_curves(all_results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for r, c in zip(all_results, colors):
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, color=c, lw=2,
                label=f"{r['name']} (AUC={auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS["roc_curves"], dpi=FIG_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("[EVAL] ROC curves saved")

def plot_pr_curves(all_results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for r, c in zip(all_results, colors):
        p, rc, _ = precision_recall_curve(y_test, r["y_proba"])
        ap = average_precision_score(y_test, r["y_proba"])
        ax.plot(rc, p, color=c, lw=2,
                label=f"{r['name']} (AP={ap:.4f})")
    baseline = y_test.mean()
    ax.axhline(y=baseline, color="gray", ls="--", alpha=0.5,
               label=f"Baseline ({baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS["pr_curves"], dpi=FIG_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("[EVAL] PR curves saved")

def plot_feature_importance(results, feature_names):
    models_fi = [(n, d) for n, d in results.items()
                 if hasattr(d["model"], "feature_importances_")]
    if not models_fi:
        return
    n_plots = min(len(models_fi), 2)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 8))
    if n_plots == 1:
        axes = [axes]
    for ax, (name, data) in zip(axes, models_fi[:2]):
        imp = data["model"].feature_importances_
        n_f = min(len(imp), len(feature_names))
        idx = np.argsort(imp[:n_f])[-15:]
        ax.barh(range(len(idx)), imp[idx], color="#3498db", edgecolor="white")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_names[i] for i in idx])
        ax.set_title(f"Feature Importance - {name}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS["feature_importance"], dpi=FIG_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("[EVAL] Feature importance saved")

def plot_threshold_tuning(model, X_test, y_test, name):
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.01)
    prec_l, rec_l, f1_l = [], [], []
    for t in thresholds:
        yp = (y_proba >= t).astype(int)
        prec_l.append(precision_score(y_test, yp, zero_division=0))
        rec_l.append(recall_score(y_test, yp, zero_division=0))
        f1_l.append(f1_score(y_test, yp, zero_division=0))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, prec_l, "b-", lw=2, label="Precision")
    ax.plot(thresholds, rec_l, "g-", lw=2, label="Recall")
    ax.plot(thresholds, f1_l, "r-", lw=2.5, label="F1-Score")
    bi = np.argmax(f1_l)
    ax.axvline(x=thresholds[bi], color="red", ls="--", alpha=0.5,
               label=f"Best={thresholds[bi]:.2f}")
    ax.scatter([thresholds[bi]], [f1_l[bi]], color="red", s=100, zorder=5)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Tuning - {name}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS["threshold_tuning"], dpi=FIG_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("[EVAL] Threshold tuning plot saved")

def plot_model_comparison(all_results):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.25
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for i, (r, c) in enumerate(zip(all_results, colors)):
        vals = [r[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=r["name"],
                      color=c, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des Modeles", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS["model_comparison"], dpi=FIG_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("[EVAL] Model comparison saved")

def run_evaluation(results, X_test, y_test, feature_names):
    """Execute l'evaluation complete sur le test set."""
    log.info("\n" + "=" * 60)
    log.info("MODEL EVALUATION (on TEST set)")
    log.info("=" * 60)

    all_results = []
    for name, data in results.items():
        # Le threshold a deja ete optimise sur le val set dans train.py
        threshold = data["threshold"]
        r = evaluate_model(data["model"], X_test, y_test, name, threshold)
        r["training_time"] = data["training_time"]
        r["cv_scores"] = data["cv_scores"]
        all_results.append(r)

    log.info("\n[EVAL] Generation des visualisations...")
    plot_confusion_matrices(all_results)
    plot_roc_curves(all_results, y_test)
    plot_pr_curves(all_results, y_test)
    plot_feature_importance(results, feature_names)
    plot_model_comparison(all_results)

    best = max(all_results, key=lambda x: x["f1"])
    plot_threshold_tuning(results[best["name"]]["model"],
                          X_test, y_test, best["name"])

    log.info("\n" + "=" * 60)
    log.info("FINAL RESULTS SUMMARY")
    log.info("=" * 60)
    log.info(f"{'Model':<25} {'F1':>8} {'Recall':>8} {'Prec':>8} "
             f"{'ROC-AUC':>8} {'Thresh':>8}")
    log.info("-" * 75)
    for r in sorted(all_results, key=lambda x: x["f1"], reverse=True):
        log.info(f"{r['name']:<25} {r['f1']:>8.4f} {r['recall']:>8.4f} "
                 f"{r['precision']:>8.4f} {r['roc_auc']:>8.4f} "
                 f"{r['threshold']:>8.2f}")
    log.info(f"\nMEILLEUR MODELE : {best['name']} "
             f"(F1={best['f1']:.4f}, threshold={best['threshold']:.2f})")
    log.info("=" * 60)
    return all_results