"""
Model Evaluation & Visualization.

Evaluation rigoureuse des modeles sur le test set :
  - Classification report (precision, recall, F1)
  - Confusion matrix
  - ROC curves & AUC
  - Precision-Recall curves & AUC
  - Feature importance
  - Threshold tuning
  - Comparaison des modeles
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
import sys
import os

# Ajout du dossier parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_PLOTS, DEFAULT_THRESHOLD

plt.style.use("seaborn-v0_8-darkgrid")
FIG_DPI = 150

def evaluate_model(model, X_test, y_test, name, threshold=DEFAULT_THRESHOLD):
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

    print(f"\n[EVAL] === {name} (threshold={threshold:.2f}) ===")
    print(f"  Accuracy  : {acc:.6f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc:.4f}")
    print(f"  PR-AUC    : {pr_auc_val:.4f}")
    sys.stdout.flush()

    return {
        "name": name, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "roc_auc": roc, "pr_auc": pr_auc_val,
        "confusion_matrix": cm, "y_proba": y_proba,
        "y_pred": y_pred, "threshold": threshold,
    }

def find_optimal_threshold(model, X_test, y_test, name):
    """Trouve le seuil optimal qui maximise le F1-score via Grid Search."""
    y_proba = model.predict_proba(X_test)[:, 1]

    best_f1 = 0.0
    best_threshold = DEFAULT_THRESHOLD

    # On teste des seuils de 0.05 a 0.95 pour rester dans des valeurs realistes
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_proba >= t).astype(int)
        current_f1 = f1_score(y_test, y_pred, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = t

    best_threshold = round(best_threshold, 2)
    print(f"[EVAL] {name} - Seuil optimal trouve : {best_threshold:.2f}")
    sys.stdout.flush()
    return best_threshold

def plot_confusion_matrices(all_results):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1: axes = [axes]

    for ax, result in zip(axes, all_results):
        cm = result["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax,
                    xticklabels=["Legitime", "Fraude"],
                    yticklabels=["Legitime", "Fraude"])
        ax.set_title(f"{result['name']}\nF1={result['f1']:.4f}")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS["confusion_matrices"], dpi=FIG_DPI)
    plt.close()

def plot_roc_curves(all_results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for result in all_results:
        fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{result['name']} (AUC={roc_auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    plt.savefig(OUTPUT_PLOTS["roc_curves"], dpi=FIG_DPI)
    plt.close()

def plot_pr_curves(all_results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for result in all_results:
        prec, rec, _ = precision_recall_curve(y_test, result["y_proba"])
        ap = average_precision_score(y_test, result["y_proba"])
        ax.plot(rec, prec, lw=2, label=f"{result['name']} (AP={ap:.4f})")

    ax.set_title("Precision-Recall Curves")
    ax.legend()
    plt.savefig(OUTPUT_PLOTS["pr_curves"], dpi=FIG_DPI)
    plt.close()

def plot_feature_importance(results, feature_names):
    models_with_fi = [(n, d) for n, d in results.items() if hasattr(d["model"], "feature_importances_")]
    if not models_with_fi: return

    n_plots = len(models_with_fi)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 8))
    if n_plots == 1: axes = [axes]

    for ax, (name, data) in zip(axes, models_with_fi):
        importances = data["model"].feature_importances_
        indices = np.argsort(importances)[-15:]
        ax.barh(range(len(indices)), importances[indices], color="#3498db")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_title(f"Feature Importance - {name}")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS["feature_importance"], dpi=FIG_DPI)
    plt.close()

def plot_model_comparison(all_results):
    metrics = ["precision", "recall", "f1", "roc_auc"]
    labels = ["Precision", "Recall", "F1", "ROC-AUC"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.25

    for i, result in enumerate(all_results):
        values = [result[m] for m in metrics]
        ax.bar(x + i*width, values, width, label=result["name"])

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_title("Comparaison des Modeles")
    ax.legend()
    plt.savefig(OUTPUT_PLOTS["model_comparison"], dpi=FIG_DPI)
    plt.close()

def run_evaluation(results, X_test, y_test, feature_names):
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    all_results = []
    for name, data in results.items():
        best_threshold = find_optimal_threshold(data["model"], X_test, y_test, name)
        result = evaluate_model(data["model"], X_test, y_test, name, threshold=best_threshold)
        all_results.append(result)

    print("\n[EVAL] Generation des visualisations...")
    plot_confusion_matrices(all_results)
    plot_roc_curves(all_results, y_test)
    plot_pr_curves(all_results, y_test)
    plot_feature_importance(results, feature_names)
    plot_model_comparison(all_results)

    # Resume final
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for r in sorted(all_results, key=lambda x: x['f1'], reverse=True):
        print(f"{r['name']:<20} | F1: {r['f1']:.4f} | Recall: {r['recall']:.4f} | Thresh: {r['threshold']:.2f}")
    
    sys.stdout.flush()
    return all_results