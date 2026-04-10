"""Exploratory Data Analysis (EDA).

Génère des visualisations pour comprendre le dataset :
  - Distribution des classes (fraude vs légitime)
  - Distribution de Amount et Time
  - Matrice de corrélation
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend non-interactif pour serveur/agent
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Ajout du path pour config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DATA_FILE, TARGET_COL, OUTPUT_PLOTS, OUTPUTS_DIR
)

# Style global pro
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
FIG_DPI = 150

def load_data() -> pd.DataFrame:
    """Charge le dataset nettoyé."""
    df = pd.read_csv(PROCESSED_DATA_FILE)
    print(f"[EDA] Dataset chargé : {df.shape}")
    return df

def plot_class_distribution(df: pd.DataFrame) -> None:
    """Visualise le déséquilibre des classes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = df[TARGET_COL].value_counts()
    colors = ["#2ecc71", "#e74c3c"]

    # Bar plot
    axes[0].bar(["Légitime", "Fraude"], counts.values, color=colors, edgecolor="white")
    axes[0].set_title("Distribution des Classes", fontsize=14, fontweight="bold")
    
    # Pie chart
    axes[1].pie(counts.values, labels=["Légitime", "Fraude"], 
                autopct="%1.3f%%", colors=colors, startangle=90, explode=(0, 0.1))
    axes[1].set_title("Proportion des Classes", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS["class_distribution"], dpi=FIG_DPI)
    plt.close()
    print("[EDA] ✅ Class distribution saved")

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Matrice de corrélation des 15 meilleures features."""
    correlations = df.corr()[TARGET_COL].abs().sort_values(ascending=False)
    top_features = correlations.head(15).index.tolist()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = df[top_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                cmap="RdBu_r", center=0, ax=ax, square=True)
    ax.set_title("Matrice de Corrélation (Top 15)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS["correlation_matrix"], dpi=FIG_DPI)
    plt.close()
    print("[EDA] ✅ Correlation matrix saved")

def run_eda() -> None:
    """Exécute l'analyse complète."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    df = load_data()
    plot_class_distribution(df)
    plot_correlation_matrix(df)
    
    print(f"\n[EDA] ✅ Visualisations sauvegardées dans {OUTPUTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()