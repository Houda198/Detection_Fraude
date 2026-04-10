"""Data Cleaning & Preprocessing Pipeline.

Ce module charge les données brutes, les nettoie et les prépare
pour l'entraînement des modèles.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import sys
import os

# Ajout du chemin pour importer config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_FILE, PROCESSED_DATA_FILE, SCALE_COLS,
    TARGET_COL, DROP_COLS
)

def load_raw_data(filepath: str = RAW_DATA_FILE) -> pd.DataFrame:
    """Charge le dataset brut depuis le CSV."""
    print(f"[CLEAN] Chargement de {filepath}...")
    df = pd.read_csv(filepath)
    print(f"[CLEAN] Shape: {df.shape}")
    return df

def check_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie la qualité des données : NaN, doublons."""
    print("\n[CLEAN] === Data Quality Check ===")
    
    # Valeurs manquantes
    missing = df.isnull().sum()
    if missing.sum() > 0:
        df = df.dropna()
        print(f"[CLEAN] Après suppression NaN: {df.shape}")
    
    # Doublons
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        df = df.drop_duplicates()
        print(f"[CLEAN] Après suppression doublons: {df.shape}")
        
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering (Heure et Log Amount)."""
    print("\n[CLEAN] === Feature Engineering ===")
    df["Hour"] = (df["Time"] / 3600) % 24
    df["Log_Amount"] = np.log1p(df["Amount"])
    return df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applique RobustScaler."""
    print("\n[CLEAN] === Feature Scaling ===")
    scaler = RobustScaler()
    cols_to_scale = SCALE_COLS + ["Hour", "Log_Amount"]
    existing_cols = [c for c in cols_to_scale if c in df.columns]
    df[existing_cols] = scaler.fit_transform(df[existing_cols])
    return df

def run_cleaning_pipeline():
    """Exécute le pipeline complet."""
    df = load_raw_data()
    df = check_data_quality(df)
    df = engineer_features(df)
    df = scale_features(df)
    
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"\n[CLEAN] ✅ Terminé : {PROCESSED_DATA_FILE}")

if __name__ == "__main__":
    run_cleaning_pipeline()