"""Training Pipeline.

Pipeline d'entraînement complet :
  1. Split stratifié (maintien du ratio de fraude)
  2. SMOTE sur Train uniquement
  3. GridSearchCV (optimisation des hyperparamètres)
  4. Calcul dynamique du scale_pos_weight pour XGBoost
"""
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
)
from imblearn.over_sampling import SMOTE
import sys
import os

# Configuration des imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DATA_FILE, TARGET_COL, TEST_SIZE, RANDOM_STATE,
    CV_FOLDS, SCORING_METRIC, SMOTE_SAMPLING_STRATEGY,
    SMOTE_K_NEIGHBORS, MODELS_DIR
)
from src.model import get_all_models

def load_processed_data():
    """Charge les données nettoyées."""
    df = pd.read_csv(PROCESSED_DATA_FILE)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y

def split_data(X, y):
    """Split stratifié pour conserver la rareté de la fraude dans les deux sets."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

def apply_smote(X_train, y_train):
    """Génère des fraudes synthétiques pour équilibrer l'apprentissage."""
    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=RANDOM_STATE
    )
    return smote.fit_resample(X_train, y_train)

def train_model_with_grid_search(model, param_grid, name, X_train, y_train):
    """Entraîne et optimise via Cross-Validation."""
    print(f"\n[TRAIN] --- Optimisation de {name} ---")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid,
        cv=cv, scoring=SCORING_METRIC, n_jobs=-1
    )
    
    start = time.time()
    grid_search.fit(X_train, y_train)
    duration = time.time() - start
    
    return grid_search.best_estimator_, grid_search.best_params_, duration

def run_training_pipeline():
    """Exécution du pipeline global."""
    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Équilibrage du set d'entraînement
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    
    # Calcul du poids pour XGBoost (Ratio Neg/Pos)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    models_list = get_all_models(scale_pos_weight)
    results = {}

    for model, param_grid, name in models_list:
        best_model, params, train_time = train_model_with_grid_search(
            model, param_grid, name, X_train_res, y_train_res
        )
        
        # Sauvegarde sur disque
        filename = f"{name.lower().replace(' ', '_')}_best.joblib"
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump(best_model, path)
        
        results[name] = {"model": best_model, "training_time": train_time}

    print("\n[TRAIN] ✅ Pipeline d'entraînement terminé avec succès.")
    return results, X_test, y_test

if __name__ == "__main__":
    run_training_pipeline()