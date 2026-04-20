"""Training Pipeline.

Pipeline d'entrainement :
  1. Chargement des donnees nettoyees
  2. Split train/val/test stratifie
  3. PAS de SMOTE (on utilise class_weight a la place)
  4. GridSearchCV avec cross-validation stratifiee
  5. Threshold tuning sur le VALIDATION set
  6. Evaluation finale sur le TEST set

Pourquoi pas SMOTE ?
  - SMOTE cree des samples synthetiques qui decalent les probabilites
  - Les thresholds optimaux deviennent extremes (0.9+)
  - class_weight='balanced' fait le meme job sans ces problemes
  - Les probas restent calibrees -> threshold tuning fiable
"""

import pandas as pd
import numpy as np
import time
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.metrics import f1_score
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PROCESSED_DATA_FILE, TARGET_COL, TEST_SIZE, VAL_SIZE,
    RANDOM_STATE, CV_FOLDS, SCORING_METRIC, MODELS_DIR, log
)
from src.model import get_all_models

def load_processed_data():
    df = pd.read_csv(PROCESSED_DATA_FILE)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    log.info(f"[TRAIN] Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    log.info(f"[TRAIN] Distribution: {dict(y.value_counts())}")
    return X, y

def split_data(X, y):
    """Split stratifie en 3 : train / val / test.

    - train : entrainement des modeles
    - val   : threshold tuning (trouve le meilleur seuil)
    - test  : evaluation finale (jamais touche avant la fin)
    """
    # D'abord separer le test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # Puis separer train et validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=y_temp
    )
    log.info(f"[TRAIN] Train : {X_train.shape[0]} samples "
             f"({y_train.sum()} fraudes)")
    log.info(f"[TRAIN] Val   : {X_val.shape[0]} samples "
             f"({y_val.sum()} fraudes)")
    log.info(f"[TRAIN] Test  : {X_test.shape[0]} samples "
             f"({y_test.sum()} fraudes)")
    return X_train, X_val, X_test, y_train, y_val, y_test

def find_best_threshold(model, X_val, y_val, name):
    """Trouve le seuil optimal sur le VALIDATION set.

    On teste de 0.10 a 0.90 par pas de 0.01.
    Le val set n'a PAS ete utilise pour l'entrainement,
    donc les probas sont realistes.
    """
    y_proba = model.predict_proba(X_val)[:, 1]
    best_f1 = 0.0
    best_t = 0.5
    for t in np.arange(0.10, 0.91, 0.01):
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = round(t, 2)
    log.info(f"[TRAIN] {name} - Best threshold (val): {best_t:.2f} "
             f"(F1={best_f1:.4f})")
    return best_t

def train_model(model, param_grid, name, X_train, y_train):
    log.info(f"\n[TRAIN] --- Training {name} ---")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=cv,
        scoring=SCORING_METRIC, n_jobs=-1, verbose=0, refit=True,
    )
    start = time.time()
    grid.fit(X_train, y_train)
    train_time = time.time() - start

    best = grid.best_estimator_
    log.info(f"[TRAIN] Best params: {grid.best_params_}")
    log.info(f"[TRAIN] Best CV {SCORING_METRIC}: {grid.best_score_:.4f}")
    log.info(f"[TRAIN] Training time: {train_time:.1f}s")

    cv_scores = cross_val_score(
        best, X_train, y_train, cv=cv,
        scoring=SCORING_METRIC, n_jobs=-1
    )
    log.info(f"[TRAIN] CV F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    return best, grid.best_params_, train_time, cv_scores

def save_model(model, name):
    filename = f"{name.lower().replace(' ', '_')}_best.joblib"
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, filepath)
    log.info(f"[TRAIN] Modele sauvegarde : {filepath}")
    return filepath

def run_training_pipeline():
    log.info("=" * 60)
    log.info("TRAINING PIPELINE")
    log.info("=" * 60)

    X, y = load_processed_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # scale_pos_weight pour XGBoost
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    log.info(f"[TRAIN] scale_pos_weight (XGBoost): {spw:.1f}")

    models_list = get_all_models(spw)
    results = {}

    for model, param_grid, name in models_list:
        best_model, best_params, train_time, cv_scores = train_model(
            model, param_grid, name, X_train, y_train
        )
        # Threshold tuning sur le VAL set (pas le test set !)
        best_threshold = find_best_threshold(
            best_model, X_val, y_val, name
        )
        save_model(best_model, name)
        results[name] = {
            "model": best_model,
            "params": best_params,
            "training_time": train_time,
            "cv_scores": cv_scores,
            "threshold": best_threshold,
        }

    log.info("\n" + "=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("=" * 60)
    return results, X_test, y_test

if __name__ == "__main__":
    run_training_pipeline()