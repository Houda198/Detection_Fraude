"""Model Definitions & Factory.

Définit les 3 modèles piliers :
  1. Logistic Regression (Baseline)
  2. Random Forest (Bagging)
  3. XGBoost (Boosting)
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RF_PARAMS, XGB_PARAMS, RANDOM_STATE

def get_random_forest():
    """Modèle robuste aux features non-linéaires."""
    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return model, RF_PARAMS, "Random Forest"

def get_xgboost(scale_pos_weight: float = 1.0):
    """Le must pour les données tabulaires (XGBoost)."""
    model = XGBClassifier(
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        tree_method = "hist",
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss",
        
    )
    return model, XGB_PARAMS, "XGBoost"

def get_all_models(scale_pos_weight: float = 1.0):
    """Retourne la liste complète pour la boucle d'entraînement."""
    return [
        get_random_forest(),
        get_xgboost(scale_pos_weight),
    ]