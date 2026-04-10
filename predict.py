"""Prediction Module.

Charge le meilleur modèle et prédit la probabilité de fraude.
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Gestion des imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, PROCESSED_DATA_FILE, TARGET_COL

class FraudPredictor:
    """Prédicteur de fraude prêt pour la production."""

    def __init__(self, model_name: str = "xgboost_best.joblib", threshold: float = 0.45):
        """Initialise le modèle avec son seuil de décision optimisé."""
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé : {model_path}. Lancez d'abord le pipeline.")

        self.model = joblib.load(model_path)
        self.threshold = threshold
        print(f"[PREDICT] Modèle chargé : {model_name} (Seuil : {threshold})")

    def predict(self, X: pd.DataFrame) -> dict:
        """Prédiction sur un lot (batch) de transactions."""
        probas = self.model.predict_proba(X)[:, 1]
        predictions = (probas >= self.threshold).astype(int)
        
        return {
            "predictions": predictions,
            "probabilities": probas,
            "n_fraud": int(predictions.sum()),
            "fraud_rate": float(predictions.mean())
        }

    def predict_single(self, transaction: dict) -> dict:
        """Prédiction pour une transaction unique (temps réel)."""
        X = pd.DataFrame([transaction])
        proba = self.model.predict_proba(X)[:, 1][0]
        
        return {
            "is_fraud": bool(proba >= self.threshold),
            "fraud_probability": float(proba),
            "risk_level": self._get_risk_level(proba)
        }

    @staticmethod
    def _get_risk_level(proba: float) -> str:
        """Traduit une probabilité en niveau de risque métier."""
        if proba >= 0.8: return "🔴 CRITICAL"
        if proba >= 0.5: return "🟠 HIGH"
        if proba >= 0.3: return "🟡 MEDIUM"
        return "🟢 LOW"

def demo_prediction():
    """Script de démonstration sur un échantillon aléatoire."""
    predictor = FraudPredictor()
    df = pd.read_csv(PROCESSED_DATA_FILE)
    
    # Simulation : On prend 5 transactions au hasard
    sample = df.sample(5, random_state=42)
    X_sample = sample.drop(columns=[TARGET_COL])
    y_sample = sample[TARGET_COL].values

    print("\n🔍 TESTS INDIVIDUELS :")
    for i in range(len(sample)):
        tx = X_sample.iloc[i].to_dict()
        res = predictor.predict_single(tx)
        status = "FRAUDE" if y_sample[i] == 1 else "LÉGITIME"
        print(f"Probabilité: {res['fraud_probability']:.4f} | Risque: {res['risk_level']} | Réalité: {status}")

if __name__ == "__main__":
    demo_prediction()