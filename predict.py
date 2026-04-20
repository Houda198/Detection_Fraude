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
    """Script de démonstration grand format avec statistiques."""
    predictor = FraudPredictor()
    df = pd.read_csv(PROCESSED_DATA_FILE)
    
    # On prend 10 légitimes et 10 fraudes pour avoir un bon mix
    legit_samples = df[df[TARGET_COL] == 0].sample(10, random_state=42)
    fraud_samples = df[df[TARGET_COL] == 1].sample(10, random_state=42)
    
    # On mélange le tout
    combined_samples = pd.concat([legit_samples, fraud_samples]).sample(frac=1, random_state=7)
    
    X_sample = combined_samples.drop(columns=[TARGET_COL])
    y_sample = combined_samples[TARGET_COL].values

    print(f"\n🔍 TEST SUR {len(combined_samples)} TRANSACTIONS MIXTES :")
    print("-" * 85)
    print(f"{'Probabilité':<15} | {'Risque':<12} | {'Réalité':<15} | {'Verdict'}")
    print("-" * 85)

    stats = {"Correct": 0, "Erreur": 0}

    for i in range(len(combined_samples)):
        tx = X_sample.iloc[i].to_dict()
        res = predictor.predict_single(tx)
        
        is_fraud_real = y_sample[i] == 1
        is_fraud_pred = res['fraud_probability'] >= predictor.threshold
        
        real_text = "FRAUDE" if is_fraud_real else "LÉGITIME"
        
        # On vérifie si le modèle a raison
        if is_fraud_real == is_fraud_pred:
            verdict = "✅ OK"
            stats["Correct"] += 1
        else:
            verdict = "❌ ERREUR"
            stats["Erreur"] += 1
            
        print(f"{res['fraud_probability']:.6f}      | "
              f"{res['risk_level']:<12} | "
              f"{real_text:<15} | "
              f"{verdict}")

    print("-" * 85)
    print(f"BILAN : {stats['Correct']} corrects, {stats['Erreur']} erreurs.")

if __name__ == "__main__":
    demo_prediction()