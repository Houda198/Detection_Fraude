"""
Main Pipeline Orchestrator.
Lance le pipeline complet : Cleaning -> EDA -> Training -> Evaluation.
"""

import sys
import time
import os
import pandas as pd

# Ajout du chemin pour l'import des modules locaux
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RAW_DATA_FILE, OUTPUTS_DIR, PROCESSED_DATA_FILE, TARGET_COL

def check_prerequisites():
    """Verifie que tout est en place avant de lancer."""
    print("\n" + "=" * 60)
    print(" CHECKING PREREQUISITES")
    print("=" * 60)

    if not os.path.exists(RAW_DATA_FILE):
        print(f"\n ERREUR: Dataset non trouve !")
        print(f"   Attendu : {RAW_DATA_FILE}")
        print(f"   Telechargez depuis : kaggle.com/mlg-ulb/creditcardfraud")
        sys.exit(1)

    print(" Dataset trouve")
    print(" Configuration chargee")
    sys.stdout.flush()

def main():
    total_start = time.time()

    print("\n" + "=" * 60)
    print("      FRAUD DETECTION ML PIPELINE")
    print("      Credit Card Fraud Detection")
    print("=" * 60)
    sys.stdout.flush()

    check_prerequisites()

    # STEP 1 - Data Cleaning
    print("\n" + "-" * 60)
    print("STEP 1/4 : DATA CLEANING")
    print("-" * 60)
    sys.stdout.flush()
    from src.data_cleaning import run_cleaning_pipeline
    run_cleaning_pipeline()

    # STEP 2 - EDA
    print("\n" + "-" * 60)
    print("STEP 2/4 : EXPLORATORY DATA ANALYSIS")
    print("-" * 60)
    sys.stdout.flush()
    from src.eda import run_eda
    run_eda()

    # STEP 3 - Training
    print("\n" + "-" * 60)
    print("STEP 3/4 : MODEL TRAINING")
    print("-" * 60)
    sys.stdout.flush()
    from src.train import run_training_pipeline
    results, X_test, y_test = run_training_pipeline()

    # STEP 4 - Evaluation
    print("\n" + "-" * 60)
    print("STEP 4/4 : MODEL EVALUATION")
    print("-" * 60)
    sys.stdout.flush()
    from src.evaluate import run_evaluation
    
    # Recupere les noms des colonnes pour le feature importance
    feature_names = [c for c in pd.read_csv(PROCESSED_DATA_FILE).columns 
                     if c != TARGET_COL]
    
    run_evaluation(results, X_test, y_test, feature_names)

    # FIN
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f" PIPELINE COMPLETE - Total time: {total_time:.1f}s")
    print("=" * 60)

    output_files = [f for f in os.listdir(OUTPUTS_DIR) if f.endswith('.png')]
    print(f" Resultats dans : {OUTPUTS_DIR}")
    print(f" {len(output_files)} visualisations generees")
    sys.stdout.flush()

if __name__ == "__main__":
    main()