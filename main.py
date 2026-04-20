"""
Main Pipeline Orchestrator.
Lance le pipeline complet si les données ont changé.
"""
import sys
import time
import os
import pandas as pd
import hashlib
from config import RAW_DATA_FILE, OUTPUTS_DIR, PROCESSED_DATA_FILE, TARGET_COL, log

# Ajout du chemin pour l'import des modules locaux
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_file_hash(filepath):
    """Génère une empreinte unique (MD5) du fichier."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def check_prerequisites():
    """Vérifie que tout est en place avant de lancer."""
    print("\n" + "=" * 60)
    print(" CHECKING PREREQUISITES")
    print("=" * 60)
    if not os.path.exists(RAW_DATA_FILE):
        print(f"\n ERREUR: Dataset non trouvé !")
        sys.exit(1)
    print(" Dataset trouvé")
    print(" Configuration chargée")

def main():
    total_start = time.time()
    
    # --- LOGIQUE DE HASH / LAZY LOADING ---
    hash_file = "data/last_run_hash.txt"
    current_hash = get_file_hash(RAW_DATA_FILE)
    must_train = True 

    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            if f.read().strip() == current_hash:
                must_train = True

    print("\n" + "=" * 60)
    print("      FRAUD DETECTION ML PIPELINE")
    if not must_train:
        print("      (MODE ANALYSE : Données inchangées)")
    print("=" * 60)

    check_prerequisites()

    if must_train:
        # STEP 1 - Data Cleaning
        print("\nSTEP 1/4 : DATA CLEANING")
        from src.data_cleaning import run_cleaning_pipeline
        run_cleaning_pipeline()

        # STEP 2 - EDA
        print("\nSTEP 2/4 : EXPLORATORY DATA ANALYSIS")
        from src.eda import run_eda
        run_eda()

        # STEP 3 - Training
        print("\nSTEP 3/4 : MODEL TRAINING")
        from src.train import run_training_pipeline
        results, X_test, y_test = run_training_pipeline()

        # On sauvegarde le hash pour ne pas refaire le travail la prochaine fois
        with open(hash_file, "w") as f:
            f.write(current_hash)
    else:
        print("\n[INFO] Skipping Steps 1, 2, 3 (Données déjà traitées).")
        print("[INFO] Chargement des modèles et données de test existants...")
        import joblib
        from sklearn.model_selection import train_test_split
        from config import TEST_SIZE, RANDOM_STATE, MODELS_DIR
        
        # On recharge les données processed
        df = pd.read_csv(PROCESSED_DATA_FILE)
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
        
        # On simule le dictionnaire 'results' pour l'évaluateur
        results = {
            "XGBoost": {"model": joblib.load(os.path.join(MODELS_DIR, "xgboost_best.joblib")), "threshold": 0.45},
            "Random Forest": {"model": joblib.load(os.path.join(MODELS_DIR, "random_forest_best.joblib")), "threshold": 0.68}
        }

    # STEP 4 - Evaluation 
    print("\nSTEP 4/4 : MODEL EVALUATION")
    from src.evaluate import run_evaluation
    feature_names = [c for c in pd.read_csv(PROCESSED_DATA_FILE).columns if c != TARGET_COL]
    run_evaluation(results, X_test, y_test, feature_names)

    total_time = time.time() - total_start
    print(f"\nPIPELINE COMPLETE - Total time: {total_time:.1f}s")

if __name__ == "__main__":
    main()