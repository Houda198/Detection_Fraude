"""Configuration centralisée du projet Fraud Detection.

Tous les chemins, hyperparamètres et constantes sont définis ici.
"""
import os

# ============================================================
# PATHS (Gestion automatique des dossiers)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RAW_DATA_FILE = os.path.join(DATA_RAW_DIR, "creditcard.csv")
PROCESSED_DATA_FILE = os.path.join(DATA_PROCESSED_DIR, "creditcard_cleaned.csv")

# Création automatique de l'arborescence
for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# CONSTANTES & SPLITTING
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "Class"
DROP_COLS = []  
SCALE_COLS = ["Time", "Amount"]

# ============================================================
# SMOTE (Oversampling)
# ============================================================
SMOTE_SAMPLING_STRATEGY = 0.5  # Ratio minorité/majorité final
SMOTE_K_NEIGHBORS = 5

# ============================================================
# GRILLES D'HYPERPARAMÈTRES (GridSearchCV)
# ============================================================
LR_PARAMS = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "class_weight": ["balanced"],
    "solver": ["lbfgs"],
}

RF_PARAMS = {
    "n_estimators": [100], 
    "max_depth": [10],
    "class_weight": ["balanced"], # Plus rapide que balanced_subsample
    "random_state": [RANDOM_STATE]
}

XGB_PARAMS = {
    "n_estimators": [100, 200],      # On teste deux tailles de forêt
    "learning_rate": [0.05, 0.1],    # 0.05 est souvent le "sweet spot"
    "max_depth": [4, 6],             # 6 permet de capturer plus de complexité
    "scale_pos_weight": [1, 10, 50], # Crucial pour la fraude !
    "tree_method": ["hist"], 
    "random_state": [RANDOM_STATE]
}

# ============================================================
# EVALUATION & OUTPUTS
# ============================================================
CV_FOLDS = 5
SCORING_METRIC = "f1"
DEFAULT_THRESHOLD = 0.5

OUTPUT_PLOTS = {
    "class_distribution": os.path.join(OUTPUTS_DIR, "01_class_distribution.png"),
    "correlation_matrix": os.path.join(OUTPUTS_DIR, "02_correlation_matrix.png"),
    "amount_distribution": os.path.join(OUTPUTS_DIR, "03_amount_distribution.png"),
    "confusion_matrices": os.path.join(OUTPUTS_DIR, "05_confusion_matrices.png"),  # Ajouté
    "roc_curves": os.path.join(OUTPUTS_DIR, "06_roc_curves.png"),
    "pr_curves": os.path.join(OUTPUTS_DIR, "07_pr_curves.png"),
    "feature_importance": os.path.join(OUTPUTS_DIR, "08_feature_importance.png"),
    "threshold_tuning": os.path.join(OUTPUTS_DIR, "09_threshold_tuning.png"),     # Ajouté
    "model_comparison": os.path.join(OUTPUTS_DIR, "10_model_comparison.png"),
}