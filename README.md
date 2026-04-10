# 🛡️ Credit Card Fraud Detection

> détection de fraudes bancaires sur 284,807 transactions avec 3 modèles ML optimisés.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)


## 🎯 Résultats Finaux

Le pipeline a optimisé les modèles pour maximiser le **F1-Score**, garantissant un équilibre entre détection et précision.

* **XGBoost (Champion)** : **84.88%** de F1-Score
  * Précision : 94.81% | Rappel (Recall) : 76.84% | Seuil : 0.91
* **Random Forest** : **79.10%** de F1-Score
  * Précision : 85.37% | Rappel (Recall) : 73.68% | Seuil : 0.71
* **Logistic Regression** : **39.48%** de F1-Score (Seuil : 0.95)

## 🚀 Quick Start

 1. Cloner le repo
git clone <repo-url> && cd fraud-detection

 2. Installer les dépendances
pip install -r requirements.txt

 3. Télécharger le dataset
 Option A : Kaggle CLI
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/raw/

 Option B : Téléchargement manuel
 -> kaggle.com/mlg-ulb/creditcardfraud
 -> Placer creditcard.csv dans data/raw/

 4. Lancer le pipeline complet
python main.py

 5. Tester les prédictions
python predict.py


## 📊 Dataset

- **Source** : [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Transactions** : 284,807
- **Fraudes** : 492 (0.173%)
- **Features** : V1-V28 (PCA), Time, Amount
- **Déséquilibre** : 577:1 (légitime:fraude)


## 🧠 Optimisations Stratégiques

- **Seuils sur mesure** : Recherche par grille (0.05 à 0.95) pour éviter les seuils extrêmes et assurer la stabilité opérationnelle.
- **Traitement de l'imbalance** : Application de **SMOTE** et ajustement du `scale_pos_weight` pour XGBoost.
- **Robustesse** : Pipeline entièrement automatisé du nettoyage (`RobustScaler`, Log Transform) jusqu'à l'évaluation finale.
## 🛡️ Bonnes Pratiques Appliquées

- ✅ SMOTE appliqué uniquement sur le train set (pas de data leakage)
- ✅ Stratified split et stratified cross-validation
- ✅ Threshold optimisé (pas le 0.5 par défaut)
- ✅ Métriques adaptées au déséquilibre (F1, PR-AUC > Accuracy)
- ✅ Overfitting monitoring (CV ≈ Test scores)
- ✅ Configuration centralisée (config.py)
- ✅ Code modulaire et documenté
- ✅ Reproductibilité (random_state=42)


## 📈 Visualisations Générées

10 plots haute qualité dans `outputs/` :
1. Distribution des classes
2. Matrice de corrélation
3. Distribution Amount
4. Distribution Time
5. Confusion matrices
6. ROC curves
7. Precision-Recall curves
8. Feature importance
9. Threshold tuning
10. Comparaison des modèles

## 🛠️ Stack Technique

- **Python** 3.10+
- **pandas** / **numpy** : manipulation de données
- **scikit-learn** : ML pipeline, métriques, preprocessing
- **XGBoost** : gradient boosting
- **imbalanced-learn** : SMOTE
- **matplotlib** / **seaborn** : visualisations
- **joblib** : sérialisation des modèles