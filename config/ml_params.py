# Cible
TARGET = "Class"
POSITIVE_CLASS = "High_BFE"
CLASSES = ["Low_BFE", POSITIVE_CLASS]

# Reproductibilité et découpage
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Modèles
MODEL_NAMES = ("elastic", "svm", "trees")

# Variables fortement asymétriques (Yeo-Johnson pour elastic)
SKEWED_FEATURES = ["minHaaCH", "maxwHBa"]

# Validation croisée pour l'optimisation
CV_SPLITS = 5
CV_REPEATS = 3
SCORING = "roc_auc"

# Paramètres fixes des estimateurs
ELASTIC_PARAMS = {"solver": "saga", "max_iter": 10_000, "random_state": RANDOM_STATE}
SVM_PARAMS = {"kernel": "rbf", "random_state": RANDOM_STATE}
TREES_PARAMS = {"n_estimators": 300, "random_state": RANDOM_STATE}

# Grilles d'hyperparamètres
ELASTIC_GRID = {
    "C": [0.01, 0.1, 1, 10, 100],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}
SVM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1, 1],
}
TREES_GRID = {
    "max_depth": [2, 3, 5, None],
    "min_samples_leaf": [1, 3, 5],
    "max_features": ["sqrt", 0.5, 1.0],
}

# Robustesse
ROBUSTNESS_CV_SPLITS = 5
ROBUSTNESS_CV_REPEATS = 10
N_BOOTSTRAPS = 200
NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]  # écart-type du bruit / écart-type de la variable
NOISE_CV_REPEATS = 3
ROBUSTNESS_METRICS = ("roc_auc", "accuracy", "balanced_accuracy", "f1")
CONFIDENCE_LEVEL = 0.95

# MLflow
MLFLOW_EXPERIMENT = "sirtuin6"
MLFLOW_MODEL_PREFIX = "sirtuin6"  # nom enregistré : sirtuin6-<model>
MLFLOW_PRODUCTION_ALIAS = "production"
MLFLOW_STAGING_ALIAS = "staging"

# Contrôles qualité (data validation et gate avant staging)
MIN_ROWS = 50
MIN_CLASS_SHARE = 0.3
STAGING_MIN_ROC_AUC = 0.85  # benchmark test ET moyenne de la CV répétée
