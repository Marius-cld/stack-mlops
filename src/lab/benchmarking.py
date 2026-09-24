import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.settings import BENCHMARK_PATH, model_path
from src.pipeline.process_data import load_processed


def _evaluate(name: str) -> dict:
    """Performances ponctuelles du modèle sur le jeu de test."""
    _, X_test, _, y_test = load_processed(name)
    model = joblib.load(model_path(name))
    pred = model.predict(X_test)
    # SVC sans probability=True : le score de décision suffit pour l'AUC
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_test)[:, 1]
    else:
        score = model.decision_function(X_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "mcc": matthews_corrcoef(y_test, pred),
        "roc_auc": roc_auc_score(y_test, score),
    }


def benchmark_elastic() -> dict:
    return _evaluate("elastic")


def benchmark_svm() -> dict:
    return _evaluate("svm")


def benchmark_trees() -> dict:
    return _evaluate("trees")


def main() -> pd.DataFrame:
    results = pd.DataFrame(
        [benchmark_elastic(), benchmark_svm(), benchmark_trees()]
    ).set_index("model")
    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(BENCHMARK_PATH)
    print(results.round(3).sort_values("roc_auc", ascending=False))
    return results


if __name__ == "__main__":
    main()
