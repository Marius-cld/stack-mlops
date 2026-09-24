import mlflow.sklearn
import pandas as pd

from config.ml_params import MLFLOW_STAGING_ALIAS, STAGING_MIN_ROC_AUC, TARGET
from config.setting import BENCHMARK_PATH
from src.pipeline.extract_data import load_clean_data
from src.monitoring.mlflow_tracking import registered_name, setup
from src.pipeline.train_final import best_model_name


def latest_version(name: str) -> str:
    """Dernière version enregistrée du modèle."""
    versions = setup().search_model_versions(f"name = '{registered_name(name)}'")
    if not versions:
        raise LookupError(
            f"Aucune version pour {registered_name(name)} : "
            "lancer src.monitoring.mlflow_tracking d'abord"
        )
    return max(versions, key=lambda v: int(v.version)).version


def stage_elastic(version: str | None = None) -> str:
    return _stage("elastic", version)


def stage_svm(version: str | None = None) -> str:
    return _stage("svm", version)


def stage_trees(version: str | None = None) -> str:
    return _stage("trees", version)


STAGERS = {"elastic": stage_elastic, "svm": stage_svm, "trees": stage_trees}


def _stage(name: str, version: str | None) -> str:
    """Place l'alias staging sur la version donnée (par défaut la dernière)."""
    version = version or latest_version(name)
    setup().set_registered_model_alias(
        registered_name(name), MLFLOW_STAGING_ALIAS, str(version)
    )
    return str(version)


def passes_quality_gate(name: str, test_auc: float | None = None) -> bool:
    """AUC test (par défaut lu dans le benchmark) et AUC moyen en CV répétée au-dessus du seuil."""
    from src.robustness.cross_validations import cv_elastic, cv_svm, cv_trees

    cv = {"elastic": cv_elastic, "svm": cv_svm, "trees": cv_trees}[name]()
    cv_auc = cv.set_index("metric").loc["roc_auc", "mean"]
    if test_auc is None:
        test_auc = pd.read_csv(BENCHMARK_PATH, index_col="model").loc[name, "roc_auc"]
    print(f"{name}: test={test_auc:.3f} cv={cv_auc:.3f} seuil={STAGING_MIN_ROC_AUC}")
    return bool(min(test_auc, cv_auc) >= STAGING_MIN_ROC_AUC)


def smoke_test(name: str) -> None:
    """Charge la version staging depuis le registre et vérifie qu'elle prédit."""
    setup()
    model = mlflow.sklearn.load_model(
        f"models:/{registered_name(name)}@{MLFLOW_STAGING_ALIAS}"
    )
    X = load_clean_data().drop(columns=TARGET).head(5)
    predictions = model.predict(X)
    if len(predictions) != len(X):
        raise RuntimeError("Le modèle staging ne prédit pas sur toutes les lignes")
    print(f"smoke test ok : {predictions.tolist()}")


def main() -> None:
    best = best_model_name()
    version = STAGERS[best]()
    print(f"{registered_name(best)} v{version} -> alias {MLFLOW_STAGING_ALIAS}")


if __name__ == "__main__":
    main()
