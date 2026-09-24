import numpy as np
import pandas as pd

from config.ml_params import CONFIDENCE_LEVEL, POSITIVE_CLASS, TARGET
from config.settings import ROBUSTNESS_ARTIFACTS_PATH, robustness_result_path
from src.pipeline.extract_data import load_clean_data


def load_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Toutes les données : les tests de robustesse rééchantillonnent eux-mêmes."""
    df = load_clean_data()
    return df.drop(columns=TARGET), (df[TARGET] == POSITIVE_CLASS).astype(int)


def positive_score(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def summarize(values: pd.DataFrame) -> pd.DataFrame:
    """Moyenne, écart-type et intervalle percentile de chaque colonne."""
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    return pd.DataFrame(
        {
            "mean": values.mean(),
            "std": values.std(),
            "ci_low": values.quantile(alpha),
            "ci_high": values.quantile(1 - alpha),
            "n": values.count(),
        }
    )


def save_results(test: str, results: list[pd.DataFrame]) -> pd.DataFrame:
    table = pd.concat(results)
    ROBUSTNESS_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    table.to_csv(robustness_result_path(test), index=False)
    print(f"\n== {test}\n{table.round(3).to_string(index=False)}")
    return table
