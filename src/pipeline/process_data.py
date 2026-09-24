import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from config.ml_params import (
    POSITIVE_CLASS,
    RANDOM_STATE,
    SKEWED_FEATURES,
    TARGET,
    TEST_SIZE,
)
from config.setting import (
    artifacts_dir,
    learned_stats_path,
    preprocessor_path,
    split_path,
)
from src.pipeline.extract_data import load_clean_data


def split(df: pd.DataFrame):
    X = df.drop(columns=TARGET)
    y = (df[TARGET] == POSITIVE_CLASS).astype(int)
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def load_processed(name: str):
    """Recharge X_train, X_test, y_train, y_test stockés dans artifacts."""
    X_train = pd.read_csv(split_path(name, "X_train"))
    X_test = pd.read_csv(split_path(name, "X_test"))
    y_train = pd.read_csv(split_path(name, "y_train")).squeeze("columns")
    y_test = pd.read_csv(split_path(name, "y_test")).squeeze("columns")
    return X_train, X_test, y_train, y_test


def _power_scale(columns) -> ColumnTransformer:
    """Yeo-Johnson sur les variables asymétriques, puis standardisation."""
    skewed = [col for col in SKEWED_FEATURES if col in columns]
    others = [col for col in columns if col not in skewed]
    return ColumnTransformer(
        [
            (
                "skewed",
                Pipeline(
                    [("power", PowerTransformer()), ("scale", StandardScaler())]
                ),
                skewed,
            ),
            ("others", StandardScaler(), others),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_preprocessor(name: str, columns):
    """Préprocesseur (non ajusté) du modèle, None si aucune transformation."""
    if name == "elastic":
        return _power_scale(columns)
    if name == "svm":
        return StandardScaler().set_output(transform="pandas")
    if name == "trees":
        return None
    raise ValueError(f"Modèle inconnu : {name}")


def learned_stats(transformer, columns=None) -> dict:
    """Paramètres appris à l'ajustement (moyennes, écarts-types, lambdas...)."""
    if isinstance(transformer, ColumnTransformer):
        return {
            label: learned_stats(fitted, cols)
            for label, fitted, cols in transformer.transformers_
            if label != "remainder"
        }
    if isinstance(transformer, Pipeline):
        return {
            label: learned_stats(step, columns) for label, step in transformer.steps
        }

    cols = list(columns) if columns is not None else list(transformer.feature_names_in_)
    stats = {}
    for attr in ("mean_", "scale_", "var_", "lambdas_"):
        if hasattr(transformer, attr):
            stats[attr] = dict(zip(cols, getattr(transformer, attr).tolist()))
    return stats


def _save(name: str, X_train, X_test, y_train, y_test) -> None:
    artifacts_dir(name).mkdir(parents=True, exist_ok=True)

    preprocessor = make_preprocessor(name, X_train.columns)
    if preprocessor is not None:
        columns = list(X_train.columns)
        preprocessor.fit(X_train)
        X_train = preprocessor.transform(X_train)[columns]
        X_test = preprocessor.transform(X_test)[columns]
        joblib.dump(preprocessor, preprocessor_path(name))
        learned_stats_path(name).write_text(
            json.dumps(learned_stats(preprocessor), indent=2)
        )

    X_train.reset_index(drop=True).to_csv(split_path(name, "X_train"), index=False)
    X_test.reset_index(drop=True).to_csv(split_path(name, "X_test"), index=False)
    y_train.reset_index(drop=True).to_csv(split_path(name, "y_train"), index=False)
    y_test.reset_index(drop=True).to_csv(split_path(name, "y_test"), index=False)


def process_elastic(df: pd.DataFrame) -> None:
    """Régression logistique elastic-net : Yeo-Johnson + standardisation."""
    _save("elastic", *split(df))


def process_svm(df: pd.DataFrame) -> None:
    """SVM RBF : standardisation."""
    _save("svm", *split(df))


def process_trees(df: pd.DataFrame) -> None:
    """Arbres : aucune transformation, insensibles à l'échelle."""
    _save("trees", *split(df))


def main() -> None:
    df = load_clean_data()
    process_elastic(df)
    process_svm(df)
    process_trees(df)


if __name__ == "__main__":
    main()
