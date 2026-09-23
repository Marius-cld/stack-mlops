import pandas as pd

from config.ml_params import CLASSES, MIN_CLASS_SHARE, MIN_ROWS, TARGET
from config.setting import CLEAN_FILE_PATH, RAW_FILE_PATH


def extract_data() -> pd.DataFrame:
    """Lit le csv brut et retourne un DataFrame propre avec les bons types."""
    df = pd.read_csv(RAW_FILE_PATH)
    df.columns = df.columns.str.strip()
    df[TARGET] = df[TARGET].str.strip()

    features = [col for col in df.columns if col != TARGET]
    df[features] = df[features].apply(pd.to_numeric, errors="raise").astype("float64")
    df[TARGET] = pd.Categorical(df[TARGET], categories=CLASSES)

    if df[TARGET].isna().any():
        raise ValueError(f"Classes inattendues dans {TARGET}, attendu {CLASSES}")

    return df.dropna().drop_duplicates().reset_index(drop=True)


def validate_data(df: pd.DataFrame) -> None:
    """Contrôles qualité des données : lève une erreur si un contrôle échoue."""
    if len(df) < MIN_ROWS:
        raise ValueError(f"{len(df)} lignes, minimum attendu {MIN_ROWS}")
    if df.isna().any().any():
        raise ValueError("Valeurs manquantes après nettoyage")
    share = df[TARGET].value_counts(normalize=True)
    if share.min() < MIN_CLASS_SHARE:
        raise ValueError(f"Classes trop déséquilibrées : {share.to_dict()}")


def load_clean_data() -> pd.DataFrame:
    """Recharge le csv propre en réappliquant les types."""
    df = pd.read_csv(CLEAN_FILE_PATH)
    features = [col for col in df.columns if col != TARGET]
    df[features] = df[features].astype("float64")
    df[TARGET] = pd.Categorical(df[TARGET], categories=CLASSES)
    return df


def main() -> None:
    df = extract_data()
    validate_data(df)
    CLEAN_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_FILE_PATH, index=False)
    print(f"{len(df)} lignes -> {CLEAN_FILE_PATH}")


if __name__ == "__main__":
    main()
