"""Publie le contrat de l'API (`openapi.json`) depuis le code, sans MLflow ni modèle chargé.

Le frontend en tire ses types TypeScript (`npm run api:types`) sans jamais importer de Python :
c'est la frontière entre les deux, comme entre deux dépôts. À relancer (`make openapi`) après
chaque changement de `schemas.py` ; `test_openapi_contract_up_to_date` le rappelle en CI.
"""

import json

from config.settings import OPENAPI_PATH
from src.deploying.app.main import create_app


def openapi_json() -> str:
    """Contrat OpenAPI sérialisé, tel que servi par `/openapi.json`."""
    # Le modèle n'est chargé qu'au démarrage du serveur (lifespan), pas à la création de l'app
    return json.dumps(create_app().openapi(), indent=2, ensure_ascii=False) + "\n"


def export_openapi() -> None:
    OPENAPI_PATH.write_text(openapi_json(), encoding="utf-8")
    print(f"Contrat de l'API publié : {OPENAPI_PATH}")


if __name__ == "__main__":
    export_openapi()
