# Image de serving : de quoi charger le modèle depuis le registre MLflow et l'exposer, rien de plus.
# Construite depuis la racine du dépôt (voir docker-compose.yml) : seuls les fichiers autorisés
# par Dockerfile.dockerignore sont envoyés au build.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false

# Versions identiques à l'entraînement (requirements.txt) : le pipeline sklearn est désérialisé
# tel quel. mlflow-skinny suffit pour lire le registre (ni serveur, ni UI).
RUN pip install --no-cache-dir \
    numpy==2.4.6 \
    pandas==2.3.3 \
    scipy==1.17.1 \
    scikit-learn==1.9.0 \
    joblib==1.5.3 \
    cloudpickle==3.1.2 \
    mlflow-skinny==3.13.0 \
    fastapi==0.136.3 \
    starlette==1.2.1 \
    uvicorn==0.49.0 \
    pydantic==2.13.4

RUN useradd --create-home --uid 10001 api
WORKDIR /app
COPY config/ config/
COPY src/__init__.py src/
COPY src/deploying/app/ src/deploying/app/
USER api

EXPOSE 8000

# Sain dès que l'API répond : elle ne démarre qu'avec un modèle chargé et conforme au contrat
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "src.deploying.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
