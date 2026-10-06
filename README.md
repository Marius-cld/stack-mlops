# stack-ml

Projet de démonstration d'une **stack MLOps locale** de bout en bout, appliquée à un problème de
classification binaire en chimie : prédire si une petite molécule a une énergie de liaison
élevée (`High_BFE`) ou faible (`Low_BFE`) sur la protéine **Sirtuin 6**.

Le sujet ML est volontairement modeste (petit jeu tabulaire). L'intérêt du dépôt est la
**chaîne industrielle** autour du modèle : données validées, entraînement reproductible, tests
de robustesse, registre de modèles, orchestration, monitoring, promotion contrôlée, API de
serving et interface web.

![Interface web : prédiction d'une molécule par la version production du modèle](image/README/visuel-frontend.png)

*Le frontend interroge l'API de serving, qui sert la version du registre MLflow pointée par
l'alias `production`. Chaque résultat indique la version qui l'a produit.*

## Culture MLOps : principes appliqués

| Principe | Comment il est appliqué ici |
|---|---|
| **Reproductibilité** | Seed unique (`RANDOM_STATE = 42`) partout (split, CV, bootstrap, bruit), version de Python figée (`.python-version`), dépendances figées à l'identique dans [requirements.txt](requirements.txt), l'image Airflow, l'image Evidently et l'image de l'API. Seules les données brutes sont versionnées. Tout le reste se régénère, au bit près, depuis un clone neuf (voir [Reproduire le projet](#reproduire-le-projet)). |
| **Configuration centralisée** | Le code Python lit ses chemins et ses ports dans [config/settings.py](config/settings.py), ses paramètres ML et ses seuils qualité dans [config/ml_params.py](config/ml_params.py). Les fichiers Docker et le frontend, qui n'importent pas ce code, reprennent les ports en dur. |
| **Séparation logique métier / orchestration** | Toute la logique vit dans `src/`, le DAG Airflow ne fait qu'enchaîner des appels. Chaque module est exécutable seul (bouton play ou `python3 -m ...`). |
| **Validation des données** | `extract_data` force les types, refuse une classe inconnue et retire les lignes incomplètes et les doublons. `validate_data` bloque ensuite le pipeline s'il reste moins de 50 lignes (`MIN_ROWS`), des NaN, ou une classe sous 30 % du jeu (`MIN_CLASS_SHARE`). |
| **Pas de fuite de données** | Le préprocesseur est ajusté uniquement sur le train (`learned_stats.json`). Le modèle final est un `Pipeline` (préprocesseur + estimateur) unique et versionné. |
| **Évaluation rigoureuse** | Optimisation par CV répétée (5 plis × 3 répétitions, ROC AUC), benchmark sur un jeu de test tenu à l'écart, puis tests de robustesse (voir plus bas). |
| **Contrôle qualité** | Dans le DAG, un modèle n'atteint le registre que si son AUC test **et** son AUC moyen en CV répétée atteignent `STAGING_MIN_ROC_AUC` (0,85). |
| **Registre et traçabilité** | Chaque modèle est logué dans MLflow (paramètres, métriques, dataset d'entraînement, signature) et enregistré comme nouvelle version de `sirtuin6-<modèle>`. Dans le DAG, la version finale est reliée à sa version évaluée (`candidate_version`) et à son rapport Evidently (`evidently_snapshot`). |
| **Promotion progressive** | `candidate` (modèle évalué sur le test) → `staging` (modèle réentraîné sur tout le dataset) → `production`. Le DAG ne pose **jamais** l'alias `production` : c'est une décision humaine, prise après revue dans MLflow. |
| **Smoke test consommateur** | Après la mise en staging, le modèle est rechargé via `models:/sirtuin6-elastic@staging` comme le ferait un client, et doit prédire sur des données réelles. |
| **Serving traçable** | L'API sert la version du registre pointée par un alias, `production` par défaut (`MODEL_ALIAS` pour un autre), et chaque réponse indique la version qui l'a produite. Elle refuse de démarrer si le modèle n'attend pas exactement les descripteurs de son schéma d'entrée ou ne fournit pas de probabilités. |
| **Contrat d'interface** | Le contrat OpenAPI de l'API est publié dans le dépôt. Le frontend en génère ses types, et la CI échoue si le contrat ou les types ne sont plus à jour. |
| **Monitoring des données** | Evidently produit des rapports de data summary et de data drift, consultables dans une UI dédiée. |
| **Infra as code** | Chaque service (MLflow, Evidently, Airflow, API) est décrit par un `docker-compose.yml` versionné. |
| **Secrets hors du dépôt** | Les secrets Airflow vivent dans `services/airflow/.env`, créé depuis `.env.example`. Il est ignoré par git et exclu du contexte de build de l'image Airflow. |

## Vue d'ensemble

Le projet se déroule en deux temps.

**1. Lab (manuel, script par script)** : comparer trois modèles et choisir le champion.

```
 SIRTUIN6.csv ──► extract + validate ──► process (par modèle) ──► optimize (CV répétée)
                                                                          │
     tests de robustesse ◄── benchmark (jeu de test) ◄── build ◄──────────┘
             │
             ▼
     champion : elastic
```

**2. Production du champion (Airflow, DAG `sirtuin6_elastic_staging`)** : chaque lundi à 3 h.

```
 extract + validate ──► process (split) ──► optimize ──► build ──► evaluate (test)
                                                                        │
                                                       contrôle qualité (AUC ≥ 0,85)
                                                                        │
                                                MLflow : nouvelle version, alias `candidate`
                                                                        │
                                                 Evidently : rapport train vs test (tagué vN)
                                                                        │
                                          réentraînement sur l'intégralité du dataset (train_final)
                                                                        │
                                                 MLflow : nouvelle version, alias `staging`
                                                                        │
                                                                   smoke test
                                                                        │
                                           revue humaine ──► alias `production` (manuel)
                                                                        │
                                                 API FastAPI : sert la version `production`
                                                                        │
                                                    Frontend React : interroge l'API
```

Les trois modèles comparés :

| Nom | Estimateur | Préprocessing |
|---|---|---|
| `elastic` | Régression logistique elastic-net (saga) | Yeo-Johnson sur les variables asymétriques, puis standardisation |
| `svm` | SVM à noyau RBF | Standardisation |
| `trees` | Random Forest (300 arbres) | Aucun |

Résultats du benchmark sur le jeu de test (`data/artifacts/benchmark.csv`, généré par
`src.lab.benchmarking`) :

| Modèle | ROC AUC | Accuracy | F1 |
|---|---|---|---|
| elastic | 0,94 | 0,90 | 0,89 |
| svm | 0,93 | 0,90 | 0,89 |
| trees | 0,91 | 0,80 | 0,78 |

> Le jeu de test est petit (20 lignes) : ces chiffres sont indicatifs, d'où les tests de
> robustesse ci-dessous.

## Structure du dépôt

```
config/                     Chemins, ports et paramètres ML lus par le code Python
data/                       Seul raw/ est versionné, le reste est régénéré par les scripts
  raw/                      Données brutes (SIRTUIN6.csv)
  processed/                Données nettoyées et validées
  artifacts/<modèle>/       Splits, préprocesseur ajusté, meilleurs hyperparamètres
  artifacts/robustness/     Résultats des tests de robustesse
  artifacts/benchmark.csv   Comparaison des modèles sur le test
  models/                   Modèles sérialisés (joblib)
  figures/                  Graphiques de l'analyse descriptive
src/
  notebooks/                Analyse exploratoire (notebook)
  pipeline/                 extract_data, process_data, build_models, train_final
  lab/                      hyperparameters_optimizations, benchmarking
  robustness/               Validation croisée répétée, bootstrap, bruit gaussien,
                            sensibilité aux hyperparamètres
  monitoring/               mlflow_tracking, mlflow_staging, evidently_reports
  deploying/                API de serving FastAPI (app/), ses tests (tests/) et son contrat
                            publié (openapi.json)
frontend/                   Interface web du modèle servi (React + TypeScript + Vite), port 5173
services/
  mlflow/                   Serveur MLflow (tracking et registre), port 5001
  evidently/                UI Evidently, port 8000
  airflow/                  Airflow et DAG `sirtuin6_elastic_staging`, port 8080
  api/                      Image Docker de l'API de serving, port 8001
image/README/               Captures d'écran du README
.github/workflows/ci.yml    CI : un job par brique (API de serving, frontend)
.env                        PYTHONPATH=. pour le bouton play (VS Code)
.python-version             Version de Python du projet
Makefile                    Raccourcis : install (dépendances Python), puis run, test, openapi,
                            build, up, down pour l'API
pytest.ini                  Configuration des tests
requirements.txt            Dépendances Python figées
```

## Robustesse

Au-delà d'un simple score de test, quatre analyses ([src/robustness/](src/robustness/)) mesurent
la fiabilité de chaque modèle sur l'ensemble du dataset, avec ses meilleurs hyperparamètres.

| Analyse | Protocole | Métriques |
|---|---|---|
| **Validation croisée répétée** | 5 plis × 10 répétitions | ROC AUC, accuracy, balanced accuracy, F1 : moyenne, écart-type, intervalle à 95 % |
| **Bootstrap** | 200 tirages avec remise, évaluation sur les lignes hors tirage (out-of-bag) | ROC AUC, accuracy : moyenne, écart-type, intervalle à 95 % |
| **Bruit gaussien** | 5 plis × 3 répétitions, bruit ajouté au pli de test (de 0 à 100 % de l'écart-type de chaque variable sur le train) | ROC AUC, accuracy par niveau de bruit : moyenne, écart-type |
| **Sensibilité aux hyperparamètres** | Un hyperparamètre varie à la fois sur sa grille, les autres restent à l'optimum, 5 plis × 3 répétitions | ROC AUC : moyenne, écart-type |

La dernière analyse répond à une question simple : la performance dépend-elle fortement du
réglage retenu ?

## Reproduire le projet

Deux circuits mènent au même résultat : lancer les scripts un par un depuis l'IDE, ou passer par
le DAG Airflow. Ils partagent les mêmes dossiers `data/`, le même registre MLflow et le même
workspace Evidently.

### Prérequis

| Outil | Version | Nécessaire pour |
|---|---|---|
| Python | 3.13 (`.python-version`) | Scripts du lab et du monitoring, API en local, tests |
| Docker (Docker Desktop sur macOS) | Démarré, avec au moins 4 Go de mémoire pour Airflow | MLflow, Evidently, Airflow, API conteneurisée |
| Node.js et npm | 24 (`frontend/.nvmrc`) | Frontend |
| VS Code | Recommandé | Bouton play, débogueur, notebook |
| make | Fourni avec les outils en ligne de commande de macOS | Raccourcis de l'API (`Makefile`) |

> Docker n'est pas nécessaire pour le lab. Il le devient dès qu'un script parle à MLflow ou à
> Evidently : le conteneur du service doit tourner, sinon le script échoue à la connexion.

### Installation

Le projet n'utilise pas d'environnement virtuel : les dépendances s'installent dans
l'interpréteur Python 3.13 de la machine.

```bash
git clone https://github.com/Marius-cld/stack-mlops.git
cd stack-mlops
make install          # python3 -m pip install -r requirements.txt
```

Si `python3` ne pointe pas sur la version 3.13, préciser l'interpréteur :
`make install PYTHON=python3.13` (même option pour `make run`, `make test` et `make openapi`).

Dans VS Code, sélectionner l'interpréteur Python 3.13 (commande `Python: Select Interpreter`).
Le fichier `.env` racine (`PYTHONPATH=.`) et `.vscode/settings.json` rendent `config` et `src`
importables : chaque script se lance directement avec le **bouton play**. En ligne de commande,
lancer depuis la racine avec `python3 -m <module>`.

### Services et ports

| Service | Démarrage (depuis la racine) | Adresse | Requis par |
|---|---|---|---|
| MLflow (tracking et registre) | `docker compose -f services/mlflow/docker-compose.yml up -d` | http://localhost:5001 | `mlflow_tracking`, `mlflow_staging`, DAG, API |
| Evidently (UI des rapports) | `docker compose -f services/evidently/docker-compose.yml up -d --build` | http://localhost:8000 | `evidently_reports`, DAG |
| Airflow | `cd services/airflow && docker compose up -d --build` | http://localhost:8080 | Circuit DAG |
| API de serving | `make up` (conteneur) ou `make run` (local) | http://localhost:8001/docs | Frontend |
| Frontend | `cd frontend && npm run dev` | http://localhost:5173 | |

Le port 5001 évite le 5000, pris par AirPlay sur macOS, et le 8001 évite le 8000 d'Evidently.
Pour arrêter un service : `docker compose -f <fichier> down`, ou `make down` pour l'API.

### Deux circuits possibles

| | Circuit IDE | Circuit DAG |
|---|---|---|
| Modèles traités | Les trois (`elastic`, `svm`, `trees`) | Le champion `elastic` |
| Déclenchement | À la main, script par script | Chaque lundi à 3 h, ou à la main depuis l'UI Airflow |
| Python local | Oui | Non pour le DAG lui-même (dépendances dans l'image Airflow) |
| Docker | À partir de la phase monitoring | Toujours |
| Alias posés dans MLflow | `production` sur le champion, puis `staging` | `candidate` puis `staging`, jamais `production` |
| Rôle | Explorer, comparer, choisir le champion | Réentraîner et versionner le champion, sous contrôle qualité |

Le DAG ne dépend pas du circuit IDE : depuis un clone neuf, il régénère lui-même les données
nettoyées, le split, les hyperparamètres et les modèles du champion. Les deux circuits écrivent
dans les mêmes `data/` (le dépôt est monté dans les conteneurs Airflow) : un passage du DAG
remplace les artefacts d'`elastic` produits depuis l'IDE.

### Circuit 1 : depuis l'IDE, script par script

Chaque script se lance avec le bouton play de VS Code, ou depuis la racine avec
`python3 -m <module>`. L'ordre compte : chaque étape lit ce que produisent les précédentes.

**Phase 1, lab : aucun service requis, Docker peut rester éteint.**

| # | Script | Lit | Produit |
|---|---|---|---|
| 0 | `src/notebooks/descriptive_analyses.ipynb` (optionnel) | `data/raw/SIRTUIN6.csv` | `data/figures/*.png` |
| 1 | `src.pipeline.extract_data` | `data/raw/SIRTUIN6.csv` (versionné) | `data/processed/sirtuin6_clean.csv` |
| 2 | `src.pipeline.process_data` | Étape 1 | `data/artifacts/<modèle>/` : splits, préprocesseur ajusté et `learned_stats.json` (sauf `trees`) |
| 3 | `src.lab.hyperparameters_optimizations` | Étape 2 | `data/artifacts/<modèle>/best_params.json` |
| 4 | `src.pipeline.build_models` | Étapes 2 et 3 | `data/models/<modèle>.joblib` |
| 5 | `src.lab.benchmarking` | Étapes 2 et 4 | `data/artifacts/benchmark.csv`, qui désigne le champion |
| 6 | `src.robustness.cross_validations`, `bootstraps`, `gaussian_noises`, `hyperparameters_sensitivities` | Étapes 1 et 3 | `data/artifacts/robustness/*.csv` |
| 7 | `src.pipeline.train_final` | Étapes 1, 3 et 5 | `data/models/final_<champion>.joblib` |

**Phase 2, monitoring : Docker actif, MLflow et Evidently démarrés.**

Avant cette phase, lancer Docker Desktop puis les deux conteneurs. Sans eux, les scripts
échouent à la connexion.

```bash
docker compose -f services/mlflow/docker-compose.yml up -d
docker compose -f services/evidently/docker-compose.yml up -d --build
```

| # | Script | Prérequis | Produit |
|---|---|---|---|
| 8 | `src.monitoring.mlflow_tracking` | MLflow démarré, étapes 1, 3 et 5 | Une nouvelle version par modèle dans le registre, alias `production` sur le champion |
| 9 | `src.monitoring.mlflow_staging` | MLflow démarré, étape 8 | Alias `staging` sur la dernière version du champion |
| 10 | `src.monitoring.evidently_reports` | Evidently démarré, étape 1 (à défaut, lit le CSV brut) | Rapport data summary et data drift dans le workspace Evidently |

L'étape 8 réentraîne chaque modèle sur tout le dataset, le logue avec ses métriques du benchmark
et pose directement l'alias `production` sur le champion : la lancer est en soi la décision
humaine, prise après lecture du benchmark et des tests de robustesse. Ce circuit ne passe ni par
le contrôle qualité ni par le smoke test, réservés au DAG. Chaque relance crée de nouvelles
versions dans le registre.

L'étape 10 compare deux parts du dataset tirées au hasard (70 % en référence, 30 % en
courant), alors que le DAG compare le train et le test du split.

**Phase 3, serving et interface : MLflow démarré, alias `production` posé.**

| # | Commande | Prérequis | Résultat |
|---|---|---|---|
| 11 | `make test` | Aucun. Si MLflow répond, le test d'intégration tourne aussi et exige l'alias `production` | Tests de l'API |
| 12 | `make run` (local) ou `make up` (conteneur, Docker actif) | MLflow démarré, alias `production` posé | API sur http://localhost:8001/docs |
| 13 | `cd frontend && npm ci && npm run dev` | Node 24, API démarrée | Interface sur http://localhost:5173 |

Le même parcours en ligne de commande, depuis la racine :

```bash
# Phase 1 : lab
python3 -m src.pipeline.extract_data
python3 -m src.pipeline.process_data
python3 -m src.lab.hyperparameters_optimizations
python3 -m src.pipeline.build_models
python3 -m src.lab.benchmarking
python3 -m src.robustness.cross_validations
python3 -m src.robustness.bootstraps
python3 -m src.robustness.gaussian_noises
python3 -m src.robustness.hyperparameters_sensitivities
python3 -m src.pipeline.train_final

# Phase 2 : monitoring (Docker actif)
docker compose -f services/mlflow/docker-compose.yml up -d
docker compose -f services/evidently/docker-compose.yml up -d --build
python3 -m src.monitoring.mlflow_tracking
python3 -m src.monitoring.mlflow_staging
python3 -m src.monitoring.evidently_reports

# Phase 3 : serving et interface
make test
make up
cd frontend && npm ci && npm run dev
```

Les résultats sont déterministes : un clone neuf retrouve exactement les mêmes splits,
hyperparamètres, métriques et prédictions.

### Circuit 2 : par le DAG Airflow

Prérequis : Docker actif avec au moins 4 Go de mémoire alloués, MLflow et Evidently démarrés
**avant** de lancer le DAG (ses tâches les joignent via `host.docker.internal`). Python local ne
sert qu'à générer la clé Fernet, puis à l'API si elle tourne avec `make run`.

1. Démarrer MLflow et Evidently (voir [Services et ports](#services-et-ports)).
2. Configurer et démarrer Airflow :

   ```bash
   cd services/airflow
   cp .env.example .env
   # remplir FERNET_KEY (commande de génération dans le fichier) et _AIRFLOW_WWW_USER_PASSWORD
   docker compose up -d --build
   ```

3. Ouvrir http://localhost:8080 avec les identifiants `_AIRFLOW_WWW_USER_USERNAME` et
   `_AIRFLOW_WWW_USER_PASSWORD` du `.env`.
4. Activer le DAG `sirtuin6_elastic_staging` (il est en pause à sa création), puis le
   déclencher. Il tourne ensuite chaque lundi à 3 h.
5. Suivre les tâches : extraction et validation, traitement, optimisation, construction,
   évaluation, contrôle qualité, version `candidate`, rapport Evidently, réentraînement complet,
   version `staging`, smoke test. Si le contrôle qualité échoue, les tâches suivantes sont
   sautées et rien n'atteint le registre.
6. Revoir la version `staging` dans MLflow, puis la promouvoir (voir
   [Promouvoir en production](#promouvoir-en-production)).
7. Servir le modèle et ouvrir l'interface comme à la phase 3 du circuit IDE.

Bon à savoir :

- Le dépôt est monté dans les conteneurs (`PYTHONPATH=/opt/airflow/project`). Le DAG importe
  directement `src/` et `config/`, et écrit dans les mêmes `data/`, workspace Evidently et
  registre MLflow (via `host.docker.internal:5001`) que les scripts locaux.
- Le compte admin n'est créé qu'au **premier** passage d'`airflow-init`. Pour changer ensuite le
  mot de passe : `docker exec airflow-airflow-apiserver-1 airflow users reset-password -u admin -p '<mdp>'`.
- Deux fichiers `.env` coexistent. Le `.env` racine (`PYTHONPATH=.`) est lu par VS Code
  (`python.envFile`), ne contient aucun secret et reste versionné. `services/airflow/.env` est lu
  par le `docker compose` d'Airflow (`AIRFLOW_UID`, `FERNET_KEY`, identifiant et mot de passe
  admin), contient des secrets et n'est **pas** versionné.
- Cette configuration est prévue pour le **développement local**, pas pour la production.

### Promouvoir en production

Après revue de la version `staging` dans l'UI MLflow (modèle `sirtuin6-elastic`), poser
manuellement l'alias `production` sur la version retenue, depuis l'UI ou en Python :

```python
from src.monitoring.mlflow_tracking import load_production, promote

promote("elastic", 12)               # alias production sur la version 12
model = load_production("elastic")   # models:/sirtuin6-elastic@production
```

L'API charge cette version à son démarrage : la redémarrer (`docker restart api`, ou relancer
`make run`) pour servir une version nouvellement promue. Pour essayer une version avant sa
promotion : `MODEL_ALIAS=staging make up`.

## API de serving

L'API ([src/deploying/app/](src/deploying/app/)) sert la version du registre pointée par l'alias
`production`, chargée une seule fois au démarrage. Elle refuse de démarrer si MLflow est
injoignable, si le modèle n'attend pas exactement les descripteurs de son schéma d'entrée ou s'il
ne fournit pas de probabilités : mieux vaut un service absent qu'un service qui prédit faux. La
racine `/` redirige vers la documentation interactive (`/docs`).

```bash
make up      # conteneur (services/api)           -> http://localhost:8001/docs
make run     # ou en local, rechargement à chaud  -> http://localhost:8001/docs
make test    # tests de l'API (le test d'intégration ne tourne que si MLflow répond)
make openapi # publie le contrat dans src/deploying/openapi.json
```

| Méthode | Route | Rôle |
|---|---|---|
| `GET` | `/health` | Sonde : modèle chargé et version servie |
| `GET` | `/v1/model` | Fiche du modèle : run MLflow, descripteurs attendus, hyperparamètres, métriques |
| `POST` | `/v1/predict` | Prédiction pour une molécule |
| `POST` | `/v1/predict/batch` | Prédiction pour 1 à 1 000 molécules, dans l'ordre de la requête |

```bash
curl -X POST http://localhost:8001/v1/predict -H "Content-Type: application/json" \
  -d '{"SC-5": 0.540936, "SP-6": 7.64192, "SHBd": 0.162171, "minHaaCH": 0.44527, "maxwHBa": 2.20557, "FMF": 0.467742}'
```

Réponse (probabilités arrondies) :

```json
{
  "label": "High_BFE",
  "probabilities": {"Low_BFE": 0.056, "High_BFE": 0.944},
  "model": {"name": "sirtuin6-elastic", "version": "10", "alias": "production"}
}
```

Les entrées sont validées : un descripteur manquant ou inconnu, une valeur non numérique, un NaN
ou un infini renvoient une erreur `422`, tout comme un lot vide ou de plus de 1 000 molécules.

Les tests (`make test`) remplacent le registre par un modèle léger entraîné sur
`data/raw/SIRTUIN6.csv` : ils tournent sans MLflow. Seul le test d'intégration interroge le
vrai registre, et seulement si MLflow répond.

| Variable d'environnement | Défaut | Rôle |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` en local, `http://host.docker.internal:5001` en conteneur | Registre MLflow |
| `MODEL_ALIAS` | `production` | Alias servi, par exemple `staging` pour essayer une version avant sa promotion |
| `CORS_ORIGINS` | `http://localhost:5173` | Origines navigateur autorisées (le frontend), séparées par des virgules |

Le schéma OpenAPI (`/openapi.json`) décrit tout le contrat de l'API. Il est publié depuis le
code, sans MLflow ni modèle chargé, dans [src/deploying/openapi.json](src/deploying/openapi.json)
(`make openapi`, à relancer après chaque changement de `schemas.py`). C'est le seul lien entre
l'API et le frontend. Un test vérifie que le fichier publié est bien celui que l'API sert.

## Frontend

Interface web du modèle servi ([frontend/](frontend/)), en React 19, TypeScript et Vite. Elle
n'a aucune dépendance d'exécution en dehors de React : les appels passent par `fetch`, typés par
le contrat de l'API.

### Ce que montre l'interface

- **Bandeau** : état de l'API et version servie (par exemple `sirtuin6-elastic v10 · production`).
- **Onglet « Une molécule »** : saisie des six descripteurs PaDEL, chacun avec son libellé. Deux
  molécules du dataset, une par classe observée, remplissent le formulaire en un clic. Le
  résultat affiche la classe prédite, la probabilité de chaque classe et la version du modèle
  qui l'a produit. Si une valeur change après la prédiction, le résultat est signalé comme
  périmé.
- **Onglet « Lot CSV »** : glisser-déposer ou choix d'un fichier, vérifié avant l'envoi. Une
  colonne manquante est signalée, une valeur non numérique l'est avec son numéro de ligne. Le
  résultat donne le décompte par classe, le tableau des prédictions et un export CSV nommé
  d'après la version du
  modèle. L'API accepte jusqu'à 1 000 molécules par lot.
- **Fiche « Modèle servi »** : nom, version, alias, run MLflow, classe positive, métriques,
  hyperparamètres, descripteurs attendus et lien vers la documentation de l'API.
- **API injoignable** : message explicite (API arrêtée ou origine refusée par CORS), rappel des
  commandes de démarrage et bouton « Réessayer ».

Les deux onglets restent montés : passer de l'un à l'autre ne perd ni la saisie ni les
résultats.

### Démarrer

Prérequis : Node 24 et l'API de serving démarrée sur http://localhost:8001 (`make up` ou
`make run`).

```bash
cd frontend
npm ci
npm run dev        # http://localhost:5173
```

- Pour viser une autre API, copier `.env.example` en `.env.local` (ignoré par git) et ajuster
  `VITE_API_URL`.
- Le port 5173 est imposé (`strictPort`) : c'est l'origine autorisée par défaut par le CORS de
  l'API. Si le port est pris, Vite s'arrête au lieu de basculer sur un port que l'API
  refuserait.

### Contrat avec l'API

Les types TypeScript de [frontend/src/api/schema.d.ts](frontend/src/api/schema.d.ts) sont
**générés** depuis [src/deploying/openapi.json](src/deploying/openapi.json), jamais écrits à la
main. Quand le contrat change côté API (`make openapi`), `npm run api:types` régénère les types
et `npm run build` échoue partout où le frontend n'est plus aligné : descripteur renommé, champ
de réponse retiré, classe ajoutée. Les libellés des descripteurs sont eux aussi vérifiés contre
le contrat à la compilation. En CI, le job frontend régénère les types et échoue s'ils diffèrent
de ceux du dépôt.

| Script | Rôle |
|---|---|
| `npm run dev` | Serveur de développement, rechargement à chaud |
| `npm run build` | Vérification des types puis build de production (`dist/`) |
| `npm run preview` | Sert le build de production sur le port 5173 |
| `npm run lint` | Analyse statique (Oxlint) |
| `npm run api:types` | Régénère les types depuis le contrat de l'API |

### Organisation du code

```
frontend/src/
  App.tsx              Bandeau, état de l'API, onglets
  api/client.ts        Appels à l'API, typés par le contrat
  api/schema.d.ts      Types générés depuis le contrat OpenAPI (ne pas modifier)
  components/          Formulaire, lot CSV, fiche du modèle, affichage des prédictions
  molecules.ts         Libellés des descripteurs, exemples tirés du dataset
  csv.ts               Lecture du CSV d'entrée, export des prédictions
  format.ts            Formats français des nombres et des pourcentages
  index.css            Charte graphique (couleurs, typographie, rayons, ombres)
```

La charte tient en une couleur d'accent, le bleu `#262E5E`, déclinée en mode clair et en mode
sombre. Le format du CSV, la charte et le détail des scripts sont dans
[frontend/README.md](frontend/README.md).

## Un monorepo pensé comme un multi-repo

Tout vit dans un seul dépôt pour qu'un clone suffise à tout reproduire. Mais chaque brique est
construite comme si elle avait son propre dépôt : elle parle aux autres par une adresse réseau
ou par un fichier de contrat, et la CI exécute un job par brique. Trois dépendances de code
traversent encore les frontières : Airflow monte le dépôt pour importer le code d'entraînement,
le serving lit les constantes de `config/`, et ses tests lisent les données brutes via
`src.pipeline.extract_data`.

### Les briques et leurs interfaces

| Brique | Dossiers | Consomme | Expose |
|---|---|---|---|
| Entraînement et monitoring | `config/`, `src/pipeline/`, `src/lab/`, `src/robustness/`, `src/monitoring/`, `src/notebooks/`, `data/raw/` | Données brutes, MLflow (`MLFLOW_TRACKING_URI`), Evidently (`EVIDENTLY_WORKSPACE_URL`) | Versions et alias dans le registre, rapports Evidently |
| Orchestration | `services/airflow/` | Code d'entraînement (dépôt monté en volume) | Exécutions planifiées du DAG |
| Plateforme | `services/mlflow/`, `services/evidently/` | Rien | Registre MLflow (port 5001), UI Evidently (port 8000) |
| Serving | `src/deploying/`, `services/api/`, `Makefile` | Registre MLflow (alias `production`), `config/` | API HTTP (port 8001), contrat `openapi.json` |
| Frontend | `frontend/` | URL de l'API, `openapi.json` | Interface web (port 5173) |

### Découpages possibles

**En deux dépôts : le backend ML et le frontend.** C'est possible dès aujourd'hui sans toucher
au code applicatif. Le frontend ne connaît de l'API que son URL (`VITE_API_URL`) et son contrat.
Il suffit de faire lire à sa CI le contrat publié par l'autre dépôt :
`OPENAPI_SPEC=<url ou chemin> npm run api:types`.

**En trois dépôts : le modèle, le service, l'interface.** C'est le découpage classique par
équipe. Les data scientists gardent l'entraînement, l'orchestration et la plateforme, l'équipe
backend le serving, l'équipe frontend l'interface. Le registre MLflow devient la frontière entre
le modèle et le service : l'API ne lit qu'un alias, jamais un fichier local. Le dépôt serving
embarque sa propre copie des quelques constantes de `config/` qu'il utilise (classes, classe
positive, nom du modèle, ports) et un petit jeu de données pour ses tests.

**En cinq dépôts : une brique par dépôt.** MLflow et Evidently forment une plateforme partagée,
gérée à part et réutilisable par d'autres projets. L'orchestration devient un dépôt de DAGs qui
installe le code d'entraînement comme un paquet versionné (ou le reçoit dans une image), au lieu
de monter le dépôt en volume.

### Ce qui rend la séparation possible

- Les adresses des services se règlent par variables d'environnement (`MLFLOW_TRACKING_URI`,
  `EVIDENTLY_WORKSPACE_URL`, `VITE_API_URL`), tout comme l'alias servi (`MODEL_ALIAS`) et les
  origines autorisées par l'API (`CORS_ORIGINS`).
- Chaque service a son propre `docker-compose.yml`.
- Chaque image a son `Dockerfile.dockerignore`, une liste blanche. L'image de l'API n'embarque
  que `config/` et `src/deploying/app/`. Les images Airflow et Evidently ne reçoivent que leur
  Dockerfile : ni secrets, ni logs, ni rapports dans le contexte de build.
- Le frontend n'importe jamais de Python. Ses types viennent du contrat publié.
- La CI exécute un job par brique, comme si chacune avait son dépôt.

## Dataset

Les données utilisées dans ce projet (`data/raw/SIRTUIN6.csv`) proviennent de :

Tardu, M. & RAHIM, F. (2016). Sirtuin6 Small Molecules [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C9Z.

Le jeu de données est distribué par l'UCI Machine Learning Repository sous licence
CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

## Licence

Le code source est publié sous licence MIT (voir [LICENSE](LICENSE)).
Le jeu de données reste sous sa licence d'origine (CC BY 4.0), voir ci-dessus.
