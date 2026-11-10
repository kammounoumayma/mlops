
# ============================================================
# TP4 - MLflow Model Registry
# Gestion des versions et déplacement de l'alias champion
# ============================================================

# ===== 1. Importer les bibliothèques =====

import mlflow
from mlflow import MlflowClient


# ===== 2. Se connecter à la base MLflow =====

# Utiliser la même base de données que le TP3
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Initialiser le client MLflow
client = MlflowClient()


# ===== 3. Retrouver le run "essai-2" =====

# Rechercher le modèle le plus faible du TP3
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'essai-2'"
)

# Récupérer l'identifiant du run
run_id = runs.iloc[0]["run_id"]


# ===== 4. Enregistrer une nouvelle version =====

# Enregistrer le modèle du run essai-2
# Cela crée une nouvelle version de churn-model

v = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-model"
)

print("Nouvelle version créée :", v.version)


# ===== 5. Vérifier le champion actuel =====

# Récupérer la version associée à l'alias champion
actuel = client.get_model_version_by_alias(
    "churn-model",
    "champion"
)

print("Champion actuel : version", actuel.version)


# ===== 6. Déplacer l'alias champion =====

# Associer temporairement champion à la nouvelle version
client.set_registered_model_alias(
    "churn-model",
    "champion",
    v.version
)

print(
    "Après déplacement : champion -> version",
    client.get_model_version_by_alias(
        "churn-model",
        "champion"
    ).version
)


# ===== 7. Restaurer le champion initial =====

# Revenir à la version précédente
client.set_registered_model_alias(
    "churn-model",
    "champion",
    actuel.version
)

print(
    "Après retour : champion -> version",
    client.get_model_version_by_alias(
        "churn-model",
        "champion"
    ).version
)
