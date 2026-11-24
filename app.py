
import os
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# =====================================
# 1. Création de l'application FastAPI
# =====================================
app = FastAPI(title="API Churn")


# =====================================
# 2. Configuration MLflow
# =====================================
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db"
)

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/churn-model@champion"
)

mlflow.set_tracking_uri(TRACKING_URI)


# =====================================
# 3. Chargement du modèle champion
# =====================================
model = mlflow.sklearn.load_model(MODEL_URI)

print(f"Modèle chargé depuis : {MODEL_URI}")


# =====================================
# 4. Définition des données du client
# =====================================
class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


# =====================================
# 5. Endpoint GET /health
# =====================================
@app.get("/health")
def health():
    return {"status": "ok"}


# =====================================
# 6. Endpoint POST /predict
# =====================================
@app.post("/predict")
def predict(client: Client):

    # Transformer les données en DataFrame
    ligne = pd.DataFrame([client.model_dump()])

    # Encoder les variables catégorielles
    ligne = pd.get_dummies(ligne)

    # Respecter les colonnes du modèle entraîné
    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Effectuer la prédiction
    resultat = int(model.predict(ligne)[0])

    return {
        "churn": resultat,
        "message": (
            "Le client va partir"
            if resultat == 1
            else "Le client reste"
        )
    }
