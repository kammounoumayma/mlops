
# ===== 1. Importer les bibliothèques =====

from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ===== 2. Créer l'application FastAPI =====

app = FastAPI(title="API Churn")


# ===== 3. Charger le modèle champion =====

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)


# ===== 4. Définir les données du client =====

class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


# ===== 5. Route GET /health =====

@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 6. Route POST /predict =====

@app.post("/predict")
def predict(client: Client):

    # Convertir le client en DataFrame
    ligne = pd.DataFrame([client.model_dump()])

    # Encoder l'abonnement
    ligne = pd.get_dummies(ligne)

    # Respecter les colonnes attendues par le modèle
    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Effectuer la prédiction
    resultat = int(model.predict(ligne)[0])

    # Retourner le résultat
    return {
        "churn": resultat,
        "message": (
            "Le client va partir"
            if resultat == 1
            else "Le client reste"
        )
    }
