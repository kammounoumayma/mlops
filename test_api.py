
# ============================================================
# TP4 - MLflow + FastAPI
# Test des routes /health et /predict avec requests
# ============================================================

# ===== 1. Importer les bibliothèques =====

import requests


# ===== 2. Définir l'adresse de l'API =====

# Utiliser le port sur lequel FastAPI fonctionne
URL = "http://localhost:8001"


# ===== 3. Vérifier que l'API fonctionne =====

response = requests.get(URL + "/health")

print("GET /health ->", response.json())


# ===== 4. Tester un premier client =====

# Client avec faible consommation et faible ancienneté
client = {
    "age": 35,
    "salaire": 2500,
    "abonnement": "Standard",
    "anciennete": 1,
    "consommation": 10
}

# Envoyer une requête POST
r = requests.post(
    URL + "/predict",
    json=client
)

print(
    "POST /predict (client 1) ->",
    r.status_code,
    r.json()
)


# ===== 5. Tester un deuxième client =====

# Client fidèle avec forte consommation
# et forte ancienneté
client2 = {
    "age": 50,
    "salaire": 4000,
    "abonnement": "Premium",
    "anciennete": 10,
    "consommation": 80
}

# Envoyer une requête POST
r = requests.post(
    URL + "/predict",
    json=client2
)

print(
    "POST /predict (client 2) ->",
    r.status_code,
    r.json()
)


# ===== 6. Tester une erreur volontaire =====

# L'âge doit être un entier, pas du texte
mauvais = dict(client, age="abc")

# Envoyer une requête avec des données invalides
r = requests.post(
    URL + "/predict",
    json=mauvais
)

print(
    "POST /predict (age='abc') ->",
    r.status_code
)

print(
    "Message :",
    r.json()["detail"][0]["msg"]
)
