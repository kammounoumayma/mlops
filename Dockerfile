
# 1. Utiliser Python 3.11
FROM python:3.11-slim

# 2. Définir le dossier de travail
WORKDIR /app

# 3. Copier les dépendances
COPY requirements.txt .

# 4. Installer les bibliothèques
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier l'application FastAPI
COPY app.py .

# 6. Copier le modèle exporté
COPY model/ model/

# 7. Utiliser le modèle local
ENV MODEL_URI=model

# 8. Exposer le port de l'API
EXPOSE 8000

# 9. Démarrer FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
