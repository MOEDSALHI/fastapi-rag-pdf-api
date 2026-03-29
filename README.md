# fastapi-rag-pdf-api

API backend FastAPI permettant d’uploader un document PDF, d’en extraire le texte, de l’indexer dans un vector store local avec FAISS, puis de poser des questions sur son contenu via une approche RAG (Retrieval-Augmented Generation).

## Objectif du projet

Ce projet a été conçu comme un démonstrateur backend moderne autour des usages LLM et RAG appliqués à des documents PDF.

L’application permet de :

- uploader un PDF texte natif
- extraire le texte du document
- découper le contenu en chunks
- générer les embeddings des chunks avec OpenAI
- indexer ces embeddings dans FAISS
- retrouver les passages les plus pertinents pour une question utilisateur
- générer une réponse contextualisée à partir du document

## Fonctionnalités

- API FastAPI structurée par routes et services
- endpoint `GET /health`
- endpoint `POST /upload` pour indexer un PDF
- endpoint `POST /ask` pour interroger le document indexé
- extraction de texte PDF avec `pypdf`
- chunking avec overlap
- génération d’embeddings OpenAI
- vector store local avec FAISS
- persistance locale de l’index et des chunks
- tests unitaires avec `pytest`

## Stack technique

- Python 3.12
- FastAPI
- Uvicorn
- Pydantic
- pydantic-settings
- OpenAI
- PyPDF
- FAISS CPU
- NumPy
- Pytest

## Architecture du projet

```bash
fastapi-rag-pdf-api/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── ask.py
│   │       ├── health.py
│   │       └── upload.py
│   ├── core/
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── logging.py
│   ├── schemas/
│   │   ├── ask.py
│   │   └── upload.py
│   ├── services/
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── llm.py
│   │   ├── pdf_extractor.py
│   │   └── vector_store.py
│   └── main.py
├── data/
│   └── vector_store/
├── tests/
├── .env.example
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
````

## Installation locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/<your-username>/fastapi-rag-pdf-api.git
cd fastapi-rag-pdf-api
```

### 2. Créer un environnement virtuel

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Créer le fichier `.env`

```bash
cp .env.example .env
```

Puis compléter avec une vraie clé OpenAI.

## Variables d’environnement

Exemple de fichier `.env` :

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_TIMEOUT=30
APP_NAME=FastAPI RAG PDF API
APP_ENV=dev
```

## Lancer l’application en local

```bash
uvicorn app.main:app --reload
```

Application disponible sur :

* API : `http://127.0.0.1:8000`
* Swagger UI : `http://127.0.0.1:8000/docs`

## Lancer les tests

```bash
pytest
```

## Utilisation de l’API

### Vérifier l’état de l’API

```bash
curl http://127.0.0.1:8000/health
```

Réponse attendue :

```json
{"status":"ok"}
```

### Indexer un PDF

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.pdf"
```

Exemple de réponse :

```json
{
  "filename": "sample.pdf",
  "content_type": "application/pdf",
  "page_count": 3,
  "extracted_text_length": 5421,
  "chunk_count": 8,
  "embedding_count": 8,
  "status": "indexed"
}
```

### Poser une question sur le document

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the monthly rent mentioned in the document?",
    "top_k": 3
  }'
```

Exemple de réponse :

```json
{
  "question": "What is the monthly rent mentioned in the document?",
  "answer": "The monthly rent mentioned in the document is 950 euros.",
  "retrieved_chunks": [
    "The monthly rent is set at 950 euros...",
    "The lease term is 3 years...",
    "The security deposit is two months..."
  ]
}
```

## Lancer le projet avec Docker

### Build de l’image

```bash
docker build -t fastapi-rag-pdf-api .
```

### Lancer le conteneur

```bash
docker run --rm -p 8000:8000 --env-file .env fastapi-rag-pdf-api
```

## Limites actuelles

Cette version du projet est volontairement simple et orientée démonstration.

Limitations actuelles :

* un seul document indexé à la fois
* chaque nouvel upload remplace l’index précédent
* pas de support OCR pour les PDF scannés
* pas de gestion multi-documents
* pas encore de métadonnées avancées sur les chunks
* pas encore de citations par page ou score de similarité

## Axes d’amélioration possibles

* support multi-documents
* ajout de métadonnées par chunk
* support OCR pour PDF image
* stockage persistant plus avancé
* ajout de scores de similarité
* ajout d’authentification
* amélioration du prompt RAG
* observabilité et logs enrichis

## Pourquoi ce projet est intéressant

Ce projet démontre plusieurs compétences backend et IA utiles en contexte professionnel :

* structuration d’une API FastAPI
* intégration de services LLM
* pipeline RAG de bout en bout
* traitement documentaire PDF
* recherche sémantique avec FAISS
* qualité de code avec tests automatisés

## Statut

Version MVP fonctionnelle, construite étape par étape dans une logique de progression backend + GenAI.
