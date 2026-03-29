# fastapi-rag-pdf-api

API backend FastAPI permettant d’uploader un document PDF, d’en extraire le texte, de l’indexer dans un vector store local avec FAISS, puis de poser des questions sur son contenu via une approche RAG (Retrieval-Augmented Generation).

## Objectif du projet

Ce projet a été construit pour monter en compétence de manière concrète sur plusieurs briques backend et GenAI modernes :

- FastAPI
- traitement de documents PDF
- chunking
- embeddings OpenAI
- recherche vectorielle avec FAISS
- pipeline RAG de bout en bout
- tests backend
- dockerisation

L’objectif n’était pas seulement de faire un POC, mais de comprendre et implémenter proprement chaque étape du pipeline.

## Ce que le projet permet de faire

- uploader un PDF texte natif
- extraire le texte du document
- découper le contenu en chunks avec overlap
- générer les embeddings des chunks
- indexer ces embeddings dans FAISS
- poser une question sur le document
- retrouver les passages les plus pertinents
- générer une réponse contextualisée via OpenAI

## Compétences travaillées

Ce repo m’a permis de pratiquer concrètement :

- conception d’API backend avec FastAPI
- structuration d’un projet Python par couches (`routes`, `schemas`, `services`)
- validation avec Pydantic
- gestion d’erreurs métier
- tests unitaires et tests de routes avec `pytest`
- intégration des APIs OpenAI
- implémentation d’un pipeline RAG simple et compréhensible
- recherche sémantique locale avec FAISS
- conteneurisation avec Docker

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
- Ruff
- Docker

## Architecture

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
├── pyproject.toml
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

### 4. Configurer les variables d’environnement

```bash
cp .env.example .env
```

Exemple :

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_TIMEOUT=30
APP_NAME=FastAPI RAG PDF API
APP_ENV=dev
```

## Lancer l’application

```bash
uvicorn app.main:app --reload
```

Swagger UI :

```bash
http://127.0.0.1:8000/docs
```

## Lancer les tests

```bash
pytest
```

## Vérifier la qualité du code

```bash
ruff check . --fix
ruff format .
```

## Exemples d’utilisation

### Healthcheck

```bash
curl http://127.0.0.1:8000/health
```

### Upload et indexation d’un PDF

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.pdf"
```

### Question sur le document

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the monthly rent mentioned in the document?",
    "top_k": 3
  }'
```

## Lancer avec Docker

### Build

```bash
docker build -t fastapi-rag-pdf-api .
```

### Run

```bash
docker run --rm -p 8000:8000 --env-file .env fastapi-rag-pdf-api
```

### Run avec persistance locale de l’index

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  fastapi-rag-pdf-api
```

## Limites actuelles

* un seul document indexé à la fois
* chaque nouvel upload remplace l’index précédent
* pas de support OCR pour les PDF scannés
* pas encore de métadonnées avancées par chunk
* pas encore de citations par page
* pas encore de support multi-documents