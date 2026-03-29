from fastapi import FastAPI

from app.api.routes.ask import router as ask_router
from app.api.routes.health import router as health_router
from app.api.routes.upload import router as upload_router
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="API FastAPI minimale pour le projet RAG PDF.",
)

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(ask_router)
