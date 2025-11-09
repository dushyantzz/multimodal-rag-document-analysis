"""Celery configuration for async task processing."""

from celery import Celery
from src.core.config import settings
from src.core.logger import logger

celery_app = Celery(
    "multimodal_rag",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "src.services.document_processor.tasks",
        "src.services.embeddings.tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
)

logger.info("Celery app configured successfully")
