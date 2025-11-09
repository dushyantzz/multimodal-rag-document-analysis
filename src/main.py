"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import make_asgi_app

from src.core.config import settings
from src.core.logger import logger
from src.api.v1 import documents, query, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Multimodal RAG Application...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Log Level: {settings.LOG_LEVEL}")
    
    # Initialize services
    from src.services.embeddings.colpali_service import ColPALIService
    from src.services.vector_db.qdrant_service import QdrantService
    
    # Load models on startup
    try:
        colpali = ColPALIService()
        await colpali.initialize()
        logger.info("ColPALI model loaded successfully")
        
        qdrant = QdrantService()
        await qdrant.initialize_collections()
        logger.info("Qdrant collections initialized")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    logger.info("Shutting down Multimodal RAG Application...")


app = FastAPI(
    title="Multimodal RAG for Document Analysis",
    description="Production-ready RAG system for processing text, images, and tables from complex documents",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multimodal RAG Document Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
    )
