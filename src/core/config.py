"""Application configuration."""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )
    
    # API Keys
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key")
    COHERE_API_KEY: str = Field(..., description="Cohere API key")
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API key")
    
    # Vector Database
    QDRANT_URL: str = Field(default="http://localhost:6333", description="Qdrant URL")
    QDRANT_API_KEY: str = Field(default="", description="Qdrant API key")
    QDRANT_COLLECTION_VISUAL: str = Field(
        default="multimodal_rag_visual",
        description="Qdrant collection for visual embeddings"
    )
    QDRANT_COLLECTION_TEXT: str = Field(
        default="multimodal_rag_text",
        description="Qdrant collection for text embeddings"
    )
    
    # PostgreSQL Database
    POSTGRES_HOST: str = Field(default="localhost", description="PostgreSQL host")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL port")
    POSTGRES_DB: str = Field(default="multimodal_rag", description="PostgreSQL database")
    POSTGRES_USER: str = Field(default="postgres", description="PostgreSQL user")
    POSTGRES_PASSWORD: str = Field(default="postgres", description="PostgreSQL password")
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    
    # Application Settings
    ENVIRONMENT: str = Field(default="development", description="Environment")
    LOG_LEVEL: str = Field(default="INFO", description="Log level")
    MAX_UPLOAD_SIZE_MB: int = Field(default=50, description="Max upload size in MB")
    CHUNK_SIZE: int = Field(default=512, description="Chunk size for text splitting")
    CHUNK_OVERLAP: int = Field(default=50, description="Chunk overlap")
    
    # Model Settings
    COLPALI_MODEL: str = Field(
        default="vidore/colpali-v1.2",
        description="ColPALI model name"
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-large",
        description="Text embedding model"
    )
    LLM_MODEL: str = Field(default="gpt-4-turbo", description="LLM model")
    VISION_MODEL: str = Field(
        default="gpt-4-vision-preview",
        description="Vision model"
    )
    YOLO_MODEL: str = Field(default="doclayout_yolo", description="YOLO model")
    
    # Performance
    BATCH_SIZE: int = Field(default=8, description="Batch size for processing")
    MAX_WORKERS: int = Field(default=4, description="Max worker threads")
    CACHE_TTL: int = Field(default=3600, description="Cache TTL in seconds")
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS origins"
    )
    
    @property
    def database_url(self) -> str:
        """Get PostgreSQL database URL."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    @property
    def async_database_url(self) -> str:
        """Get async PostgreSQL database URL."""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()
