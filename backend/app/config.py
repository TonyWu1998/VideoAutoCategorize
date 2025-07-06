"""
Configuration management for the Media Semantic Search application.
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # =============================================================================
    # DATABASE CONFIGURATION
    # =============================================================================
    
    CHROMA_DB_PATH: str = "./data/chroma_db"
    
    # =============================================================================
    # OLLAMA CONFIGURATION
    # =============================================================================
    
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3:4b"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_KEEP_ALIVE: int = 300
    
    # =============================================================================
    # API CONFIGURATION
    # =============================================================================
    
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000", 
        "http://localhost:5173", 
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:5173"
    ]
    API_V1_STR: str = "/api/v1"
    MAX_REQUEST_SIZE_MB: int = 100
    
    # =============================================================================
    # MEDIA PROCESSING CONFIGURATION
    # =============================================================================
    
    SUPPORTED_IMAGE_FORMATS: List[str] = [
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"
    ]
    SUPPORTED_VIDEO_FORMATS: List[str] = [
        ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".webm"
    ]
    MAX_FILE_SIZE_MB: int = 500
    MAX_IMAGE_DIMENSION: int = 1024
    IMAGE_QUALITY: int = 85
    VIDEO_FRAME_INTERVAL: int = 30
    MAX_VIDEO_FRAMES: int = 10
    
    # =============================================================================
    # INDEXING CONFIGURATION
    # =============================================================================
    
    BATCH_SIZE: int = 10
    MAX_CONCURRENT_PROCESSING: int = 4
    ENABLE_FILE_WATCHER: bool = True
    WATCH_INTERVAL_SECONDS: int = 5
    INDEXING_QUEUE_SIZE: int = 1000
    RETRY_ATTEMPTS: int = 3
    RETRY_DELAY_SECONDS: int = 5
    
    # =============================================================================
    # SEARCH CONFIGURATION
    # =============================================================================
    
    DEFAULT_SEARCH_LIMIT: int = 20
    MAX_SEARCH_LIMIT: int = 100
    MIN_SIMILARITY_THRESHOLD: float = 0.3
    ENABLE_SEARCH_CACHE: bool = True
    SEARCH_CACHE_TTL_SECONDS: int = 300
    
    # =============================================================================
    # LOGGING CONFIGURATION
    # =============================================================================
    
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: Optional[str] = "./logs/app.log"
    LOG_MAX_SIZE_MB: int = 10
    LOG_BACKUP_COUNT: int = 5
    
    # =============================================================================
    # SECURITY CONFIGURATION
    # =============================================================================
    
    SECRET_KEY: str = "your-secret-key-here-change-this-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_MINUTES: int = 15
    
    # =============================================================================
    # DEVELOPMENT CONFIGURATION
    # =============================================================================
    
    DEBUG: bool = False
    RELOAD: bool = False
    RESET_DB_ON_STARTUP: bool = False
    ENABLE_DB_LOGGING: bool = False
    FRONTEND_DEV_SERVER: str = "http://localhost:5173"
    
    # =============================================================================
    # PERFORMANCE TUNING
    # =============================================================================
    
    MAX_MEMORY_USAGE_MB: int = 4096
    GARBAGE_COLLECTION_THRESHOLD: int = 100
    WORKER_THREADS: int = 4
    ASYNC_POOL_SIZE: int = 10
    VECTOR_DB_BATCH_SIZE: int = 100
    ENABLE_DB_COMPRESSION: bool = True
    
    # =============================================================================
    # MONITORING AND METRICS
    # =============================================================================
    
    ENABLE_METRICS: bool = False
    METRICS_PORT: int = 9090
    HEALTH_CHECK_INTERVAL: int = 30
    ENABLE_DETAILED_HEALTH_CHECK: bool = False
    
    # =============================================================================
    # BACKUP AND RECOVERY
    # =============================================================================
    
    ENABLE_AUTO_BACKUP: bool = False
    BACKUP_INTERVAL_HOURS: int = 24
    BACKUP_RETENTION_DAYS: int = 7
    BACKUP_PATH: str = "./backups"
    
    # =============================================================================
    # EXPERIMENTAL FEATURES
    # =============================================================================
    
    ENABLE_EXPERIMENTAL_FEATURES: bool = False
    ENABLE_ADVANCED_ANALYSIS: bool = False
    USE_GPU_ACCELERATION: bool = False
    ENABLE_MULTILINGUAL: bool = False
    DEFAULT_LANGUAGE: str = "en"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        directories = [
            Path(self.CHROMA_DB_PATH).parent,
            Path(self.LOG_FILE_PATH).parent if self.LOG_FILE_PATH else None,
            Path(self.BACKUP_PATH),
            Path("./data"),
            Path("./logs"),
        ]
        
        for directory in directories:
            if directory and not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def all_supported_formats(self) -> List[str]:
        """Get all supported media formats."""
        return self.SUPPORTED_IMAGE_FORMATS + self.SUPPORTED_VIDEO_FORMATS
    
    @property
    def database_url(self) -> str:
        """Get the database URL for ChromaDB."""
        return f"sqlite:///{self.CHROMA_DB_PATH}/metadata.db"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings instance."""
    return settings
