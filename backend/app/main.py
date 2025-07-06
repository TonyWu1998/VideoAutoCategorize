"""
Main FastAPI application for Local Media Semantic Search.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import logging
import sys
from pathlib import Path

# Import application modules
from app.config import settings
from app.api import search, indexing, media, health, config_api
from app.database.vector_db import VectorDatabase

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE_PATH) if settings.LOG_FILE_PATH else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Media Semantic Search API",
    description="API for semantic search of local media files using AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Media Semantic Search API...")
    
    try:
        # Initialize database
        vector_db = VectorDatabase()
        logger.info("Vector database initialized successfully")
        
        # Reset database if configured
        if settings.RESET_DB_ON_STARTUP:
            logger.warning("Resetting database on startup...")
            # Add database reset logic here if needed
        
        # Create required directories
        Path(settings.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
        if settings.LOG_FILE_PATH:
            Path(settings.LOG_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Media Semantic Search API...")
    # Add cleanup logic here if needed
    logger.info("Application shutdown completed")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Include API routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(indexing.router, prefix="/api/index", tags=["indexing"])
app.include_router(media.router, prefix="/api/media", tags=["media"])
app.include_router(config_api.router, prefix="/api/config", tags=["config"])

# Serve static files (media files)
try:
    # Create media directory if it doesn't exist
    media_dir = Path("./data/media")
    media_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/media", StaticFiles(directory=str(media_dir)), name="media")
except Exception as e:
    logger.warning(f"Could not mount media directory: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Media Semantic Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "name": "Media Semantic Search API",
        "version": "1.0.0",
        "description": "API for semantic search of local media files using AI",
        "endpoints": {
            "health": "/api/health",
            "search": "/api/search",
            "indexing": "/api/index",
            "media": "/api/media"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )
