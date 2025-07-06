"""
Health check API endpoints.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import time
import psutil
import logging
from pathlib import Path

from app.models.common import HealthResponse
from app.config import settings
from app.database.vector_db import VectorDatabase

router = APIRouter()
logger = logging.getLogger(__name__)

# Track application start time
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns the health status of the application and its dependencies.
    """
    try:
        # Calculate uptime
        uptime_seconds = time.time() - _start_time
        
        # Check database health
        database_healthy = await _check_database_health()
        
        # Check Ollama health
        ollama_healthy = await _check_ollama_health()
        
        # Get system metrics
        memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
        cpu_usage = psutil.cpu_percent(interval=1)
        disk_usage = psutil.disk_usage('/').percent
        
        # Get application metrics
        indexed_files_count = await _get_indexed_files_count()
        active_indexing_jobs = await _get_active_indexing_jobs()
        
        # Determine overall status
        if database_healthy and ollama_healthy:
            status = "healthy"
        elif database_healthy or ollama_healthy:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            uptime_seconds=uptime_seconds,
            database_healthy=database_healthy,
            ollama_healthy=ollama_healthy,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            disk_usage_percent=disk_usage,
            indexed_files_count=indexed_files_count,
            active_indexing_jobs=active_indexing_jobs
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/health/simple")
async def simple_health_check():
    """
    Simple health check endpoint for load balancers.
    
    Returns a simple OK response if the service is running.
    """
    return {"status": "ok", "timestamp": datetime.utcnow()}


@router.get("/health/database")
async def database_health_check():
    """
    Check database connectivity and health.
    """
    try:
        database_healthy = await _check_database_health()
        
        if database_healthy:
            return {"status": "healthy", "message": "Database is accessible"}
        else:
            raise HTTPException(status_code=503, detail="Database is not accessible")
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(status_code=503, detail="Database health check failed")


@router.get("/health/ollama")
async def ollama_health_check():
    """
    Check Ollama service connectivity and model availability.
    """
    try:
        ollama_healthy = await _check_ollama_health()
        
        if ollama_healthy:
            return {"status": "healthy", "message": "Ollama service is accessible"}
        else:
            raise HTTPException(status_code=503, detail="Ollama service is not accessible")
            
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        raise HTTPException(status_code=503, detail="Ollama health check failed")


async def _check_database_health() -> bool:
    """Check if the vector database is accessible."""
    try:
        # Try to initialize database connection
        vector_db = VectorDatabase()
        
        # Try a simple operation
        collections = vector_db.client.list_collections()
        return True
        
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return False


async def _check_ollama_health() -> bool:
    """Check if Ollama service is accessible and model is available."""
    try:
        import ollama
        
        # Try to connect to Ollama
        client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        
        # Check if the required model is available
        models = client.list()
        model_names = [model['name'] for model in models.get('models', [])]
        
        return settings.OLLAMA_MODEL in model_names
        
    except Exception as e:
        logger.warning(f"Ollama health check failed: {e}")
        return False


async def _get_indexed_files_count() -> int:
    """Get the total number of indexed files."""
    try:
        vector_db = VectorDatabase()
        collection = vector_db.get_collection()
        return collection.count()
        
    except Exception as e:
        logger.warning(f"Failed to get indexed files count: {e}")
        return 0


async def _get_active_indexing_jobs() -> int:
    """Get the number of active indexing jobs."""
    try:
        # This would be implemented when we have the indexing service
        # For now, return 0
        return 0
        
    except Exception as e:
        logger.warning(f"Failed to get active indexing jobs: {e}")
        return 0
