"""
API endpoints for application settings management.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging

from ..config import settings
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["config"])


class LLMConfigRequest(BaseModel):
    """Request model for LLM configuration updates."""
    max_video_frames: Optional[int] = Field(None, ge=1, le=50, description="Maximum frames to extract from videos")
    video_frame_interval: Optional[int] = Field(None, ge=1, le=300, description="Frame extraction interval in seconds")
    max_image_dimension: Optional[int] = Field(None, ge=256, le=4096, description="Maximum image dimension in pixels")
    image_quality: Optional[int] = Field(None, ge=50, le=100, description="JPEG quality percentage")
    ollama_model: Optional[str] = Field(None, description="Ollama vision model name")
    ollama_embedding_model: Optional[str] = Field(None, description="Ollama embedding model name")
    ollama_timeout: Optional[int] = Field(None, ge=30, le=600, description="Ollama request timeout in seconds")
    enable_advanced_analysis: Optional[bool] = Field(None, description="Enable advanced AI analysis features")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_tags_per_item: Optional[int] = Field(None, ge=1, le=50, description="Maximum tags to generate per item")


class LLMConfigResponse(BaseModel):
    """Response model for current LLM configuration."""
    max_video_frames: int
    video_frame_interval: int
    max_image_dimension: int
    image_quality: int
    ollama_model: str
    ollama_embedding_model: str
    ollama_timeout: int
    enable_advanced_analysis: bool
    confidence_threshold: float
    max_tags_per_item: int


class SettingsUpdateResponse(BaseModel):
    """Response model for settings update operations."""
    success: bool
    message: str
    updated_settings: Dict[str, Any]


@router.get("/llm", response_model=LLMConfigResponse)
async def get_llm_config():
    """Get current LLM configuration."""
    try:
        return LLMConfigResponse(
            max_video_frames=settings.MAX_VIDEO_FRAMES,
            video_frame_interval=settings.VIDEO_FRAME_INTERVAL,
            max_image_dimension=settings.MAX_IMAGE_DIMENSION,
            image_quality=settings.IMAGE_QUALITY,
            ollama_model=settings.OLLAMA_MODEL,
            ollama_embedding_model=settings.OLLAMA_EMBEDDING_MODEL,
            ollama_timeout=settings.OLLAMA_TIMEOUT,
            enable_advanced_analysis=settings.ENABLE_ADVANCED_ANALYSIS,
            confidence_threshold=0.5,  # Default value, could be made configurable
            max_tags_per_item=10,  # Default value, could be made configurable
        )
    except Exception as e:
        logger.error(f"Failed to get LLM config: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve LLM configuration")


@router.put("/llm", response_model=SettingsUpdateResponse)
async def update_llm_config(config: LLMConfigRequest):
    """Update LLM configuration settings."""
    try:
        updated_settings = {}
        
        # Update settings that are provided
        if config.max_video_frames is not None:
            settings.MAX_VIDEO_FRAMES = config.max_video_frames
            updated_settings["max_video_frames"] = config.max_video_frames
            
        if config.video_frame_interval is not None:
            settings.VIDEO_FRAME_INTERVAL = config.video_frame_interval
            updated_settings["video_frame_interval"] = config.video_frame_interval
            
        if config.max_image_dimension is not None:
            settings.MAX_IMAGE_DIMENSION = config.max_image_dimension
            updated_settings["max_image_dimension"] = config.max_image_dimension
            
        if config.image_quality is not None:
            settings.IMAGE_QUALITY = config.image_quality
            updated_settings["image_quality"] = config.image_quality
            
        if config.ollama_model is not None:
            settings.OLLAMA_MODEL = config.ollama_model
            updated_settings["ollama_model"] = config.ollama_model
            
        if config.ollama_embedding_model is not None:
            settings.OLLAMA_EMBEDDING_MODEL = config.ollama_embedding_model
            updated_settings["ollama_embedding_model"] = config.ollama_embedding_model
            
        if config.ollama_timeout is not None:
            settings.OLLAMA_TIMEOUT = config.ollama_timeout
            updated_settings["ollama_timeout"] = config.ollama_timeout
            
        if config.enable_advanced_analysis is not None:
            settings.ENABLE_ADVANCED_ANALYSIS = config.enable_advanced_analysis
            updated_settings["enable_advanced_analysis"] = config.enable_advanced_analysis
        
        logger.info(f"Updated LLM configuration: {updated_settings}")
        
        return SettingsUpdateResponse(
            success=True,
            message=f"Successfully updated {len(updated_settings)} settings",
            updated_settings=updated_settings
        )
        
    except Exception as e:
        logger.error(f"Failed to update LLM config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update LLM configuration: {str(e)}")


@router.post("/llm/reset", response_model=SettingsUpdateResponse)
async def reset_llm_config():
    """Reset LLM configuration to default values."""
    try:
        # Reset to default values from config
        default_settings = {
            "max_video_frames": 10,
            "video_frame_interval": 30,
            "max_image_dimension": 1024,
            "image_quality": 85,
            "ollama_model": "gemma3:4b",
            "ollama_embedding_model": "nomic-embed-text",
            "ollama_timeout": 120,
            "enable_advanced_analysis": False,
        }
        
        # Apply default settings
        settings.MAX_VIDEO_FRAMES = default_settings["max_video_frames"]
        settings.VIDEO_FRAME_INTERVAL = default_settings["video_frame_interval"]
        settings.MAX_IMAGE_DIMENSION = default_settings["max_image_dimension"]
        settings.IMAGE_QUALITY = default_settings["image_quality"]
        settings.OLLAMA_MODEL = default_settings["ollama_model"]
        settings.OLLAMA_EMBEDDING_MODEL = default_settings["ollama_embedding_model"]
        settings.OLLAMA_TIMEOUT = default_settings["ollama_timeout"]
        settings.ENABLE_ADVANCED_ANALYSIS = default_settings["enable_advanced_analysis"]
        
        logger.info("Reset LLM configuration to defaults")
        
        return SettingsUpdateResponse(
            success=True,
            message="LLM configuration reset to default values",
            updated_settings=default_settings
        )
        
    except Exception as e:
        logger.error(f"Failed to reset LLM config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset LLM configuration: {str(e)}")


@router.get("/system")
async def get_system_settings():
    """Get current system settings and status."""
    try:
        return {
            "database_path": settings.CHROMA_DB_PATH,
            "ollama_base_url": settings.OLLAMA_BASE_URL,
            "debug_mode": settings.DEBUG,
            "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
            "batch_size": settings.BATCH_SIZE,
            "max_concurrent_processing": settings.MAX_CONCURRENT_PROCESSING,
            "enable_file_watcher": settings.ENABLE_FILE_WATCHER,
            "search_cache_enabled": settings.ENABLE_SEARCH_CACHE,
            "supported_image_formats": settings.SUPPORTED_IMAGE_FORMATS,
            "supported_video_formats": settings.SUPPORTED_VIDEO_FORMATS,
        }
    except Exception as e:
        logger.error(f"Failed to get system settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system settings")


@router.get("/export")
async def export_settings():
    """Export current settings configuration."""
    try:
        config_data = {
            "llm_config": {
                "max_video_frames": settings.MAX_VIDEO_FRAMES,
                "video_frame_interval": settings.VIDEO_FRAME_INTERVAL,
                "max_image_dimension": settings.MAX_IMAGE_DIMENSION,
                "image_quality": settings.IMAGE_QUALITY,
                "ollama_model": settings.OLLAMA_MODEL,
                "ollama_embedding_model": settings.OLLAMA_EMBEDDING_MODEL,
                "ollama_timeout": settings.OLLAMA_TIMEOUT,
                "enable_advanced_analysis": settings.ENABLE_ADVANCED_ANALYSIS,
            },
            "system_config": {
                "batch_size": settings.BATCH_SIZE,
                "max_concurrent_processing": settings.MAX_CONCURRENT_PROCESSING,
                "enable_file_watcher": settings.ENABLE_FILE_WATCHER,
                "search_cache_enabled": settings.ENABLE_SEARCH_CACHE,
            },
            "exported_at": "2025-01-06T00:00:00Z",  # Would use actual timestamp
            "version": "1.0.0"
        }
        
        return config_data
        
    except Exception as e:
        logger.error(f"Failed to export settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to export settings")
