"""
API endpoints for application settings management.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import ollama

from ..config import settings
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["config"])


class LLMConfigRequest(BaseModel):
    """Request model for LLM configuration updates."""
    video_frame_interval: Optional[int] = Field(None, ge=1, le=300, description="Frame extraction interval in seconds")
    max_image_dimension: Optional[int] = Field(None, ge=256, le=4096, description="Maximum image dimension in pixels")
    image_quality: Optional[int] = Field(None, ge=50, le=100, description="JPEG quality percentage")
    ollama_model: Optional[str] = Field(None, description="Ollama vision model name")
    ollama_embedding_model: Optional[str] = Field(None, description="Ollama embedding model name")
    ollama_base_url: Optional[str] = Field(None, description="Ollama server base URL (e.g., http://192.168.50.188:11434)")
    ollama_timeout: Optional[int] = Field(None, ge=30, le=600, description="Ollama request timeout in seconds")
    enable_advanced_analysis: Optional[bool] = Field(None, description="Enable advanced AI analysis features")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_tags_per_item: Optional[int] = Field(None, ge=1, le=50, description="Maximum tags to generate per item")


class LLMConfigResponse(BaseModel):
    """Response model for current LLM configuration."""
    video_frame_interval: int
    max_image_dimension: int
    image_quality: int
    ollama_model: str
    ollama_embedding_model: str
    ollama_base_url: str
    ollama_timeout: int
    enable_advanced_analysis: bool
    confidence_threshold: float
    max_tags_per_item: int


class SettingsUpdateResponse(BaseModel):
    """Response model for settings update operations."""
    success: bool
    message: str
    updated_settings: Dict[str, Any]


class OllamaModelInfo(BaseModel):
    """Information about an Ollama model."""
    name: str
    size: Optional[str] = None
    modified_at: Optional[str] = None
    digest: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    class Config:
        # Allow extra fields and be more flexible with types
        extra = "allow"


class AvailableModelsResponse(BaseModel):
    """Response model for available Ollama models."""
    success: bool
    models: List[OllamaModelInfo]
    vision_models: List[str]
    embedding_models: List[str]
    total_count: int
    ollama_connected: bool
    message: Optional[str] = None


class OllamaEndpointTestRequest(BaseModel):
    """Request model for testing Ollama endpoint connectivity."""
    base_url: str = Field(..., description="Ollama server base URL to test (e.g., http://192.168.50.188:11434)")


class OllamaEndpointTestResponse(BaseModel):
    """Response model for Ollama endpoint test results."""
    success: bool
    accessible: bool
    base_url: str
    response_time_ms: Optional[float] = None
    models_count: Optional[int] = None
    vision_models_count: Optional[int] = None
    embedding_models_count: Optional[int] = None
    error_message: Optional[str] = None
    message: str


@router.get("/llm", response_model=LLMConfigResponse)
async def get_llm_config():
    """Get current LLM configuration."""
    try:
        return LLMConfigResponse(
            video_frame_interval=settings.VIDEO_FRAME_INTERVAL,
            max_image_dimension=settings.MAX_IMAGE_DIMENSION,
            image_quality=settings.IMAGE_QUALITY,
            ollama_model=settings.OLLAMA_MODEL,
            ollama_embedding_model=settings.OLLAMA_EMBEDDING_MODEL,
            ollama_base_url=settings.OLLAMA_BASE_URL,
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
            logger.info(f"🔍 Updating OLLAMA_MODEL from '{settings.OLLAMA_MODEL}' to '{config.ollama_model}'")
            logger.info(f"🔍 Settings object ID in API: {id(settings)}")
            settings.OLLAMA_MODEL = config.ollama_model
            logger.info(f"🔍 OLLAMA_MODEL after update: '{settings.OLLAMA_MODEL}'")
            updated_settings["ollama_model"] = config.ollama_model

            # Refresh LLM service if model changed
            try:
                from app.services.llm_service import LLMService
                LLMService.refresh_if_model_changed()
                logger.info("🔄 LLM service refresh triggered after model update")
            except Exception as e:
                logger.warning(f"Failed to refresh LLM service: {e}")
                # Don't fail the API call if refresh fails
            
        if config.ollama_embedding_model is not None:
            settings.OLLAMA_EMBEDDING_MODEL = config.ollama_embedding_model
            updated_settings["ollama_embedding_model"] = config.ollama_embedding_model

        if config.ollama_base_url is not None:
            logger.info(f"🔍 Updating OLLAMA_BASE_URL from '{settings.OLLAMA_BASE_URL}' to '{config.ollama_base_url}'")
            settings.OLLAMA_BASE_URL = config.ollama_base_url
            updated_settings["ollama_base_url"] = config.ollama_base_url

            # Refresh LLM service if base URL changed
            try:
                from app.services.llm_service import LLMService
                LLMService.refresh_if_model_changed()
                logger.info("🔄 LLM service refresh triggered after base URL update")
            except Exception as e:
                logger.warning(f"Failed to refresh LLM service: {e}")
                # Don't fail the API call if refresh fails

        if config.ollama_timeout is not None:
            settings.OLLAMA_TIMEOUT = config.ollama_timeout
            updated_settings["ollama_timeout"] = config.ollama_timeout
            
        if config.enable_advanced_analysis is not None:
            settings.ENABLE_ADVANCED_ANALYSIS = config.enable_advanced_analysis
            updated_settings["enable_advanced_analysis"] = config.enable_advanced_analysis
        
        logger.info(f"Updated LLM configuration: {updated_settings}")

        # Save configuration for persistence
        if updated_settings:
            try:
                from app.services.config_manager import ConfigManager
                config_manager = ConfigManager()

                # Save the complete current configuration, not just the updated fields
                # This ensures all settings are persisted together
                complete_config = {
                    "video_frame_interval": settings.VIDEO_FRAME_INTERVAL,
                    "max_image_dimension": settings.MAX_IMAGE_DIMENSION,
                    "image_quality": settings.IMAGE_QUALITY,
                    "ollama_model": settings.OLLAMA_MODEL,
                    "ollama_embedding_model": settings.OLLAMA_EMBEDDING_MODEL,
                    "ollama_base_url": settings.OLLAMA_BASE_URL,
                    "ollama_timeout": settings.OLLAMA_TIMEOUT,
                    "enable_advanced_analysis": settings.ENABLE_ADVANCED_ANALYSIS,
                }

                await config_manager.save_config(complete_config)
                logger.info(f"Saved complete LLM configuration for persistence: {complete_config}")
            except Exception as e:
                logger.warning(f"Failed to save config for persistence: {e}")
                # Don't fail the API call if persistence fails

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
            "video_frame_interval": 30,
            "max_image_dimension": 1024,
            "image_quality": 85,
            "ollama_model": "gemma3:4b",
            "ollama_embedding_model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434",
            "ollama_timeout": 120,
            "enable_advanced_analysis": False,
        }

        # Apply default settings
        settings.VIDEO_FRAME_INTERVAL = default_settings["video_frame_interval"]
        settings.MAX_IMAGE_DIMENSION = default_settings["max_image_dimension"]
        settings.IMAGE_QUALITY = default_settings["image_quality"]
        settings.OLLAMA_MODEL = default_settings["ollama_model"]
        settings.OLLAMA_EMBEDDING_MODEL = default_settings["ollama_embedding_model"]
        settings.OLLAMA_BASE_URL = default_settings["ollama_base_url"]
        settings.OLLAMA_TIMEOUT = default_settings["ollama_timeout"]
        settings.ENABLE_ADVANCED_ANALYSIS = default_settings["enable_advanced_analysis"]

        # Save the complete default configuration for persistence
        try:
            from app.services.config_manager import ConfigManager
            config_manager = ConfigManager()
            await config_manager.save_config(default_settings)
            logger.info(f"Saved default LLM configuration for persistence: {default_settings}")
        except Exception as e:
            logger.warning(f"Failed to save default config for persistence: {e}")
            # Don't fail the API call if persistence fails

        # Refresh LLM service after resetting model
        try:
            from app.services.llm_service import LLMService
            LLMService.refresh_if_model_changed()
            logger.info("🔄 LLM service refresh triggered after reset")
        except Exception as e:
            logger.warning(f"Failed to refresh LLM service after reset: {e}")
            # Don't fail the API call if refresh fails

        logger.info("Reset LLM configuration to defaults")
        
        return SettingsUpdateResponse(
            success=True,
            message="LLM configuration reset to default values",
            updated_settings=default_settings
        )
        
    except Exception as e:
        logger.error(f"Failed to reset LLM config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset LLM configuration: {str(e)}")


@router.get("/ollama/models", response_model=AvailableModelsResponse)
async def get_available_ollama_models():
    """Get list of available Ollama models."""
    try:
        logger.info("Fetching available Ollama models")

        # Try to connect to Ollama
        client = ollama.Client(host=settings.OLLAMA_BASE_URL)

        try:
            # Get models from Ollama
            models_response = client.list()

            # Handle different response formats
            if hasattr(models_response, 'models'):
                model_list = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                model_list = models_response['models']
            else:
                model_list = models_response if isinstance(models_response, list) else []

            # Parse model information
            models = []
            vision_models = []
            embedding_models = []

            for model in model_list:
                # Extract model name
                model_name = None
                if hasattr(model, 'model'):
                    model_name = model.model
                elif hasattr(model, 'name'):
                    model_name = model.name
                elif isinstance(model, dict) and 'model' in model:
                    model_name = model['model']
                elif isinstance(model, dict) and 'name' in model:
                    model_name = model['name']
                elif isinstance(model, str):
                    model_name = model

                if not model_name:
                    continue

                # Extract additional model info
                size = None
                modified_at = None
                digest = None
                details = None

                if isinstance(model, dict):
                    size = model.get('size')
                    modified_at = model.get('modified_at')
                    digest = model.get('digest')
                    details = model.get('details')
                elif hasattr(model, 'size'):
                    size = getattr(model, 'size', None)
                    modified_at = getattr(model, 'modified_at', None)
                    digest = getattr(model, 'digest', None)
                    details = getattr(model, 'details', None)

                # Convert data types to strings for consistency
                size_str = str(size) if size is not None else None
                modified_str = str(modified_at) if modified_at is not None else None
                digest_str = str(digest) if digest is not None else None

                # Ensure details is a dict or None
                details_dict = details if isinstance(details, dict) else None

                # Create model info
                model_info = OllamaModelInfo(
                    name=model_name,
                    size=size_str,
                    modified_at=modified_str,
                    digest=digest_str,
                    details=details_dict
                )
                models.append(model_info)

                # Categorize models
                model_lower = model_name.lower()

                # Vision models (models that can process images)
                if any(keyword in model_lower for keyword in [
                    'llava', 'bakllava', 'gemma', 'minicpm', 'moondream',
                    'vision', 'visual', 'multimodal', 'mm', 'qwen2.5vl', 'qwenvl'
                ]):
                    vision_models.append(model_name)

                # Embedding models
                if any(keyword in model_lower for keyword in [
                    'embed', 'embedding', 'nomic', 'minilm', 'bge', 'gte', 'e5'
                ]):
                    embedding_models.append(model_name)

            logger.info(f"Found {len(models)} total models, {len(vision_models)} vision models, {len(embedding_models)} embedding models")

            return AvailableModelsResponse(
                success=True,
                models=models,
                vision_models=vision_models,
                embedding_models=embedding_models,
                total_count=len(models),
                ollama_connected=True,
                message=f"Successfully retrieved {len(models)} models from Ollama"
            )

        except Exception as ollama_error:
            logger.warning(f"Failed to fetch models from Ollama: {ollama_error}")

            # Return empty response with connection error
            return AvailableModelsResponse(
                success=False,
                models=[],
                vision_models=[],
                embedding_models=[],
                total_count=0,
                ollama_connected=False,
                message=f"Failed to connect to Ollama: {str(ollama_error)}"
            )

    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve available models: {str(e)}")


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


@router.post("/ollama/test-endpoint", response_model=OllamaEndpointTestResponse)
async def test_ollama_endpoint(request: OllamaEndpointTestRequest):
    """Test connectivity to a custom Ollama endpoint."""
    import time
    import re

    try:
        logger.info(f"Testing Ollama endpoint: {request.base_url}")

        # Validate URL format
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        if not url_pattern.match(request.base_url):
            return OllamaEndpointTestResponse(
                success=False,
                accessible=False,
                base_url=request.base_url,
                error_message="Invalid URL format. Expected format: http://hostname:port or http://ip:port",
                message="Invalid URL format"
            )

        # Test connection with timeout
        start_time = time.time()

        try:
            # Create a temporary client for testing
            test_client = ollama.Client(host=request.base_url)

            # Try to list models to test connectivity
            models_response = test_client.list()

            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000

            # Handle different response formats
            if hasattr(models_response, 'models'):
                model_list = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                model_list = models_response['models']
            else:
                model_list = models_response if isinstance(models_response, list) else []

            # Count different types of models
            vision_models = []
            embedding_models = []

            for model in model_list:
                model_name = model.get('name', '') if isinstance(model, dict) else str(model)
                model_lower = model_name.lower()

                # Vision models detection (including qwen2.5vl)
                if any(keyword in model_lower for keyword in [
                    'llava', 'bakllava', 'gemma', 'minicpm', 'moondream',
                    'vision', 'visual', 'multimodal', 'mm', 'qwen2.5vl', 'qwenvl'
                ]):
                    vision_models.append(model_name)

                # Embedding models
                if any(keyword in model_lower for keyword in [
                    'embed', 'embedding', 'nomic', 'minilm', 'bge', 'gte', 'e5'
                ]):
                    embedding_models.append(model_name)

            return OllamaEndpointTestResponse(
                success=True,
                accessible=True,
                base_url=request.base_url,
                response_time_ms=round(response_time_ms, 2),
                models_count=len(model_list),
                vision_models_count=len(vision_models),
                embedding_models_count=len(embedding_models),
                message=f"Successfully connected to Ollama server. Found {len(model_list)} models ({len(vision_models)} vision, {len(embedding_models)} embedding)"
            )

        except Exception as conn_error:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000

            logger.warning(f"Failed to connect to Ollama endpoint {request.base_url}: {conn_error}")

            return OllamaEndpointTestResponse(
                success=False,
                accessible=False,
                base_url=request.base_url,
                response_time_ms=round(response_time_ms, 2) if response_time_ms < 30000 else None,
                error_message=str(conn_error),
                message=f"Failed to connect to Ollama server: {str(conn_error)}"
            )

    except Exception as e:
        logger.error(f"Error testing Ollama endpoint: {e}")
        return OllamaEndpointTestResponse(
            success=False,
            accessible=False,
            base_url=request.base_url,
            error_message=str(e),
            message=f"Error testing endpoint: {str(e)}"
        )
