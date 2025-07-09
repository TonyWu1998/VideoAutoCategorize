"""
Configuration Manager for persistent LLM settings.

This module handles loading and saving LLM configuration settings to ChromaDB
for persistence across application restarts.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

logger = logging.getLogger(__name__)

# ChromaDB collection name for LLM configuration
LLM_CONFIG_COLLECTION = "llm_configuration"
LLM_CONFIG_DOCUMENT_ID = "active_llm_config"


class ConfigManager:
    """
    Manages persistent LLM configuration using ChromaDB.
    
    This class handles loading and saving LLM configuration settings
    to ensure they persist across application restarts.
    """
    
    def __init__(self):
        """Initialize the ConfigManager."""
        self.client = None
        self.config_collection = None
        self._initialize_chroma()
        logger.info("ConfigManager initialized")
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_DB_PATH,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection for LLM configuration
            self.config_collection = self.client.get_or_create_collection(
                name=LLM_CONFIG_COLLECTION,
                metadata={"description": "LLM configuration storage"}
            )
            
            logger.info("LLM configuration collection ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB for config: {e}")
            raise
    
    async def load_and_apply_config(self) -> bool:
        """
        Load persistent LLM configuration from ChromaDB and apply to settings.
        
        Returns:
            True if configuration was loaded and applied, False otherwise
        """
        try:
            config_data = await self._load_config_from_db()
            
            if not config_data:
                logger.info("No persistent LLM configuration found")
                return False
            
            # Apply configuration to settings
            applied_changes = []

            if "video_frame_interval" in config_data:
                settings.VIDEO_FRAME_INTERVAL = config_data["video_frame_interval"]
                applied_changes.append(f"VIDEO_FRAME_INTERVAL: {config_data['video_frame_interval']}")

            if "max_image_dimension" in config_data:
                settings.MAX_IMAGE_DIMENSION = config_data["max_image_dimension"]
                applied_changes.append(f"MAX_IMAGE_DIMENSION: {config_data['max_image_dimension']}")

            if "image_quality" in config_data:
                settings.IMAGE_QUALITY = config_data["image_quality"]
                applied_changes.append(f"IMAGE_QUALITY: {config_data['image_quality']}")

            if "ollama_model" in config_data:
                settings.OLLAMA_MODEL = config_data["ollama_model"]
                applied_changes.append(f"OLLAMA_MODEL: {config_data['ollama_model']}")

            if "ollama_embedding_model" in config_data:
                settings.OLLAMA_EMBEDDING_MODEL = config_data["ollama_embedding_model"]
                applied_changes.append(f"OLLAMA_EMBEDDING_MODEL: {config_data['ollama_embedding_model']}")

            if "ollama_base_url" in config_data:
                settings.OLLAMA_BASE_URL = config_data["ollama_base_url"]
                applied_changes.append(f"OLLAMA_BASE_URL: {config_data['ollama_base_url']}")

            if "ollama_timeout" in config_data:
                settings.OLLAMA_TIMEOUT = config_data["ollama_timeout"]
                applied_changes.append(f"OLLAMA_TIMEOUT: {config_data['ollama_timeout']}")

            if "enable_advanced_analysis" in config_data:
                settings.ENABLE_ADVANCED_ANALYSIS = config_data["enable_advanced_analysis"]
                applied_changes.append(f"ENABLE_ADVANCED_ANALYSIS: {config_data['enable_advanced_analysis']}")
            
            logger.info(f"Loaded LLM configuration from ChromaDB")
            logger.info(f"Applied persistent configuration to settings: {applied_changes}")

            # Refresh LLM service if configuration was loaded to ensure it uses the correct settings
            if applied_changes:
                try:
                    from app.services.llm_service import LLMService
                    LLMService.refresh_if_model_changed()
                    logger.info("🔄 LLM service refresh triggered after loading persistent config")
                except Exception as e:
                    logger.warning(f"Failed to refresh LLM service after loading config: {e}")
                    # Don't fail the startup if refresh fails

            return True
            
        except Exception as e:
            logger.error(f"Failed to load and apply config: {e}")
            return False
    
    async def save_config(self, config_data: Dict[str, Any]) -> bool:
        """
        Save LLM configuration to ChromaDB for persistence.
        
        Args:
            config_data: Dictionary containing configuration values
            
        Returns:
            True if configuration was saved successfully, False otherwise
        """
        try:
            # Add metadata
            config_with_metadata = {
                **config_data,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "updated_by": "api"
            }
            
            # Check if configuration already exists
            try:
                existing = self.config_collection.get(ids=[LLM_CONFIG_DOCUMENT_ID])
                if existing['ids']:
                    # Update existing
                    self.config_collection.update(
                        ids=[LLM_CONFIG_DOCUMENT_ID],
                        documents=[f"LLM configuration updated {config_with_metadata['last_updated']}"],
                        metadatas=[config_with_metadata]
                    )
                    logger.info("Updated LLM configuration in ChromaDB")
                else:
                    raise Exception("Not found")  # Fall through to add
            except:
                # Add new configuration
                self.config_collection.add(
                    ids=[LLM_CONFIG_DOCUMENT_ID],
                    documents=[f"LLM configuration created {config_with_metadata['last_updated']}"],
                    metadatas=[config_with_metadata]
                )
                logger.info("Saved new LLM configuration to ChromaDB")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    async def _load_config_from_db(self) -> Optional[Dict[str, Any]]:
        """Load configuration from ChromaDB."""
        try:
            result = self.config_collection.get(ids=[LLM_CONFIG_DOCUMENT_ID])
            
            if not result['ids']:
                logger.info("No LLM configuration found in ChromaDB")
                return None
            
            metadata = result['metadatas'][0]
            
            # Remove internal metadata fields
            config_data = {k: v for k, v in metadata.items() 
                          if k not in ['last_updated', 'updated_by']}
            
            logger.info("Loaded LLM configuration from ChromaDB")
            return config_data
            
        except Exception as e:
            logger.error(f"Failed to load config from DB: {e}")
            return None
