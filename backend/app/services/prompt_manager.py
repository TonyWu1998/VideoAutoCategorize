"""
Service for managing LLM prompt templates and configurations.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
from app.database.schemas import PromptTemplate, PromptConfiguration, PROMPT_TEMPLATES_COLLECTION_NAME
from app.config import settings
from app.models.prompts import (
    PromptTemplateRequest, 
    PromptTemplateResponse,
    PromptConfigurationRequest,
    PromptConfigurationResponse,
    MediaType
)
from app.config import settings

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Service for managing prompt templates and active configurations.
    
    Handles CRUD operations for prompt templates, manages active prompt
    configurations, and provides caching for performance.
    """
    
    def __init__(self):
        """Initialize the prompt manager."""
        # Initialize ChromaDB client
        db_path = Path(settings.CHROMA_DB_PATH)
        db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        self._active_config_cache: Optional[PromptConfiguration] = None
        self._prompt_cache: Dict[str, PromptTemplate] = {}

        # Ensure prompt templates collection exists
        self._ensure_collection_exists()

        logger.info("PromptManager initialized")
    
    def _ensure_collection_exists(self):
        """Ensure the prompt templates collection exists in ChromaDB."""
        try:
            # Get or create the collection
            self.collection = self.client.get_or_create_collection(
                name=PROMPT_TEMPLATES_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Prompt templates collection ready: {PROMPT_TEMPLATES_COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to ensure prompt templates collection exists: {e}")
            raise
    
    async def create_template(self, request: PromptTemplateRequest) -> PromptTemplateResponse:
        """
        Create a new prompt template.
        
        Args:
            request: Template creation request
            
        Returns:
            Created template response
        """
        try:
            # Create template object
            template = PromptTemplate(
                template_id=str(uuid.uuid4()),
                name=request.name,
                description=request.description,
                media_type=request.media_type.value,
                prompt_text=request.prompt_text,
                version=request.version or "1.0",
                author="user",
                is_default=False,
                is_active=False
            )
            
            # Store in ChromaDB
            self.collection.add(
                ids=[template.template_id],
                documents=[template.prompt_text],  # Use prompt text as document for search
                metadatas=[template.to_chroma_metadata()]
            )
            
            # Update cache
            self._prompt_cache[template.template_id] = template
            
            logger.info(f"Created prompt template: {template.name} ({template.template_id})")
            
            return self._template_to_response(template)
            
        except Exception as e:
            logger.error(f"Failed to create prompt template: {e}")
            raise
    
    async def get_template(self, template_id: str) -> Optional[PromptTemplateResponse]:
        """
        Get a specific prompt template by ID.
        
        Args:
            template_id: Template identifier
            
        Returns:
            Template response or None if not found
        """
        try:
            # Check cache first
            if template_id in self._prompt_cache:
                return self._template_to_response(self._prompt_cache[template_id])
            
            # Query ChromaDB
            result = self.collection.get(ids=[template_id])
            
            if not result["ids"]:
                return None
            
            # Convert to template object
            metadata = result["metadatas"][0]
            template = PromptTemplate.from_chroma_metadata(template_id, metadata)
            
            # Update cache
            self._prompt_cache[template_id] = template
            
            return self._template_to_response(template)
            
        except Exception as e:
            logger.error(f"Failed to get prompt template {template_id}: {e}")
            raise
    
    async def list_templates(self, media_type: Optional[MediaType] = None) -> List[PromptTemplateResponse]:
        """
        List all prompt templates, optionally filtered by media type.
        
        Args:
            media_type: Optional media type filter
            
        Returns:
            List of template responses
        """
        try:
            # Build query filter
            where_filter = {}
            if media_type:
                where_filter["media_type"] = media_type.value

            # Query all templates
            if where_filter:
                result = self.collection.get(where=where_filter)
            else:
                result = self.collection.get()
            
            templates = []
            for i, template_id in enumerate(result["ids"]):
                metadata = result["metadatas"][i]
                template = PromptTemplate.from_chroma_metadata(template_id, metadata)
                templates.append(self._template_to_response(template))
                
                # Update cache
                self._prompt_cache[template_id] = template
            
            # Sort by creation date (newest first)
            templates.sort(key=lambda t: t.created_date, reverse=True)
            
            logger.info(f"Listed {len(templates)} prompt templates")
            return templates
            
        except Exception as e:
            logger.error(f"Failed to list prompt templates: {e}")
            raise
    
    async def update_template(self, template_id: str, request: PromptTemplateRequest) -> Optional[PromptTemplateResponse]:
        """
        Update an existing prompt template.
        
        Args:
            template_id: Template identifier
            request: Template update request
            
        Returns:
            Updated template response or None if not found
        """
        try:
            # Get existing template
            existing = await self.get_template(template_id)
            if not existing:
                return None
            
            # Create updated template
            template = PromptTemplate(
                template_id=template_id,
                name=request.name,
                description=request.description,
                media_type=request.media_type.value,
                prompt_text=request.prompt_text,
                version=request.version or existing.version,
                author=existing.author,
                is_default=existing.is_default,
                is_active=existing.is_active,
                created_date=existing.created_date,
                modified_date=datetime.now(timezone.utc)
            )
            
            # Update in ChromaDB
            self.collection.update(
                ids=[template_id],
                documents=[template.prompt_text],
                metadatas=[template.to_chroma_metadata()]
            )
            
            # Update cache
            self._prompt_cache[template_id] = template
            
            logger.info(f"Updated prompt template: {template.name} ({template_id})")
            
            return self._template_to_response(template)
            
        except Exception as e:
            logger.error(f"Failed to update prompt template {template_id}: {e}")
            raise
    
    async def delete_template(self, template_id: str) -> bool:
        """
        Delete a prompt template.
        
        Args:
            template_id: Template identifier
            
        Returns:
            True if deleted, False if not found
        """
        try:
            # Check if template exists
            existing = await self.get_template(template_id)
            if not existing:
                return False
            
            # Don't allow deletion of default templates
            if existing.is_default:
                raise ValueError("Cannot delete default prompt templates")
            
            # Don't allow deletion of active templates
            if existing.is_active:
                raise ValueError("Cannot delete active prompt templates. Please set a different template as active first.")
            
            # Delete from ChromaDB
            self.collection.delete(ids=[template_id])
            
            # Remove from cache
            self._prompt_cache.pop(template_id, None)
            
            logger.info(f"Deleted prompt template: {template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete prompt template {template_id}: {e}")
            raise
    
    async def get_active_configuration(self) -> PromptConfigurationResponse:
        """
        Get the current active prompt configuration.

        Returns:
            Current active configuration
        """
        try:
            # Check cache first
            if self._active_config_cache:
                config = self._active_config_cache
            else:
                # Load from settings or create default
                config = PromptConfiguration(
                    active_image_prompt_id=getattr(settings, 'ACTIVE_IMAGE_PROMPT_ID', None),
                    active_video_prompt_id=getattr(settings, 'ACTIVE_VIDEO_PROMPT_ID', None),
                    last_updated=datetime.now(timezone.utc),
                    updated_by="system"
                )
                self._active_config_cache = config

            # Get active prompt details
            active_image_prompt = None
            active_video_prompt = None

            if config.active_image_prompt_id:
                active_image_prompt = await self.get_template(config.active_image_prompt_id)

            if config.active_video_prompt_id:
                active_video_prompt = await self.get_template(config.active_video_prompt_id)

            return PromptConfigurationResponse(
                active_image_prompt_id=config.active_image_prompt_id,
                active_video_prompt_id=config.active_video_prompt_id,
                active_image_prompt=active_image_prompt,
                active_video_prompt=active_video_prompt,
                last_updated=config.last_updated,
                updated_by=config.updated_by
            )

        except Exception as e:
            logger.error(f"Failed to get active configuration: {e}")
            raise

    async def update_active_configuration(self, request: PromptConfigurationRequest) -> PromptConfigurationResponse:
        """
        Update the active prompt configuration.

        Args:
            request: Configuration update request

        Returns:
            Updated configuration
        """
        try:
            # Validate template IDs exist
            if request.active_image_prompt_id:
                image_template = await self.get_template(request.active_image_prompt_id)
                if not image_template:
                    raise ValueError(f"Image prompt template not found: {request.active_image_prompt_id}")
                if image_template.media_type != MediaType.IMAGE:
                    raise ValueError(f"Template {request.active_image_prompt_id} is not for images")

            if request.active_video_prompt_id:
                video_template = await self.get_template(request.active_video_prompt_id)
                if not video_template:
                    raise ValueError(f"Video prompt template not found: {request.active_video_prompt_id}")
                if video_template.media_type != MediaType.VIDEO_FRAME:
                    raise ValueError(f"Template {request.active_video_prompt_id} is not for video frames")

            # Update active flags in existing templates
            await self._update_active_flags(request)

            # Create new configuration
            config = PromptConfiguration(
                active_image_prompt_id=request.active_image_prompt_id,
                active_video_prompt_id=request.active_video_prompt_id,
                last_updated=datetime.now(timezone.utc),
                updated_by="user"
            )

            # Update settings (in a real app, this would persist to database)
            if hasattr(settings, 'ACTIVE_IMAGE_PROMPT_ID'):
                settings.ACTIVE_IMAGE_PROMPT_ID = request.active_image_prompt_id
            if hasattr(settings, 'ACTIVE_VIDEO_PROMPT_ID'):
                settings.ACTIVE_VIDEO_PROMPT_ID = request.active_video_prompt_id

            # Update cache
            self._active_config_cache = config

            logger.info(f"Updated active prompt configuration")

            return await self.get_active_configuration()

        except Exception as e:
            logger.error(f"Failed to update active configuration: {e}")
            raise

    async def _update_active_flags(self, request: PromptConfigurationRequest):
        """Update is_active flags for all templates based on new configuration."""
        try:
            # Get all templates
            all_templates = await self.list_templates()

            for template in all_templates:
                new_active_status = False

                # Check if this template should be active
                if (template.media_type == MediaType.IMAGE and
                    template.template_id == request.active_image_prompt_id):
                    new_active_status = True
                elif (template.media_type == MediaType.VIDEO_FRAME and
                      template.template_id == request.active_video_prompt_id):
                    new_active_status = True

                # Update if status changed
                if template.is_active != new_active_status:
                    # Get current template data
                    current_template = self._prompt_cache.get(template.template_id)
                    if current_template:
                        current_template.is_active = new_active_status
                        current_template.modified_date = datetime.now(timezone.utc)

                        # Update in ChromaDB
                        self.collection.update(
                            ids=[template.template_id],
                            metadatas=[current_template.to_chroma_metadata()]
                        )

                        # Update cache
                        self._prompt_cache[template.template_id] = current_template

        except Exception as e:
            logger.error(f"Failed to update active flags: {e}")
            raise

    async def get_active_prompt_text(self, media_type: MediaType) -> Optional[str]:
        """
        Get the active prompt text for a specific media type.

        This is the main method used by the LLM service to get prompts.

        Args:
            media_type: Media type to get prompt for

        Returns:
            Active prompt text or None if no active prompt
        """
        try:
            config = await self.get_active_configuration()

            if media_type == MediaType.IMAGE and config.active_image_prompt:
                return config.active_image_prompt.prompt_text
            elif media_type == MediaType.VIDEO_FRAME and config.active_video_prompt:
                return config.active_video_prompt.prompt_text

            return None

        except Exception as e:
            logger.error(f"Failed to get active prompt text for {media_type}: {e}")
            return None

    def _template_to_response(self, template: PromptTemplate) -> PromptTemplateResponse:
        """Convert PromptTemplate to response model."""
        return PromptTemplateResponse(
            template_id=template.template_id,
            name=template.name,
            description=template.description,
            media_type=MediaType(template.media_type),
            prompt_text=template.prompt_text,
            is_default=template.is_default,
            is_active=template.is_active,
            version=template.version,
            author=template.author,
            created_date=template.created_date,
            modified_date=template.modified_date
        )
