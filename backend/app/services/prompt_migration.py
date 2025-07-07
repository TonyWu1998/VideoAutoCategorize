"""
Migration service for setting up default prompt templates.
"""

import logging
from datetime import datetime, timezone
import uuid

from app.services.prompt_manager import PromptManager
from app.models.prompts import PromptTemplateRequest, MediaType, PromptConfigurationRequest

logger = logging.getLogger(__name__)


class PromptMigrationService:
    """
    Service for migrating and setting up default prompt templates.
    
    Handles the initial setup of default prompts from the current
    hardcoded values in the LLM service.
    """
    
    def __init__(self):
        """Initialize the migration service."""
        self.prompt_manager = PromptManager()
    
    async def setup_default_prompts(self) -> bool:
        """
        Set up default prompt templates if they don't exist.
        
        Creates system default prompts for image and video analysis
        using the current hardcoded prompt text.
        
        Returns:
            True if setup was successful, False otherwise
        """
        try:
            logger.info("Setting up default prompt templates")
            
            # Check if default prompts already exist
            existing_templates = await self.prompt_manager.list_templates()
            default_templates = [t for t in existing_templates if t.is_default]
            
            if default_templates:
                logger.info(f"Default prompts already exist ({len(default_templates)} found)")
                return True
            
            # Create default image prompt
            image_prompt_id = await self._create_default_image_prompt()
            
            # Create default video prompt
            video_prompt_id = await self._create_default_video_prompt()
            
            # Set as active configuration
            await self._set_default_active_configuration(image_prompt_id, video_prompt_id)
            
            logger.info("Default prompt templates setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup default prompt templates: {e}")
            return False
    
    async def _create_default_image_prompt(self) -> str:
        """Create the default image analysis prompt template."""
        try:
            prompt_text = """Analyze this image and provide a JSON response with the following structure:

{
  "description": "A detailed 2-3 sentence description of what you see in the image",
  "objects": ["list", "of", "key", "objects", "people", "subjects"],
  "setting": "location or environment description",
  "mood": "atmosphere or emotional tone",
  "colors": "dominant colors and visual style",
  "tags": ["relevant", "descriptive", "keywords"]
}

IMPORTANT: Respond ONLY with valid JSON. Do not include markdown code blocks, explanations, or any text outside the JSON object."""
            
            request = PromptTemplateRequest(
                name="Default Image Analysis",
                description="System default prompt for analyzing images. Provides structured analysis including description, objects, setting, mood, colors, and tags.",
                media_type=MediaType.IMAGE,
                prompt_text=prompt_text,
                version="1.0"
            )
            
            # Create template with system author
            template = await self.prompt_manager.create_template(request)
            
            # Update to mark as default and system-created
            await self._mark_as_default_system_template(template.template_id)
            
            logger.info(f"Created default image prompt: {template.template_id}")
            return template.template_id
            
        except Exception as e:
            logger.error(f"Failed to create default image prompt: {e}")
            raise
    
    async def _create_default_video_prompt(self) -> str:
        """Create the default video frame analysis prompt template."""
        try:
            prompt_text = """Analyze this video frame and provide a JSON response with the following structure:

{
  "description": "A detailed 2-3 sentence description of what you see in the frame",
  "objects": ["list", "of", "key", "objects"],
  "setting": "location or environment description",
  "mood": "atmosphere or emotional tone",
  "tags": ["relevant", "descriptive", "keywords"]
}

IMPORTANT: Respond ONLY with valid JSON. Do not include markdown code blocks, explanations, or any text outside the JSON object."""
            
            request = PromptTemplateRequest(
                name="Default Video Frame Analysis",
                description="System default prompt for analyzing video frames. Provides structured analysis including description, objects, setting, mood, and tags.",
                media_type=MediaType.VIDEO_FRAME,
                prompt_text=prompt_text,
                version="1.0"
            )
            
            # Create template with system author
            template = await self.prompt_manager.create_template(request)
            
            # Update to mark as default and system-created
            await self._mark_as_default_system_template(template.template_id)
            
            logger.info(f"Created default video prompt: {template.template_id}")
            return template.template_id
            
        except Exception as e:
            logger.error(f"Failed to create default video prompt: {e}")
            raise
    
    async def _mark_as_default_system_template(self, template_id: str):
        """Mark a template as a default system template."""
        try:
            # Get the template from ChromaDB and update its metadata
            result = self.prompt_manager.collection.get(ids=[template_id])
            if result["ids"]:
                metadata = result["metadatas"][0]
                metadata["is_default"] = True
                metadata["author"] = "system"
                metadata["modified_date"] = datetime.now(timezone.utc).isoformat()

                self.prompt_manager.collection.update(
                    ids=[template_id],
                    metadatas=[metadata]
                )
                
                # Update cache
                if template_id in self.prompt_manager._prompt_cache:
                    cached_template = self.prompt_manager._prompt_cache[template_id]
                    cached_template.is_default = True
                    cached_template.author = "system"
                    cached_template.modified_date = datetime.now(timezone.utc)
                
                logger.info(f"Marked template {template_id} as default system template")
            
        except Exception as e:
            logger.error(f"Failed to mark template {template_id} as default: {e}")
            raise
    
    async def _set_default_active_configuration(self, image_prompt_id: str, video_prompt_id: str):
        """Set the default prompts as the active configuration."""
        try:
            config_request = PromptConfigurationRequest(
                active_image_prompt_id=image_prompt_id,
                active_video_prompt_id=video_prompt_id
            )
            
            await self.prompt_manager.update_active_configuration(config_request)
            
            logger.info("Set default prompts as active configuration")
            
        except Exception as e:
            logger.error(f"Failed to set default active configuration: {e}")
            raise
    
    async def reset_to_defaults(self) -> bool:
        """
        Reset prompt configuration to use only default prompts.
        
        This removes all custom prompts and resets to system defaults.
        Use with caution as this will delete user-created prompts.
        
        Returns:
            True if reset was successful, False otherwise
        """
        try:
            logger.warning("Resetting prompt configuration to defaults")
            
            # Get all templates
            all_templates = await self.prompt_manager.list_templates()
            
            # Delete all non-default templates
            for template in all_templates:
                if not template.is_default:
                    await self.prompt_manager.delete_template(template.template_id)
                    logger.info(f"Deleted custom template: {template.name}")
            
            # Find default templates
            default_templates = [t for t in all_templates if t.is_default]
            image_defaults = [t for t in default_templates if t.media_type == MediaType.IMAGE]
            video_defaults = [t for t in default_templates if t.media_type == MediaType.VIDEO_FRAME]
            
            if not image_defaults or not video_defaults:
                # Recreate defaults if missing
                logger.info("Default templates missing, recreating...")
                await self.setup_default_prompts()
            else:
                # Set defaults as active
                config_request = PromptConfigurationRequest(
                    active_image_prompt_id=image_defaults[0].template_id,
                    active_video_prompt_id=video_defaults[0].template_id
                )
                await self.prompt_manager.update_active_configuration(config_request)
            
            logger.info("Reset to default prompts completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset to default prompts: {e}")
            return False
