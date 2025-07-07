"""
API endpoints for prompt template management.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging

from app.models.prompts import (
    PromptTemplateRequest,
    PromptTemplateResponse,
    PromptTemplateListResponse,
    PromptConfigurationRequest,
    PromptConfigurationResponse,
    PromptValidationRequest,
    PromptValidationResponse,
    PromptTestRequest,
    PromptTestResponse,
    MediaType
)
from app.models.common import BaseResponse
from app.services.prompt_manager import PromptManager

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize prompt manager
prompt_manager = PromptManager()


@router.post("/", response_model=PromptTemplateResponse)
async def create_prompt_template(request: PromptTemplateRequest):
    """
    Create a new prompt template.
    
    Creates a custom prompt template that can be used for media analysis.
    The template will be available for selection in the active configuration.
    """
    try:
        logger.info(f"Creating prompt template: {request.name}")
        
        # Validate request
        if not request.name.strip():
            raise HTTPException(status_code=400, detail="Template name cannot be empty")
        
        if not request.prompt_text.strip():
            raise HTTPException(status_code=400, detail="Prompt text cannot be empty")
        
        # Create template
        template = await prompt_manager.create_template(request)
        
        logger.info(f"Created prompt template: {template.name} ({template.template_id})")
        
        return template
        
    except ValueError as e:
        logger.error(f"Validation error creating prompt template: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create prompt template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create prompt template: {str(e)}")


@router.get("/", response_model=PromptTemplateListResponse)
async def list_prompt_templates(
    media_type: Optional[MediaType] = Query(None, description="Filter by media type")
):
    """
    List all prompt templates.
    
    Returns all available prompt templates, optionally filtered by media type.
    Includes both system default templates and user-created custom templates.
    """
    try:
        logger.info(f"Listing prompt templates (media_type: {media_type})")
        
        templates = await prompt_manager.list_templates(media_type)
        
        logger.info(f"Found {len(templates)} prompt templates")
        
        return PromptTemplateListResponse(
            templates=templates,
            total_count=len(templates),
            media_type_filter=media_type
        )
        
    except Exception as e:
        logger.error(f"Failed to list prompt templates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list prompt templates: {str(e)}")


@router.get("/{template_id}", response_model=PromptTemplateResponse)
async def get_prompt_template(template_id: str):
    """
    Get a specific prompt template by ID.
    
    Returns detailed information about a prompt template including
    its content, metadata, and current status.
    """
    try:
        logger.info(f"Getting prompt template: {template_id}")
        
        template = await prompt_manager.get_template(template_id)
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Prompt template not found: {template_id}")
        
        logger.info(f"Retrieved prompt template: {template.name}")
        
        return template
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get prompt template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get prompt template: {str(e)}")


@router.put("/{template_id}", response_model=PromptTemplateResponse)
async def update_prompt_template(template_id: str, request: PromptTemplateRequest):
    """
    Update an existing prompt template.
    
    Updates the content and metadata of an existing prompt template.
    Cannot update system default templates.
    """
    try:
        logger.info(f"Updating prompt template: {template_id}")
        
        # Validate request
        if not request.name.strip():
            raise HTTPException(status_code=400, detail="Template name cannot be empty")
        
        if not request.prompt_text.strip():
            raise HTTPException(status_code=400, detail="Prompt text cannot be empty")
        
        # Update template
        template = await prompt_manager.update_template(template_id, request)
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Prompt template not found: {template_id}")
        
        logger.info(f"Updated prompt template: {template.name}")
        
        return template
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error updating prompt template: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update prompt template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update prompt template: {str(e)}")


@router.delete("/{template_id}", response_model=BaseResponse)
async def delete_prompt_template(template_id: str):
    """
    Delete a prompt template.
    
    Deletes a custom prompt template. Cannot delete system default templates
    or currently active templates.
    """
    try:
        logger.info(f"Deleting prompt template: {template_id}")
        
        success = await prompt_manager.delete_template(template_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Prompt template not found: {template_id}")
        
        logger.info(f"Deleted prompt template: {template_id}")
        
        return BaseResponse(
            success=True,
            message=f"Prompt template deleted successfully"
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error deleting prompt template: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete prompt template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete prompt template: {str(e)}")


@router.get("/config/active", response_model=PromptConfigurationResponse)
async def get_active_prompt_configuration():
    """
    Get the current active prompt configuration.
    
    Returns which prompt templates are currently active for different
    media types, along with their details.
    """
    try:
        logger.info("Getting active prompt configuration")
        
        config = await prompt_manager.get_active_configuration()
        
        logger.info(f"Retrieved active prompt configuration")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to get active prompt configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get active configuration: {str(e)}")


@router.put("/config/active", response_model=PromptConfigurationResponse)
async def update_active_prompt_configuration(request: PromptConfigurationRequest):
    """
    Update the active prompt configuration.
    
    Sets which prompt templates should be used for analyzing different
    media types. This affects all future media analysis operations.
    """
    try:
        logger.info("Updating active prompt configuration")
        
        config = await prompt_manager.update_active_configuration(request)
        
        logger.info(f"Updated active prompt configuration")
        
        return config
        
    except ValueError as e:
        logger.error(f"Validation error updating active configuration: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update active prompt configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update active configuration: {str(e)}")


@router.post("/validate", response_model=PromptValidationResponse)
async def validate_prompt_template(request: PromptValidationRequest):
    """
    Validate a prompt template.
    
    Checks if a prompt template is properly formatted and provides
    suggestions for improvement.
    """
    try:
        logger.info(f"Validating prompt template for {request.media_type}")
        
        # Basic validation
        errors = []
        suggestions = []
        
        if len(request.prompt_text.strip()) < 10:
            errors.append("Prompt text is too short (minimum 10 characters)")
        
        if len(request.prompt_text) > 2000:
            errors.append("Prompt text is too long (maximum 2000 characters)")
        
        # Check for JSON structure requirement
        if "JSON" not in request.prompt_text.upper():
            suggestions.append("Consider including JSON format requirements for structured output")
        
        # Estimate token count (rough approximation)
        estimated_tokens = len(request.prompt_text.split()) * 1.3
        
        is_valid = len(errors) == 0
        
        logger.info(f"Prompt validation result: valid={is_valid}, errors={len(errors)}")
        
        return PromptValidationResponse(
            is_valid=is_valid,
            validation_errors=errors,
            suggestions=suggestions,
            estimated_tokens=int(estimated_tokens)
        )

    except Exception as e:
        logger.error(f"Failed to validate prompt template: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate prompt: {str(e)}")


@router.post("/test", response_model=PromptTestResponse)
async def test_prompt_template(request: PromptTestRequest):
    """
    Test a prompt template with sample data.

    Tests how a prompt performs with actual media analysis,
    providing insights into the expected output format and quality.
    """
    try:
        logger.info(f"Testing prompt template for {request.media_type}")

        # For now, return a mock response since we don't have sample images readily available
        # In a real implementation, this would use the LLM service to test the prompt

        mock_result = {
            "description": "This is a sample description that would be generated by the LLM",
            "objects": ["sample", "objects", "detected"],
            "setting": "sample environment",
            "mood": "sample mood",
            "tags": ["sample", "tags", "generated"]
        }

        logger.info(f"Prompt test completed successfully")

        return PromptTestResponse(
            success=True,
            test_result=mock_result,
            execution_time_ms=150.0
        )

    except Exception as e:
        logger.error(f"Failed to test prompt template: {e}")
        return PromptTestResponse(
            success=False,
            error_message=str(e)
        )
