"""
Pydantic models for prompt management API endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class MediaType(str, Enum):
    """Supported media types for prompt templates."""
    IMAGE = "image"
    VIDEO_FRAME = "video_frame"


class PromptAuthor(str, Enum):
    """Prompt template authors."""
    SYSTEM = "system"
    USER = "user"


class PromptTemplateRequest(BaseModel):
    """Request model for creating or updating prompt templates."""
    name: str = Field(..., min_length=1, max_length=100, description="Template name")
    description: str = Field(..., min_length=1, max_length=500, description="Template description")
    media_type: MediaType = Field(..., description="Media type this prompt is for")
    prompt_text: str = Field(..., min_length=10, description="The actual prompt text")
    version: Optional[str] = Field("1.0", description="Template version")


class PromptTemplateResponse(BaseModel):
    """Response model for prompt template data."""
    template_id: str
    name: str
    description: str
    media_type: MediaType
    prompt_text: str
    is_default: bool
    is_active: bool
    version: str
    author: PromptAuthor
    created_date: datetime
    modified_date: datetime


class PromptTemplateListResponse(BaseModel):
    """Response model for listing prompt templates."""
    templates: List[PromptTemplateResponse]
    total_count: int
    media_type_filter: Optional[MediaType] = None


class PromptConfigurationRequest(BaseModel):
    """Request model for updating active prompt configuration."""
    active_image_prompt_id: Optional[str] = Field(None, description="Active prompt ID for images")
    active_video_prompt_id: Optional[str] = Field(None, description="Active prompt ID for video frames")


class PromptConfigurationResponse(BaseModel):
    """Response model for current prompt configuration."""
    active_image_prompt_id: Optional[str]
    active_video_prompt_id: Optional[str]
    active_image_prompt: Optional[PromptTemplateResponse] = None
    active_video_prompt: Optional[PromptTemplateResponse] = None
    last_updated: datetime
    updated_by: str


class PromptValidationRequest(BaseModel):
    """Request model for validating prompt templates."""
    prompt_text: str = Field(..., min_length=10, description="Prompt text to validate")
    media_type: MediaType = Field(..., description="Media type for validation context")


class PromptValidationResponse(BaseModel):
    """Response model for prompt validation results."""
    is_valid: bool
    validation_errors: List[str] = []
    suggestions: List[str] = []
    estimated_tokens: Optional[int] = None


class PromptTestRequest(BaseModel):
    """Request model for testing prompts with sample data."""
    prompt_text: str = Field(..., description="Prompt text to test")
    media_type: MediaType = Field(..., description="Media type for testing")
    sample_image_path: Optional[str] = Field(None, description="Path to sample image for testing")


class PromptTestResponse(BaseModel):
    """Response model for prompt testing results."""
    success: bool
    test_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None


class PromptImportRequest(BaseModel):
    """Request model for importing prompt templates."""
    templates: List[PromptTemplateRequest]
    overwrite_existing: bool = Field(False, description="Whether to overwrite existing templates with same name")


class PromptImportResponse(BaseModel):
    """Response model for prompt import results."""
    imported_count: int
    skipped_count: int
    error_count: int
    imported_template_ids: List[str]
    errors: List[str] = []


class PromptExportResponse(BaseModel):
    """Response model for exporting prompt templates."""
    templates: List[PromptTemplateResponse]
    export_date: datetime
    total_count: int
