"""
Media-related Pydantic models.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pathlib import Path

from .common import BaseResponse, PaginationResponse


class MediaType(str, Enum):
    """Media type enumeration."""
    
    IMAGE = "image"
    VIDEO = "video"
    ALL = "all"


class MediaMetadata(BaseModel):
    """Media file metadata."""
    
    file_path: str = Field(description="Full path to the media file")
    file_name: str = Field(description="File name with extension")
    file_size: int = Field(description="File size in bytes")
    created_date: datetime = Field(description="File creation date")
    modified_date: datetime = Field(description="File last modified date")
    media_type: MediaType = Field(description="Type of media (image/video)")
    
    # Media-specific metadata
    dimensions: Optional[str] = Field(default=None, description="Image/video dimensions (e.g., '1920x1080')")
    duration: Optional[float] = Field(default=None, description="Video duration in seconds")
    format: Optional[str] = Field(default=None, description="File format (e.g., 'JPEG', 'MP4')")
    
    # AI-generated metadata
    ai_description: Optional[str] = Field(default=None, description="AI-generated description")
    ai_tags: List[str] = Field(default_factory=list, description="AI-generated tags")
    ai_confidence: Optional[float] = Field(default=None, description="AI analysis confidence score")
    
    # Indexing metadata
    indexed_date: Optional[datetime] = Field(default=None, description="When the file was indexed")
    index_version: Optional[str] = Field(default=None, description="Version of indexing algorithm used")
    
    @validator('file_path')
    def validate_file_path(cls, v):
        """Validate that file path exists."""
        if not Path(v).exists():
            raise ValueError(f"File path does not exist: {v}")
        return v
    
    @validator('ai_confidence')
    def validate_confidence(cls, v):
        """Validate confidence score is between 0 and 1."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Confidence score must be between 0 and 1")
        return v


class MediaItem(BaseModel):
    """Media item with search relevance information."""
    
    file_id: str = Field(description="Unique identifier for the file")
    metadata: MediaMetadata = Field(description="File metadata")
    similarity_score: Optional[float] = Field(default=None, description="Similarity score for search results")
    thumbnail_url: Optional[str] = Field(default=None, description="URL to thumbnail image")
    preview_url: Optional[str] = Field(default=None, description="URL to preview/full image")
    
    @validator('similarity_score')
    def validate_similarity_score(cls, v):
        """Validate similarity score is between 0 and 1."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Similarity score must be between 0 and 1")
        return v


class SearchFilters(BaseModel):
    """Search filters for media queries."""
    
    media_type: MediaType = Field(default=MediaType.ALL, description="Filter by media type")
    date_range: Optional[Dict[str, datetime]] = Field(default=None, description="Date range filter")
    min_similarity: float = Field(default=0.3, ge=0, le=1, description="Minimum similarity threshold")
    max_results: int = Field(default=20, ge=1, le=100, description="Maximum number of results")
    
    # File size filters
    min_file_size: Optional[int] = Field(default=None, description="Minimum file size in bytes")
    max_file_size: Optional[int] = Field(default=None, description="Maximum file size in bytes")
    
    # Dimension filters
    min_width: Optional[int] = Field(default=None, description="Minimum image/video width")
    max_width: Optional[int] = Field(default=None, description="Maximum image/video width")
    min_height: Optional[int] = Field(default=None, description="Minimum image/video height")
    max_height: Optional[int] = Field(default=None, description="Maximum image/video height")
    
    # Duration filter (for videos)
    min_duration: Optional[float] = Field(default=None, description="Minimum video duration in seconds")
    max_duration: Optional[float] = Field(default=None, description="Maximum video duration in seconds")
    
    # Tag filters
    include_tags: List[str] = Field(default_factory=list, description="Tags that must be present")
    exclude_tags: List[str] = Field(default_factory=list, description="Tags that must not be present")
    
    # Path filters
    include_paths: List[str] = Field(default_factory=list, description="Paths to include in search")
    exclude_paths: List[str] = Field(default_factory=list, description="Paths to exclude from search")


class SearchRequest(BaseModel):
    """Search request model."""
    
    query: str = Field(description="Natural language search query")
    filters: SearchFilters = Field(default_factory=SearchFilters, description="Search filters")
    include_metadata: bool = Field(default=True, description="Whether to include full metadata")
    include_thumbnails: bool = Field(default=True, description="Whether to include thumbnail URLs")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate search query is not empty."""
        if not v or not v.strip():
            raise ValueError("Search query cannot be empty")
        return v.strip()


class SearchResponse(BaseResponse):
    """Search response model."""
    
    query: str = Field(description="Original search query")
    results: List[MediaItem] = Field(description="Search results")
    total_results: int = Field(description="Total number of matching results")
    search_time_ms: float = Field(description="Search execution time in milliseconds")
    filters_applied: SearchFilters = Field(description="Filters that were applied")
    pagination: Optional[PaginationResponse] = Field(default=None, description="Pagination information")


class MediaUploadRequest(BaseModel):
    """Media upload request model."""
    
    file_path: str = Field(description="Path where the file should be stored")
    overwrite: bool = Field(default=False, description="Whether to overwrite existing files")
    auto_index: bool = Field(default=True, description="Whether to automatically index the uploaded file")


class MediaUploadResponse(BaseResponse):
    """Media upload response model."""
    
    file_id: str = Field(description="Unique identifier for the uploaded file")
    file_path: str = Field(description="Path where the file was stored")
    file_size: int = Field(description="Size of the uploaded file in bytes")
    indexed: bool = Field(description="Whether the file was automatically indexed")


class MediaDeleteRequest(BaseModel):
    """Media delete request model."""
    
    file_ids: List[str] = Field(description="List of file IDs to delete")
    delete_from_disk: bool = Field(default=False, description="Whether to delete files from disk")
    force: bool = Field(default=False, description="Force deletion even if files are in use")


class MediaDeleteResponse(BaseResponse):
    """Media delete response model."""
    
    deleted_count: int = Field(description="Number of files successfully deleted")
    failed_count: int = Field(description="Number of files that failed to delete")
    errors: List[Dict[str, str]] = Field(default_factory=list, description="List of deletion errors")


class MediaStatsResponse(BaseResponse):
    """Media statistics response model."""
    
    total_files: int = Field(description="Total number of indexed files")
    total_size_bytes: int = Field(description="Total size of all files in bytes")
    
    # Breakdown by type
    image_count: int = Field(description="Number of image files")
    video_count: int = Field(description="Number of video files")
    
    # Breakdown by format
    format_breakdown: Dict[str, int] = Field(description="File count by format")
    
    # Date statistics
    oldest_file: Optional[datetime] = Field(default=None, description="Date of oldest file")
    newest_file: Optional[datetime] = Field(default=None, description="Date of newest file")
    
    # Indexing statistics
    last_index_date: Optional[datetime] = Field(default=None, description="Last indexing date")
    pending_indexing: int = Field(description="Number of files pending indexing")
