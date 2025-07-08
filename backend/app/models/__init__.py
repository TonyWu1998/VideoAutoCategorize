"""
Pydantic models for the Media Semantic Search application.
"""

from .media import (
    MediaItem,
    MediaMetadata,
    MediaType,
    SearchRequest,
    SearchResponse,
    SearchFilters
)

from .indexing import (
    IndexingRequest,
    IndexingStatus,
    IndexingProgress,
    IndexingResult
)

from .common import (
    BaseResponse,
    ErrorResponse,
    HealthResponse
)

__all__ = [
    # Media models
    "MediaItem",
    "MediaMetadata", 
    "MediaType",
    "SearchRequest",
    "SearchResponse",
    "SearchFilters",
    
    # Indexing models
    "IndexingRequest",
    "IndexingStatus",
    "IndexingProgress",
    "IndexingResult",
    
    # Common models
    "BaseResponse",
    "ErrorResponse",
    "HealthResponse"
]
