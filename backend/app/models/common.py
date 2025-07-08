"""
Common Pydantic models used across the application.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from datetime import datetime
from enum import Enum


class BaseResponse(BaseModel):
    """Base response model for all API responses."""
    
    success: bool = Field(default=True, description="Whether the request was successful")
    message: Optional[str] = Field(default=None, description="Optional message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class ErrorResponse(BaseResponse):
    """Error response model."""
    
    success: bool = Field(default=False, description="Always false for error responses")
    error_code: Optional[str] = Field(default=None, description="Error code for programmatic handling")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(description="Health status (healthy, unhealthy, degraded)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(description="Application version")
    uptime_seconds: float = Field(description="Application uptime in seconds")
    
    # Service health checks
    database_healthy: bool = Field(description="Vector database health")
    ollama_healthy: bool = Field(description="Ollama service health")
    
    # System metrics
    memory_usage_mb: Optional[float] = Field(default=None, description="Current memory usage in MB")
    cpu_usage_percent: Optional[float] = Field(default=None, description="Current CPU usage percentage")
    disk_usage_percent: Optional[float] = Field(default=None, description="Current disk usage percentage")
    
    # Application metrics
    indexed_files_count: Optional[int] = Field(default=None, description="Total number of indexed files")
    active_indexing_jobs: Optional[int] = Field(default=None, description="Number of active indexing jobs")


class StatusEnum(str, Enum):
    """Status enumeration for various operations."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PaginationRequest(BaseModel):
    """Pagination request parameters."""
    
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    page_size: int = Field(default=20, ge=1, le=100, description="Number of items per page")
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.page_size


class PaginationResponse(BaseModel):
    """Pagination response metadata."""
    
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")
    total_items: int = Field(description="Total number of items")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there are more pages")
    has_previous: bool = Field(description="Whether there are previous pages")


class SortOrder(str, Enum):
    """Sort order enumeration."""
    
    ASC = "asc"
    DESC = "desc"


class SortRequest(BaseModel):
    """Sort request parameters."""
    
    field: str = Field(description="Field to sort by")
    order: SortOrder = Field(default=SortOrder.ASC, description="Sort order")


class FilterOperator(str, Enum):
    """Filter operator enumeration."""
    
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IN = "in"
    NOT_IN = "not_in"


class FilterRequest(BaseModel):
    """Filter request parameters."""
    
    field: str = Field(description="Field to filter by")
    operator: FilterOperator = Field(description="Filter operator")
    value: Any = Field(description="Filter value")


class BulkOperationRequest(BaseModel):
    """Bulk operation request."""
    
    operation: str = Field(description="Operation to perform")
    items: list[str] = Field(description="List of item IDs to operate on")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Operation parameters")


class BulkOperationResponse(BaseResponse):
    """Bulk operation response."""
    
    total_items: int = Field(description="Total number of items processed")
    successful_items: int = Field(description="Number of successfully processed items")
    failed_items: int = Field(description="Number of failed items")
    errors: Optional[list[Dict[str, Any]]] = Field(default=None, description="List of errors for failed items")
