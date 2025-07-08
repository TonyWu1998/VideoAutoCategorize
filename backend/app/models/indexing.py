"""
Indexing-related Pydantic models.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pathlib import Path

from .common import BaseResponse, StatusEnum


class IndexingStatus(str, Enum):
    """Indexing status enumeration."""
    
    IDLE = "idle"
    SCANNING = "scanning"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class IndexingRequest(BaseModel):
    """Request to start indexing media files."""
    
    paths: List[str] = Field(description="List of directory paths to index")
    recursive: bool = Field(default=True, description="Whether to scan directories recursively")
    force_reindex: bool = Field(default=False, description="Whether to reindex already processed files")
    batch_size: Optional[int] = Field(default=None, description="Number of files to process in each batch")
    max_concurrent: Optional[int] = Field(default=None, description="Maximum concurrent processing jobs")
    
    # File filters
    include_patterns: List[str] = Field(default_factory=list, description="File patterns to include (glob)")
    exclude_patterns: List[str] = Field(default_factory=list, description="File patterns to exclude (glob)")
    min_file_size: Optional[int] = Field(default=None, description="Minimum file size in bytes")
    max_file_size: Optional[int] = Field(default=None, description="Maximum file size in bytes")
    
    @validator('paths')
    def validate_paths(cls, v):
        """Validate that all paths exist and are either directories or supported media files."""
        from ..config import Settings
        settings = Settings()
        supported_formats = settings.all_supported_formats

        for path_str in v:
            path = Path(path_str)
            if not path.exists():
                raise ValueError(f"Path does not exist: {path_str}")

            # Allow both directories and supported media files
            if path.is_dir():
                continue  # Directory is valid
            elif path.is_file():
                # Check if it's a supported media file
                file_extension = path.suffix.lower()
                if file_extension not in supported_formats:
                    raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: {', '.join(supported_formats)}")
            else:
                raise ValueError(f"Path is neither a directory nor a regular file: {path_str}")
        return v
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        """Validate batch size is reasonable."""
        if v is not None and (v < 1 or v > 1000):
            raise ValueError("Batch size must be between 1 and 1000")
        return v
    
    @validator('max_concurrent')
    def validate_max_concurrent(cls, v):
        """Validate max concurrent jobs is reasonable."""
        if v is not None and (v < 1 or v > 20):
            raise ValueError("Max concurrent jobs must be between 1 and 20")
        return v


class IndexingProgress(BaseModel):
    """Indexing progress information."""

    total_files: int = Field(description="Total number of files to process")
    processed_files: int = Field(description="Number of files processed so far")
    successful_files: int = Field(description="Number of successfully processed files")
    failed_files: int = Field(description="Number of files that failed processing")
    skipped_files: int = Field(description="Number of files skipped (already indexed)")

    current_file: Optional[str] = Field(default=None, description="Currently processing file")
    estimated_remaining_seconds: Optional[float] = Field(default=None, description="Estimated time remaining")

    # Frame-level progress tracking for video analysis
    current_file_frames_total: Optional[int] = Field(default=None, description="Total frames in current file being processed")
    current_file_frames_processed: Optional[int] = Field(default=None, description="Frames processed in current file")
    current_frame_activity: Optional[str] = Field(default=None, description="Current frame processing activity")

    # Performance metrics
    files_per_second: Optional[float] = Field(default=None, description="Processing rate")
    average_file_size: Optional[float] = Field(default=None, description="Average file size in bytes")
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage including frame-level progress."""
        if self.total_files == 0:
            return 100.0

        # Base progress from completed files
        base_progress = self.processed_files / self.total_files

        # Add fractional progress from current file's frame processing
        if (self.current_file_frames_total and
            self.current_file_frames_processed and
            self.current_file_frames_total > 0):

            current_file_progress = self.current_file_frames_processed / self.current_file_frames_total
            fractional_progress = current_file_progress / self.total_files
            base_progress += fractional_progress

        return min(base_progress * 100.0, 100.0)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.processed_files == 0:
            return 100.0
        return (self.successful_files / self.processed_files) * 100.0


class IndexingResult(BaseModel):
    """Result of processing a single file."""
    
    file_path: str = Field(description="Path to the processed file")
    file_id: Optional[str] = Field(default=None, description="Generated file ID")
    status: StatusEnum = Field(description="Processing status")
    processing_time_seconds: float = Field(description="Time taken to process the file")
    
    # AI analysis results
    description: Optional[str] = Field(default=None, description="AI-generated description")
    tags: List[str] = Field(default_factory=list, description="AI-generated tags")
    confidence: Optional[float] = Field(default=None, description="AI analysis confidence")
    
    # Error information
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
    error_code: Optional[str] = Field(default=None, description="Error code for programmatic handling")
    
    # Metadata
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    dimensions: Optional[str] = Field(default=None, description="Image/video dimensions")
    duration: Optional[float] = Field(default=None, description="Video duration in seconds")


class IndexingStatusResponse(BaseResponse):
    """Response containing current indexing status."""
    
    status: IndexingStatus = Field(description="Current indexing status")
    job_id: Optional[str] = Field(default=None, description="Current job ID")
    started_at: Optional[datetime] = Field(default=None, description="When indexing started")
    estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time")
    
    progress: Optional[IndexingProgress] = Field(default=None, description="Progress information")
    recent_results: List[IndexingResult] = Field(default_factory=list, description="Recent processing results")
    
    # Configuration
    current_paths: List[str] = Field(default_factory=list, description="Paths being indexed")
    batch_size: int = Field(description="Current batch size")
    max_concurrent: int = Field(description="Maximum concurrent jobs")


class IndexingHistoryItem(BaseModel):
    """Historical indexing job information."""
    
    job_id: str = Field(description="Unique job identifier")
    started_at: datetime = Field(description="When the job started")
    completed_at: Optional[datetime] = Field(default=None, description="When the job completed")
    status: IndexingStatus = Field(description="Final job status")
    
    paths: List[str] = Field(description="Paths that were indexed")
    total_files: int = Field(description="Total files processed")
    successful_files: int = Field(description="Successfully processed files")
    failed_files: int = Field(description="Failed files")
    
    duration_seconds: Optional[float] = Field(default=None, description="Total job duration")
    average_processing_time: Optional[float] = Field(default=None, description="Average time per file")


class IndexingHistoryResponse(BaseResponse):
    """Response containing indexing history."""
    
    jobs: List[IndexingHistoryItem] = Field(description="List of historical indexing jobs")
    total_jobs: int = Field(description="Total number of jobs in history")


class IndexingControlRequest(BaseModel):
    """Request to control indexing operations."""
    
    action: str = Field(description="Action to perform (pause, resume, cancel)")
    job_id: Optional[str] = Field(default=None, description="Specific job ID (if applicable)")


class IndexingStatsResponse(BaseResponse):
    """Indexing statistics response."""
    
    total_indexed_files: int = Field(description="Total number of indexed files")
    total_processing_time: float = Field(description="Total processing time in seconds")
    average_processing_time: float = Field(description="Average processing time per file")
    
    # Success/failure rates
    total_successful: int = Field(description="Total successful indexing operations")
    total_failed: int = Field(description="Total failed indexing operations")
    success_rate: float = Field(description="Overall success rate percentage")
    
    # Performance metrics
    files_per_hour: float = Field(description="Average files processed per hour")
    peak_processing_rate: Optional[float] = Field(default=None, description="Peak processing rate")
    
    # File type breakdown
    images_indexed: int = Field(description="Number of images indexed")
    videos_indexed: int = Field(description="Number of videos indexed")
    
    # Recent activity
    last_indexing_date: Optional[datetime] = Field(default=None, description="Last indexing activity")
    files_indexed_today: int = Field(description="Files indexed today")
    files_indexed_this_week: int = Field(description="Files indexed this week")


class FileWatcherStatus(BaseModel):
    """File system watcher status."""
    
    enabled: bool = Field(description="Whether file watching is enabled")
    watched_paths: List[str] = Field(description="Paths being watched")
    events_processed: int = Field(description="Number of file system events processed")
    pending_files: int = Field(description="Number of files pending processing")
    last_event_time: Optional[datetime] = Field(default=None, description="Time of last file system event")
