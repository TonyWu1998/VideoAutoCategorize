"""
Database schemas and data models for ChromaDB and metadata storage.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
import json
import uuid


@dataclass
class MediaDocument:
    """
    Document structure for media files stored in ChromaDB.
    
    This represents how media files are stored in the vector database
    with their embeddings and metadata.
    """
    
    # Unique identifier
    file_id: str
    
    # File information
    file_path: str
    file_name: str
    file_size: int
    created_date: datetime
    modified_date: datetime
    media_type: str  # "image" or "video"
    
    # Media-specific metadata
    dimensions: Optional[str] = None  # "1920x1080"
    duration: Optional[float] = None  # Video duration in seconds
    format: Optional[str] = None  # File format (JPEG, MP4, etc.)
    
    # AI-generated content
    ai_description: str = ""
    ai_tags: List[str] = None
    ai_confidence: Optional[float] = None
    
    # Indexing metadata
    indexed_date: datetime = None
    index_version: str = "1.0"
    
    # Vector embedding (stored separately in ChromaDB)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.ai_tags is None:
            self.ai_tags = []
        if self.indexed_date is None:
            self.indexed_date = datetime.utcnow()
    
    def to_chroma_metadata(self) -> Dict[str, Any]:
        """
        Convert to ChromaDB metadata format.

        ChromaDB metadata must be JSON-serializable and cannot contain
        complex objects like datetime directly. None values are filtered out.
        """
        metadata = {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "created_date": self.created_date.isoformat(),
            "modified_date": self.modified_date.isoformat(),
            "media_type": self.media_type,
            "ai_description": self.ai_description or "",
            "ai_tags": json.dumps(self.ai_tags),  # Store as JSON string
            "indexed_date": self.indexed_date.isoformat(),
            "index_version": self.index_version
        }

        # Add optional fields only if they are not None
        if self.dimensions is not None:
            metadata["dimensions"] = self.dimensions
        if self.duration is not None:
            metadata["duration"] = self.duration
        if self.format is not None:
            metadata["format"] = self.format
        if self.ai_confidence is not None:
            metadata["ai_confidence"] = self.ai_confidence

        return metadata
    
    @classmethod
    def from_chroma_metadata(cls, file_id: str, metadata: Dict[str, Any]) -> "MediaDocument":
        """
        Create MediaDocument from ChromaDB metadata.
        
        Converts the stored metadata back to proper Python types.
        """
        return cls(
            file_id=file_id,
            file_path=metadata["file_path"],
            file_name=metadata["file_name"],
            file_size=metadata["file_size"],
            created_date=datetime.fromisoformat(metadata["created_date"]),
            modified_date=datetime.fromisoformat(metadata["modified_date"]),
            media_type=metadata["media_type"],
            dimensions=metadata.get("dimensions"),
            duration=metadata.get("duration"),
            format=metadata.get("format"),
            ai_description=metadata.get("ai_description", ""),
            ai_tags=json.loads(metadata.get("ai_tags", "[]")),
            ai_confidence=metadata.get("ai_confidence"),
            indexed_date=datetime.fromisoformat(metadata["indexed_date"]),
            index_version=metadata.get("index_version", "1.0")
        )
    
    def to_searchable_text(self) -> str:
        """
        Generate searchable text content for the document.
        
        This text is used for full-text search and as the document
        content in ChromaDB.
        """
        parts = []
        
        # Add description
        if self.ai_description:
            parts.append(self.ai_description)
        
        # Add tags
        if self.ai_tags:
            parts.extend(self.ai_tags)
        
        # Add file name (without extension)
        file_name_without_ext = self.file_name.rsplit('.', 1)[0]
        parts.append(file_name_without_ext.replace('_', ' ').replace('-', ' '))
        
        # Add media type
        parts.append(self.media_type)
        
        return " ".join(parts)


@dataclass
class IndexingJob:
    """
    Represents an indexing job for tracking progress and history.
    
    This is stored separately from the vector database to track
    indexing operations and their status.
    """
    
    job_id: str
    status: str  # "pending", "running", "completed", "failed", "cancelled"
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Job configuration
    paths: List[str] = None
    recursive: bool = True
    force_reindex: bool = False
    batch_size: int = 10
    max_concurrent: int = 4
    
    # Progress tracking
    total_files: int = 0
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    
    # Performance metrics
    files_per_second: Optional[float] = None
    average_processing_time: Optional[float] = None
    
    # Error tracking
    errors: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.paths is None:
            self.paths = []
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "paths": self.paths,
            "recursive": self.recursive,
            "force_reindex": self.force_reindex,
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "skipped_files": self.skipped_files,
            "files_per_second": self.files_per_second,
            "average_processing_time": self.average_processing_time,
            "errors": self.errors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexingJob":
        """Create IndexingJob from dictionary."""
        return cls(
            job_id=data["job_id"],
            status=data["status"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            paths=data.get("paths", []),
            recursive=data.get("recursive", True),
            force_reindex=data.get("force_reindex", False),
            batch_size=data.get("batch_size", 10),
            max_concurrent=data.get("max_concurrent", 4),
            total_files=data.get("total_files", 0),
            processed_files=data.get("processed_files", 0),
            successful_files=data.get("successful_files", 0),
            failed_files=data.get("failed_files", 0),
            skipped_files=data.get("skipped_files", 0),
            files_per_second=data.get("files_per_second"),
            average_processing_time=data.get("average_processing_time"),
            errors=data.get("errors", [])
        )
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_files == 0:
            return 100.0
        return (self.processed_files / self.total_files) * 100.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.processed_files == 0:
            return 100.0
        return (self.successful_files / self.processed_files) * 100.0
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds."""
        if not self.completed_at:
            return None
        return (self.completed_at - self.started_at).total_seconds()


@dataclass
class PromptTemplate:
    """
    Represents a customizable LLM prompt template for media analysis.

    This stores user-defined or system default prompts that can be used
    for analyzing images and video frames with configurable instructions.
    """

    # Unique identifier
    template_id: str

    # Template metadata
    name: str
    description: str
    media_type: str  # "image" or "video_frame"

    # Prompt content
    prompt_text: str

    # Template properties
    is_default: bool = False
    is_active: bool = False
    version: str = "1.0"
    author: str = "user"  # "system" for defaults, "user" for custom

    # Timestamps
    created_date: datetime = None
    modified_date: datetime = None

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.created_date is None:
            self.created_date = datetime.now(timezone.utc)
        if self.modified_date is None:
            self.modified_date = self.created_date
        if not self.template_id:
            self.template_id = str(uuid.uuid4())

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """
        Convert to ChromaDB metadata format.

        ChromaDB requires all metadata values to be strings, numbers, or booleans.
        """
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "media_type": self.media_type,
            "prompt_text": self.prompt_text,
            "is_default": self.is_default,
            "is_active": self.is_active,
            "version": self.version,
            "author": self.author,
            "created_date": self.created_date.isoformat(),
            "modified_date": self.modified_date.isoformat(),
        }

    @classmethod
    def from_chroma_metadata(cls, template_id: str, metadata: Dict[str, Any]) -> "PromptTemplate":
        """
        Create PromptTemplate from ChromaDB metadata.

        Converts the stored metadata back to proper Python types.
        """
        return cls(
            template_id=template_id,
            name=metadata["name"],
            description=metadata["description"],
            media_type=metadata["media_type"],
            prompt_text=metadata["prompt_text"],
            is_default=metadata.get("is_default", False),
            is_active=metadata.get("is_active", False),
            version=metadata.get("version", "1.0"),
            author=metadata.get("author", "user"),
            created_date=datetime.fromisoformat(metadata["created_date"]),
            modified_date=datetime.fromisoformat(metadata["modified_date"]),
        )


@dataclass
class PromptConfiguration:
    """
    Represents the active prompt configuration for the system.

    This tracks which prompt templates are currently active for
    different media types.
    """

    # Active prompt template IDs
    active_image_prompt_id: Optional[str] = None
    active_video_prompt_id: Optional[str] = None

    # Configuration metadata
    last_updated: datetime = None
    updated_by: str = "system"

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.last_updated is None:
            self.last_updated = datetime.now(timezone.utc)


# Collection names for ChromaDB
MEDIA_COLLECTION_NAME = "media_embeddings"
JOBS_COLLECTION_NAME = "indexing_jobs"
PROMPT_TEMPLATES_COLLECTION_NAME = "prompt_templates"

# Metadata field names
METADATA_FIELDS = [
    "file_path",
    "file_name", 
    "file_size",
    "created_date",
    "modified_date",
    "media_type",
    "dimensions",
    "duration",
    "format",
    "ai_description",
    "ai_tags",
    "ai_confidence",
    "indexed_date",
    "index_version"
]

# Searchable fields for filtering
SEARCHABLE_FIELDS = [
    "media_type",
    "format",
    "ai_tags",
    "file_name"
]

# Prompt template metadata field names
PROMPT_TEMPLATE_FIELDS = [
    "template_id",
    "name",
    "description",
    "media_type",
    "prompt_text",
    "is_default",
    "is_active",
    "version",
    "author",
    "created_date",
    "modified_date"
]

# Searchable fields for prompt templates
PROMPT_SEARCHABLE_FIELDS = [
    "media_type",
    "is_default",
    "is_active",
    "author",
    "name"
]
