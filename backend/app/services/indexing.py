"""
Indexing service for processing and indexing media files.
"""

from typing import List, Dict, Any, Optional
import logging
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
import os

from app.models.indexing import (
    IndexingRequest, IndexingStatusResponse, IndexingStatus, 
    IndexingProgress, IndexingResult, IndexingHistoryResponse,
    IndexingStatsResponse, FileWatcherStatus
)
from app.database.vector_db import VectorDatabase
from app.services.llm_service import LLMService
from app.config import settings

logger = logging.getLogger(__name__)


class IndexingService:
    """
    Service for indexing media files and managing indexing operations.
    
    Handles file discovery, AI analysis, and storage in the vector database.
    """
    
    def __init__(self):
        """Initialize the indexing service."""
        self.vector_db = VectorDatabase()
        self.llm_service = LLMService()
        self.current_job = None
        self.job_history = []
        self._file_watcher = None
        
        logger.info("Indexing service initialized")
    
    async def start_indexing(self, request: IndexingRequest) -> str:
        """
        Start indexing media files in the specified directories.
        
        Args:
            request: IndexingRequest with paths and configuration
            
        Returns:
            Job ID for tracking progress
        """
        try:
            # Generate job ID
            job_id = str(uuid.uuid4())
            
            # Create job tracking object
            self.current_job = {
                "job_id": job_id,
                "status": IndexingStatus.SCANNING,
                "started_at": datetime.utcnow(),
                "request": request,
                "progress": IndexingProgress(
                    total_files=0,
                    processed_files=0,
                    successful_files=0,
                    failed_files=0,
                    skipped_files=0
                )
            }
            
            logger.info(f"Starting indexing job {job_id} for paths: {request.paths}")
            
            # Start indexing in background
            asyncio.create_task(self._process_indexing_job(request))
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to start indexing: {e}")
            raise
    
    async def get_status(self) -> IndexingStatusResponse:
        """
        Get the current status of indexing operations.
        
        Returns:
            IndexingStatusResponse with current status and progress
        """
        try:
            if not self.current_job:
                return IndexingStatusResponse(
                    status=IndexingStatus.IDLE,
                    current_paths=[],
                    batch_size=settings.BATCH_SIZE,
                    max_concurrent=settings.MAX_CONCURRENT_PROCESSING,
                    recent_results=[],
                    success=True,
                    message="No active indexing job"
                )
            
            job = self.current_job
            
            return IndexingStatusResponse(
                status=IndexingStatus(job["status"]),
                job_id=job["job_id"],
                started_at=job["started_at"],
                progress=job["progress"],
                current_paths=job["request"].paths,
                batch_size=job["request"].batch_size or settings.BATCH_SIZE,
                max_concurrent=job["request"].max_concurrent or settings.MAX_CONCURRENT_PROCESSING,
                success=True,
                message="Indexing status retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to get indexing status: {e}")
            raise
    
    async def control_indexing(self, action: str, job_id: Optional[str] = None) -> bool:
        """
        Control indexing operations (pause, resume, cancel).
        
        Args:
            action: Action to perform
            job_id: Optional specific job ID
            
        Returns:
            True if successful
        """
        try:
            if not self.current_job:
                raise ValueError("No active indexing job")
            
            if action == "cancel" or action == "stop":
                self.current_job["status"] = IndexingStatus.CANCELLED
                logger.info(f"Indexing job cancelled: {self.current_job['job_id']}")
            elif action == "pause":
                self.current_job["status"] = IndexingStatus.PAUSED
                logger.info(f"Indexing job paused: {self.current_job['job_id']}")
            elif action == "resume":
                if self.current_job["status"] == IndexingStatus.PAUSED:
                    self.current_job["status"] = IndexingStatus.PROCESSING
                    logger.info(f"Indexing job resumed: {self.current_job['job_id']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to control indexing: {e}")
            raise
    
    async def get_history(self, limit: int = 20, offset: int = 0) -> IndexingHistoryResponse:
        """
        Get the history of indexing operations.
        
        Args:
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip
            
        Returns:
            IndexingHistoryResponse with job history
        """
        try:
            # For now, return empty history
            # In a full implementation, this would be stored in a database
            return IndexingHistoryResponse(
                jobs=[],
                total_jobs=0,
                success=True,
                message="Indexing history retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to get indexing history: {e}")
            raise
    
    async def get_stats(self) -> IndexingStatsResponse:
        """
        Get comprehensive indexing statistics.
        
        Returns:
            IndexingStatsResponse with statistics
        """
        try:
            # Get database stats
            db_stats = self.vector_db.get_stats()
            total_files = db_stats.get("total_documents", 0)
            
            return IndexingStatsResponse(
                total_indexed_files=total_files,
                total_processing_time=0.0,  # Would be tracked in full implementation
                average_processing_time=0.0,
                total_successful=total_files,
                total_failed=0,
                success_rate=100.0 if total_files > 0 else 0.0,
                files_per_hour=0.0,
                images_indexed=0,  # Would be calculated from database
                videos_indexed=0,
                files_indexed_today=0,
                files_indexed_this_week=0,
                success=True,
                message="Indexing statistics retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to get indexing stats: {e}")
            raise
    
    async def reindex_files(self, file_ids: List[str]) -> str:
        """
        Reindex specific files by their IDs.
        
        Args:
            file_ids: List of file IDs to reindex
            
        Returns:
            Job ID for tracking progress
        """
        try:
            # For now, just return a placeholder job ID
            job_id = str(uuid.uuid4())
            logger.info(f"Reindexing {len(file_ids)} files with job ID: {job_id}")
            
            # In a full implementation, this would:
            # 1. Get file paths from database
            # 2. Re-analyze files with LLM
            # 3. Update embeddings and metadata
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to start reindexing: {e}")
            raise
    
    async def get_watcher_status(self) -> FileWatcherStatus:
        """
        Get the status of the file system watcher.
        
        Returns:
            FileWatcherStatus with watcher information
        """
        return FileWatcherStatus(
            enabled=False,  # Not implemented yet
            watched_paths=[],
            events_processed=0,
            pending_files=0
        )
    
    async def start_file_watcher(self, paths: List[str]) -> None:
        """
        Start monitoring directories for new files.
        
        Args:
            paths: List of directory paths to monitor
        """
        logger.info(f"File watcher requested for paths: {paths}")
        # Placeholder - would implement file system monitoring
        pass
    
    async def stop_file_watcher(self) -> None:
        """Stop the file system watcher."""
        logger.info("File watcher stop requested")
        # Placeholder - would stop file system monitoring
        pass
    
    async def clear_index(self) -> None:
        """
        Clear all indexed data.
        
        WARNING: This operation cannot be undone.
        """
        try:
            success = self.vector_db.clear_collection()
            if success:
                logger.warning("Index cleared successfully")
            else:
                raise Exception("Failed to clear index")
                
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            raise
    
    async def validate_index(self) -> List[Dict[str, Any]]:
        """
        Validate the integrity of the indexed data.
        
        Returns:
            List of validation results
        """
        try:
            validation_results = []
            
            # Get all media documents
            media_docs = self.vector_db.list_media(limit=1000)
            
            for doc in media_docs:
                result = {
                    "file_id": doc.file_id,
                    "file_path": doc.file_path,
                    "valid": True,
                    "issues": []
                }
                
                # Check if file exists
                if not Path(doc.file_path).exists():
                    result["valid"] = False
                    result["issues"].append("File not found on disk")
                
                # Check if embedding exists
                if not doc.embedding:
                    result["valid"] = False
                    result["issues"].append("Missing embedding")
                
                # Check if description exists
                if not doc.ai_description:
                    result["issues"].append("Missing AI description")
                
                validation_results.append(result)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate index: {e}")
            raise
    
    async def monitor_indexing_job(self, job_id: str) -> None:
        """
        Monitor an indexing job (background task).
        
        Args:
            job_id: Job ID to monitor
        """
        logger.info(f"Monitoring indexing job: {job_id}")
        # Placeholder for job monitoring logic
        pass
    
    async def _process_indexing_job(self, request: IndexingRequest) -> None:
        """
        Process an indexing job in the background.
        
        Args:
            request: IndexingRequest with configuration
        """
        try:
            if not self.current_job:
                return
            
            job = self.current_job
            job["status"] = IndexingStatus.SCANNING
            
            # Discover files
            all_files = []
            for path in request.paths:
                files = self._discover_files(path, request.recursive)
                all_files.extend(files)
            
            # Filter files
            filtered_files = self._filter_files(all_files, request)
            
            job["progress"].total_files = len(filtered_files)
            job["status"] = IndexingStatus.PROCESSING
            
            logger.info(f"Found {len(filtered_files)} files to process")
            
            # Process files in batches
            batch_size = request.batch_size or settings.BATCH_SIZE
            
            for i in range(0, len(filtered_files), batch_size):
                if job["status"] == IndexingStatus.CANCELLED:
                    break
                
                batch = filtered_files[i:i + batch_size]
                await self._process_file_batch(batch)
                
                job["progress"].processed_files += len(batch)
            
            # Complete job
            job["status"] = IndexingStatus.COMPLETED
            job["completed_at"] = datetime.utcnow()
            
            logger.info(f"Indexing job completed: {job['job_id']}")
            
        except Exception as e:
            logger.error(f"Indexing job failed: {e}")
            if self.current_job:
                self.current_job["status"] = IndexingStatus.FAILED
    
    def _discover_files(self, path: str, recursive: bool) -> List[str]:
        """
        Discover media files in a directory.
        
        Args:
            path: Directory path to scan
            recursive: Whether to scan recursively
            
        Returns:
            List of file paths
        """
        files = []
        path_obj = Path(path)
        
        if not path_obj.exists() or not path_obj.is_dir():
            return files
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in path_obj.glob(pattern):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix in settings.all_supported_formats:
                    files.append(str(file_path))
        
        return files
    
    def _filter_files(self, files: List[str], request: IndexingRequest) -> List[str]:
        """
        Filter files based on request criteria.
        
        Args:
            files: List of file paths
            request: IndexingRequest with filter criteria
            
        Returns:
            Filtered list of file paths
        """
        filtered = []
        
        for file_path in files:
            path_obj = Path(file_path)
            
            # Check file size
            try:
                file_size = path_obj.stat().st_size
                if request.min_file_size and file_size < request.min_file_size:
                    continue
                if request.max_file_size and file_size > request.max_file_size:
                    continue
            except OSError:
                continue
            
            # Check if already indexed (unless force reindex)
            if not request.force_reindex:
                file_id = self.vector_db._generate_file_id(file_path)
                existing = self.vector_db.get_media(file_id)
                if existing:
                    continue
            
            filtered.append(file_path)
        
        return filtered
    
    async def _process_file_batch(self, files: List[str]) -> None:
        """
        Process a batch of files.
        
        Args:
            files: List of file paths to process
        """
        # For now, just log the files that would be processed
        # In a full implementation, this would:
        # 1. Analyze each file with LLM
        # 2. Generate embeddings
        # 3. Store in vector database
        
        for file_path in files:
            logger.debug(f"Would process file: {file_path}")
            
            if self.current_job:
                self.current_job["progress"].successful_files += 1
