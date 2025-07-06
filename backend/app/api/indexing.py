"""
Indexing API endpoints for media file processing.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
import logging

from app.models.indexing import (
    IndexingRequest,
    IndexingStatusResponse,
    IndexingHistoryResponse,
    IndexingControlRequest,
    IndexingStatsResponse,
    FileWatcherStatus
)
from app.models.common import BaseResponse
from app.services.indexing import IndexingService
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize indexing service
indexing_service = IndexingService()


@router.post("/start", response_model=BaseResponse)
async def start_indexing(request: IndexingRequest, background_tasks: BackgroundTasks):
    """
    Start indexing media files in the specified directories.
    
    This operation runs in the background and can be monitored
    using the status endpoint.
    """
    try:
        logger.info(f"Starting indexing for paths: {request.paths}")
        
        # Validate request
        if not request.paths:
            raise HTTPException(status_code=400, detail="At least one path must be specified")
        
        # Start indexing in background
        job_id = await indexing_service.start_indexing(request)
        
        # Add background task to monitor progress
        background_tasks.add_task(indexing_service.monitor_indexing_job, job_id)
        
        logger.info(f"Indexing started with job ID: {job_id}")
        
        return BaseResponse(
            success=True,
            message=f"Indexing started successfully. Job ID: {job_id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start indexing: {str(e)}")


@router.get("/status", response_model=IndexingStatusResponse)
async def get_indexing_status():
    """
    Get the current status of indexing operations.
    
    Returns detailed information about ongoing and recent
    indexing jobs.
    """
    try:
        status = await indexing_service.get_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get indexing status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.post("/control", response_model=BaseResponse)
async def control_indexing(request: IndexingControlRequest):
    """
    Control indexing operations (pause, resume, cancel).
    
    Allows fine-grained control over running indexing jobs.
    """
    try:
        valid_actions = ["pause", "resume", "cancel", "stop"]
        if request.action not in valid_actions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid action. Must be one of: {valid_actions}"
            )
        
        result = await indexing_service.control_indexing(request.action, request.job_id)
        
        return BaseResponse(
            success=True,
            message=f"Indexing {request.action} completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to control indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to control indexing: {str(e)}")


@router.get("/history", response_model=IndexingHistoryResponse)
async def get_indexing_history(
    limit: int = 20,
    offset: int = 0
):
    """
    Get the history of indexing operations.
    
    Returns information about past indexing jobs including
    performance metrics and results.
    """
    try:
        history = await indexing_service.get_history(limit, offset)
        return history
        
    except Exception as e:
        logger.error(f"Failed to get indexing history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.get("/stats", response_model=IndexingStatsResponse)
async def get_indexing_stats():
    """
    Get comprehensive indexing statistics.
    
    Returns performance metrics, success rates, and other
    statistical information about indexing operations.
    """
    try:
        stats = await indexing_service.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get indexing stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/reindex", response_model=BaseResponse)
async def reindex_files(
    file_ids: List[str],
    background_tasks: BackgroundTasks
):
    """
    Reindex specific files by their IDs.
    
    Useful for updating the analysis of files that may have
    been processed with an older version of the AI model.
    """
    try:
        if not file_ids:
            raise HTTPException(status_code=400, detail="At least one file ID must be specified")
        
        if len(file_ids) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 files can be reindexed at once")
        
        job_id = await indexing_service.reindex_files(file_ids)
        
        # Add background task to monitor progress
        background_tasks.add_task(indexing_service.monitor_indexing_job, job_id)
        
        return BaseResponse(
            success=True,
            message=f"Reindexing started for {len(file_ids)} files. Job ID: {job_id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start reindexing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start reindexing: {str(e)}")


@router.get("/watcher", response_model=FileWatcherStatus)
async def get_file_watcher_status():
    """
    Get the status of the file system watcher.
    
    Returns information about which directories are being
    monitored for new files.
    """
    try:
        status = await indexing_service.get_watcher_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get file watcher status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get watcher status: {str(e)}")


@router.post("/watcher/start", response_model=BaseResponse)
async def start_file_watcher(paths: List[str]):
    """
    Start monitoring directories for new files.
    
    Automatically indexes new files as they are added
    to the monitored directories.
    """
    try:
        if not paths:
            raise HTTPException(status_code=400, detail="At least one path must be specified")
        
        await indexing_service.start_file_watcher(paths)
        
        return BaseResponse(
            success=True,
            message=f"File watcher started for {len(paths)} paths"
        )
        
    except Exception as e:
        logger.error(f"Failed to start file watcher: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start file watcher: {str(e)}")


@router.post("/watcher/stop", response_model=BaseResponse)
async def stop_file_watcher():
    """
    Stop the file system watcher.
    
    Disables automatic indexing of new files.
    """
    try:
        await indexing_service.stop_file_watcher()
        
        return BaseResponse(
            success=True,
            message="File watcher stopped successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to stop file watcher: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop file watcher: {str(e)}")


@router.delete("/clear", response_model=BaseResponse)
async def clear_index():
    """
    Clear all indexed data.
    
    WARNING: This will remove all indexed media information
    and cannot be undone. Use with caution.
    """
    try:
        await indexing_service.clear_index()
        
        return BaseResponse(
            success=True,
            message="Index cleared successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to clear index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {str(e)}")


@router.get("/validate")
async def validate_index():
    """
    Validate the integrity of the indexed data.
    
    Checks for missing files, corrupted data, and other
    potential issues with the index.
    """
    try:
        validation_results = await indexing_service.validate_index()
        
        return {
            "validation_results": validation_results,
            "total_issues": len([r for r in validation_results if not r["valid"]]),
            "total_checked": len(validation_results)
        }
        
    except Exception as e:
        logger.error(f"Failed to validate index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate index: {str(e)}")
