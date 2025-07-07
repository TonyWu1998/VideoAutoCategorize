"""
Media file API endpoints for serving and managing media files.
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Optional
import logging
from pathlib import Path
import mimetypes
import os

from app.models.media import (
    MediaItem,
    MediaMetadata,
    MediaUploadRequest,
    MediaUploadResponse,
    MediaDeleteRequest,
    MediaDeleteResponse,
    MediaStatsResponse,
    MediaType
)
from app.models.common import BaseResponse
from app.services.media import MediaService
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize media service
media_service = MediaService()


# =============================================================================
# LIBRARY MANAGEMENT ENDPOINTS (Must come before /{file_id} route)
# =============================================================================

@router.get("/library")
async def get_library_contents(
    media_type: Optional[MediaType] = Query(None, description="Filter by media type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    sort_by: str = Query("created_date", description="Field to sort by"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")
) -> List[MediaItem]:
    """
    Get all files currently in the media library.

    Returns a paginated list of all indexed media files with their metadata.
    This represents the persistent library contents.
    """
    try:
        files = await media_service.list_files(
            media_type=media_type,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order
        )
        return files

    except Exception as e:
        logger.error(f"Failed to get library contents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get library contents: {str(e)}")


@router.delete("/library")
async def remove_from_library(
    file_ids: List[str] = Query(..., description="List of file IDs to remove from library"),
    delete_from_disk: bool = Query(False, description="Whether to also delete files from disk"),
    force: bool = Query(False, description="Force removal even if files are missing")
) -> MediaDeleteResponse:
    """
    Remove files from the media library.

    This removes files from the vector database, metadata database,
    and optionally from disk. Thumbnails and cached data are also cleaned up.
    """
    try:
        result = await media_service.delete_files(
            file_ids=file_ids,
            delete_from_disk=delete_from_disk,
            force=force
        )
        return result

    except Exception as e:
        logger.error(f"Failed to remove files from library: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove files from library: {str(e)}")


@router.get("/library/stats")
async def get_library_stats() -> MediaStatsResponse:
    """
    Get statistics about the current media library.

    Returns information about total files, file types, storage usage, etc.
    """
    try:
        stats = await media_service.get_stats()
        return stats

    except Exception as e:
        logger.error(f"Failed to get library stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get library stats: {str(e)}")


@router.post("/library/validate")
async def validate_library():
    """
    Validate the integrity of the media library.

    Checks for missing files, orphaned database entries, and other issues.
    """
    try:
        # Import here to avoid circular imports
        from app.services.indexing import IndexingService
        indexing_service = IndexingService()

        validation_results = await indexing_service.validate_index()

        return {
            "validation_results": validation_results,
            "total_issues": len([r for r in validation_results if not r.get("valid", True)]),
            "total_checked": len(validation_results),
            "success": True
        }

    except Exception as e:
        logger.error(f"Failed to validate library: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate library: {str(e)}")


@router.get("/formats/supported")
async def get_supported_formats():
    """
    Get list of supported media file formats.

    Returns the file extensions that can be processed by the system.
    """
    return {
        "image_formats": settings.SUPPORTED_IMAGE_FORMATS,
        "video_formats": settings.SUPPORTED_VIDEO_FORMATS,
        "all_formats": settings.all_supported_formats
    }


# =============================================================================
# MEDIA FILE SERVING ENDPOINTS
# =============================================================================

@router.get("/{file_id}")
async def get_media_file(file_id: str):
    """
    Serve a media file by its ID.
    
    Returns the actual media file content with appropriate
    headers for browser display.
    """
    try:
        file_path = await media_service.get_file_path(file_id)
        
        if not file_path or not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="Media file not found")
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"
        
        return FileResponse(
            path=file_path,
            media_type=mime_type,
            filename=Path(file_path).name
        )
        
    except Exception as e:
        logger.error(f"Failed to serve media file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve file: {str(e)}")


@router.get("/{file_id}/thumbnail")
async def get_media_thumbnail(
    file_id: str,
    size: int = Query(200, ge=50, le=500, description="Thumbnail size in pixels")
):
    """
    Get a thumbnail for a media file.

    Generates and caches thumbnails for efficient preview display.
    """
    try:
        thumbnail_path = await media_service.get_thumbnail(file_id, size)

        if not thumbnail_path or not Path(thumbnail_path).exists():
            raise HTTPException(status_code=404, detail="Thumbnail not found")

        return FileResponse(
            path=thumbnail_path,
            media_type="image/jpeg",
            filename=f"thumbnail_{file_id}_{size}.jpg"
        )

    except Exception as e:
        logger.error(f"Failed to serve thumbnail for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve thumbnail: {str(e)}")


@router.get("/{file_id}/frames")
async def get_video_frames(
    file_id: str,
    frame_count: int = Query(5, ge=1, le=10, description="Number of frames to extract"),
    size: int = Query(300, ge=100, le=500, description="Frame size in pixels")
):
    """
    Get multiple preview frames from a video file.

    Returns a list of frame URLs for detailed video preview.
    """
    try:
        frame_paths = await media_service.get_video_frames(file_id, frame_count, size)

        if not frame_paths:
            raise HTTPException(status_code=404, detail="Video frames not found")

        # Return relative URLs for the frames (frontend will add base URL)
        base_url = f"/api/media/{file_id}/frame"
        frame_urls = [f"{base_url}/{i}?size={size}" for i in range(len(frame_paths))]

        return {
            "file_id": file_id,
            "frame_count": len(frame_urls),
            "frame_urls": frame_urls
        }

    except Exception as e:
        logger.error(f"Failed to get video frames for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get video frames: {str(e)}")


@router.get("/{file_id}/frame/{frame_index}")
async def get_video_frame(
    file_id: str,
    frame_index: int,
    size: int = Query(300, ge=100, le=500, description="Frame size in pixels")
):
    """
    Get a specific frame from a video file.

    Serves individual video frames for detailed preview.
    """
    try:
        frame_paths = await media_service.get_video_frames(file_id, 5, size)

        if not frame_paths or frame_index >= len(frame_paths):
            raise HTTPException(status_code=404, detail="Video frame not found")

        frame_path = frame_paths[frame_index]
        if not Path(frame_path).exists():
            raise HTTPException(status_code=404, detail="Video frame not found")

        return FileResponse(
            path=frame_path,
            media_type="image/jpeg",
            filename=f"frame_{file_id}_{frame_index}_{size}.jpg"
        )

    except Exception as e:
        logger.error(f"Failed to serve video frame {frame_index} for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve video frame: {str(e)}")


@router.post("/{file_id}/open-in-explorer")
async def open_file_in_explorer(file_id: str):
    """
    Open a media file in the system's default file explorer.

    Cross-platform support for Windows, macOS, and Linux.
    """
    try:
        success = await media_service.open_file_in_explorer(file_id)

        if not success:
            raise HTTPException(status_code=404, detail="File not found or could not open explorer")

        return {
            "success": True,
            "message": "File opened in explorer successfully"
        }

    except Exception as e:
        logger.error(f"Failed to open file {file_id} in explorer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to open file in explorer: {str(e)}")


@router.get("/{file_id}/metadata", response_model=MediaMetadata)
async def get_media_metadata(file_id: str):
    """
    Get detailed metadata for a media file.
    
    Returns comprehensive information about the file including
    AI-generated descriptions and tags.
    """
    try:
        metadata = await media_service.get_metadata(file_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Media metadata not found")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to get metadata for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")


@router.post("/upload", response_model=MediaUploadResponse)
async def upload_media_file(
    file: UploadFile = File(...),
    auto_index: bool = Query(True, description="Whether to automatically index the uploaded file")
):
    """
    Upload a new media file.
    
    Stores the file and optionally indexes it for search.
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.all_supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file_extension}"
            )
        
        # Upload the file
        upload_result = await media_service.upload_file(file, auto_index)
        
        return upload_result
        
    except Exception as e:
        logger.error(f"Failed to upload file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.delete("/", response_model=MediaDeleteResponse)
async def delete_media_files(request: MediaDeleteRequest):
    """
    Delete media files by their IDs.
    
    Can optionally delete the actual files from disk or just
    remove them from the index.
    """
    try:
        if not request.file_ids:
            raise HTTPException(status_code=400, detail="No file IDs provided")
        
        delete_result = await media_service.delete_files(request)
        
        return delete_result
        
    except Exception as e:
        logger.error(f"Failed to delete files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete files: {str(e)}")


@router.get("/", response_model=List[MediaItem])
async def list_media_files(
    media_type: MediaType = Query(MediaType.ALL, description="Filter by media type"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of files to return"),
    offset: int = Query(0, ge=0, description="Number of files to skip"),
    sort_by: str = Query("created_date", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)")
):
    """
    List media files with optional filtering and pagination.
    
    Returns a paginated list of media files with basic metadata.
    """
    try:
        files = await media_service.list_files(
            media_type=media_type,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        return files
        
    except Exception as e:
        logger.error(f"Failed to list media files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@router.get("/stats", response_model=MediaStatsResponse)
async def get_media_stats():
    """
    Get comprehensive statistics about indexed media files.
    
    Returns counts, sizes, and other statistical information
    about the media collection.
    """
    try:
        stats = await media_service.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get media stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/{file_id}/regenerate-thumbnail", response_model=BaseResponse)
async def regenerate_thumbnail(file_id: str):
    """
    Regenerate thumbnail for a specific media file.
    
    Useful when thumbnails are corrupted or need to be updated.
    """
    try:
        await media_service.regenerate_thumbnail(file_id)
        
        return BaseResponse(
            success=True,
            message=f"Thumbnail regenerated for file {file_id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to regenerate thumbnail for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to regenerate thumbnail: {str(e)}")


@router.get("/{file_id}/download")
async def download_media_file(file_id: str):
    """
    Download a media file with appropriate headers for file download.
    
    Forces browser to download the file rather than display it.
    """
    try:
        file_path = await media_service.get_file_path(file_id)
        
        if not file_path or not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="Media file not found")
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"
        
        return FileResponse(
            path=file_path,
            media_type=mime_type,
            filename=Path(file_path).name,
            headers={"Content-Disposition": f"attachment; filename={Path(file_path).name}"}
        )
        
    except Exception as e:
        logger.error(f"Failed to download media file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")


@router.post("/{file_id}/update-metadata", response_model=BaseResponse)
async def update_media_metadata(
    file_id: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None
):
    """
    Update metadata for a media file.
    
    Allows manual editing of AI-generated descriptions and tags.
    """
    try:
        await media_service.update_metadata(file_id, description, tags)
        
        return BaseResponse(
            success=True,
            message=f"Metadata updated for file {file_id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to update metadata for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update metadata: {str(e)}")



