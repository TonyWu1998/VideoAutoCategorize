"""
Media service for file management and serving operations.
"""

from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import shutil
import mimetypes
from datetime import datetime
from PIL import Image
import io

from app.models.media import (
    MediaItem, MediaMetadata, MediaType, MediaUploadResponse,
    MediaDeleteRequest, MediaDeleteResponse, MediaStatsResponse
)
from app.database.vector_db import VectorDatabase
from app.config import settings

logger = logging.getLogger(__name__)


class MediaService:
    """
    Service for managing media files and metadata operations.
    
    Handles file serving, thumbnail generation, and metadata management.
    """
    
    def __init__(self):
        """Initialize the media service."""
        self.vector_db = VectorDatabase()
        self.thumbnail_cache = {}
        
        # Ensure media directories exist
        self.media_dir = Path("./data/media")
        self.thumbnail_dir = Path("./data/thumbnails")
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Media service initialized")
    
    async def get_file_path(self, file_id: str) -> Optional[str]:
        """
        Get the file path for a media file by its ID.
        
        Args:
            file_id: Unique identifier for the file
            
        Returns:
            File path if found, None otherwise
        """
        try:
            media_doc = self.vector_db.get_media(file_id)
            if media_doc and Path(media_doc.file_path).exists():
                return media_doc.file_path
            return None
            
        except Exception as e:
            logger.error(f"Failed to get file path for {file_id}: {e}")
            return None
    
    async def get_thumbnail(self, file_id: str, size: int = 200) -> Optional[str]:
        """
        Get or generate a thumbnail for a media file.
        
        Args:
            file_id: Unique identifier for the file
            size: Thumbnail size in pixels
            
        Returns:
            Path to thumbnail file if successful, None otherwise
        """
        try:
            # Check cache first
            cache_key = f"{file_id}_{size}"
            if cache_key in self.thumbnail_cache:
                thumbnail_path = self.thumbnail_cache[cache_key]
                if Path(thumbnail_path).exists():
                    return thumbnail_path
            
            # Get original file path
            file_path = await self.get_file_path(file_id)
            if not file_path:
                return None
            
            # Generate thumbnail path
            thumbnail_filename = f"{file_id}_{size}.jpg"
            thumbnail_path = self.thumbnail_dir / thumbnail_filename
            
            # Generate thumbnail if it doesn't exist
            if not thumbnail_path.exists():
                success = await self._generate_thumbnail(file_path, str(thumbnail_path), size)
                if not success:
                    return None
            
            # Cache the result
            self.thumbnail_cache[cache_key] = str(thumbnail_path)
            
            return str(thumbnail_path)
            
        except Exception as e:
            logger.error(f"Failed to get thumbnail for {file_id}: {e}")
            return None
    
    async def get_metadata(self, file_id: str) -> Optional[MediaMetadata]:
        """
        Get detailed metadata for a media file.
        
        Args:
            file_id: Unique identifier for the file
            
        Returns:
            MediaMetadata if found, None otherwise
        """
        try:
            media_doc = self.vector_db.get_media(file_id)
            if not media_doc:
                return None
            
            return MediaMetadata(
                file_path=media_doc.file_path,
                file_name=media_doc.file_name,
                file_size=media_doc.file_size,
                created_date=media_doc.created_date,
                modified_date=media_doc.modified_date,
                media_type=MediaType(media_doc.media_type),
                dimensions=media_doc.dimensions,
                duration=media_doc.duration,
                format=media_doc.format,
                ai_description=media_doc.ai_description,
                ai_tags=media_doc.ai_tags,
                ai_confidence=media_doc.ai_confidence,
                indexed_date=media_doc.indexed_date,
                index_version=media_doc.index_version
            )
            
        except Exception as e:
            logger.error(f"Failed to get metadata for {file_id}: {e}")
            return None
    
    async def upload_file(self, file, auto_index: bool = True) -> MediaUploadResponse:
        """
        Upload a new media file.
        
        Args:
            file: Uploaded file object
            auto_index: Whether to automatically index the file
            
        Returns:
            MediaUploadResponse with upload results
        """
        try:
            # Generate unique filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{file.filename}"
            file_path = self.media_dir / filename
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_size = file_path.stat().st_size
            file_id = self.vector_db._generate_file_id(str(file_path))
            
            logger.info(f"Uploaded file: {filename} ({file_size} bytes)")
            
            return MediaUploadResponse(
                file_id=file_id,
                file_path=str(file_path),
                file_size=file_size,
                indexed=False,  # Would be True if auto_index was implemented
                success=True,
                message="File uploaded successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            raise
    
    async def delete_files(self, request: MediaDeleteRequest) -> MediaDeleteResponse:
        """
        Delete media files by their IDs.
        
        Args:
            request: MediaDeleteRequest with file IDs and options
            
        Returns:
            MediaDeleteResponse with deletion results
        """
        try:
            deleted_count = 0
            failed_count = 0
            errors = []
            
            for file_id in request.file_ids:
                try:
                    # Get file path
                    file_path = await self.get_file_path(file_id)
                    
                    # Delete from database
                    db_success = self.vector_db.delete_media(file_id)
                    
                    # Delete from disk if requested
                    if request.delete_from_disk and file_path:
                        Path(file_path).unlink(missing_ok=True)
                    
                    # Delete thumbnail
                    await self._delete_thumbnails(file_id)
                    
                    if db_success:
                        deleted_count += 1
                    else:
                        failed_count += 1
                        errors.append({"file_id": file_id, "error": "Database deletion failed"})
                        
                except Exception as e:
                    failed_count += 1
                    errors.append({"file_id": file_id, "error": str(e)})
            
            return MediaDeleteResponse(
                deleted_count=deleted_count,
                failed_count=failed_count,
                errors=errors,
                success=True,
                message=f"Deleted {deleted_count} files, {failed_count} failed"
            )
            
        except Exception as e:
            logger.error(f"Failed to delete files: {e}")
            raise
    
    async def list_files(
        self,
        media_type: MediaType = MediaType.ALL,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "created_date",
        sort_order: str = "desc"
    ) -> List[MediaItem]:
        """
        List media files with optional filtering and pagination.
        
        Args:
            media_type: Filter by media type
            limit: Maximum number of files to return
            offset: Number of files to skip
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            
        Returns:
            List of MediaItem objects
        """
        try:
            # Build filters
            filters = {}
            if media_type != MediaType.ALL:
                filters["media_type"] = media_type.value
            
            # Get documents from database
            media_docs = self.vector_db.list_media(
                limit=limit,
                offset=offset,
                filters=filters
            )
            
            # Convert to MediaItem objects
            media_items = []
            for doc in media_docs:
                metadata = MediaMetadata(
                    file_path=doc.file_path,
                    file_name=doc.file_name,
                    file_size=doc.file_size,
                    created_date=doc.created_date,
                    modified_date=doc.modified_date,
                    media_type=MediaType(doc.media_type),
                    dimensions=doc.dimensions,
                    duration=doc.duration,
                    format=doc.format,
                    ai_description=doc.ai_description,
                    ai_tags=doc.ai_tags,
                    ai_confidence=doc.ai_confidence,
                    indexed_date=doc.indexed_date,
                    index_version=doc.index_version
                )
                
                media_item = MediaItem(
                    file_id=doc.file_id,
                    metadata=metadata,
                    thumbnail_url=f"/api/media/{doc.file_id}/thumbnail",
                    preview_url=f"/api/media/{doc.file_id}"
                )
                
                media_items.append(media_item)
            
            # Sort if needed (basic implementation)
            if sort_by == "created_date":
                reverse = sort_order == "desc"
                media_items.sort(key=lambda x: x.metadata.created_date, reverse=reverse)
            
            return media_items
            
        except Exception as e:
            logger.error(f"Failed to list media files: {e}")
            return []
    
    async def get_stats(self) -> MediaStatsResponse:
        """
        Get comprehensive statistics about indexed media files.
        
        Returns:
            MediaStatsResponse with statistics
        """
        try:
            # Get database stats
            db_stats = self.vector_db.get_stats()
            
            return MediaStatsResponse(
                total_files=db_stats.get("total_documents", 0),
                total_size_bytes=db_stats.get("estimated_total_size_bytes", 0),
                image_count=db_stats.get("media_types", {}).get("image", 0),
                video_count=db_stats.get("media_types", {}).get("video", 0),
                format_breakdown=db_stats.get("formats", {}),
                oldest_file=None,  # Would be calculated from database
                newest_file=None,  # Would be calculated from database
                last_index_date=None,  # Would be tracked
                pending_indexing=0,  # Would be calculated
                success=True,
                message="Media statistics retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to get media stats: {e}")
            raise
    
    async def regenerate_thumbnail(self, file_id: str) -> None:
        """
        Regenerate thumbnail for a specific media file.
        
        Args:
            file_id: Unique identifier for the file
        """
        try:
            # Delete existing thumbnails
            await self._delete_thumbnails(file_id)
            
            # Clear cache
            keys_to_remove = [key for key in self.thumbnail_cache.keys() if key.startswith(file_id)]
            for key in keys_to_remove:
                del self.thumbnail_cache[key]
            
            logger.info(f"Thumbnails regenerated for file: {file_id}")
            
        except Exception as e:
            logger.error(f"Failed to regenerate thumbnail for {file_id}: {e}")
            raise
    
    async def update_metadata(
        self,
        file_id: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Update metadata for a media file.
        
        Args:
            file_id: Unique identifier for the file
            description: New description
            tags: New tags list
        """
        try:
            # Get existing document
            media_doc = self.vector_db.get_media(file_id)
            if not media_doc:
                raise ValueError(f"File not found: {file_id}")
            
            # Update fields
            if description is not None:
                media_doc.ai_description = description
            if tags is not None:
                media_doc.ai_tags = tags
            
            # Update in database
            success = self.vector_db.update_media(media_doc)
            if not success:
                raise Exception("Failed to update database")
            
            logger.info(f"Metadata updated for file: {file_id}")
            
        except Exception as e:
            logger.error(f"Failed to update metadata for {file_id}: {e}")
            raise
    
    async def _generate_thumbnail(self, file_path: str, thumbnail_path: str, size: int) -> bool:
        """
        Generate a thumbnail for a media file.
        
        Args:
            file_path: Path to the original file
            thumbnail_path: Path where thumbnail should be saved
            size: Thumbnail size in pixels
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in settings.SUPPORTED_IMAGE_FORMATS:
                return await self._generate_image_thumbnail(file_path, thumbnail_path, size)
            elif file_ext in settings.SUPPORTED_VIDEO_FORMATS:
                return await self._generate_video_thumbnail(file_path, thumbnail_path, size)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to generate thumbnail: {e}")
            return False
    
    async def _generate_image_thumbnail(self, file_path: str, thumbnail_path: str, size: int) -> bool:
        """Generate thumbnail for an image file."""
        try:
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create thumbnail
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                
                # Save thumbnail
                img.save(thumbnail_path, 'JPEG', quality=85)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to generate image thumbnail: {e}")
            return False
    
    async def _generate_video_thumbnail(self, file_path: str, thumbnail_path: str, size: int) -> bool:
        """Generate thumbnail for a video file."""
        try:
            import cv2
            import asyncio

            def extract_frame():
                # Open video file
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    logger.error(f"Could not open video file: {file_path}")
                    return False

                try:
                    # Get video properties
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    if total_frames <= 0 or fps <= 0:
                        logger.error(f"Invalid video properties: frames={total_frames}, fps={fps}")
                        return False

                    # Extract frame at 10% of video duration
                    target_frame = int(total_frames * 0.1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        # Fallback to first frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            logger.error(f"Could not read frame from video: {file_path}")
                            return False

                    # Resize frame to thumbnail size
                    height, width = frame.shape[:2]
                    aspect_ratio = width / height

                    if aspect_ratio > 1:  # Landscape
                        new_width = size
                        new_height = int(size / aspect_ratio)
                    else:  # Portrait or square
                        new_height = size
                        new_width = int(size * aspect_ratio)

                    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

                    # Save thumbnail
                    success = cv2.imwrite(thumbnail_path, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not success:
                        logger.error(f"Failed to save thumbnail: {thumbnail_path}")
                        return False

                    logger.info(f"Generated video thumbnail: {thumbnail_path}")
                    return True

                finally:
                    cap.release()

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, extract_frame)

        except Exception as e:
            logger.error(f"Failed to generate video thumbnail: {e}")
            return False

    async def get_video_frames(self, file_id: str, frame_count: int = 5, size: int = 300) -> List[str]:
        """
        Get multiple preview frames from a video file.

        Args:
            file_id: Unique identifier for the video file
            frame_count: Number of frames to extract
            size: Frame size in pixels

        Returns:
            List of paths to frame files
        """
        try:
            # Get original file path
            file_path = await self.get_file_path(file_id)
            if not file_path:
                return []

            # Check if it's a video file
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in settings.SUPPORTED_VIDEO_FORMATS:
                return []

            # Generate frame paths
            frame_paths = []
            for i in range(frame_count):
                frame_filename = f"{file_id}_frame_{i}_{size}.jpg"
                frame_path = self.thumbnail_dir / frame_filename
                frame_paths.append(str(frame_path))

            # Check if frames already exist
            if all(Path(path).exists() for path in frame_paths):
                return frame_paths

            # Generate frames
            success = await self._generate_video_frames(file_path, frame_paths, frame_count, size)
            if success:
                return frame_paths
            else:
                return []

        except Exception as e:
            logger.error(f"Failed to get video frames for {file_id}: {e}")
            return []

    async def _generate_video_frames(self, file_path: str, frame_paths: List[str], frame_count: int, size: int) -> bool:
        """Generate multiple preview frames from a video file."""
        try:
            import cv2
            import asyncio

            def extract_frames():
                # Open video file
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    logger.error(f"Could not open video file: {file_path}")
                    return False

                try:
                    # Get video properties
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    if total_frames <= 0 or fps <= 0:
                        logger.error(f"Invalid video properties: frames={total_frames}, fps={fps}")
                        return False

                    # Calculate frame positions (evenly distributed)
                    frame_positions = []
                    for i in range(frame_count):
                        # Extract frames at 20%, 40%, 60%, 80%, etc. of video duration
                        position = (i + 1) * 0.2
                        if position > 0.9:  # Don't go too close to the end
                            position = 0.9
                        frame_positions.append(int(total_frames * position))

                    # Extract and save frames
                    for i, frame_pos in enumerate(frame_positions):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                        ret, frame = cap.read()

                        if not ret or frame is None:
                            logger.warning(f"Could not read frame at position {frame_pos}")
                            continue

                        # Resize frame
                        height, width = frame.shape[:2]
                        aspect_ratio = width / height

                        if aspect_ratio > 1:  # Landscape
                            new_width = size
                            new_height = int(size / aspect_ratio)
                        else:  # Portrait or square
                            new_height = size
                            new_width = int(size * aspect_ratio)

                        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

                        # Save frame
                        success = cv2.imwrite(frame_paths[i], resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if not success:
                            logger.error(f"Failed to save frame: {frame_paths[i]}")
                            return False

                    logger.info(f"Generated {frame_count} video frames for: {file_path}")
                    return True

                finally:
                    cap.release()

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, extract_frames)

        except Exception as e:
            logger.error(f"Failed to generate video frames: {e}")
            return False

    async def open_file_in_explorer(self, file_id: str) -> bool:
        """
        Open a file in the system's default file explorer.

        Args:
            file_id: Unique identifier for the file

        Returns:
            True if successful, False otherwise
        """
        try:
            import subprocess
            import platform
            import asyncio

            # Get file path
            file_path = await self.get_file_path(file_id)
            if not file_path or not Path(file_path).exists():
                logger.error(f"File not found: {file_id}")
                return False

            def open_explorer():
                system = platform.system()
                try:
                    if system == "Windows":
                        # Windows: Use explorer with /select flag
                        subprocess.run(["explorer", "/select,", file_path], check=True)
                    elif system == "Darwin":  # macOS
                        # macOS: Use open with -R flag to reveal in Finder
                        subprocess.run(["open", "-R", file_path], check=True)
                    elif system == "Linux":
                        # Linux: Open directory containing the file
                        directory = str(Path(file_path).parent)
                        subprocess.run(["xdg-open", directory], check=True)
                    else:
                        logger.error(f"Unsupported operating system: {system}")
                        return False

                    logger.info(f"Opened file in explorer: {file_path}")
                    return True

                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to open file in explorer: {e}")
                    return False
                except FileNotFoundError as e:
                    logger.error(f"Explorer command not found: {e}")
                    return False

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, open_explorer)

        except Exception as e:
            logger.error(f"Failed to open file in explorer: {e}")
            return False
    
    async def _delete_thumbnails(self, file_id: str) -> None:
        """Delete all thumbnails for a file."""
        try:
            # Find and delete thumbnail files
            for thumbnail_file in self.thumbnail_dir.glob(f"{file_id}_*.jpg"):
                thumbnail_file.unlink(missing_ok=True)
                
        except Exception as e:
            logger.warning(f"Failed to delete thumbnails for {file_id}: {e}")
