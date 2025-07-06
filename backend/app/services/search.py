"""
Search service for semantic media search operations.
"""

from typing import List, Dict, Any, Optional
import logging
import time
from collections import defaultdict

from app.models.media import MediaItem, SearchFilters, MediaMetadata, MediaType
from app.database.vector_db import VectorDatabase
from app.services.llm_service import LLMService
from app.config import settings

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service for performing semantic search operations on media files.
    
    Handles query processing, vector similarity search, and result formatting.
    """
    
    def __init__(self):
        """Initialize the search service."""
        self.vector_db = VectorDatabase()
        self.llm_service = LLMService()
        self._search_cache = {}  # Simple in-memory cache
        
        logger.info("Search service initialized")
    
    async def semantic_search(
        self,
        query: str,
        filters: SearchFilters,
        include_metadata: bool = True,
        include_thumbnails: bool = True
    ) -> List[MediaItem]:
        """
        Perform semantic search on indexed media files.
        
        Args:
            query: Natural language search query
            filters: Search filters to apply
            include_metadata: Whether to include full metadata
            include_thumbnails: Whether to include thumbnail URLs
            
        Returns:
            List of matching MediaItem objects
        """
        try:
            logger.debug(f"Performing semantic search for: '{query}'")
            
            # Check cache first
            cache_key = self._generate_cache_key(query, filters)
            if settings.ENABLE_SEARCH_CACHE and cache_key in self._search_cache:
                cached_result = self._search_cache[cache_key]
                if time.time() - cached_result["timestamp"] < settings.SEARCH_CACHE_TTL_SECONDS:
                    logger.debug("Returning cached search results")
                    return cached_result["results"]
            
            # Generate query embedding
            query_embedding = await self.llm_service.generate_query_embedding(query)
            
            # Build search filters for vector database
            db_filters = self._build_db_filters(filters)
            
            # Perform vector similarity search
            search_results = self.vector_db.search_similar(
                query_embedding=query_embedding,
                limit=filters.max_results,
                filters=db_filters
            )
            
            # Filter results by similarity threshold
            filtered_results = [
                result for result in search_results
                if result["similarity_score"] >= filters.min_similarity
            ]
            
            # Convert to MediaItem objects
            media_items = []
            for result in filtered_results:
                try:
                    media_item = await self._create_media_item(
                        result,
                        include_metadata,
                        include_thumbnails
                    )
                    media_items.append(media_item)
                except Exception as e:
                    logger.warning(f"Failed to create media item: {e}")
                    continue
            
            # Apply additional filters
            media_items = self._apply_additional_filters(media_items, filters)
            
            # Sort by similarity score
            media_items.sort(key=lambda x: x.similarity_score or 0, reverse=True)
            
            # Cache results
            if settings.ENABLE_SEARCH_CACHE:
                self._search_cache[cache_key] = {
                    "results": media_items,
                    "timestamp": time.time()
                }
            
            logger.debug(f"Search completed, returning {len(media_items)} results")
            return media_items
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    async def find_similar_media(
        self,
        file_id: str,
        limit: int = 10,
        min_similarity: float = 0.5
    ) -> List[MediaItem]:
        """
        Find media files similar to a specific file.
        
        Args:
            file_id: ID of the reference file
            limit: Maximum number of similar items to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar MediaItem objects
        """
        try:
            # Get the reference document
            reference_doc = self.vector_db.get_media(file_id)
            if not reference_doc or not reference_doc.embedding:
                raise ValueError(f"File not found or has no embedding: {file_id}")
            
            # Search for similar items
            search_results = self.vector_db.search_similar(
                query_embedding=reference_doc.embedding,
                limit=limit + 1  # +1 to account for the reference file itself
            )
            
            # Filter out the reference file and apply similarity threshold
            similar_items = []
            for result in search_results:
                if (result["file_id"] != file_id and 
                    result["similarity_score"] >= min_similarity):
                    
                    media_item = await self._create_media_item(result, True, True)
                    similar_items.append(media_item)
            
            return similar_items[:limit]
            
        except Exception as e:
            logger.error(f"Failed to find similar media for {file_id}: {e}")
            raise
    
    async def get_search_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        Args:
            partial_query: Partial search query
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested search terms
        """
        try:
            # For now, return simple suggestions based on common tags
            # In a full implementation, this could use a more sophisticated approach
            suggestions = []
            
            # Get popular tags that match the partial query
            popular_tags = await self.get_popular_tags(limit * 2, 1)
            
            for tag_info in popular_tags:
                tag = tag_info["tag"]
                if partial_query.lower() in tag.lower():
                    suggestions.append(tag)
            
            # Add some common search patterns
            common_patterns = [
                "beach sunset", "family photo", "landscape", "portrait",
                "city skyline", "nature", "animals", "food", "travel",
                "celebration", "sports", "architecture", "art"
            ]
            
            for pattern in common_patterns:
                if (partial_query.lower() in pattern.lower() and 
                    pattern not in suggestions):
                    suggestions.append(pattern)
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return []
    
    async def get_popular_tags(self, limit: int = 50, min_frequency: int = 2) -> List[Dict[str, Any]]:
        """
        Get popular tags from indexed media files.
        
        Args:
            limit: Maximum number of tags to return
            min_frequency: Minimum tag frequency
            
        Returns:
            List of tag information dictionaries
        """
        try:
            # Get all media documents
            all_media = self.vector_db.list_media(limit=1000)  # Sample for performance
            
            # Count tag frequencies
            tag_counts = defaultdict(int)
            for media_doc in all_media:
                for tag in media_doc.ai_tags:
                    if tag and len(tag) > 2:  # Filter out very short tags
                        tag_counts[tag.lower()] += 1
            
            # Filter by minimum frequency and sort
            popular_tags = [
                {"tag": tag, "frequency": count}
                for tag, count in tag_counts.items()
                if count >= min_frequency
            ]
            
            popular_tags.sort(key=lambda x: x["frequency"], reverse=True)
            
            return popular_tags[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get popular tags: {e}")
            return []
    
    async def clear_cache(self) -> None:
        """Clear the search result cache."""
        self._search_cache.clear()
        logger.info("Search cache cleared")
    
    def _generate_cache_key(self, query: str, filters: SearchFilters) -> str:
        """Generate a cache key for the search query and filters."""
        # Create a simple hash of the query and key filter parameters
        key_parts = [
            query.lower().strip(),
            filters.media_type.value,
            str(filters.min_similarity),
            str(filters.max_results)
        ]
        return "|".join(key_parts)
    
    def _build_db_filters(self, filters: SearchFilters) -> Dict[str, Any]:
        """
        Build database filters from search filters.
        
        Args:
            filters: SearchFilters object
            
        Returns:
            Dictionary of database filters
        """
        db_filters = {}
        
        # Media type filter
        if filters.media_type != MediaType.ALL:
            db_filters["media_type"] = filters.media_type.value
        
        # File size filters
        if filters.min_file_size:
            db_filters["file_size_min"] = filters.min_file_size
        if filters.max_file_size:
            db_filters["file_size_max"] = filters.max_file_size
        
        return db_filters
    
    async def _create_media_item(
        self,
        search_result: Dict[str, Any],
        include_metadata: bool,
        include_thumbnails: bool
    ) -> MediaItem:
        """
        Create a MediaItem from search result.
        
        Args:
            search_result: Raw search result from vector database
            include_metadata: Whether to include full metadata
            include_thumbnails: Whether to include thumbnail URLs
            
        Returns:
            MediaItem object
        """
        file_id = search_result["file_id"]
        metadata_doc = search_result["metadata"]
        similarity_score = search_result["similarity_score"]
        
        # Create MediaMetadata
        metadata = MediaMetadata(
            file_path=metadata_doc.file_path,
            file_name=metadata_doc.file_name,
            file_size=metadata_doc.file_size,
            created_date=metadata_doc.created_date,
            modified_date=metadata_doc.modified_date,
            media_type=MediaType(metadata_doc.media_type),
            dimensions=metadata_doc.dimensions,
            duration=metadata_doc.duration,
            format=metadata_doc.format,
            ai_description=metadata_doc.ai_description if include_metadata else None,
            ai_tags=metadata_doc.ai_tags if include_metadata else [],
            ai_confidence=metadata_doc.ai_confidence,
            indexed_date=metadata_doc.indexed_date,
            index_version=metadata_doc.index_version
        )
        
        # Generate URLs
        thumbnail_url = f"/api/media/{file_id}/thumbnail" if include_thumbnails else None
        preview_url = f"/api/media/{file_id}" if include_metadata else None
        
        return MediaItem(
            file_id=file_id,
            metadata=metadata,
            similarity_score=similarity_score,
            thumbnail_url=thumbnail_url,
            preview_url=preview_url
        )
    
    def _apply_additional_filters(
        self,
        media_items: List[MediaItem],
        filters: SearchFilters
    ) -> List[MediaItem]:
        """
        Apply additional filters that couldn't be handled at the database level.
        
        Args:
            media_items: List of MediaItem objects
            filters: SearchFilters to apply
            
        Returns:
            Filtered list of MediaItem objects
        """
        filtered_items = media_items
        
        # Date range filter
        if filters.date_range:
            start_date = filters.date_range.get("start")
            end_date = filters.date_range.get("end")
            
            if start_date or end_date:
                filtered_items = [
                    item for item in filtered_items
                    if self._date_in_range(item.metadata.created_date, start_date, end_date)
                ]
        
        # Dimension filters
        if any([filters.min_width, filters.max_width, filters.min_height, filters.max_height]):
            filtered_items = [
                item for item in filtered_items
                if self._dimensions_match(item.metadata.dimensions, filters)
            ]
        
        # Duration filter (for videos)
        if filters.min_duration or filters.max_duration:
            filtered_items = [
                item for item in filtered_items
                if self._duration_matches(item.metadata.duration, filters)
            ]
        
        # Tag filters
        if filters.include_tags or filters.exclude_tags:
            filtered_items = [
                item for item in filtered_items
                if self._tags_match(item.metadata.ai_tags, filters)
            ]
        
        # Path filters
        if filters.include_paths or filters.exclude_paths:
            filtered_items = [
                item for item in filtered_items
                if self._path_matches(item.metadata.file_path, filters)
            ]
        
        return filtered_items
    
    def _date_in_range(self, date, start_date, end_date) -> bool:
        """Check if date is within the specified range."""
        if start_date and date < start_date:
            return False
        if end_date and date > end_date:
            return False
        return True
    
    def _dimensions_match(self, dimensions: Optional[str], filters: SearchFilters) -> bool:
        """Check if dimensions match the filter criteria."""
        if not dimensions:
            return True
        
        try:
            width, height = map(int, dimensions.split('x'))
            
            if filters.min_width and width < filters.min_width:
                return False
            if filters.max_width and width > filters.max_width:
                return False
            if filters.min_height and height < filters.min_height:
                return False
            if filters.max_height and height > filters.max_height:
                return False
            
            return True
        except (ValueError, AttributeError):
            return True
    
    def _duration_matches(self, duration: Optional[float], filters: SearchFilters) -> bool:
        """Check if duration matches the filter criteria."""
        if duration is None:
            return True
        
        if filters.min_duration and duration < filters.min_duration:
            return False
        if filters.max_duration and duration > filters.max_duration:
            return False
        
        return True
    
    def _tags_match(self, tags: List[str], filters: SearchFilters) -> bool:
        """Check if tags match the filter criteria."""
        tag_set = set(tag.lower() for tag in tags)
        
        # Check include tags
        if filters.include_tags:
            include_set = set(tag.lower() for tag in filters.include_tags)
            if not include_set.issubset(tag_set):
                return False
        
        # Check exclude tags
        if filters.exclude_tags:
            exclude_set = set(tag.lower() for tag in filters.exclude_tags)
            if exclude_set.intersection(tag_set):
                return False
        
        return True
    
    def _path_matches(self, file_path: str, filters: SearchFilters) -> bool:
        """Check if file path matches the filter criteria."""
        # Check include paths
        if filters.include_paths:
            if not any(include_path in file_path for include_path in filters.include_paths):
                return False
        
        # Check exclude paths
        if filters.exclude_paths:
            if any(exclude_path in file_path for exclude_path in filters.exclude_paths):
                return False
        
        return True
