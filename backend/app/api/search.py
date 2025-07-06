"""
Search API endpoints for semantic media search.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import time
import logging

from app.models.media import (
    SearchRequest, 
    SearchResponse, 
    SearchFilters, 
    MediaItem,
    MediaType
)
from app.models.common import BaseResponse
from app.services.search import SearchService
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize search service
search_service = SearchService()


@router.post("/", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search on indexed media files.
    
    Uses AI-powered vector similarity to find media files that match
    the natural language query.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Performing semantic search for query: '{request.query}'")
        
        # Perform the search
        results = await search_service.semantic_search(
            query=request.query,
            filters=request.filters,
            include_metadata=request.include_metadata,
            include_thumbnails=request.include_thumbnails
        )
        
        search_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Search completed in {search_time_ms:.2f}ms, found {len(results)} results")
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time_ms,
            filters_applied=request.filters
        )
        
    except Exception as e:
        logger.error(f"Search failed for query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/", response_model=SearchResponse)
async def simple_search(
    q: str = Query(..., description="Search query"),
    media_type: MediaType = Query(MediaType.ALL, description="Filter by media type"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
    min_similarity: float = Query(0.3, ge=0, le=1, description="Minimum similarity threshold")
):
    """
    Simple GET-based search endpoint for easy integration.
    
    Provides a simplified interface for performing semantic searches
    without requiring a POST request body.
    """
    try:
        # Create search request from query parameters
        filters = SearchFilters(
            media_type=media_type,
            max_results=limit,
            min_similarity=min_similarity
        )
        
        request = SearchRequest(
            query=q,
            filters=filters,
            include_metadata=True,
            include_thumbnails=True
        )
        
        # Use the main search endpoint
        return await semantic_search(request)
        
    except Exception as e:
        logger.error(f"Simple search failed for query '{q}': {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/suggestions")
async def get_search_suggestions(
    q: str = Query(..., description="Partial search query"),
    limit: int = Query(10, ge=1, le=20, description="Maximum number of suggestions")
):
    """
    Get search suggestions based on partial query.
    
    Returns suggested search terms based on indexed content
    and common search patterns.
    """
    try:
        suggestions = await search_service.get_search_suggestions(q, limit)
        
        return {
            "query": q,
            "suggestions": suggestions
        }
        
    except Exception as e:
        logger.error(f"Failed to get suggestions for query '{q}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


@router.get("/similar/{file_id}")
async def find_similar_media(
    file_id: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum number of similar items"),
    min_similarity: float = Query(0.5, ge=0, le=1, description="Minimum similarity threshold")
):
    """
    Find media files similar to a specific file.
    
    Uses the vector embedding of the specified file to find
    other files with similar content.
    """
    try:
        similar_items = await search_service.find_similar_media(
            file_id=file_id,
            limit=limit,
            min_similarity=min_similarity
        )
        
        return {
            "file_id": file_id,
            "similar_items": similar_items,
            "total_found": len(similar_items)
        }
        
    except Exception as e:
        logger.error(f"Failed to find similar media for file_id '{file_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find similar media: {str(e)}")


@router.get("/tags")
async def get_popular_tags(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of tags"),
    min_frequency: int = Query(2, ge=1, description="Minimum tag frequency")
):
    """
    Get popular tags from indexed media files.
    
    Returns the most frequently occurring tags across all
    indexed media files.
    """
    try:
        tags = await search_service.get_popular_tags(limit, min_frequency)
        
        return {
            "tags": tags,
            "total_tags": len(tags)
        }
        
    except Exception as e:
        logger.error(f"Failed to get popular tags: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tags: {str(e)}")


@router.post("/batch")
async def batch_search(
    queries: List[str],
    filters: Optional[SearchFilters] = None
):
    """
    Perform multiple searches in a single request.
    
    Useful for applications that need to perform multiple
    related searches efficiently.
    """
    try:
        if len(queries) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 queries per batch")
        
        if not filters:
            filters = SearchFilters()
        
        results = []
        for query in queries:
            try:
                search_results = await search_service.semantic_search(
                    query=query,
                    filters=filters,
                    include_metadata=False,  # Reduce response size
                    include_thumbnails=False
                )
                
                results.append({
                    "query": query,
                    "results": search_results,
                    "success": True
                })
                
            except Exception as e:
                logger.warning(f"Batch search failed for query '{query}': {e}")
                results.append({
                    "query": query,
                    "results": [],
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "batch_results": results,
            "total_queries": len(queries),
            "successful_queries": sum(1 for r in results if r["success"])
        }
        
    except Exception as e:
        logger.error(f"Batch search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}")


@router.delete("/cache")
async def clear_search_cache():
    """
    Clear the search result cache.
    
    Forces fresh searches for all subsequent queries.
    """
    try:
        await search_service.clear_cache()
        
        return BaseResponse(
            success=True,
            message="Search cache cleared successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to clear search cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")
