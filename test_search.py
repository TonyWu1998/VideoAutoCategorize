#!/usr/bin/env python3
"""
Test search functionality with the indexed video.
"""

import asyncio
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.services.search import SearchService
from app.models.media import SearchFilters


async def test_search():
    """Test search functionality."""
    print("🔍 Testing Search Functionality")
    print("=" * 50)
    
    try:
        # Initialize search service
        search_service = SearchService()
        
        # Test queries related to the video content
        test_queries = [
            "traditional Chinese courtyard",
            "red pillars",
            "serene elegant",
            "mirror",
            "jimingyue"
        ]
        
        for query in test_queries:
            print(f"\n🔎 Testing query: '{query}'")
            
            try:
                # Create search request
                filters = SearchFilters(
                    max_results=10,
                    min_similarity=0.0  # Very low threshold to ensure we find results
                )
                
                # Perform search
                results = await search_service.semantic_search(
                    query=query,
                    filters=filters,
                    include_metadata=True,
                    include_thumbnails=True
                )
                
                print(f"🔎 Found {len(results)} results")
                
                if len(results) > 0:
                    for i, result in enumerate(results):
                        print(f"   📄 Result {i+1}:")
                        print(f"      File: {result.metadata.file_name}")
                        print(f"      Similarity: {result.similarity_score}")
                        print(f"      Description: {result.metadata.ai_description[:100]}...")
                        print(f"      Tags: {result.metadata.ai_tags[:3]}")
                        
                        # Check for error descriptions
                        if "Error analyzing" in result.metadata.ai_description:
                            print(f"      ❌ Contains error description")
                        else:
                            print(f"      ✅ Analysis looks good")
                else:
                    print(f"   ❌ No results found")
                    
            except Exception as e:
                print(f"   ❌ Search failed for query '{query}': {e}")
        
        print(f"\n📈 Search test completed!")
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_search())
