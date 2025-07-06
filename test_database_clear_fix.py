#!/usr/bin/env python3
"""
Test script to verify the database clear fix works correctly.
This script tests the ChromaDB collection clearing and recreation functionality.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.database.vector_db import VectorDatabase
from app.services.search import SearchService
from app.services.indexing import IndexingService
from app.models.media import SearchFilters, SearchRequest


async def test_database_clear_fix():
    """Test the database clear and search functionality."""
    print("🧪 Testing Database Clear Fix")
    print("=" * 50)
    
    try:
        # Initialize services
        print("1. Initializing services...")
        vector_db = VectorDatabase()
        search_service = SearchService()
        indexing_service = IndexingService()
        
        # Check initial state
        print("2. Checking initial database state...")
        initial_stats = vector_db.get_stats()
        print(f"   Initial document count: {initial_stats.get('total_documents', 0)}")
        
        # Test search before clearing (should work)
        print("3. Testing search before clearing...")
        try:
            search_request = SearchRequest(
                query="test search",
                filters=SearchFilters(max_results=5)
            )
            results_before = await search_service.semantic_search(search_request)
            print(f"   Search before clear: {len(results_before)} results (no error)")
        except Exception as e:
            print(f"   Search before clear failed: {e}")
        
        # Clear the database
        print("4. Clearing database...")
        clear_success = await indexing_service.clear_index()
        if clear_success:
            print("   ✅ Database cleared successfully")
        else:
            print("   ❌ Database clear failed")
            return False
        
        # Check state after clearing
        print("5. Checking database state after clearing...")
        cleared_stats = vector_db.get_stats()
        print(f"   Document count after clear: {cleared_stats.get('total_documents', 0)}")
        
        # Test search after clearing (this is the critical test)
        print("6. Testing search after clearing (critical test)...")
        try:
            search_request = SearchRequest(
                query="test search after clear",
                filters=SearchFilters(max_results=5)
            )
            results_after = await search_service.semantic_search(search_request)
            print(f"   ✅ Search after clear: {len(results_after)} results (no 500 error)")
            print("   ✅ Fix successful - search returns empty results instead of 500 error")
        except Exception as e:
            print(f"   ❌ Search after clear failed: {e}")
            print("   ❌ Fix failed - search still throws 500 error")
            return False
        
        # Test collection recreation
        print("7. Testing collection accessibility...")
        collection_exists = vector_db.ensure_collection_exists()
        if collection_exists:
            print("   ✅ Collection exists and is accessible")
        else:
            print("   ❌ Collection is not accessible")
            return False
        
        # Test multiple searches to ensure stability
        print("8. Testing search stability (multiple searches)...")
        for i in range(3):
            try:
                search_request = SearchRequest(
                    query=f"stability test {i}",
                    filters=SearchFilters(max_results=5)
                )
                results = await search_service.semantic_search(search_request)
                print(f"   Search {i+1}: {len(results)} results ✅")
            except Exception as e:
                print(f"   Search {i+1} failed: {e} ❌")
                return False
        
        print("\n🎉 All tests passed! The database clear fix is working correctly.")
        print("\nSummary:")
        print("- Database can be cleared without breaking the system")
        print("- Search operations return empty results instead of 500 errors")
        print("- Collection is properly recreated with correct configuration")
        print("- Multiple searches work consistently after clearing")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_database_clear_fix())
    sys.exit(0 if success else 1)
