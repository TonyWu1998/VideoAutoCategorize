#!/usr/bin/env python3
"""
End-to-end test script to verify the complete VideoAutoCategorize workflow.
Tests indexing, database storage, and search functionality for video files.
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.services.indexing import IndexingService
from app.services.search import SearchService
from app.database.vector_db import VectorDatabase
from app.models.indexing import IndexingRequest
from app.models.media import SearchFilters, SearchRequest


async def test_video_indexing():
    """Test indexing of the video file."""
    print("🎬 STEP 1: Testing Video File Indexing")
    print("=" * 60)
    
    video_path = "/Volumes/T7 Shield/final_cut/jimingyue_4k_dv.mp4"
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    print(f"📁 Video file found: {video_path}")
    
    try:
        # Initialize indexing service
        indexing_service = IndexingService()
        
        # Create indexing request
        request = IndexingRequest(
            paths=[video_path],
            recursive=False,
            force_reindex=True,  # Force reindex to ensure fresh analysis
            batch_size=1
        )
        
        print(f"🚀 Starting indexing process...")
        job_id = await indexing_service.start_indexing(request)
        print(f"📋 Indexing job started with ID: {job_id}")
        
        # Wait for indexing to complete
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = await indexing_service.get_status()
            print(f"📊 Indexing status: {status.status}")
            print(f"📊 Progress: {status.progress.processed_files}/{status.progress.total_files}")
            print(f"📊 Successful: {status.progress.successful_files}")
            print(f"📊 Failed: {status.progress.failed_files}")
            
            if status.status.value in ["COMPLETED", "FAILED", "CANCELLED"]:
                break
            
            await asyncio.sleep(5)  # Wait 5 seconds before checking again
        
        final_status = await indexing_service.get_status()
        if final_status.status.value == "COMPLETED" and final_status.progress.successful_files > 0:
            print(f"✅ Indexing completed successfully!")
            return True
        else:
            print(f"❌ Indexing failed or incomplete")
            return False
            
    except Exception as e:
        print(f"❌ Indexing failed with exception: {e}")
        return False


async def test_database_storage():
    """Test that the video is properly stored in the database."""
    print(f"\n🗄️ STEP 2: Testing Database Storage")
    print("=" * 60)
    
    try:
        # Initialize vector database
        vector_db = VectorDatabase()
        
        # Search for our specific video file
        video_path = "/Volumes/T7 Shield/final_cut/jimingyue_4k_dv.mp4"
        
        # Get all media documents to find our video
        print(f"🔍 Searching for video in database...")
        
        # We'll search by filename since we don't have a direct "get by path" method
        all_docs = vector_db.get_all_media()
        
        video_doc = None
        for doc in all_docs:
            if video_path in doc.file_path:
                video_doc = doc
                break
        
        if not video_doc:
            print(f"❌ Video not found in database")
            return False
        
        print(f"✅ Video found in database!")
        print(f"📄 File ID: {video_doc.file_id}")
        print(f"📄 File name: {video_doc.file_name}")
        print(f"📄 Media type: {video_doc.media_type}")
        print(f"📄 AI Description: {video_doc.ai_description[:100]}...")
        print(f"📄 AI Tags: {video_doc.ai_tags[:10]}")  # Show first 10 tags
        print(f"📄 Embedding length: {len(video_doc.embedding) if video_doc.embedding else 0}")
        
        # Validate the stored data
        validation_passed = True
        
        if not video_doc.ai_description or len(video_doc.ai_description) < 10:
            print(f"❌ AI description is missing or too short")
            validation_passed = False
        
        if not video_doc.ai_tags or len(video_doc.ai_tags) == 0:
            print(f"❌ AI tags are missing")
            validation_passed = False
        
        if not video_doc.embedding or len(video_doc.embedding) == 0:
            print(f"❌ Embedding is missing")
            validation_passed = False
        
        # Check for error descriptions
        if "Error analyzing" in video_doc.ai_description:
            print(f"❌ AI description contains error message")
            validation_passed = False
        
        if validation_passed:
            print(f"✅ Database storage validation passed!")
            return video_doc
        else:
            print(f"❌ Database storage validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Database storage test failed with exception: {e}")
        return False


async def test_search_functionality(video_doc):
    """Test search functionality with the indexed video."""
    print(f"\n🔍 STEP 3: Testing Search Functionality")
    print("=" * 60)
    
    try:
        # Initialize search service
        search_service = SearchService()
        
        # Test queries related to the video content
        test_queries = [
            "traditional Chinese courtyard",
            "woman in white dress",
            "serene peaceful scene",
            "red pillars architecture",
            "jimingyue"  # Part of the filename
        ]
        
        search_results = {}
        
        for query in test_queries:
            print(f"\n🔎 Testing query: '{query}'")
            
            # Create search request
            filters = SearchFilters(
                max_results=10,
                min_similarity=0.1  # Low threshold to ensure we find results
            )
            
            # Perform search
            results = await search_service.semantic_search(
                query=query,
                filters=filters,
                include_metadata=True,
                include_thumbnails=True
            )
            
            print(f"🔎 Found {len(results)} results")
            
            # Check if our video is in the results
            video_found = False
            video_result = None
            
            for result in results:
                if video_doc.file_id == result.file_id:
                    video_found = True
                    video_result = result
                    break
            
            if video_found:
                print(f"✅ Video found in search results!")
                print(f"📊 Similarity score: {video_result.similarity_score}")
                print(f"📊 Description: {video_result.metadata.ai_description[:100]}...")
                print(f"📊 Tags: {video_result.metadata.ai_tags[:5]}")  # Show first 5 tags
                
                # Check for error descriptions in search results
                if "Error analyzing" in video_result.metadata.ai_description:
                    print(f"❌ Search result contains error description")
                    search_results[query] = False
                else:
                    print(f"✅ Search result looks good!")
                    search_results[query] = True
            else:
                print(f"❌ Video not found in search results for this query")
                search_results[query] = False
        
        # Summary
        successful_queries = sum(1 for success in search_results.values() if success)
        total_queries = len(test_queries)
        
        print(f"\n📈 Search Test Summary:")
        print(f"📈 Successful queries: {successful_queries}/{total_queries}")
        
        if successful_queries > 0:
            print(f"✅ Search functionality is working!")
            return True
        else:
            print(f"❌ Search functionality failed")
            return False
            
    except Exception as e:
        print(f"❌ Search functionality test failed with exception: {e}")
        return False


async def main():
    """Run the complete end-to-end test."""
    print("🚀 VideoAutoCategorize End-to-End Test")
    print("=" * 80)
    print("Testing complete workflow: Indexing → Database Storage → Search")
    print("=" * 80)
    
    # Step 1: Index the video file
    indexing_success = await test_video_indexing()
    if not indexing_success:
        print(f"\n❌ End-to-end test FAILED at indexing step")
        return
    
    # Step 2: Verify database storage
    video_doc = await test_database_storage()
    if not video_doc:
        print(f"\n❌ End-to-end test FAILED at database storage step")
        return
    
    # Step 3: Test search functionality
    search_success = await test_search_functionality(video_doc)
    if not search_success:
        print(f"\n❌ End-to-end test FAILED at search functionality step")
        return
    
    print(f"\n🎉 END-TO-END TEST COMPLETED SUCCESSFULLY!")
    print(f"🎉 All components are working correctly:")
    print(f"   ✅ Video file indexing")
    print(f"   ✅ AI analysis and parsing")
    print(f"   ✅ Database storage")
    print(f"   ✅ Search functionality")
    print(f"   ✅ No error descriptions in results")


if __name__ == "__main__":
    asyncio.run(main())
