#!/usr/bin/env python3
"""
Check what's currently in the database after indexing.
"""

import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.database.vector_db import VectorDatabase


def check_database():
    """Check what's in the database."""
    print("🗄️ Checking Database Contents")
    print("=" * 50)
    
    try:
        # Initialize vector database
        vector_db = VectorDatabase()
        
        # Get all media documents
        all_docs = vector_db.list_media(limit=100)
        
        print(f"📊 Total documents in database: {len(all_docs)}")
        
        if len(all_docs) == 0:
            print(f"❌ No documents found in database")
            return
        
        # Look for our video file
        video_path = "/Volumes/T7 Shield/final_cut/jimingyue_4k_dv.mp4"
        
        for i, doc in enumerate(all_docs):
            print(f"\n📄 Document {i+1}:")
            print(f"   File ID: {doc.file_id}")
            print(f"   File name: {doc.file_name}")
            print(f"   File path: {doc.file_path}")
            print(f"   Media type: {doc.media_type}")
            print(f"   AI Description: {doc.ai_description[:100]}...")
            print(f"   AI Tags: {doc.ai_tags[:5] if doc.ai_tags else []}")
            print(f"   Embedding length: {len(doc.embedding) if doc.embedding else 0}")
            
            # Check if this is our target video
            if video_path in doc.file_path:
                print(f"   🎯 THIS IS OUR TARGET VIDEO!")
                
                # Check for errors
                if "Error analyzing" in doc.ai_description:
                    print(f"   ❌ Contains error description")
                else:
                    print(f"   ✅ Analysis looks good")
                    
    except Exception as e:
        print(f"❌ Database check failed: {e}")


if __name__ == "__main__":
    check_database()
