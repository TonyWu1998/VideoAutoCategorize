#!/usr/bin/env python3
"""
Debug script to test file type detection and AI analysis workflow.
This script helps identify why video files might be treated as images.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.config import settings
from app.models.media import MediaType
from app.services.llm_service import LLMService


def detect_media_type(file_path: str) -> MediaType:
    """Test the file type detection logic."""
    file_ext = os.path.splitext(file_path)[1].lower()
    print(f"🔍 File: {file_path}")
    print(f"🔍 Extension: {file_ext}")
    print(f"🔍 Supported image formats: {settings.SUPPORTED_IMAGE_FORMATS}")
    print(f"🔍 Supported video formats: {settings.SUPPORTED_VIDEO_FORMATS}")
    
    if file_ext in settings.SUPPORTED_IMAGE_FORMATS:
        media_type = MediaType.IMAGE
        print(f"🔍 ✅ Detected as IMAGE")
    elif file_ext in settings.SUPPORTED_VIDEO_FORMATS:
        media_type = MediaType.VIDEO
        print(f"🔍 ✅ Detected as VIDEO")
    else:
        print(f"🔍 ❌ Unsupported file type")
        return None
    
    return media_type


async def test_analysis(file_path: str):
    """Test the AI analysis workflow for a specific file."""
    print(f"\n{'='*60}")
    print(f"Testing analysis for: {file_path}")
    print(f"{'='*60}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        return
    
    # Test file type detection
    media_type = detect_media_type(file_path)
    if not media_type:
        print(f"❌ Unsupported file type")
        return
    
    # Initialize LLM service
    try:
        print(f"\n🤖 Initializing LLM service...")
        llm_service = LLMService()
        print(f"🤖 ✅ LLM service initialized successfully")
    except Exception as e:
        print(f"🤖 ❌ Failed to initialize LLM service: {e}")
        return
    
    # Test analysis based on media type
    try:
        print(f"\n🎯 Starting analysis...")
        if media_type == MediaType.VIDEO:
            print(f"🎯 Calling analyze_video...")
            result = await llm_service.analyze_video(file_path)
        else:
            print(f"🎯 Calling analyze_image...")
            result = await llm_service.analyze_image(file_path)
        
        print(f"\n📊 Analysis Results:")
        print(f"📊 Description: {result.get('description', 'N/A')}")
        print(f"📊 Tags: {result.get('tags', [])}")
        print(f"📊 Confidence: {result.get('confidence', 'N/A')}")
        print(f"📊 Processing time: {result.get('processing_time', 'N/A')}s")
        print(f"📊 Model used: {result.get('model_used', 'N/A')}")
        
        if 'frames_analyzed' in result:
            print(f"📊 Frames analyzed: {result.get('frames_analyzed', 'N/A')}")
        
        # Check for errors
        if result.get('description', '').startswith('Error'):
            print(f"⚠️  Analysis returned an error!")
        else:
            print(f"✅ Analysis completed successfully!")
            
    except Exception as e:
        print(f"❌ Analysis failed with exception: {e}")


async def main():
    """Main test function."""
    print("🚀 Starting AI Analysis Debug Script")
    print("="*60)
    
    # Test files
    test_files = [
        "/Users/tony/Desktop/projects/VideoAutoCategorize/test_media/videos/moving_circle.mp4",
        "/Users/tony/Desktop/projects/VideoAutoCategorize/test_media/images/beach_scene.jpg",
        "/Volumes/T7 Shield/final_cut/jimingyue_4k_dv.mp4"  # User's test file
    ]
    
    for file_path in test_files:
        await test_analysis(file_path)
    
    print(f"\n🏁 Debug script completed!")


if __name__ == "__main__":
    asyncio.run(main())
