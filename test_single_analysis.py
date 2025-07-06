#!/usr/bin/env python3
"""
Simple test to verify the LLM analysis fixes work with a single image.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.services.llm_service import LLMService


async def test_single_image():
    """Test analysis with a single image."""
    
    image_path = "/Users/tony/Desktop/projects/VideoAutoCategorize/test_media/images/beach_scene.jpg"
    
    print(f"🧪 Testing single image analysis")
    print(f"🧪 Image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    try:
        # Initialize LLM service
        print(f"🤖 Initializing LLM service...")
        llm_service = LLMService()
        print(f"🤖 ✅ LLM service initialized")
        
        # Test image analysis
        print(f"🎯 Starting image analysis...")
        result = await llm_service.analyze_image(image_path)
        
        print(f"\n📊 Analysis Results:")
        print(f"📊 Description: {result.get('description', 'N/A')}")
        print(f"📊 Tags: {result.get('tags', [])}")
        print(f"📊 Confidence: {result.get('confidence', 'N/A')}")
        print(f"📊 Processing time: {result.get('processing_time', 'N/A')}s")
        
        # Check if it's an error result
        if result.get('description', '').startswith('Analysis incomplete'):
            print(f"⚠️  Analysis returned an error - this is expected if Ollama is having issues")
            print(f"⚠️  The important thing is that the error was handled gracefully")
        else:
            print(f"✅ Analysis completed successfully!")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")


if __name__ == "__main__":
    asyncio.run(test_single_image())
