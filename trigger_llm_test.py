#!/usr/bin/env python3
"""
Script to trigger LLM service usage and test the singleton pattern.
"""

import requests
import base64
import io
from PIL import Image

API_BASE_URL = "http://localhost:8000"

def create_test_image():
    """Create a simple test image."""
    # Create a simple 100x100 red image
    img = Image.new('RGB', (100, 100), color='red')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def trigger_llm_via_search():
    """Try to trigger LLM via search endpoint."""
    print("🔍 Attempting to trigger LLM via search...")
    
    try:
        # Try a simple search that might trigger embedding generation
        response = requests.get(f"{API_BASE_URL}/api/search/simple?q=test&limit=1")
        if response.status_code == 200:
            print("   ✅ Search request successful")
            return True
        else:
            print(f"   ❌ Search failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Search error: {e}")
        return False

def trigger_llm_via_indexing():
    """Try to trigger LLM via indexing status check."""
    print("🔄 Attempting to trigger LLM via indexing status...")
    
    try:
        # Check indexing status - this might trigger LLM service creation
        response = requests.get(f"{API_BASE_URL}/api/index/status")
        if response.status_code == 200:
            print("   ✅ Indexing status request successful")
            return True
        else:
            print(f"   ❌ Indexing status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Indexing status error: {e}")
        return False

def check_logs_for_singleton():
    """Check if singleton debug messages appeared in logs."""
    print("\n📋 Checking for singleton debug messages...")
    
    try:
        # Read recent logs
        import subprocess
        result = subprocess.run(
            ["tail", "-n", "50", "backend/logs/app.log"],
            capture_output=True,
            text=True,
            cwd="/Users/tony/Desktop/projects/VideoAutoCategorize"
        )
        
        if result.returncode == 0:
            logs = result.stdout
            
            # Check for our debug messages
            singleton_messages = [
                "🔍 Creating NEW LLM service singleton instance",
                "🔍 Returning EXISTING LLM service singleton instance",
                "🔍 LLM service __init__ starting initialization",
                "🔍 LLM service model property accessed",
                "huihui_ai/gemma3-abliterated:4b"
            ]
            
            found_messages = []
            for message in singleton_messages:
                if message in logs:
                    found_messages.append(message)
            
            if found_messages:
                print("   ✅ Found singleton debug messages:")
                for msg in found_messages:
                    print(f"      - {msg}")
                return True
            else:
                print("   ❌ No singleton debug messages found")
                print("   💡 LLM service hasn't been recreated yet")
                return False
        else:
            print(f"   ❌ Failed to read logs: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Log check error: {e}")
        return False

def main():
    """Main function to trigger LLM and test singleton."""
    print("🧪 Trigger LLM Service Test")
    print("=" * 50)
    
    # Try different ways to trigger LLM service
    print("1. Trying to trigger LLM service...")
    
    # Method 1: Search
    search_success = trigger_llm_via_search()
    
    # Method 2: Indexing status
    indexing_success = trigger_llm_via_indexing()
    
    # Check logs
    singleton_found = check_logs_for_singleton()
    
    print("\n📋 Results Summary:")
    print("=" * 50)
    print(f"🔍 Search trigger: {'✅ Success' if search_success else '❌ Failed'}")
    print(f"🔄 Indexing trigger: {'✅ Success' if indexing_success else '❌ Failed'}")
    print(f"🔧 Singleton messages: {'✅ Found' if singleton_found else '❌ Not found'}")
    
    if not singleton_found:
        print("\n💡 LLM service hasn't been recreated yet.")
        print("💡 Try one of these to force LLM service creation:")
        print("   1. Upload an image via the web UI")
        print("   2. Start an indexing operation")
        print("   3. Restart the server to clear existing instances")
        print("")
        print("🔍 Then check logs for:")
        print("   - '🔍 Creating NEW LLM service singleton instance'")
        print("   - 'LLM service initialized with model: huihui_ai/gemma3-abliterated:4b'")
        print("   - '🤖 Calling Ollama API with model: huihui_ai/gemma3-abliterated:4b'")
    else:
        print("\n🎉 Singleton pattern appears to be working!")
        print("✅ Check the logs to verify the model is now 'huihui_ai/gemma3-abliterated:4b'")

if __name__ == "__main__":
    main()
