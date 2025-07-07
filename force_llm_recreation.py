#!/usr/bin/env python3
"""
Script to force LLM service recreation and test the singleton pattern.
"""

import requests
import time

API_BASE_URL = "http://localhost:8000"

def test_api_connection():
    """Test if the API is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def force_llm_service_recreation():
    """Force the creation of a new LLM service instance."""
    print("🔄 Forcing LLM Service Recreation")
    print("=" * 50)
    
    if not test_api_connection():
        print("❌ Cannot connect to API server. Please start the backend first.")
        return False
    
    try:
        # Step 1: Update model to force a change
        print("1. Updating model to 'huihui_ai/gemma3'...")
        update_data = {"ollama_model": "huihui_ai/gemma3"}
        response = requests.put(f"{API_BASE_URL}/api/config/llm", json=update_data)
        if response.status_code == 200:
            print("   ✅ Model update successful")
        else:
            print(f"   ❌ Model update failed: {response.status_code}")
            return False
        
        # Step 2: Trigger LLM service usage by uploading a test image
        print("2. Triggering LLM service usage...")
        print("   💡 We need to trigger an analysis operation to force LLM service creation")
        print("   💡 Options:")
        print("   💡   a) Upload an image via the UI")
        print("   💡   b) Start indexing operation")
        print("   💡   c) Use the search endpoint (if it triggers LLM)")
        print("")
        print("   🔍 After triggering, check server logs for:")
        print("   🔍   - '🔍 Creating NEW LLM service singleton instance'")
        print("   🔍   - '🔍 LLM service __init__ starting initialization...'")
        print("   🔍   - '🔍 LLM service model property accessed, returning: huihui_ai/gemma3'")
        print("   🔍   - 'LLM service initialized with model: huihui_ai/gemma3'")
        print("   🔍   - '🤖 Calling Ollama API with model: huihui_ai/gemma3'")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to force recreation: {e}")
        return False

def check_current_status():
    """Check the current model status."""
    print("\n📋 Current Status Check")
    print("=" * 50)
    
    try:
        # Check API configuration
        response = requests.get(f"{API_BASE_URL}/api/config/llm")
        if response.status_code == 200:
            config = response.json()
            current_model = config.get('ollama_model', 'Unknown')
            print(f"   📋 API reports current model: {current_model}")
        else:
            print(f"   ❌ Failed to get config: {response.status_code}")
        
        # Check available models
        response = requests.get(f"{API_BASE_URL}/api/config/ollama/models")
        if response.status_code == 200:
            models_data = response.json()
            available_models = models_data.get('vision_models', [])
            print(f"   📊 Available models: {available_models}")
        else:
            print(f"   ❌ Failed to get models: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Status check failed: {e}")

def main():
    """Main function."""
    print("🔧 Force LLM Service Recreation Test")
    print("=" * 60)
    
    check_current_status()
    success = force_llm_service_recreation()
    
    print("\n📝 Next Steps:")
    print("=" * 50)
    if success:
        print("✅ Model updated successfully")
        print("")
        print("🎯 Now trigger an analysis operation:")
        print("   1. Go to the web UI")
        print("   2. Upload an image OR start indexing")
        print("   3. Watch server logs for singleton debug messages")
        print("")
        print("🔍 Expected log sequence:")
        print("   1. '🔍 Creating NEW LLM service singleton instance'")
        print("   2. '🔍 LLM service __init__ starting initialization...'")
        print("   3. '🔍 Initial OLLAMA_MODEL value: huihui_ai/gemma3'")
        print("   4. 'LLM service initialized with model: huihui_ai/gemma3'")
        print("   5. '🔍 LLM service model property accessed, returning: huihui_ai/gemma3'")
        print("   6. '🤖 Calling Ollama API with model: huihui_ai/gemma3'")
        print("")
        print("❌ If you see 'gemma3:4b' instead of 'huihui_ai/gemma3':")
        print("   - Singleton pattern is not working")
        print("   - Settings are not being updated properly")
        print("   - Multiple instances still being created")
    else:
        print("❌ Failed to update model")
        print("🔧 Check API connectivity and model availability")

if __name__ == "__main__":
    main()
