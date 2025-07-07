#!/usr/bin/env python3
"""
Test script to verify the model selection fix.
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_model_selection_fix():
    """Test that model selection changes are applied immediately."""
    print("🧪 Testing Model Selection Fix")
    print("=" * 50)
    
    try:
        # 1. Get current LLM configuration
        print("1. Getting current LLM configuration...")
        response = requests.get(f"{API_BASE_URL}/api/config/llm")
        if response.status_code == 200:
            current_config = response.json()
            print(f"   📋 Current model: {current_config.get('ollama_model', 'Unknown')}")
        else:
            print(f"   ❌ Failed to get current config: {response.status_code}")
            return
        
        # 2. Get available models
        print("2. Getting available models...")
        response = requests.get(f"{API_BASE_URL}/api/config/ollama/models")
        if response.status_code == 200:
            models_data = response.json()
            available_models = models_data.get('vision_models', [])
            print(f"   📊 Available vision models: {available_models}")
            
            if 'huihui_ai' in available_models:
                target_model = 'huihui_ai'
                print(f"   ✅ Target model 'huihui_ai' is available")
            elif len(available_models) > 1:
                # Use a different model for testing
                current_model = current_config.get('ollama_model', 'gemma3:4b')
                target_model = next((m for m in available_models if m != current_model), available_models[0])
                print(f"   🔄 Using '{target_model}' for testing (huihui_ai not available)")
            else:
                print(f"   ⚠️  Only one model available, cannot test switching")
                return
        else:
            print(f"   ❌ Failed to get available models: {response.status_code}")
            return
        
        # 3. Update model configuration
        print(f"3. Updating model to '{target_model}'...")
        update_data = {"ollama_model": target_model}
        response = requests.put(f"{API_BASE_URL}/api/config/llm", json=update_data)
        if response.status_code == 200:
            print(f"   ✅ Model update request successful")
        else:
            print(f"   ❌ Failed to update model: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return
        
        # 4. Verify the change took effect immediately
        print("4. Verifying model change took effect...")
        time.sleep(1)  # Brief pause
        response = requests.get(f"{API_BASE_URL}/api/config/llm")
        if response.status_code == 200:
            updated_config = response.json()
            new_model = updated_config.get('ollama_model', 'Unknown')
            print(f"   📋 New model setting: {new_model}")
            
            if new_model == target_model:
                print(f"   ✅ Model change successful! Now using: {new_model}")
            else:
                print(f"   ❌ Model change failed. Expected: {target_model}, Got: {new_model}")
        else:
            print(f"   ❌ Failed to verify model change: {response.status_code}")
        
        # 5. Test with a simple analysis (if possible)
        print("5. Testing if new model is used in analysis...")
        print("   💡 Check the server logs for: '🤖 Calling Ollama API with model: {}'".format(target_model))
        print("   💡 You can trigger an analysis by uploading an image in the UI")
        
        print("\n🎉 Model selection test completed!")
        print(f"📝 Summary: Model should now be '{target_model}' and changes should persist")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to the API server. Make sure the backend is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model_selection_fix()
