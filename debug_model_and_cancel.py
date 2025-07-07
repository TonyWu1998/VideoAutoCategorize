#!/usr/bin/env python3
"""
Debug script to test model selection and cancel functionality.
"""

import asyncio
import sys
import os
from pathlib import Path
import requests
import time

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

API_BASE_URL = "http://localhost:8000"

def test_api_connection():
    """Test if the API is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/config/llm", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_model_selection_issue():
    """Test the model selection issue."""
    print("🔍 Testing Model Selection Issue")
    print("=" * 50)
    
    if not test_api_connection():
        print("❌ Cannot connect to API server. Please start the backend first.")
        return
    
    try:
        # 1. Get current configuration
        print("1. Getting current LLM configuration...")
        response = requests.get(f"{API_BASE_URL}/api/config/llm")
        if response.status_code == 200:
            current_config = response.json()
            current_model = current_config.get('ollama_model', 'Unknown')
            print(f"   📋 Current model in API: {current_model}")
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
            
            # Check if huihui_ai/gemma3 is available
            target_models = ['huihui_ai/gemma3', 'huihui_ai', 'gemma3:4b']
            available_target = None
            for model in target_models:
                if model in available_models:
                    available_target = model
                    break
            
            if available_target:
                print(f"   ✅ Target model '{available_target}' is available")
            else:
                print(f"   ⚠️  None of the target models {target_models} are available")
                if available_models:
                    available_target = available_models[0]
                    print(f"   🔄 Using '{available_target}' for testing")
                else:
                    print("   ❌ No vision models available")
                    return
        else:
            print(f"   ❌ Failed to get available models: {response.status_code}")
            return
        
        # 3. Update model configuration
        print(f"3. Updating model to '{available_target}'...")
        update_data = {"ollama_model": available_target}
        response = requests.put(f"{API_BASE_URL}/api/config/llm", json=update_data)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Model update successful")
            print(f"   📄 Updated settings: {result.get('updated_settings', {})}")
        else:
            print(f"   ❌ Failed to update model: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return
        
        # 4. Verify the change immediately
        print("4. Verifying model change took effect...")
        time.sleep(0.5)  # Brief pause
        response = requests.get(f"{API_BASE_URL}/api/config/llm")
        if response.status_code == 200:
            updated_config = response.json()
            new_model = updated_config.get('ollama_model', 'Unknown')
            print(f"   📋 New model setting: {new_model}")
            
            if new_model == available_target:
                print(f"   ✅ Model change verified in API!")
            else:
                print(f"   ❌ Model change not reflected. Expected: {available_target}, Got: {new_model}")
        else:
            print(f"   ❌ Failed to verify model change: {response.status_code}")
        
        # 5. Check what the logs show
        print("5. Next steps for verification:")
        print(f"   💡 Check server logs for: '🤖 Calling Ollama API with model: {available_target}'")
        print("   💡 Trigger an analysis by uploading an image or starting indexing")
        print("   💡 If logs still show old model, there may be multiple LLM service instances")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_cancel_functionality():
    """Test the cancel job functionality."""
    print("\n🛑 Testing Cancel Job Functionality")
    print("=" * 50)
    
    if not test_api_connection():
        print("❌ Cannot connect to API server. Please start the backend first.")
        return
    
    try:
        # 1. Check current indexing status
        print("1. Checking current indexing status...")
        response = requests.get(f"{API_BASE_URL}/api/index/status")
        if response.status_code == 200:
            status = response.json()
            current_status = status.get('status', 'unknown')
            job_id = status.get('job_id')
            print(f"   📋 Current status: {current_status}")
            print(f"   📋 Job ID: {job_id}")
            
            if current_status in ['processing', 'scanning']:
                print("   ✅ Active job found - testing cancel...")
                
                # 2. Try to cancel the job
                cancel_data = {"action": "cancel", "job_id": job_id}
                response = requests.post(f"{API_BASE_URL}/api/index/control", json=cancel_data)
                if response.status_code == 200:
                    print("   ✅ Cancel request sent successfully")
                    
                    # 3. Check if cancellation took effect
                    time.sleep(2)
                    response = requests.get(f"{API_BASE_URL}/api/index/status")
                    if response.status_code == 200:
                        new_status = response.json()
                        new_status_value = new_status.get('status', 'unknown')
                        print(f"   📋 Status after cancel: {new_status_value}")
                        
                        if new_status_value == 'cancelled':
                            print("   ✅ Job successfully cancelled!")
                        else:
                            print("   ⚠️  Job status not changed to cancelled")
                            print("   💡 This might indicate the cancel mechanism needs improvement")
                    else:
                        print(f"   ❌ Failed to check status after cancel: {response.status_code}")
                else:
                    print(f"   ❌ Failed to send cancel request: {response.status_code}")
                    print(f"   📄 Response: {response.text}")
            else:
                print("   ℹ️  No active job to cancel")
                print("   💡 Start an indexing operation first to test cancel functionality")
        else:
            print(f"   ❌ Failed to get indexing status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Cancel test failed with error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debug function."""
    print("🔧 Model Selection and Cancel Job Debug Tool")
    print("=" * 60)
    
    # Test model selection
    test_model_selection_issue()
    
    # Test cancel functionality
    test_cancel_functionality()
    
    print("\n📋 Summary and Recommendations:")
    print("=" * 50)
    print("1. Model Selection:")
    print("   - If API shows correct model but logs show old model:")
    print("     → Multiple LLM service instances issue")
    print("   - If API doesn't update:")
    print("     → Settings update mechanism issue")
    print("")
    print("2. Cancel Functionality:")
    print("   - If status changes to 'cancelled' but processing continues:")
    print("     → LLM calls are not being interrupted")
    print("   - If status doesn't change:")
    print("     → Cancel mechanism not working")
    print("")
    print("3. Next Steps:")
    print("   - Check server logs during these tests")
    print("   - Look for multiple LLM service initialization messages")
    print("   - Monitor actual model usage in analysis logs")

if __name__ == "__main__":
    main()
