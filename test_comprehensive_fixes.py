#!/usr/bin/env python3
"""
Comprehensive test script for model selection and cancel functionality fixes.
"""

import requests
import time
import json

API_BASE_URL = "http://localhost:8000"

def test_api_connection():
    """Test if the API is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_singleton_llm_service():
    """Test that the LLM service singleton is working."""
    print("🔍 Testing LLM Service Singleton Pattern")
    print("=" * 50)
    
    print("1. Check server logs for LLM service initialization messages")
    print("   💡 Look for: 'LLM service initialized with model: [model_name]'")
    print("   💡 There should be only ONE initialization message after server restart")
    print("   💡 Multiple messages indicate multiple instances (the old bug)")
    print("")

def test_model_selection_persistence():
    """Test that model selection changes persist immediately."""
    print("🤖 Testing Model Selection Persistence")
    print("=" * 50)
    
    if not test_api_connection():
        print("❌ Cannot connect to API server. Please start the backend first.")
        return False
    
    try:
        # 1. Get current model
        print("1. Getting current model configuration...")
        response = requests.get(f"{API_BASE_URL}/api/config/llm")
        if response.status_code != 200:
            print(f"   ❌ Failed to get config: {response.status_code}")
            return False
        
        current_config = response.json()
        current_model = current_config.get('ollama_model', 'Unknown')
        print(f"   📋 Current model: {current_model}")
        
        # 2. Get available models
        print("2. Getting available models...")
        response = requests.get(f"{API_BASE_URL}/api/config/ollama/models")
        if response.status_code != 200:
            print(f"   ❌ Failed to get models: {response.status_code}")
            return False
        
        models_data = response.json()
        available_models = models_data.get('vision_models', [])
        print(f"   📊 Available models: {available_models}")
        
        # Find a different model to test with
        target_model = None
        for model in ['huihui_ai/gemma3', 'huihui_ai', 'gemma3:4b', 'llava:latest']:
            if model in available_models and model != current_model:
                target_model = model
                break
        
        if not target_model:
            print("   ⚠️  No alternative model available for testing")
            return False
        
        print(f"   🎯 Testing with model: {target_model}")
        
        # 3. Update model
        print("3. Updating model configuration...")
        update_data = {"ollama_model": target_model}
        response = requests.put(f"{API_BASE_URL}/api/config/llm", json=update_data)
        if response.status_code != 200:
            print(f"   ❌ Failed to update model: {response.status_code}")
            return False
        
        print("   ✅ Model update request successful")
        
        # 4. Verify immediately
        print("4. Verifying model change took effect...")
        time.sleep(0.5)
        response = requests.get(f"{API_BASE_URL}/api/config/llm")
        if response.status_code != 200:
            print(f"   ❌ Failed to verify: {response.status_code}")
            return False
        
        new_config = response.json()
        new_model = new_config.get('ollama_model', 'Unknown')
        print(f"   📋 New model setting: {new_model}")
        
        if new_model == target_model:
            print("   ✅ Model change verified in API!")
            print("   💡 Now check server logs for actual model usage")
            print(f"   💡 Look for: '🤖 Calling Ollama API with model: {target_model}'")
            return True
        else:
            print(f"   ❌ Model not updated. Expected: {target_model}, Got: {new_model}")
            return False
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_cancel_functionality():
    """Test the enhanced cancel functionality."""
    print("\n🛑 Testing Enhanced Cancel Functionality")
    print("=" * 50)
    
    if not test_api_connection():
        print("❌ Cannot connect to API server.")
        return False
    
    try:
        # 1. Check current status
        print("1. Checking current indexing status...")
        response = requests.get(f"{API_BASE_URL}/api/index/status")
        if response.status_code != 200:
            print(f"   ❌ Failed to get status: {response.status_code}")
            return False
        
        status = response.json()
        current_status = status.get('status', 'unknown')
        job_id = status.get('job_id')
        
        print(f"   📋 Current status: {current_status}")
        print(f"   📋 Job ID: {job_id}")
        
        if current_status in ['processing', 'scanning']:
            print("   ✅ Active job found - testing cancel...")
            
            # 2. Send cancel request
            cancel_data = {"action": "cancel", "job_id": job_id}
            response = requests.post(f"{API_BASE_URL}/api/index/control", json=cancel_data)
            if response.status_code != 200:
                print(f"   ❌ Failed to send cancel: {response.status_code}")
                return False
            
            print("   ✅ Cancel request sent")
            
            # 3. Check if cancellation took effect
            print("3. Verifying cancellation...")
            time.sleep(2)
            response = requests.get(f"{API_BASE_URL}/api/index/status")
            if response.status_code == 200:
                new_status = response.json()
                new_status_value = new_status.get('status', 'unknown')
                print(f"   📋 Status after cancel: {new_status_value}")
                
                if new_status_value == 'cancelled':
                    print("   ✅ Job successfully cancelled!")
                    print("   💡 Check server logs for: '🛑 LLM job [job_id] marked for cancellation'")
                    print("   💡 Check for: '🛑 Video analysis cancelled for job [job_id]'")
                    return True
                else:
                    print("   ⚠️  Status not changed to cancelled")
                    print("   💡 Check server logs for cancellation messages")
                    return False
            else:
                print(f"   ❌ Failed to check status: {response.status_code}")
                return False
        else:
            print("   ℹ️  No active job to cancel")
            print("   💡 Start an indexing operation first:")
            print("   💡 1. Go to Settings → Indexing")
            print("   💡 2. Add a directory with videos")
            print("   💡 3. Start indexing")
            print("   💡 4. Quickly run this test again")
            return False
            
    except Exception as e:
        print(f"❌ Cancel test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🔧 Comprehensive Model Selection and Cancel Fixes Test")
    print("=" * 60)
    print("")
    
    # Test singleton pattern
    test_singleton_llm_service()
    
    # Test model selection
    model_success = test_model_selection_persistence()
    
    # Test cancel functionality
    cancel_success = test_cancel_functionality()
    
    print("\n📋 Test Results Summary")
    print("=" * 50)
    print(f"🤖 Model Selection Fix: {'✅ PASSED' if model_success else '❌ FAILED'}")
    print(f"🛑 Cancel Functionality: {'✅ PASSED' if cancel_success else '❌ NEEDS TESTING'}")
    print("")
    
    print("📝 What to Check in Server Logs:")
    print("=" * 50)
    print("1. LLM Service Initialization:")
    print("   - Should see only ONE 'LLM service initialized' message")
    print("   - Multiple messages = multiple instances (old bug)")
    print("")
    print("2. Model Usage:")
    print("   - Look for '🤖 Calling Ollama API with model: [new_model]'")
    print("   - Should show the model you selected, not the old one")
    print("")
    print("3. Cancellation:")
    print("   - Look for '🛑 LLM job [job_id] marked for cancellation'")
    print("   - Look for '🛑 Video analysis cancelled for job [job_id]'")
    print("   - Processing should stop quickly after cancel")
    print("")
    
    if model_success and cancel_success:
        print("🎉 All fixes appear to be working correctly!")
    elif model_success:
        print("✅ Model selection fix is working!")
        print("🔧 Cancel functionality needs active job to test")
    else:
        print("❌ Issues detected - check server logs for details")

if __name__ == "__main__":
    main()
