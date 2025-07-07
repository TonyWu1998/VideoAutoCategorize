#!/usr/bin/env python3
"""
Test script to verify that the LLM service model refresh mechanism works correctly.
"""

import sys
import requests
import time
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def test_model_refresh():
    """Test the model refresh mechanism."""
    print("🔄 Testing LLM Service Model Refresh Mechanism")
    print("=" * 60)
    
    # Test configuration
    api_base = "http://localhost:8000"
    original_model = "gemma3:4b"
    test_model = "huihui_ai/gemma3-abliterated:4b"
    
    try:
        # Step 1: Get current configuration
        print("1. Getting current LLM configuration...")
        response = requests.get(f"{api_base}/api/config/llm")
        if response.status_code == 200:
            current_config = response.json()
            print(f"   📋 Current model: {current_config.get('ollama_model', 'Unknown')}")
        else:
            print(f"   ❌ Failed to get current config: {response.status_code}")
            return False
        
        # Step 2: Create LLM service instance (this will initialize it)
        print("\n2. Initializing LLM service...")
        try:
            from app.services.llm_service import LLMService
            llm_service = LLMService()
            initial_model = llm_service.model
            print(f"   📋 LLM service initialized with model: {initial_model}")
            print(f"   📋 LLM service instance ID: {id(llm_service)}")
            print(f"   📋 Initialized model tracked: {getattr(llm_service, '_initialized_model', 'Not set')}")
        except Exception as e:
            print(f"   ❌ Failed to initialize LLM service: {e}")
            return False
        
        # Step 3: Update model via API
        print(f"\n3. Updating model to '{test_model}' via API...")
        update_data = {"ollama_model": test_model}
        response = requests.put(f"{api_base}/api/config/llm", json=update_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ API update successful: {result.get('message', 'No message')}")
            print(f"   📋 Updated settings: {result.get('updated_settings', {})}")
        else:
            print(f"   ❌ API update failed: {response.status_code}")
            print(f"   📋 Response: {response.text}")
            return False
        
        # Step 4: Check if LLM service picked up the change
        print("\n4. Checking if LLM service picked up the model change...")
        
        # Get a fresh instance (should be the same singleton)
        llm_service_after = LLMService()
        updated_model = llm_service_after.model
        
        print(f"   📋 LLM service instance ID after update: {id(llm_service_after)}")
        print(f"   📋 Same instance: {llm_service is llm_service_after}")
        print(f"   📋 Model after update: {updated_model}")
        print(f"   📋 Tracked initialized model: {getattr(llm_service_after, '_initialized_model', 'Not set')}")
        
        # Step 5: Verify the change
        if updated_model == test_model:
            print(f"   ✅ SUCCESS: Model correctly updated to '{test_model}'")
            success = True
        else:
            print(f"   ❌ FAILURE: Model is still '{updated_model}', expected '{test_model}'")
            success = False
        
        # Step 6: Test manual refresh (should be no-op if already refreshed)
        print("\n5. Testing manual refresh call...")
        try:
            LLMService.refresh_if_model_changed()
            print("   ✅ Manual refresh call completed")
        except Exception as e:
            print(f"   ❌ Manual refresh failed: {e}")
        
        # Step 7: Reset to original model
        print(f"\n6. Resetting model to original '{original_model}'...")
        reset_data = {"ollama_model": original_model}
        response = requests.put(f"{api_base}/api/config/llm", json=reset_data)
        
        if response.status_code == 200:
            # Check if reset worked
            final_model = LLMService().model
            print(f"   📋 Model after reset: {final_model}")
            if final_model == original_model:
                print(f"   ✅ Successfully reset to '{original_model}'")
            else:
                print(f"   ⚠️  Reset may not have worked: expected '{original_model}', got '{final_model}'")
        else:
            print(f"   ❌ Reset failed: {response.status_code}")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_refresh_without_api():
    """Test the refresh mechanism directly without API calls."""
    print("\n🔧 Testing Direct Refresh Mechanism")
    print("=" * 40)
    
    try:
        from app.config import settings
        from app.services.llm_service import LLMService
        
        # Get initial state
        print("1. Getting initial state...")
        llm_service = LLMService()
        initial_model = llm_service.model
        initial_tracked = getattr(llm_service, '_initialized_model', 'Not set')
        
        print(f"   📋 Current model: {initial_model}")
        print(f"   📋 Tracked model: {initial_tracked}")
        
        # Manually change settings
        print("\n2. Manually changing settings...")
        test_model = "test-model-direct-123"
        old_model = settings.OLLAMA_MODEL
        settings.OLLAMA_MODEL = test_model
        print(f"   📋 Changed settings.OLLAMA_MODEL to: {test_model}")
        
        # Test refresh
        print("\n3. Calling refresh_if_model_changed()...")
        LLMService.refresh_if_model_changed()
        
        # Check result
        new_model = llm_service.model
        new_tracked = getattr(llm_service, '_initialized_model', 'Not set')
        
        print(f"   📋 Model after refresh: {new_model}")
        print(f"   📋 Tracked after refresh: {new_tracked}")
        
        if new_model == test_model and new_tracked == test_model:
            print("   ✅ Direct refresh mechanism working correctly")
            success = True
        else:
            print("   ❌ Direct refresh mechanism failed")
            success = False
        
        # Restore original
        settings.OLLAMA_MODEL = old_model
        LLMService.refresh_if_model_changed()
        print(f"   📋 Restored to: {llm_service.model}")
        
        return success
        
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 LLM Service Model Refresh Test Suite")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not responding correctly")
            return
    except requests.exceptions.RequestException:
        print("❌ Server is not running. Please start the backend server first.")
        print("   Run: cd backend && python -m uvicorn app.main:app --reload")
        return
    
    print("✅ Server is running\n")
    
    # Run tests
    test1_result = test_refresh_without_api()
    test2_result = test_model_refresh()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Direct Refresh Test: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"API Refresh Test:    {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 All tests passed! Model refresh mechanism is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
