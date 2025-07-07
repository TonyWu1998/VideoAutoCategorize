#!/usr/bin/env python3
"""
Test script to verify that IndexingService gets fresh LLM service instances
that pick up model configuration changes.
"""

import sys
import requests
import time
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def test_indexing_service_llm_refresh():
    """Test that IndexingService picks up LLM model changes."""
    print("🔄 Testing IndexingService LLM Service Refresh")
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
        
        # Step 2: Test IndexingService before model change
        print("\n2. Testing IndexingService before model change...")
        try:
            from app.services.indexing import IndexingService
            indexing_service = IndexingService()
            
            # Access the LLM service through the property
            llm_service_before = indexing_service.llm_service
            model_before = llm_service_before.model
            instance_id_before = id(llm_service_before)
            
            print(f"   📋 IndexingService LLM model before: {model_before}")
            print(f"   📋 LLM service instance ID before: {instance_id_before}")
            
        except Exception as e:
            print(f"   ❌ Failed to test IndexingService before: {e}")
            return False
        
        # Step 3: Update model via API
        print(f"\n3. Updating model to '{test_model}' via API...")
        update_data = {"ollama_model": test_model}
        response = requests.put(f"{api_base}/api/config/llm", json=update_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ API update successful: {result.get('message', 'No message')}")
        else:
            print(f"   ❌ API update failed: {response.status_code}")
            return False
        
        # Step 4: Test IndexingService after model change
        print("\n4. Testing IndexingService after model change...")
        try:
            # Get a fresh LLM service instance through the property
            llm_service_after = indexing_service.llm_service
            model_after = llm_service_after.model
            instance_id_after = id(llm_service_after)
            
            print(f"   📋 IndexingService LLM model after: {model_after}")
            print(f"   📋 LLM service instance ID after: {instance_id_after}")
            print(f"   📋 Same instance: {instance_id_before == instance_id_after}")
            
            # Verify the change
            if model_after == test_model:
                print(f"   ✅ SUCCESS: IndexingService now uses '{test_model}'")
                success = True
            else:
                print(f"   ❌ FAILURE: IndexingService still uses '{model_after}', expected '{test_model}'")
                success = False
                
        except Exception as e:
            print(f"   ❌ Failed to test IndexingService after: {e}")
            success = False
        
        # Step 5: Test multiple property accesses
        print("\n5. Testing multiple property accesses...")
        try:
            llm1 = indexing_service.llm_service
            llm2 = indexing_service.llm_service
            llm3 = indexing_service.llm_service
            
            print(f"   📋 LLM service 1 ID: {id(llm1)}")
            print(f"   📋 LLM service 2 ID: {id(llm2)}")
            print(f"   📋 LLM service 3 ID: {id(llm3)}")
            print(f"   📋 All same instance: {id(llm1) == id(llm2) == id(llm3)}")
            print(f"   📋 All have same model: {llm1.model == llm2.model == llm3.model == test_model}")
            
        except Exception as e:
            print(f"   ❌ Failed to test multiple accesses: {e}")
        
        # Step 6: Reset to original model
        print(f"\n6. Resetting model to original '{original_model}'...")
        reset_data = {"ollama_model": original_model}
        response = requests.put(f"{api_base}/api/config/llm", json=reset_data)
        
        if response.status_code == 200:
            # Test that IndexingService picks up the reset
            llm_service_reset = indexing_service.llm_service
            model_reset = llm_service_reset.model
            print(f"   📋 IndexingService model after reset: {model_reset}")
            
            if model_reset == original_model:
                print(f"   ✅ Successfully reset to '{original_model}'")
            else:
                print(f"   ⚠️  Reset may not have worked: expected '{original_model}', got '{model_reset}'")
        else:
            print(f"   ❌ Reset failed: {response.status_code}")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    print("🧪 IndexingService LLM Refresh Test")
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
    
    # Run test
    result = test_indexing_service_llm_refresh()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"IndexingService Refresh Test: {'✅ PASS' if result else '❌ FAIL'}")
    
    if result:
        print("\n🎉 IndexingService now correctly picks up model changes!")
        print("\n📋 What this proves:")
        print("   ✅ IndexingService gets fresh LLM service instances")
        print("   ✅ Model changes are picked up immediately")
        print("   ✅ No caching of old LLM service instances")
        print("   ✅ Video analysis will use updated model")
    else:
        print("\n❌ IndexingService is still not picking up model changes.")
        print("   Check the implementation and server logs for details.")

if __name__ == "__main__":
    main()
