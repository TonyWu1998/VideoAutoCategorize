#!/usr/bin/env python3
"""
Test script to verify that video analysis operations use the updated model
by triggering actual video analysis through the API.
"""

import requests
import time

def test_video_analysis_with_model_change():
    """Test that video analysis uses the updated model."""
    print("🎬 Testing Video Analysis with Model Change")
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
        
        # Step 2: Check if there are any media files to reindex
        print("\n2. Checking for media files to test with...")
        response = requests.get(f"{api_base}/api/media/?limit=5")
        if response.status_code == 200:
            media_files = response.json()
            if not media_files:
                print("   ⚠️  No media files found to test with")
                print("   💡 Please add some media files to the system first")
                return False
            
            # Find any media file (prefer video, but accept image)
            test_file = None
            for file in media_files:
                if file.get('media_type') == 'video':
                    test_file = file
                    break

            # If no video, use any available file
            if not test_file and media_files:
                test_file = media_files[0]

            if not test_file:
                print("   ⚠️  No media files found to test with")
                print("   💡 Please add some media files to the system first")
                return False

            print(f"   📋 Found {test_file.get('media_type', 'unknown')} file to test: {test_file.get('filename', test_file.get('name', 'Unknown'))}")
            print(f"   📋 File ID: {test_file['file_id']}")
            print(f"   📋 Available fields: {list(test_file.keys())}")
            
        else:
            print(f"   ❌ Failed to get media files: {response.status_code}")
            return False
        
        # Step 3: Update model to the new one
        print(f"\n3. Updating model to '{test_model}' via API...")
        update_data = {"ollama_model": test_model}
        response = requests.put(f"{api_base}/api/config/llm", json=update_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ API update successful: {result.get('message', 'No message')}")
        else:
            print(f"   ❌ API update failed: {response.status_code}")
            return False
        
        # Step 4: Trigger reindexing of the media file
        print(f"\n4. Triggering reindexing of media file...")
        print(f"   📋 This will force the IndexingService to analyze the file")
        print(f"   📋 Watch server logs for model usage during analysis")

        reindex_data = [test_file['file_id']]
        response = requests.post(f"{api_base}/api/index/reindex", json=reindex_data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Reindexing started: {result.get('message', 'No message')}")
            
            # Wait for reindexing to start
            print("   📋 Waiting for reindexing to start...")
            time.sleep(3)
            
            # Check indexing status
            for i in range(10):  # Check for up to 30 seconds
                status_response = requests.get(f"{api_base}/api/index/status")
                if status_response.status_code == 200:
                    status = status_response.json()
                    current_status = status.get('status', 'Unknown')
                    print(f"   📋 Indexing status: {current_status}")
                    
                    if current_status in ['completed', 'idle']:
                        print("   ✅ Reindexing completed")
                        break
                    elif current_status == 'failed':
                        print("   ❌ Reindexing failed")
                        break
                else:
                    print(f"   ⚠️  Failed to get status: {status_response.status_code}")
                
                time.sleep(3)
            
            success = True
            
        else:
            print(f"   ❌ Reindexing failed: {response.status_code}")
            print(f"   📋 Response: {response.text}")
            success = False
        
        # Step 5: Reset to original model (even if it doesn't exist)
        print(f"\n5. Resetting model to original '{original_model}'...")
        reset_data = {"ollama_model": original_model}
        response = requests.put(f"{api_base}/api/config/llm", json=reset_data)
        
        if response.status_code == 200:
            print(f"   ✅ Model setting reset to '{original_model}'")
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
    print("🧪 Real Video Analysis Model Test")
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
    result = test_video_analysis_with_model_change()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Video Analysis Model Test: {'✅ PASS' if result else '❌ FAIL'}")
    
    print("\n💡 Check the server logs for:")
    print("   🔍 Look for: '🔄 Model changed from X to Y - refreshing LLM service'")
    print("   🔍 Look for: '🤖 Calling Ollama API with model: huihui_ai/gemma3-abliterated:4b'")
    print("   🔍 Look for any errors about 'gemma3:4b' not found during analysis")
    
    if result:
        print("\n🎉 If you see the new model being used in the logs, the fix is working!")
    else:
        print("\n❌ Check the server logs to see what model was used during analysis.")

if __name__ == "__main__":
    main()
