#!/usr/bin/env python3
"""
Debug script to investigate the model selection issue.
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

def debug_model_selection_issue():
    """Debug the model selection issue step by step."""
    print("🔍 Debugging Model Selection Issue")
    print("=" * 60)
    
    if not test_api_connection():
        print("❌ Cannot connect to API server. Please start the backend first.")
        return
    
    try:
        # Step 1: Check current API configuration
        print("1. Checking current API configuration...")
        response = requests.get(f"{API_BASE_URL}/api/config/llm")
        if response.status_code == 200:
            current_config = response.json()
            current_model = current_config.get('ollama_model', 'Unknown')
            print(f"   📋 API reports current model: {current_model}")
        else:
            print(f"   ❌ Failed to get config: {response.status_code}")
            return
        
        # Step 2: Check available models
        print("\n2. Checking available models...")
        response = requests.get(f"{API_BASE_URL}/api/config/ollama/models")
        if response.status_code == 200:
            models_data = response.json()
            available_models = models_data.get('vision_models', [])
            print(f"   📊 Available vision models: {available_models}")
            
            # Check if huihui_ai/gemma3 is available
            if 'huihui_ai/gemma3' in available_models:
                target_model = 'huihui_ai/gemma3'
                print(f"   ✅ Target model '{target_model}' is available")
            elif 'huihui_ai' in available_models:
                target_model = 'huihui_ai'
                print(f"   ✅ Alternative target model '{target_model}' is available")
            else:
                print("   ❌ Neither 'huihui_ai/gemma3' nor 'huihui_ai' are available")
                print("   💡 Available models:", available_models)
                if available_models:
                    target_model = available_models[0]
                    print(f"   🔄 Using '{target_model}' for testing")
                else:
                    print("   ❌ No vision models available")
                    return
        else:
            print(f"   ❌ Failed to get models: {response.status_code}")
            return
        
        # Step 3: Update model via API
        print(f"\n3. Updating model to '{target_model}' via API...")
        update_data = {"ollama_model": target_model}
        response = requests.put(f"{API_BASE_URL}/api/config/llm", json=update_data)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ API update successful")
            print(f"   📄 Response: {json.dumps(result, indent=2)}")
        else:
            print(f"   ❌ Failed to update model: {response.status_code}")
            print(f"   📄 Error response: {response.text}")
            return
        
        # Step 4: Verify API reflects the change immediately
        print("\n4. Verifying API reflects the change...")
        time.sleep(0.5)
        response = requests.get(f"{API_BASE_URL}/api/config/llm")
        if response.status_code == 200:
            updated_config = response.json()
            new_model = updated_config.get('ollama_model', 'Unknown')
            print(f"   📋 API now reports model: {new_model}")
            
            if new_model == target_model:
                print(f"   ✅ API correctly shows updated model!")
            else:
                print(f"   ❌ API still shows old model. Expected: {target_model}, Got: {new_model}")
                print("   🔧 This indicates the settings.OLLAMA_MODEL is not being updated")
                return
        else:
            print(f"   ❌ Failed to verify: {response.status_code}")
            return
        
        # Step 5: Test with a simple analysis to see what model is actually used
        print("\n5. Testing actual model usage...")
        print("   💡 Now trigger an analysis operation (upload image or start indexing)")
        print("   💡 Check server logs for:")
        print(f"   💡   - 'LLM service initialized with model: [model_name]'")
        print(f"   💡   - '🤖 Calling Ollama API with model: {target_model}'")
        print("")
        print("   🔍 If logs still show 'gemma3:4b', the issue is:")
        print("   🔍   1. Singleton pattern not working (multiple instances)")
        print("   🔍   2. LLM service was initialized before model change")
        print("   🔍   3. Settings object is not the same instance")
        
        # Step 6: Additional debugging info
        print("\n6. Additional debugging steps:")
        print("   📋 Check server logs for:")
        print("   📋   - How many 'LLM service initialized' messages appear")
        print("   📋   - When they appear relative to model changes")
        print("   📋   - What model each initialization shows")
        print("")
        print("   🔧 If multiple 'LLM service initialized' messages:")
        print("   🔧   → Singleton pattern is not working")
        print("   🔧   → Multiple instances are being created")
        print("")
        print("   🔧 If only one initialization but wrong model:")
        print("   🔧   → LLM service was created before model update")
        print("   🔧   → Need to ensure dynamic property is working")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_settings_object_identity():
    """Test if the settings object is consistent."""
    print("\n🔬 Testing Settings Object Identity")
    print("=" * 50)
    
    print("This test requires backend code inspection:")
    print("1. Check if 'from app.config import settings' imports the same object everywhere")
    print("2. Verify that settings.OLLAMA_MODEL is actually being modified")
    print("3. Confirm that the LLM service's self.model property reads from the same settings")
    print("")
    print("💡 Add debug logging to LLM service model property:")
    print("   @property")
    print("   def model(self):")
    print("       logger.info(f'🔍 LLM service model property called, returning: {settings.OLLAMA_MODEL}')")
    print("       return settings.OLLAMA_MODEL")

def main():
    """Main debug function."""
    print("🐛 Model Selection Issue Debug Tool")
    print("=" * 60)
    
    success = debug_model_selection_issue()
    test_settings_object_identity()
    
    print("\n📋 Summary of Potential Issues:")
    print("=" * 50)
    print("1. 🔄 Singleton Pattern Issues:")
    print("   - Multiple LLM service instances still being created")
    print("   - Class-level variables not working as expected")
    print("")
    print("2. ⚙️  Settings Object Issues:")
    print("   - Different settings instances in different modules")
    print("   - settings.OLLAMA_MODEL not actually being updated")
    print("")
    print("3. 🕐 Timing Issues:")
    print("   - LLM service initialized before model change")
    print("   - Property not being called when expected")
    print("")
    print("4. 🔍 Debugging Steps:")
    print("   - Add logging to model property")
    print("   - Check singleton instance creation")
    print("   - Verify settings object identity")
    print("   - Monitor initialization timing")
    
    if success:
        print("\n✅ API level changes are working correctly")
        print("🔧 Issue is likely in LLM service singleton or property access")
    else:
        print("\n❌ API level changes are not working")
        print("🔧 Issue is in the settings update mechanism")

if __name__ == "__main__":
    main()
