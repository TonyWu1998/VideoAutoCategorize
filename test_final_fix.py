#!/usr/bin/env python3
"""
Final test to verify the model selection fix is working.
"""

import requests
import time
import subprocess

API_BASE_URL = "http://localhost:8000"

def check_current_model_setting():
    """Check what model is currently set in the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/config/llm")
        if response.status_code == 200:
            config = response.json()
            return config.get('ollama_model', 'Unknown')
        return None
    except:
        return None

def get_recent_llm_logs():
    """Get recent LLM-related logs."""
    try:
        result = subprocess.run(
            ["grep", "-E", "(LLM service initialized|🤖 Calling Ollama|🔍)", "backend/logs/app.log"],
            capture_output=True,
            text=True,
            cwd="/Users/tony/Desktop/projects/VideoAutoCategorize"
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            return lines[-10:]  # Last 10 lines
        return []
    except:
        return []

def main():
    """Main test function."""
    print("🔬 Final Model Selection Fix Test")
    print("=" * 60)
    
    # Check current model setting
    current_model = check_current_model_setting()
    print(f"📋 Current API model setting: {current_model}")
    
    # Check recent logs
    print("\n📋 Recent LLM-related logs:")
    recent_logs = get_recent_llm_logs()
    for log in recent_logs[-5:]:  # Show last 5
        print(f"   {log}")
    
    print("\n🎯 Test Status:")
    print("=" * 50)
    
    if current_model == "huihui_ai/gemma3-abliterated:4b":
        print("✅ API model setting is correct")
        
        # Check if any recent logs show the new model
        new_model_in_logs = any("huihui_ai/gemma3-abliterated:4b" in log for log in recent_logs)
        
        if new_model_in_logs:
            print("✅ New model found in recent logs")
            print("🎉 MODEL SELECTION FIX IS WORKING!")
        else:
            print("⚠️  New model not yet used in LLM calls")
            print("💡 This means the LLM service hasn't been recreated since the fix")
            
            # Check if there are any recent LLM calls with old model
            old_model_in_recent_logs = any("gemma3:4b" in log and "🤖 Calling Ollama" in log for log in recent_logs)
            
            if old_model_in_recent_logs:
                print("❌ Recent logs still show old model being used")
                print("🔧 This indicates the singleton pattern may not be working")
            else:
                print("✅ No recent LLM calls with old model")
                print("💡 Need to trigger an analysis to test the fix")
    else:
        print(f"❌ API model setting is wrong: {current_model}")
        print("🔧 Settings update mechanism is not working")
    
    print("\n📝 Recommendations:")
    print("=" * 50)
    
    if current_model == "huihui_ai/gemma3-abliterated:4b":
        print("1. ✅ Settings update is working correctly")
        print("2. 🔄 To test the singleton fix:")
        print("   a) Go to the web UI")
        print("   b) Upload an image or start indexing")
        print("   c) Check logs for: '🤖 Calling Ollama API with model: huihui_ai/gemma3-abliterated:4b'")
        print("   d) Look for singleton debug messages:")
        print("      - '🔍 Creating NEW LLM service singleton instance'")
        print("      - '🔍 LLM service model property accessed, returning: huihui_ai/gemma3-abliterated:4b'")
        print("")
        print("3. 🎯 Expected outcome:")
        print("   - If singleton is working: New model will be used immediately")
        print("   - If singleton is broken: Old model will still be used")
    else:
        print("1. ❌ Fix the settings update mechanism first")
        print("2. 🔧 Check API endpoint and settings object")
    
    print("\n🔍 Quick Test Command:")
    print("=" * 50)
    print("To immediately test, run this after triggering an analysis:")
    print("grep '🤖 Calling Ollama' backend/logs/app.log | tail -1")
    print("")
    print("Expected result:")
    print("🤖 Calling Ollama API with model: huihui_ai/gemma3-abliterated:4b")

if __name__ == "__main__":
    main()
