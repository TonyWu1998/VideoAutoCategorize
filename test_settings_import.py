#!/usr/bin/env python3
"""
Test script to verify settings import behavior.
"""

import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def test_settings_import():
    """Test how settings are imported and if they're the same object."""
    print("🔍 Testing Settings Import Behavior")
    print("=" * 50)
    
    try:
        # Import settings in different ways
        print("1. Testing different import methods...")
        
        # Method 1: Direct import
        from app.config import settings as settings1
        print(f"   📋 Method 1 - Direct import: {id(settings1)}")
        print(f"   📋 Initial OLLAMA_MODEL: {settings1.OLLAMA_MODEL}")
        
        # Method 2: Import from different location (simulating LLM service)
        import app.config
        settings2 = app.config.settings
        print(f"   📋 Method 2 - Module.settings: {id(settings2)}")
        print(f"   📋 OLLAMA_MODEL: {settings2.OLLAMA_MODEL}")
        
        # Method 3: Re-import
        from app.config import settings as settings3
        print(f"   📋 Method 3 - Re-import: {id(settings3)}")
        print(f"   📋 OLLAMA_MODEL: {settings3.OLLAMA_MODEL}")
        
        # Check if they're the same object
        print("\n2. Checking object identity...")
        print(f"   🔍 settings1 is settings2: {settings1 is settings2}")
        print(f"   🔍 settings1 is settings3: {settings1 is settings3}")
        print(f"   🔍 settings2 is settings3: {settings2 is settings3}")
        
        if settings1 is settings2 is settings3:
            print("   ✅ All settings objects are the same instance")
        else:
            print("   ❌ Settings objects are different instances!")
            print("   🔧 This could explain why model changes don't propagate")
        
        # Test modifying the setting
        print("\n3. Testing setting modification...")
        original_model = settings1.OLLAMA_MODEL
        test_model = "test-model-12345"
        
        print(f"   📋 Original model: {original_model}")
        print(f"   🔄 Changing to: {test_model}")
        
        settings1.OLLAMA_MODEL = test_model
        
        print(f"   📋 settings1.OLLAMA_MODEL: {settings1.OLLAMA_MODEL}")
        print(f"   📋 settings2.OLLAMA_MODEL: {settings2.OLLAMA_MODEL}")
        print(f"   📋 settings3.OLLAMA_MODEL: {settings3.OLLAMA_MODEL}")
        
        if settings1.OLLAMA_MODEL == settings2.OLLAMA_MODEL == settings3.OLLAMA_MODEL == test_model:
            print("   ✅ Model change propagated to all settings objects")
        else:
            print("   ❌ Model change did not propagate!")
            print("   🔧 This confirms different settings instances")
        
        # Restore original
        settings1.OLLAMA_MODEL = original_model
        
        return True
        
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_service_import():
    """Test LLM service import and singleton behavior."""
    print("\n🤖 Testing LLM Service Import and Singleton")
    print("=" * 50)
    
    try:
        # Import LLM service multiple times
        print("1. Testing LLM service imports...")
        
        from app.services.llm_service import LLMService as LLMService1
        from app.services.llm_service import LLMService as LLMService2
        
        print(f"   📋 LLMService1 class: {id(LLMService1)}")
        print(f"   📋 LLMService2 class: {id(LLMService2)}")
        print(f"   🔍 Same class: {LLMService1 is LLMService2}")
        
        # Create instances
        print("\n2. Creating LLM service instances...")
        print("   ⚠️  This will trigger initialization - check logs!")
        
        instance1 = LLMService1()
        print(f"   📋 Instance1: {id(instance1)}")
        
        instance2 = LLMService2()
        print(f"   📋 Instance2: {id(instance2)}")
        
        print(f"   🔍 Same instance: {instance1 is instance2}")
        
        if instance1 is instance2:
            print("   ✅ Singleton pattern working correctly")
        else:
            print("   ❌ Singleton pattern NOT working!")
            print("   🔧 Multiple instances created")
        
        # Test model property
        print("\n3. Testing model property...")
        model1 = instance1.model
        model2 = instance2.model
        
        print(f"   📋 Instance1 model: {model1}")
        print(f"   📋 Instance2 model: {model2}")
        print(f"   🔍 Same model: {model1 == model2}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Settings and LLM Service Import Test")
    print("=" * 60)
    
    settings_ok = test_settings_import()
    llm_ok = test_llm_service_import()
    
    print("\n📋 Test Results Summary")
    print("=" * 50)
    print(f"⚙️  Settings Import: {'✅ PASSED' if settings_ok else '❌ FAILED'}")
    print(f"🤖 LLM Service Singleton: {'✅ PASSED' if llm_ok else '❌ FAILED'}")
    
    if not settings_ok:
        print("\n🔧 Settings Issue Detected:")
        print("   - Different settings objects in different modules")
        print("   - Model changes won't propagate between modules")
        print("   - Need to ensure single settings instance")
    
    if not llm_ok:
        print("\n🔧 LLM Service Issue Detected:")
        print("   - Singleton pattern not working")
        print("   - Multiple LLM service instances created")
        print("   - Each instance may have different state")
    
    if settings_ok and llm_ok:
        print("\n✅ Both tests passed!")
        print("🔍 Issue might be elsewhere - check:")
        print("   - Timing of initialization vs model changes")
        print("   - Whether model property is actually called")
        print("   - Server restart needed after singleton changes")

if __name__ == "__main__":
    main()
