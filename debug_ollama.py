#!/usr/bin/env python3
"""
Debug script to test Ollama connection and model availability.
"""

import sys
import json

def test_ollama_connection():
    """Test Ollama connection using the same method as the backend."""
    try:
        import ollama
        
        print("✅ Ollama module imported successfully")
        
        # Try to connect to Ollama
        client = ollama.Client(host="http://localhost:11434")
        print("✅ Ollama client created")
        
        # Check if we can list models
        models = client.list()
        print(f"✅ Models response received: {type(models)}")
        print(f"Raw response: {models}")
        
        # Handle different response formats
        if hasattr(models, 'models'):
            model_list = models.models
            print(f"✅ Using models.models: {type(model_list)}")
        elif isinstance(models, dict) and 'models' in models:
            model_list = models['models']
            print(f"✅ Using models['models']: {type(model_list)}")
        else:
            model_list = models if isinstance(models, list) else []
            print(f"✅ Using models directly: {type(model_list)}")
        
        print(f"Model list: {model_list}")
        
        model_names = []
        for i, model in enumerate(model_list):
            print(f"Model {i}: {model} (type: {type(model)})")
            if hasattr(model, 'model'):
                model_names.append(model.model)
                print(f"  - Using model.model: {model.model}")
            elif hasattr(model, 'name'):
                model_names.append(model.name)
                print(f"  - Using model.name: {model.name}")
            elif isinstance(model, dict) and 'name' in model:
                model_names.append(model['name'])
                print(f"  - Using model['name']: {model['name']}")
            elif isinstance(model, dict) and 'model' in model:
                model_names.append(model['model'])
                print(f"  - Using model['model']: {model['model']}")
            elif isinstance(model, str):
                model_names.append(model)
                print(f"  - Using model as string: {model}")
            else:
                print(f"  - Unknown model format: {model}")
        
        print(f"✅ Extracted model names: {model_names}")
        
        target_model = "gemma3:4b"
        is_available = target_model in model_names
        print(f"✅ Model '{target_model}' available: {is_available}")
        
        return is_available
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Testing Ollama connection...")
    result = test_ollama_connection()
    print(f"\n🎯 Final result: {'SUCCESS' if result else 'FAILED'}")
    sys.exit(0 if result else 1)
