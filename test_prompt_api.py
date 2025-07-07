#!/usr/bin/env python3
"""
Test script for the prompt management API endpoints.
"""

import requests
import json
import sys
from typing import Dict, Any

# API configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def test_endpoint(method: str, endpoint: str, data: Dict[Any, Any] = None) -> Dict[Any, Any]:
    """Test an API endpoint and return the response."""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=HEADERS)
        elif method == "POST":
            response = requests.post(url, headers=HEADERS, json=data)
        elif method == "PUT":
            response = requests.put(url, headers=HEADERS, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=HEADERS)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        print(f"🔍 {method} {endpoint}")
        print(f"   Status: {response.status_code}")
        
        if response.status_code >= 400:
            print(f"   Error: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
        
        try:
            result = response.json()
            print(f"   Response: {json.dumps(result, indent=2)[:200]}...")
            return result
        except:
            print(f"   Response: {response.text[:200]}...")
            return {"text": response.text}
            
    except Exception as e:
        print(f"   Exception: {e}")
        return {"error": str(e)}

def main():
    """Run prompt API tests."""
    print("🧪 Testing Prompt Management API Endpoints")
    print("=" * 50)
    
    # Test 1: List prompt templates (should be empty initially)
    print("\n1. List prompt templates")
    templates = test_endpoint("GET", "/api/config/prompts/")
    
    # Test 2: Get active configuration (should have defaults after startup)
    print("\n2. Get active configuration")
    config = test_endpoint("GET", "/api/config/prompts/config/active")
    
    # Test 3: Create a new prompt template
    print("\n3. Create new prompt template")
    new_template = {
        "name": "Test Image Prompt",
        "description": "A test prompt for image analysis",
        "media_type": "image",
        "prompt_text": "Analyze this image and describe what you see in detail.",
        "version": "1.0"
    }
    created = test_endpoint("POST", "/api/config/prompts/", new_template)
    
    if "template_id" in created:
        template_id = created["template_id"]
        
        # Test 4: Get the created template
        print(f"\n4. Get template {template_id}")
        template = test_endpoint("GET", f"/api/config/prompts/{template_id}")
        
        # Test 5: Update the template
        print(f"\n5. Update template {template_id}")
        updated_template = {
            "name": "Updated Test Image Prompt",
            "description": "An updated test prompt for image analysis",
            "media_type": "image",
            "prompt_text": "Analyze this image and provide a detailed description with objects and mood.",
            "version": "1.1"
        }
        updated = test_endpoint("PUT", f"/api/config/prompts/{template_id}", updated_template)
        
        # Test 6: Validate a prompt
        print("\n6. Validate prompt")
        validation_request = {
            "prompt_text": "Analyze this image and provide JSON output.",
            "media_type": "image"
        }
        validation = test_endpoint("POST", "/api/config/prompts/validate", validation_request)
        
        # Test 7: Test a prompt
        print("\n7. Test prompt")
        test_request = {
            "prompt_text": "Analyze this image and describe what you see.",
            "media_type": "image"
        }
        test_result = test_endpoint("POST", "/api/config/prompts/test", test_request)
        
        # Test 8: Update active configuration
        print("\n8. Update active configuration")
        config_update = {
            "active_image_prompt_id": template_id
        }
        updated_config = test_endpoint("PUT", "/api/config/prompts/config/active", config_update)
        
        # Test 9: List templates again (should show our new template)
        print("\n9. List templates again")
        templates_after = test_endpoint("GET", "/api/config/prompts/")
        
        # Test 10: Delete the template (should fail if it's active)
        print(f"\n10. Try to delete active template {template_id}")
        delete_result = test_endpoint("DELETE", f"/api/config/prompts/{template_id}")
        
    else:
        print("❌ Failed to create template, skipping dependent tests")
    
    print("\n" + "=" * 50)
    print("✅ Prompt API testing completed")

if __name__ == "__main__":
    main()
