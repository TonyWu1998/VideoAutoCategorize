#!/usr/bin/env python3
"""
Test script to verify all API endpoints are working correctly.
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_endpoint(method, endpoint, data=None, expected_status=200):
    """Test a single API endpoint."""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        if response.status_code == expected_status:
            print(f"✅ {method} {endpoint} - Status: {response.status_code}")
            if response.content:
                try:
                    result = response.json()
                    print(f"   Response: {json.dumps(result, indent=2)[:200]}...")
                except:
                    print(f"   Response: {response.text[:200]}...")
            return True
        else:
            print(f"❌ {method} {endpoint} - Expected: {expected_status}, Got: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ {method} {endpoint} - Error: {e}")
        return False

def main():
    """Run all API endpoint tests."""
    print("🧪 Testing API Endpoints")
    print("=" * 50)
    
    tests = [
        # Health endpoints
        ("GET", "/api/health"),
        ("GET", "/api/health/simple"),
        ("GET", "/api/health/database"),
        ("GET", "/api/health/ollama"),
        
        # Config endpoints
        ("GET", "/api/config/llm"),
        ("PUT", "/api/config/llm", {"video_frame_interval": 45, "ollama_model": "gemma3:4b"}),
        ("GET", "/api/config/llm"),  # Verify the update worked
        ("POST", "/api/config/llm/reset"),
        ("GET", "/api/config/llm"),  # Verify reset worked
        ("GET", "/api/config/system"),
        
        # Indexing endpoints
        ("GET", "/api/index/status"),
        
        # Root endpoints
        ("GET", "/"),
        ("GET", "/api"),
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if len(test) == 2:
            method, endpoint = test
            data = None
        else:
            method, endpoint, data = test
            
        if test_endpoint(method, endpoint, data):
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
