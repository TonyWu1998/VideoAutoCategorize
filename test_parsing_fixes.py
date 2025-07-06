#!/usr/bin/env python3
"""
Test script to verify the LLM response parsing fixes.
This script tests the parsing logic with various response formats.
"""

import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.services.llm_service import LLMService


def test_response_parsing():
    """Test the response parsing with various malformed responses."""
    
    # Initialize LLM service (just for the parsing methods)
    llm_service = LLMService()
    
    # Test cases with different malformed responses
    test_cases = [
        {
            "name": "Markdown wrapped JSON",
            "response": '''```json
{
  "description": "A beautiful sunset over the ocean",
  "objects": ["sun", "ocean", "clouds"],
  "setting": "beach",
  "mood": "peaceful",
  "tags": ["sunset", "ocean", "peaceful"]
}
```''',
            "expected_valid": True
        },
        {
            "name": "JSON with prefix text",
            "response": '''Here's the JSON analysis of the image you provided:
{
  "description": "A busy city street with cars and people",
  "objects": ["cars", "people", "buildings"],
  "setting": "urban street",
  "mood": "busy",
  "tags": ["city", "street", "urban", "busy"]
}''',
            "expected_valid": True
        },
        {
            "name": "Malformed response with partial JSON",
            "response": '''```json ```json Okay, here's the JSON analysis:
{
  "description": "A forest scene with tall trees"''',
            "expected_valid": False
        },
        {
            "name": "Plain text response",
            "response": '''This image shows a beautiful landscape with mountains in the background. The scene is very peaceful and serene.''',
            "expected_valid": True  # Should use unstructured parsing
        },
        {
            "name": "Empty response",
            "response": '',
            "expected_valid": False
        },
        {
            "name": "Valid JSON without markdown",
            "response": '''{
  "description": "A cat sitting on a windowsill",
  "objects": ["cat", "windowsill", "window"],
  "setting": "indoor",
  "mood": "calm",
  "tags": ["cat", "indoor", "calm"]
}''',
            "expected_valid": True
        }
    ]
    
    print("🧪 Testing LLM Response Parsing Fixes")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print(f"📝 Input: {test_case['response'][:100]}...")
        
        try:
            # Test the parsing
            result = llm_service._parse_llm_response(test_case['response'])
            
            # Validate the result
            is_valid = llm_service._validate_analysis_result(result)
            
            print(f"📝 Parsed result: {result}")
            print(f"📝 Validation result: {is_valid}")
            print(f"📝 Expected valid: {test_case['expected_valid']}")
            
            if is_valid == test_case['expected_valid']:
                print(f"✅ Test PASSED")
            else:
                print(f"❌ Test FAILED - Expected {test_case['expected_valid']}, got {is_valid}")
                
        except Exception as e:
            print(f"❌ Test FAILED with exception: {e}")
    
    print(f"\n🏁 Parsing tests completed!")


def test_cleaning_logic():
    """Test the response cleaning logic specifically."""
    
    llm_service = LLMService()
    
    test_cases = [
        {
            "input": "```json\n{\"test\": \"value\"}\n```",
            "expected": "{\"test\": \"value\"}"
        },
        {
            "input": "Here's the analysis:\n{\"description\": \"test\"}",
            "expected": "{\"description\": \"test\"}"
        },
        {
            "input": "```json ```json {\"nested\": \"markdown\"}",
            "expected": "{\"nested\": \"markdown\"}"
        }
    ]
    
    print("\n🧹 Testing Response Cleaning Logic")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧹 Clean Test {i}:")
        print(f"🧹 Input: {test_case['input']}")
        
        cleaned = llm_service._clean_llm_response(test_case['input'])
        print(f"🧹 Cleaned: {cleaned}")
        print(f"🧹 Expected: {test_case['expected']}")
        
        if cleaned == test_case['expected']:
            print(f"✅ Clean test PASSED")
        else:
            print(f"❌ Clean test FAILED")


if __name__ == "__main__":
    test_response_parsing()
    test_cleaning_logic()
