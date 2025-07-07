#!/usr/bin/env python3
"""
Test script to verify prompt template functionality fixes.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.services.prompt_manager import PromptManager
from app.models.prompts import MediaType, PromptTemplateRequest, PromptConfigurationRequest


async def test_prompt_functionality():
    """Test the prompt template functionality."""
    print("🧪 Testing Prompt Template Functionality")
    print("=" * 50)
    
    try:
        # Initialize prompt manager
        print("1. Initializing PromptManager...")
        prompt_manager = PromptManager()
        
        # Test creating a custom prompt
        print("2. Creating a test Chinese prompt...")
        chinese_prompt_request = PromptTemplateRequest(
            name="Test Chinese Prompt",
            description="A test prompt with Chinese instructions",
            media_type=MediaType.IMAGE,
            prompt_text="""请分析这张图片并用中文提供JSON响应：

{
  "description": "详细描述图片内容（2-3句话）",
  "objects": ["关键", "物体", "列表"],
  "setting": "环境或场景描述",
  "mood": "氛围或情感基调",
  "colors": "主要颜色和视觉风格",
  "tags": ["相关", "描述性", "关键词"]
}

重要：只返回有效的JSON格式，不要包含markdown代码块或其他解释文字。"""
        )
        
        created_template = await prompt_manager.create_template(chinese_prompt_request)
        print(f"   ✅ Created template: {created_template.name} (ID: {created_template.template_id})")
        
        # Test setting it as active
        print("3. Setting Chinese prompt as active...")
        config_request = PromptConfigurationRequest(
            active_image_prompt_id=created_template.template_id
        )
        
        updated_config = await prompt_manager.update_active_configuration(config_request)
        print(f"   ✅ Updated configuration - Active image prompt: {updated_config.active_image_prompt_id}")
        
        # Test retrieving active prompt text
        print("4. Testing active prompt retrieval...")
        active_prompt_text = await prompt_manager.get_active_prompt_text(MediaType.IMAGE)
        
        if active_prompt_text:
            print(f"   ✅ Retrieved active prompt (length: {len(active_prompt_text)} chars)")
            print(f"   📝 Prompt preview: {active_prompt_text[:100]}...")
            
            # Check if it contains Chinese characters
            if any('\u4e00' <= char <= '\u9fff' for char in active_prompt_text):
                print("   ✅ Prompt contains Chinese characters")
            else:
                print("   ❌ Prompt does not contain Chinese characters")
        else:
            print("   ❌ Failed to retrieve active prompt text")
            
        # Test persistence by creating a new prompt manager instance
        print("5. Testing persistence with new PromptManager instance...")
        new_prompt_manager = PromptManager()
        
        persistent_config = await new_prompt_manager.get_active_configuration()
        print(f"   📋 Persistent config - Image prompt ID: {persistent_config.active_image_prompt_id}")
        
        if persistent_config.active_image_prompt_id == created_template.template_id:
            print("   ✅ Configuration persisted correctly")
        else:
            print("   ❌ Configuration did not persist")
            
        persistent_prompt_text = await new_prompt_manager.get_active_prompt_text(MediaType.IMAGE)
        if persistent_prompt_text and persistent_prompt_text == active_prompt_text:
            print("   ✅ Active prompt text persisted correctly")
        else:
            print("   ❌ Active prompt text did not persist")
            
        print("\n🎉 Prompt functionality test completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


async def test_ui_layout():
    """Test UI layout with long template names."""
    print("\n🎨 Testing UI Layout with Long Names")
    print("=" * 50)
    
    try:
        prompt_manager = PromptManager()
        
        # Create a template with a very long name
        long_name_request = PromptTemplateRequest(
            name="This is a very long prompt template name that should test the UI layout and ensure the edit button remains visible even with extremely long names that might cause overflow issues",
            description="A test prompt to verify UI layout handling",
            media_type=MediaType.VIDEO_FRAME,
            prompt_text="Test prompt for UI layout verification."
        )
        
        created_template = await prompt_manager.create_template(long_name_request)
        print(f"✅ Created template with long name (length: {len(created_template.name)} chars)")
        print(f"📝 Template name: {created_template.name}")
        
        # List all templates to verify it appears correctly
        templates = await prompt_manager.list_templates()
        long_template = next((t for t in templates if t.template_id == created_template.template_id), None)
        
        if long_template:
            print("✅ Template appears in list correctly")
            print("💡 Check the frontend UI to verify the edit button is visible")
        else:
            print("❌ Template not found in list")
            
    except Exception as e:
        print(f"❌ UI layout test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_prompt_functionality())
    asyncio.run(test_ui_layout())
