#!/usr/bin/env python3
"""
Debug script to check current prompt configuration.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

try:
    from app.services.prompt_manager import PromptManager
    from app.models.prompts import MediaType
    from app.config import settings
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


async def debug_prompt_configuration():
    """Debug the current prompt configuration."""
    print("🔍 Debugging Prompt Configuration")
    print("=" * 50)
    
    try:
        # Initialize prompt manager
        print("1. Initializing PromptManager...")
        prompt_manager = PromptManager()
        
        # Get current active configuration
        print("2. Getting active configuration...")
        config = await prompt_manager.get_active_configuration()
        
        print(f"   📋 Active image prompt ID: {config.active_image_prompt_id}")
        print(f"   📋 Active video prompt ID: {config.active_video_prompt_id}")
        print(f"   📋 Last updated: {config.last_updated}")
        print(f"   📋 Updated by: {config.updated_by}")
        
        # Check if active prompts exist
        if config.active_image_prompt:
            print(f"   ✅ Active image prompt: {config.active_image_prompt.name}")
            print(f"      📝 Preview: {config.active_image_prompt.prompt_text[:100]}...")
        else:
            print("   ❌ No active image prompt configured")
            
        if config.active_video_prompt:
            print(f"   ✅ Active video prompt: {config.active_video_prompt.name}")
            print(f"      📝 Preview: {config.active_video_prompt.prompt_text[:100]}...")
        else:
            print("   ❌ No active video prompt configured")
        
        # List all available templates
        print("\n3. Listing all available templates...")
        templates = await prompt_manager.list_templates()
        
        image_templates = [t for t in templates if t.media_type == MediaType.IMAGE.value]
        video_templates = [t for t in templates if t.media_type == MediaType.VIDEO_FRAME.value]
        
        print(f"   📊 Total templates: {len(templates)}")
        print(f"   🖼️  Image templates: {len(image_templates)}")
        print(f"   🎬 Video frame templates: {len(video_templates)}")
        
        print("\n   Image Templates:")
        for template in image_templates:
            active_marker = "🟢" if template.is_active else "⚪"
            print(f"   {active_marker} {template.name} (ID: {template.template_id})")
            
        print("\n   Video Frame Templates:")
        for template in video_templates:
            active_marker = "🟢" if template.is_active else "⚪"
            print(f"   {active_marker} {template.name} (ID: {template.template_id})")
        
        # Test prompt retrieval
        print("\n4. Testing prompt retrieval...")
        image_prompt = await prompt_manager.get_active_prompt_text(MediaType.IMAGE)
        video_prompt = await prompt_manager.get_active_prompt_text(MediaType.VIDEO_FRAME)
        
        if image_prompt:
            print(f"   ✅ Image prompt retrieved (length: {len(image_prompt)} chars)")
            # Check if it contains Chinese characters
            if any('\u4e00' <= char <= '\u9fff' for char in image_prompt):
                print("   🇨🇳 Image prompt contains Chinese characters")
            else:
                print("   🇺🇸 Image prompt is in English")
        else:
            print("   ❌ No image prompt retrieved")
            
        if video_prompt:
            print(f"   ✅ Video prompt retrieved (length: {len(video_prompt)} chars)")
            # Check if it contains Chinese characters
            if any('\u4e00' <= char <= '\u9fff' for char in video_prompt):
                print("   🇨🇳 Video prompt contains Chinese characters")
            else:
                print("   🇺🇸 Video prompt is in English")
        else:
            print("   ❌ No video prompt retrieved")
            
        # Check current model setting
        print(f"\n5. Current model configuration:")
        print(f"   🤖 OLLAMA_MODEL setting: {settings.OLLAMA_MODEL}")
        
        print("\n🎉 Prompt configuration debug completed!")
        
    except Exception as e:
        print(f"❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_prompt_configuration())
